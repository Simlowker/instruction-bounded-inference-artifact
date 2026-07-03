#if defined(__wasm_simd128__)

#include <wasm_simd128.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"

/* ---------------------------------------------------------------------------
 * Task 1 — SIMD helper functions
 * -------------------------------------------------------------------------*/

/*
 * wasm_fast_expf: fast SIMD exp approximation via 2^(x * log2e).
 *
 * Decompose x*log2e = n + f, n integer, f in [-0.5, 0.5).
 * Approximate 2^f with degree-4 minimax polynomial; reconstruct 2^n by
 * injecting n+127 into the IEEE-754 exponent field.
 *
 * Max relative error ≈ 4e-5 — sufficient for softmax attention weights.
 */
static inline v128_t wasm_fast_expf(v128_t x) {
    const v128_t log2e = wasm_f32x4_splat(1.4426950408889634f);
    const v128_t one   = wasm_f32x4_splat(1.0f);
    /* Rounding bias: 1.5 * 2^23 */
    const v128_t bias  = wasm_f32x4_splat(12582912.0f);
    /* Degree-4 polynomial for 2^f - 1 - f on [-0.5, 0.5]:
       p(f) ≈ c2*f^2 + c3*f^3 + c4*f^4                        */
    const v128_t c2 = wasm_f32x4_splat(0.24022650695f);
    const v128_t c3 = wasm_f32x4_splat(0.05550410866f);
    const v128_t c4 = wasm_f32x4_splat(0.00961864782f);

    /* t = x * log2e */
    v128_t t = wasm_f32x4_mul(x, log2e);

    /* n_f = round(t) via bias trick; f = t - n_f */
    v128_t n_f = wasm_f32x4_sub(wasm_f32x4_add(t, bias), bias);
    v128_t f   = wasm_f32x4_sub(t, n_f);

    /* p = 2^f ≈ 1 + f + f^2*(c2 + f*(c3 + f*c4)) */
    v128_t p = wasm_f32x4_add(c3, wasm_f32x4_mul(f, c4));
    p = wasm_f32x4_add(c2, wasm_f32x4_mul(f, p));
    p = wasm_f32x4_mul(wasm_f32x4_mul(f, f), p); /* f^2*(c2+..) */
    p = wasm_f32x4_add(wasm_f32x4_add(one, f), p);

    /* 2^n: build IEEE-754 float with exponent = n+127, mantissa = 0 */
    v128_t ni    = wasm_i32x4_trunc_sat_f32x4(n_f);
    v128_t pow2n = wasm_i32x4_shl(wasm_i32x4_add(ni, wasm_i32x4_splat(127)), 23);

    return wasm_f32x4_mul(p, pow2n);
}

/*
 * wasm_f16x4_load_f32: load 4 consecutive ggml_fp16_t values → F32x4.
 *
 * ggml_fp16_to_fp32() is the public GGML API (ggml.h), safe to call from C.
 */
// F16→F32 via lookup table (inline, no function call overhead).
// The GGML_CPU_FP16_TO_FP32 macro uses ggml_table_f32_f16[] — fast and correct for
// all F16 values including denormals, inf, NaN, zero.
static inline v128_t wasm_f16x4_load_f32(const ggml_fp16_t * p) {
    return wasm_f32x4_make(
        GGML_CPU_FP16_TO_FP32(p[0]),
        GGML_CPU_FP16_TO_FP32(p[1]),
        GGML_CPU_FP16_TO_FP32(p[2]),
        GGML_CPU_FP16_TO_FP32(p[3])
    );
}

// Inline F16×F16 dot product — fully SIMD, no function calls.
// Unrolled ×2 (8 F16 per iteration) for DK=64 typical case.
static inline float wasm_dot_f16(int64_t n, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    v128_t sum0 = wasm_f32x4_splat(0.0f);
    v128_t sum1 = wasm_f32x4_splat(0.0f);

    int64_t i = 0;
    for (; i <= n - 8; i += 8) {
        v128_t xv0 = wasm_f16x4_load_f32(x + i);
        v128_t yv0 = wasm_f16x4_load_f32(y + i);
        sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(xv0, yv0));

        v128_t xv1 = wasm_f16x4_load_f32(x + i + 4);
        v128_t yv1 = wasm_f16x4_load_f32(y + i + 4);
        sum1 = wasm_f32x4_add(sum1, wasm_f32x4_mul(xv1, yv1));
    }
    for (; i <= n - 4; i += 4) {
        v128_t xv = wasm_f16x4_load_f32(x + i);
        v128_t yv = wasm_f16x4_load_f32(y + i);
        sum0 = wasm_f32x4_add(sum0, wasm_f32x4_mul(xv, yv));
    }

    sum0 = wasm_f32x4_add(sum0, sum1);
    float s = wasm_f32x4_extract_lane(sum0, 0) + wasm_f32x4_extract_lane(sum0, 1)
            + wasm_f32x4_extract_lane(sum0, 2) + wasm_f32x4_extract_lane(sum0, 3);

    for (; i < n; i++) {
        s += GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]);
    }
    return s;
}

/* ---------------------------------------------------------------------------
 * Task 2 — SIMD attention inner loop
 * -------------------------------------------------------------------------*/

/*
 * ggml_fa_f16_vec_dot_simd
 *
 * Online softmax + weighted V accumulation for Flash Attention with F16 V.
 * VKQ is a F32 accumulator — no precision loss from repeated F16 round-trips.
 *
 * Parameters
 * ----------
 *   DV         : value head dimension (e.g. 64 or 128)
 *   mask_row   : F16 mask row for current Q position, or NULL
 *   slope      : ALiBi slope scalar
 *   scale      : QK scale scalar (already fused with logit_softcap if needed)
 *   K_data     : byte pointer to K tensor base
 *   nbk1       : byte stride between consecutive K vectors
 *   V_data     : byte pointer to V tensor base
 *   nbv1       : byte stride between consecutive V vectors
 *   Q_q        : query, converted to K's vec_dot type
 *   DK         : key head dimension
 *   ic_start   : first KV position (inclusive)
 *   ic_end     : last  KV position (exclusive)
 *   kq_vec_dot : Q·K dot product function pointer
 *   VKQ        : [DV] F32 output accumulator (in/out, caller zero-initialised)
 *   M_out      : running maximum (in/out)
 *   S_out      : running softmax sum (in/out)
 */
void ggml_fa_f16_vec_dot_simd(
        int64_t              DV,
        const ggml_fp16_t  * mask_row,
        float                slope,
        float                scale,
        const char         * K_data,
        size_t               nbk1,
        const char         * V_data,
        size_t               nbv1,
        const void         * Q_q,
        int64_t              DK,
        int64_t              ic_start,
        int64_t              ic_end,
        float              * VKQ,
        float              * M_out,
        float              * S_out)
{
    float M = *M_out;
    float S = *S_out;

    for (int64_t ic = ic_start; ic < ic_end; ++ic) {

        /* --- mask --- */
        const float mv = mask_row
            ? slope * GGML_CPU_FP16_TO_FP32(mask_row[ic])
            : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }

        /* --- Q·K dot product — inline SIMD, avoids indirect call overhead --- */
        const ggml_fp16_t * k_ptr = (const ggml_fp16_t *)(K_data + ic * (int64_t)nbk1);
        float s = wasm_dot_f16(DK, (const ggml_fp16_t *)Q_q, k_ptr);

        s = s * scale + mv;

        /* --- online softmax --- */
        const float Mold = M;
        float ms; /* rescale factor for old VKQ */
        float vs; /* weight for current V row   */

        if (s > M) {
            M  = s;
            ms = expf(Mold - M); /* < 1 */
            vs = 1.0f;            /* expf(0) */
        } else {
            ms = 1.0f;
            vs = expf(s - M);
        }

        /* --- rescale existing VKQ (SIMD) when new maximum found --- */
        if (ms != 1.0f) {
            const v128_t ms_vec = wasm_f32x4_splat(ms);
            int64_t d = 0;
            for (; d <= DV - 4; d += 4) {
                v128_t acc = wasm_v128_load(VKQ + d);
                acc = wasm_f32x4_mul(acc, ms_vec);
                wasm_v128_store(VKQ + d, acc);
            }
            for (; d < DV; ++d) {
                VKQ[d] *= ms;
            }
        }

        S = S * ms + vs;

        /* --- SIMD F16→F32 V accumulation --- */
        const ggml_fp16_t * v_ptr =
            (const ggml_fp16_t *)(V_data + ic * (int64_t)nbv1);
        const v128_t vs_vec = wasm_f32x4_splat(vs);

        int64_t d = 0;
        for (; d <= DV - 4; d += 4) {
            v128_t v_f32 = wasm_f16x4_load_f32(v_ptr + d);
            v128_t acc   = wasm_v128_load(VKQ + d);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(v_f32, vs_vec));
            wasm_v128_store(VKQ + d, acc);
        }
        for (; d < DV; ++d) {
            VKQ[d] += ggml_fp16_to_fp32(v_ptr[d]) * vs;
        }
    }

    *M_out = M;
    *S_out = S;
}

#endif /* __wasm_simd128__ */
