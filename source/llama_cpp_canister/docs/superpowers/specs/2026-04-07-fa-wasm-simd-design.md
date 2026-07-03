# Flash Attention WASM SIMD Kernel — Design Spec

**Date:** 2026-04-07
**Goal:** Surpass 103 tok/call on SmolLM2-135M Q4_0 (currently 86 tok/call)
**Target:** ~105-110 tok/call via SIMD optimization of FLASH_ATTN_EXT

## Problem

The upstream llama.cpp uses `FLASH_ATTN_EXT` (fused online-softmax attention) which costs **50M instructions/token** (17.8% of budget at 86 tokens). The kernel is 100% scalar on WASM — no SIMD path exists. The old onicai fork used separate MUL_MAT ops for attention at ~22M/tok. The 28M/tok delta is the primary regression source.

### Profiling Evidence (SmolLM2-135M Q4_0, 50 tokens, n_ctx=256)

| Component | Instructions/tok | % of total |
|-----------|-----------------|------------|
| MUL_MAT (vec_dot SIMD) | 366M | 85.1% |
| **FLASH_ATTN_EXT (scalar)** | **50M** | **11.6%** |
| from_float | 3.8M | 0.9% |
| ROPE + norms + misc | 10M | 2.3% |
| Framework | 0.2M | 0.05% |
| **Total** | **430M** | → **86 tok/call** |

### Target After Optimization

| Component | Instructions/tok | % of total |
|-----------|-----------------|------------|
| MUL_MAT (vec_dot SIMD) | 366M | 91.5% |
| **FLASH_ATTN_EXT (SIMD)** | **~20M** | **~5%** |
| from_float + misc | 14M | 3.5% |
| **Total** | **~400M** | → **~100 tok/call** |

With aggressive expf optimization: **~385M/tok → ~104 tok/call**.

## Architecture

### Location

New file: `src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/flash-attn.c`

Follows the same pattern as `arch/wasm/quants.c` (WASM SIMD kernels).

### Integration Point

In `ops.cpp`, function `ggml_compute_forward_flash_attn_ext_f16_one_chunk` (line 8161).

**Strategy:** Add a `#if defined(__wasm_simd128__)` fast-path at the top of the per-head loop (line 8245) that dispatches to our SIMD kernel when conditions are met:
- `v->type == GGML_TYPE_F16`
- `k->type == GGML_TYPE_F16`
- `DK % 4 == 0` and `DV % 4 == 0` (SIMD alignment)
- `logit_softcap == 0` (no soft-capping, covers SmolLM2/Qwen/most models)
- `sinks == NULL` (no attention sinks, covers most models)

Fallback to existing scalar code when conditions aren't met.

### Kernel Signature

```c
void ggml_fa_f16_one_head_wasm_simd(
    int64_t DK,              // head dim for keys (typically 64)
    int64_t DV,              // head dim for values (typically 64)
    const void * Q_q,        // quantized query [DK] (already converted by caller)
    const char * K_data,     // key cache base pointer
    size_t nbk1,             // stride between KV positions in K
    const char * V_data,     // value cache base pointer (F16)
    size_t nbv1,             // stride between KV positions in V
    const ggml_fp16_t * mask,// attention mask (F16, NULL = no mask)
    float scale,             // attention scale
    float slope,             // ALiBi slope (1.0 if no ALiBi)
    int64_t ic_start,        // KV range start
    int64_t ic_end,          // KV range end
    float * VKQ_out,         // output accumulator [DV] (F32)
    float * M_out,           // output max value
    float * S_out,           // output sum
    ggml_vec_dot_t kq_vec_dot // Q·K dot product function (reuse existing SIMD kernel)
);
```

## Hot-Path Optimization Details

### 1. Q·K Dot Product — Already SIMD (no change)

The `kq_vec_dot` function pointer dispatches to the existing SIMD kernel (e.g., `ggml_vec_dot_q8_0_q8_0` or `ggml_vec_dot_f16`). No optimization needed here.

### 2. Fast expf via SIMD — The Main Win

Replace scalar `expf()` with a SIMD polynomial approximation.

**Algorithm:** Clamped degree-4 polynomial on `[−88, 88]`:

```c
// Fast exp approximation for 4 floats via WASM SIMD
static inline v128_t wasm_fast_expf(v128_t x) {
    // Clamp to avoid overflow/underflow
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-88.0f));
    x = wasm_f32x4_min(x, wasm_f32x4_splat(88.0f));

    // exp(x) = 2^(x * log2(e))
    // Use the integer trick: reinterpret (float_bias + x*log2e) as float
    const v128_t log2e = wasm_f32x4_splat(1.44269504f);
    const v128_t half  = wasm_f32x4_splat(0.5f);
    v128_t t = wasm_f32x4_add(wasm_f32x4_mul(x, log2e), half);

    // Floor via truncation
    v128_t ti = wasm_i32x4_trunc_sat_f32x4(t);
    v128_t tf = wasm_f32x4_convert_i32x4(ti);
    // Correct for negative: if tf > t, subtract 1
    v128_t mask = wasm_f32x4_gt(tf, t);
    ti = wasm_i32x4_sub(ti, wasm_v128_and(mask, wasm_i32x4_splat(1)));
    tf = wasm_f32x4_convert_i32x4(ti);

    // Fractional part: f = x*log2e - floor
    v128_t f = wasm_f32x4_sub(wasm_f32x4_mul(x, log2e), tf);

    // Polynomial approx of 2^f for f in [0, 1)
    // p(f) = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    const v128_t c1 = wasm_f32x4_splat(0.693147180f); // ln(2)
    const v128_t c2 = wasm_f32x4_splat(0.240226507f);
    const v128_t c3 = wasm_f32x4_splat(0.055504109f);
    const v128_t c4 = wasm_f32x4_splat(0.009618129f);
    const v128_t one = wasm_f32x4_splat(1.0f);

    v128_t p = wasm_f32x4_add(wasm_f32x4_mul(c4, f), c3);
    p = wasm_f32x4_add(wasm_f32x4_mul(p, f), c2);
    p = wasm_f32x4_add(wasm_f32x4_mul(p, f), c1);
    p = wasm_f32x4_add(wasm_f32x4_mul(p, f), one);

    // Scale by 2^n: add n to float exponent
    v128_t pow2n = wasm_i32x4_shl(wasm_i32x4_add(ti, wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(p, pow2n);
}
```

**Accuracy:** ~1e-4 relative error, sufficient for softmax (dominated by max subtraction).

### 3. VKQ Accumulation in F32 with SIMD

Instead of accumulating in F16 (`ggml_vec_scale_f16` + `ggml_vec_mad_f16`), work entirely in F32 with SIMD:

```c
// For each KV position ic:
// 1. Compute s = Q·K[ic] (already SIMD via kq_vec_dot)
// 2. Online softmax update (scalar — only 2 values M, S)
// 3. If new max: scale VKQ by ms (SIMD over DV/4 iterations)
// 4. Load V[ic] F16→F32, fused multiply-add VKQ += V * vs (SIMD)
```

For DV=64: 16 SIMD iterations instead of 64 scalar ops for steps 3-4.

### 4. F16 Load with SIMD Widening

```c
// Load 4 F16 values and convert to F32
// WASM SIMD doesn't have native F16, so use the extend trick:
// Load 8 bytes (4 × f16), unpack via f16→f32 conversion
for (int d = 0; d < DV; d += 4) {
    // Load 4 F16 values
    uint64_t v_raw;
    memcpy(&v_raw, v_data_f16 + d, 8);
    v128_t v_f16_bits = wasm_u32x4_make(
        ((uint16_t*)&v_raw)[0], ((uint16_t*)&v_raw)[1],
        ((uint16_t*)&v_raw)[2], ((uint16_t*)&v_raw)[3]);
    // Convert F16 bits to F32 (manual bit manipulation)
    v128_t v_f32 = wasm_f16x4_to_f32x4(v_f16_bits); // helper function

    // FMA: VKQ[d..d+3] += v_f32 * vs_splat
    v128_t vkq = wasm_v128_load(VKQ + d);
    vkq = wasm_f32x4_add(vkq, wasm_f32x4_mul(v_f32, vs_vec));
    wasm_v128_store(VKQ + d, vkq);
}
```

## Implementation Plan

### Files to Create/Modify

1. **CREATE** `src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/flash-attn.c`
   - `wasm_fast_expf()` — SIMD exp approximation
   - `wasm_f16x4_to_f32x4()` — F16→F32 SIMD conversion helper
   - `ggml_fa_f16_one_head_wasm_simd()` — main kernel

2. **MODIFY** `src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/ops.cpp`
   - Add `#include` and dispatch to SIMD kernel in `ggml_compute_forward_flash_attn_ext_f16_one_chunk`

3. **MODIFY** `icpp.toml`
   - Add `arch/wasm/flash-attn.c` to `c_paths`

### Build & Test Sequence

1. Build WASM with new kernel
2. Deploy locally, upload SmolLM2-135M Q4_0
3. Run profiling: compare FA instructions before/after
4. Bisect max tok/call
5. Test Qwen 2.5 0.5B for cross-model validation
6. Verify output quality (same prompt, compare outputs)

## Success Criteria

- [ ] SmolLM2-135M Q4_0: ≥100 tok/call (stretch: ≥105)
- [ ] Qwen 2.5-0.5B Q4_0: ≥27 tok/call (match old fork)
- [ ] FA instructions/tok reduced by ≥50% (from 50M to ≤25M)
- [ ] Output quality: semantic equivalence (FP divergence at token 3+ is OK)
- [ ] No regression on model loading or multi-call stability

## Risks

- **F16↔F32 conversion overhead:** WASM SIMD lacks native F16 support. Bit manipulation may eat some gains. Mitigation: accumulate in F32 throughout, only convert on load.
- **expf accuracy:** Polynomial degree-4 may not be accurate enough for edge cases. Mitigation: clamp inputs, test against scalar output.
- **Conditional branches in online softmax:** The `if (s > M)` branch is data-dependent, can't be vectorized across KV positions. Mitigation: this is per-position (scalar), the SIMD gains are on the DV-dimension vectors.
