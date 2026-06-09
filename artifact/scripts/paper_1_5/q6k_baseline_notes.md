# Q6_K WASM SIMD baseline analysis (Task 24)

Source : `llama_cpp_canister/src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/quants.c:1097-1195`, function `ggml_vec_dot_q6_K_q8_K`.

## Block layout (Q6_K, 256 elements per block, 6 bpw)

For `block_q6_K` in upstream ggml:
- `ql[128]`: low 4 bits of each weight (packed 2 weights per byte; 256 weights / 2 = 128 bytes) — but the WASM path only indexes `q4[0..63]` twice per block (see structure below)
- `qh[64]`: high 2 bits of each weight (packed 4 weights per byte; 256 weights / 4 = 64 bytes) — WASM path uses `qh[0..31]` twice per block
- `scales[16]`: per-16-element int8 scales
- `d`: block delta (fp16)

**Actual sizes used** per `j`-iteration (128 output weights):
- `q4`: 64 bytes consumed (indices 0–63 in first iter, 64–127 in second)
- `qh`: 32 bytes consumed
- Output `aux8`: 128 bytes produced (aux8[0..127] in first iter, aux8[128..255] in second)

## Current implementation structure

```
ggml_vec_dot_q6_K_q8_K(n, s, vx, vy):
    for each block i in [0, nb):
        # PHASE 1: Unpack 6-bit weights → int8 aux8[256]   (SCALAR, bottleneck)
        for j in [0, 256) step 128:
            for l in [0, 32):
                aux8[l+0]  = ((q4[l+0]  & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
                aux8[l+32] = ((q4[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
                aux8[l+64] = ((q4[l+0]  >> 4)  | ((qh[l] >> 4) & 3) << 4) - 32
                aux8[l+96] = ((q4[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4) - 32
            aux8 += 128; q4 += 64; qh += 32

        # PHASE 2: dot(aux8, q8) with per-16 scaling             (SIMD, already optimized)
        for j in [0, 16):
            scale = x[i].scales[j]
            a_vec = load(aux8 + j*16)      # 16 int8
            q8_vec = load(q8 + j*16)       # 16 int8
            # extend i8→i16 low/high, multiply, extend i16→i32, multiply by scale, accumulate
            acc0,acc1 += dot(a_vec, q8_vec) * scale

        sumf += fp16_to_fp32(d) * sum(acc0 + acc1)
    *s = sumf
```

## Bottleneck: Phase 1 unpack is fully scalar

- **32 × 4 = 128 stores per j-iteration** (one for each of `aux8[l+0]`, `aux8[l+32]`, `aux8[l+64]`, `aux8[l+96]`)
- Each store requires an int8 compute with 5 scalar operations (AND, shift+AND, shl, OR, SUB)
- Per block: 2 × 32 × 4 = 256 scalar stores (QK_K/128 × 32 × 4)
- No SIMD intrinsics used in the unpack — direct byte arithmetic

## Opportunities for vectorization (Task 26 target)

Process 16 `l`-iterations at once using `v128_t`:

1. **Load**: `wasm_v128_load(q4 + offset)` → 16 bytes; similarly for `qh`
2. **Extract low nibbles**: `wasm_v128_and(q4_vec, splat(0x0F))`
3. **Extract high nibbles**: `wasm_u8x16_shr(q4_vec, 4)` (unsigned shift to avoid sign bit pollution)
4. **Extract qh bits 0-1**: `wasm_v128_and(qh_vec, splat(0x03))`
5. **Extract qh bits 2-3**: `wasm_v128_and(wasm_u8x16_shr(qh_vec, 2), splat(0x03))`
6. **Extract qh bits 4-5**: `wasm_v128_and(wasm_u8x16_shr(qh_vec, 4), splat(0x03))`
7. **Extract qh bits 6-7**: `wasm_u8x16_shr(qh_vec, 6)` (only 2 bits left, no mask needed)
8. **Shift qh bits left by 4**: `wasm_i8x16_shl(qh_bits, 4)` — makes them occupy the upper 4 bits
9. **Combine with q4 nibbles**: `wasm_v128_or(q4_nibbles, qh_shifted)`
10. **Subtract bias 32**: `wasm_i8x16_sub(combined, splat(32))`
11. **Store**: `wasm_v128_store(aux8 + offset, result)`

**Expected layout per j-iteration (128 outputs):**
- Chunk 1 (`l = 0..15`): loads `q4[0..15]`, `q4[32..47]`, `qh[0..15]` → stores `aux8[0..15]`, `aux8[32..47]`, `aux8[64..79]`, `aux8[96..111]`
- Chunk 2 (`l = 16..31`): loads `q4[16..31]`, `q4[48..63]`, `qh[16..31]` → stores `aux8[16..31]`, `aux8[48..63]`, `aux8[80..95]`, `aux8[112..127]`

**Op count per j-iteration (128 outputs):**
- Scalar baseline: 32 × 4 stores + ~5 ops each = 640 ops
- SIMD proposal: 2 × (3 loads + ~20 SIMD ops + 4 stores) = 54 ops
- Expected speedup on this phase: ~12×

**Speedup on overall Q6_K dot product:** less than 12× because Phase 2 (already SIMD) is a significant fraction of the runtime. A reasonable estimate is **+15-25%** on the whole `ggml_vec_dot_q6_K_q8_K` function.

## Determinism constraints for ICP

All proposed intrinsics are **strict SIMD** (not relaxed-simd). Specifically avoid:
- `wasm_*_relaxed_*` family (implementation-defined behavior, non-deterministic)
- `i32x4.relaxed_dot_i8x16_i7x16` (signed/unsigned ambiguity)
- `f32x4.relaxed_madd` (FMA double/single rounding)

All of `wasm_v128_load/store`, `wasm_i8x16_*`, `wasm_u8x16_shr`, `wasm_v128_and/or/not` produce bit-exact deterministic output across implementations.

## Reference implementations for comparison

The same Q6_K unpack is vectorized on other architectures (already in upstream ggml). For inspiration:
- `arch/x86/quants.c` (AVX2, 256-bit lanes — we can't copy directly, but the byte-manipulation pattern transfers)
- `arch/arm/quants.c` (NEON 128-bit — closest match to WASM SIMD128 since both are 128-bit-wide)
- `arch/riscv/quants.c:1745` (RVV — reference for the reduction pattern)

The NEON version is the best pattern to study for the unpack logic (same 128-bit vector width, similar byte intrinsics).

## Measurement plan (Tasks 25-28)

1. **Task 25** — Microbench:
   - Generate deterministic random Q6_K and Q8_K blocks (seed=42)
   - Call `ggml_vec_dot_q6_K_q8_K` in a tight loop (N=10000)
   - Record ns/iter
   - Baseline before patch

2. **Task 26** — Implement vectorized unpack (this file, function body, only the `#if defined __wasm_simd128__` branch's unpack phase)

3. **Task 27** — Bit-exact regression test:
   - Same seed, same inputs
   - Compare output float `*s` to baseline stored value
   - Must match to full precision (deterministic integer math → bit-exact)

4. **Task 28** — Re-run microbench with patched kernel, compute speedup

## Non-goals

- NOT modifying Phase 2 (already SIMD-optimized)
- NOT changing the `#else` scalar fallback (generic kernel remains untouched)
- NOT introducing relaxed SIMD
- NOT changing the function signature or call interface
