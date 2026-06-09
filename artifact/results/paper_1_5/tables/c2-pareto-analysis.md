# Paper 1.5 Phase 2 C2 — Pareto analysis (SmolLM2-135M)

## Methodology

Nine GGUF variants of SmolLM2-135M (6 mixed-precision + 3 uniform baselines Q4_0 /
Q5_K_M / Q8_0) were evaluated on three objectives: (1) on-disk size in MB,
(2) WikiText-2 perplexity at `n_ctx=512` computed with upstream `llama-perplexity`,
and (3) tok/call on ICP local replica (`n_3_mean`, binary-search N_MAX, 3 reps at
max). We compute the Pareto frontier with dominance defined as: variant A dominates B
iff A is <= B on size and PPL AND A >= B on tok/call, with strict inequality on at
least one axis. The BASE-Q5_K_M baseline is the only dominated point. The following
table is sorted by tok/call (descending). C4 PPL is shown for sanity; the frontier is
computed on WikiText-2 only. F16 reference PPL on WikiText-2 is 13.574 (from Task 5),
so BASE-Q8_0 sits within one stderr of the floor.

## Results table

| variant | size_MB | tok/call | PPL_WT2 | PPL_C4 | Pareto |
|---------|--------:|---------:|--------:|-------:|:------:|
| BASE-Q8_0 | 144.8 | 97 | 13.603 | 20.007 | yes |
| BASE-Q4_0 | 91.7 | 95 | 17.208 | 25.464 | yes |
| V3c | 121.1 | 82 | 13.892 | 20.390 | yes |
| V3b | 115.6 | 80 | 13.946 | 20.494 | yes |
| V4 | 113.0 | 78 | 13.853 | 20.395 | yes |
| V2 | 111.1 | 77 | 13.858 | 20.369 | yes |
| V3a | 110.1 | 77 | 13.965 | 20.560 | yes |
| V1 | 105.5 | 74 | 14.008 | 20.563 | yes |
| BASE-Q5_K_M | 112.1 | 66 | 14.081 | 20.610 | - |

## Pareto frontier (practical winners)

Strict 3-objective Pareto has 8/9 non-dominated variants — each mixed variant stakes
out a unique trade-off somewhere between the two uniform extremes. To rank them for
deployment we collapse to a 2-axis read (size vs `tok/call * (F16_PPL / PPL_WT2)`):

1. **BASE-Q8_0 — overall winner**: 97 tok/call (highest), 13.603 PPL (within 0.03 of
   F16 floor at 13.574), 144.8 MB. Pays +59% size vs Q4_0 for near-lossless quality
   AND the fastest throughput. On a 135M model the 144 MB is still comfortably within
   the 1 GB Q6_K canister ceiling.
2. **BASE-Q4_0 — size-constrained winner**: 91.7 MB (smallest), 95 tok/call (nearly as
   fast as Q8_0), but PPL +3.61 over Q8_0 (+26.5% relative) — quality is visibly
   degraded. Only justified when memory headroom is binding (multi-model canister,
   vault co-residency, etc.).
3. **V4 — best mixed variant** (informational): 113.0 MB, 78 tok/call, 13.853 PPL.
   Uses top-6 ΔPPL layers (11, 17, 19, 20, 28, 29) as Q8_0 and the rest as Q4_K_M. It
   recovers 92% of the Q8_0→Q4_0 quality gap for only +23 MB, but loses 19 tok/call
   (-19.6%) to BASE-Q8_0 — the Q4_K_M kernel fallbacks (hidden_dim 576 not divisible
   by 256) cost more inference cycles than the Q8_0 "keep" layers save.

## Dominated variants

- **BASE-Q5_K_M** (66 tok/call, 14.081 PPL, 112.1 MB) is strictly dominated by
  **V3a** (77 tok/call, 13.965 PPL, 110.1 MB) on all three axes — a clean
  demonstration that Q5_K_M's quantization blocksize constraint (ncols must be
  divisible by 256) forces 181 fallback lines for SmolLM2-135M's 576-wide tensors
  and collapses throughput.

Within the mixed family, no variant strictly dominates another because each
sacrifices some axis for another (e.g. V3c buys +2 PPL-points-below-V3b at the cost
of +5.5 MB and only +2 tok/call). All mixed variants lose to BASE-Q8_0 on tok/call
AND PPL; they only win on size — which is rarely the binding constraint on a 135M
model deployed in a 1 GB ICP canister.

## Findings (headline)

1. **BASE-Q8_0 is the clear practical winner on SmolLM2-135M.** Best quality, best
   throughput, and the 145 MB footprint is negligible on ICP. The mixed-precision
   hypothesis (trade a few PPL-sensitive Q8 layers against mostly-Q4 bulk to beat
   uniform Q8_0 on size-adjusted quality) is not vindicated here.
2. **Mixed precision on SmolLM2-135M costs tok/call, not saves it.** Every mixed
   variant is slower than both BASE-Q4_0 (-17 to -21 tok/call) and BASE-Q8_0 (-15
   to -23 tok/call). Root cause: `hidden_dim=576` is not divisible by 256, so
   Q4_K_M blocks fall back to a non-SIMD path (91-181 warn lines during quantize)
   — the fallback penalty swamps the few Q8_0 layers' cost.
3. **Q4_0 remains the size-performance sweet spot when quality is slack.** Only 2
   fallback warnings (embedding + tied output) because Q4_0 uses 32-element blocks
   that divide 576. 91.7 MB, 95 tok/call, fully SIMD. The PPL hit is the cost.
4. **ΔPPL-guided layer selection (V4) beats naive first-k (V3a/b/c).** V4 has the
   lowest mixed-variant PPL (13.853) and the second-smallest mixed footprint, but
   it still loses to BASE-Q8_0 because the Q4_K_M penalty is structural, not
   per-layer. The ΔPPL methodology is sound; the format choice is the problem.
5. **Paper framing (for §C2 discussion):** "Mixed precision is conditional on the
   base format already being SIMD-efficient on the target hardware's kernel. On
   models whose hidden dimension is not divisible by the Q*_K blocksize (256),
   uniform Q8_0 dominates the mixed frontier on both quality and throughput."

## Next steps

Qwen 0.5B comparison pending Task 10; SSN cross-env pending Task 12.
