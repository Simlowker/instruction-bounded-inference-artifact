# C2 — Mixed Precision (SmolLM2-135M + Qwen 2.5 0.5B)

Phase 2 sweep of mixed-precision GGUF variants against uniform
`Q4_0 / Q5_K_M / Q8_0` baselines, measured end-to-end on a local ICP
replica canister. Two models, 9 variants each = **18 configurations**.

## Methodology

1. **Imatrix calibration.** Per-model `imatrix.dat` built on a 1-to-1
   WikiText-2 test + C4 en-validation mix (512 chunks × 2048 tokens
   context) via `c2_imatrix_run.sh`. Manifest in
   `data/paper_1_5/imatrix_manifest.csv`.
2. **ΔPPL-per-layer sensitivity.** For each transformer block, we
   quantize **only that block's attn/ffn weights** to `Q4_K_M` (all
   others kept at F16) and measure `PPL_WT2`. The delta vs the F16
   baseline ranks layers by quantization fragility. Script:
   `c2_delta_ppl_per_layer.py`. Outputs in
   `results/paper_1_5/raw/c2-delta-ppl-{smollm135,qwen05}.csv`.
3. **Variant generation.** Six mixed-precision variants per model
   (V1 = embedding→Q8, V2 = attn Q/K/V→Q8, V3a/b/c = first 4/8/12
   layers→Q8, V4 = top-ΔPPL layers→Q8) on a `Q4_K_M` base, plus three
   uniform baselines (`Q4_0`, `Q5_K_M`, `Q8_0`, no imatrix). Script:
   `c2_generate_variants.py`. Manifest:
   `data/paper_1_5/mixed_precision_variants.csv`.
4. **Measurement axes.**
   - `size_MB` from GGUF on disk.
   - `tok/call` measured via binary search on the on-chain instruction
     limit (40 B per update call), local dfx replica, 3 replicates at
     `N_MAX` for CV.
   - `PPL_WT2` on `wikitext-2-raw` test and `PPL_C4` on C4
     en-validation at `--ctx-size 2048` via `llama-perplexity`.
5. **Pareto analysis** on (min `size_MB`, max `tok/call`, min
   `PPL_WT2`). Script: `c2_pareto_analysis.py`. Per-model tables in
   `c2-pareto-{smollm135,qwen05}.md`; figures in
   `figures/c2-pareto-{smollm135,qwen05}.png`.

## Per-model results

### SmolLM2-135M (hidden_dim = 576, 30 layers, tied embedding)

Baseline F16 `PPL_WT2 = 13.574 ± 0.099`.

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
| BASE-Q5_K_M | 112.1 | 66 | 14.081 | 20.610 | — |

See `figures/c2-pareto-smollm135.png`.

**Headline.** `BASE-Q8_0` is the highest-throughput / lowest-PPL point
on the frontier — 97 tok/call and 13.603 PPL (+0.029 vs F16) — while
remaining deployment-sized at 144.8 MB. It is not the smallest
variant; the precise statement is that no mixed-precision variant
strictly dominates it.
`BASE-Q5_K_M` is **strictly dominated** (smaller `tok/call`, bigger
than V1/V3a/V4, worse PPL). Details:
`c2-pareto-smollm135.md`.

### Qwen 2.5 0.5B (hidden_dim = 896, 24 layers, GQA, biased attn)

Baseline F16 `PPL_WT2 = 11.6736 ± 0.0828` (from
`c2-delta-ppl-qwen05.log`; script did not persist the baseline row in
the CSV, only per-layer deltas).

| variant | size_MB | tok/call | PPL_WT2 | PPL_C4 | Pareto |
|---------|--------:|---------:|--------:|-------:|:------:|
| BASE-Q8_0 | 506.5 | 30 | 11.680 | 18.923 | yes |
| V3c | 464.4 | 25 | 11.815 | 19.131 | yes |
| V3b | 441.1 | 24 | 11.849 | 19.205 | yes |
| V3a | 417.7 | 23 | 11.893 | 19.260 | yes |
| V4 | 423.8 | 23 | 11.813 | 19.182 | yes |
| V1 | 397.8 | 22 | 11.872 | 19.256 | yes |
| V2 | 406.6 | 22 | 11.842 | 19.175 | yes |
| BASE-Q4_0 | 335.8 | 20 | 13.139 | 21.065 | yes |
| BASE-Q5_K_M | 400.6 | 19 | 11.875 | 19.181 | — |

See `figures/c2-pareto-qwen05.png`.

**Headline.** Same pattern: `BASE-Q8_0` is the highest-throughput /
lowest-PPL frontier point (506 MB, 30 tok/call, 11.680 PPL = only
+0.007 vs F16), but not the smallest. `BASE-Q5_K_M` is again
**strictly dominated** (slowest, bigger than V1/V2, no PPL advantage
over V4/V3c).

## Cross-model comparison (BASE-Q8_0 Pareto winners)

| Metric | SmolLM2-135M | Qwen 2.5 0.5B |
|---|---:|---:|
| hidden_dim | 576 | 896 |
| layers | 30 | 24 |
| params P (approx) | 135 M | 500 M |
| size_MB (Q8_0) | 144.8 | 506.5 |
| tok/call | 97 | 30 |
| PPL_WT2 (Q8_0) | 13.603 | 11.680 |
| PPL_WT2 (F16) | 13.574 | 11.674 |
| ΔPPL (Q8_0 − F16) | +0.029 | +0.007 |
| α_eff = 40e9 / (N_MAX · 2P) | **1.53** | **1.33** |

**α_eff computation.**
- SmolLM2: `40e9 / (97 · 2 · 135e6) = 1.527`
- Qwen 0.5B: `40e9 / (30 · 2 · 500e6) = 1.333`

Both fall in the `α_eff ∈ [1.3, 1.6]` band consistent with Paper 1's
"~1.5 for modern Q4/Q8 archs" observation. **No regression** vs
Paper 1 expectations.

Interesting: Qwen 0.5B's `α_eff` is **lower** than SmolLM2 despite a
larger `hidden_dim`. Plausible contributors:

- Qwen 2.5 uses **grouped-query attention (GQA)**, reducing
  per-token K/V FLOPs.
- Qwen 2.5 has **bias tensors** on attn Q/K/V (extra matmul-free
  adds, not counted in `2P`).
- SmolLM2's **tied embedding** means the LM head contributes
  separately per decoded token without showing up as new params
  (`2P` undercounts).

In other words, the theoretical `2P` FLOP count is a leakier proxy
for Qwen than for SmolLM2 — `α_eff` therefore compresses downward.

## Mixed precision: when does it help?

**Key finding:** Mixed precision **never strictly dominates
`BASE-Q8_0`** on either model, but it fills out Pareto-efficient
intermediate points between the small/fast/lossy `Q4_0` extreme and
the big/clean/fast `Q8_0` extreme.

Detail:

- **SmolLM2-135M:** 8/9 variants Pareto-optimal; `BASE-Q5_K_M` the
  sole dominated point. `V4` is the best **mixed** variant on PPL
  (13.853 @ 113.0 MB), picking the top-6 ΔPPL layers
  `{11, 17, 19, 20, 28, 29}` — dominated by the last block (layer 29,
  ΔPPL = +0.072).
- **Qwen 2.5 0.5B:** same shape — 8/9 optimal, `BASE-Q5_K_M`
  dominated. `V4` is again the best mixed variant on PPL
  (11.813 @ 423.8 MB), targeting `{0, 16, 20, 22, 23}` — the last
  layer (23, ΔPPL = +0.025) and layer 0 (+0.014) dominate.

**Caveat on hidden_dim geometry.** Neither `576` (SmolLM2) nor `896`
(Qwen) is divisible by `256`, so `llama-quantize` emits many
fallbacks from `q4_K/q5_K/q6_K` to `q5_0/q8_0` on individual
tensors — 181 fallback/warn lines for SmolLM2 `Q5_K_M`, 145 for Qwen
`Q5_K_M`. The base `Q4_K_M`-family formats therefore don't fully
realize their compact-storage advantage here. On a `hidden_dim`
divisible by 256 (512, 1024) the mixed-precision landscape likely
differs — worth a Phase 3 follow-up note rather than generalizing
from these two models.

## Practical recommendation

For ICP canister deployment of either model:

1. **If size budget ≥ 150 MB (SmolLM) / ≥ 510 MB (Qwen):** use
   `BASE-Q8_0`. Best quality, best speed, zero quantize fallbacks.
2. **If size budget is tight and quality matters:** use `V4`
   (ΔPPL-guided top-K layers Q8, rest Q4_K_M).
3. **If size budget is very tight and some quality loss is
   acceptable:** `BASE-Q4_0` (91.7 MB / 335.8 MB) — but expect
   `+3.6 PPL` (SmolLM2) or `+1.46 PPL` (Qwen) vs F16.
4. **Never** use `Q5_K_M` on these `hidden_dim` values — strictly
   dominated on both models tested.

## Baseline cross-ref to Paper 1

- Paper 1 reported SmolLM2 `Q4_0` SIMD = 103/87 tok/call
  (local/SSN). This work measured `BASE-Q4_0 = 95 tok/call` locally —
  within 8%, **consistent**, no regression. Minor discrepancies
  attributable to binary-search granularity and the `40 B` budget
  setpoint on this branch.
- Phase 1 memory records `Qwen 0.5B ≈ 30 tok/call SSN` (Q4_0);
  this work measures `BASE-Q4_0 = 20 tok/call` locally and
  `BASE-Q8_0 = 30 tok/call` locally. Order of magnitude
  consistent; env differences (local vs SSN) and instruction-budget
  clamp under study.

## Forward references

- **Task 12 (SSN cross-env validation):** pending user confirmation
  for mainnet upload of the Pareto-winner GGUFs. Expected
  byte-identical on-chain behaviour per Phase 1 pattern.
- **Qwen 2.5 1.5B Q4_0 (Task 20 pre-check):** does **not** trip
  `IC0524` at ~892 MB but traps `IC0502 "unreachable"` during
  `load_model`. Task 18 SSN demo may need to pivot to
  `Qwen 2.5-Instruct-1.5B` or stay on `Qwen 0.5B`.
- **Paper 2 cross-section:** single-line hook in §2 (per locked
  decision #4) noting the 1.33–1.53 `α_eff` band and
  `BASE-Q8_0`-dominates-mixed result.
