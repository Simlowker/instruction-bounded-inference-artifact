# Independent Verification Package — α_eff Refit with Variance Data

**Context for the verifier.** This package is self-contained so that any independent party (human reviewer or automated assistant) can verify the statistical re-fit of an empirical scaling law for LLM inference on the Internet Computer Protocol (ICP). The paper claims:

> **tok/call ≈ B / (α_eff · 2P)** with B = 4 × 10¹⁰ instructions per message, P = total parameter count, α_eff a dimensionless cost multiplier.

This package compares the **baseline fit** (March 2026, peer-review submission) against a **v2 refit** that integrates 5 new measurements each run with 3 repetitions (CV < 1.05% intra-point).

---

## 1. Summary of changes

| Model | Quant | tok/call (baseline, 1 run) | tok/call (v2, 3 reps) | Δ % | α_eff baseline | α_eff v2 |
|---|---|---:|---:|---:|---:|---:|
| Pythia-70M | Q4_0 SIMD | 185 | **208** | +12.4% | 1.54 | **1.374** |
| SmolLM2-135M | Q4_0 SIMD | 103 | **97** | −5.8% | 1.44 | **1.527** |
| Qwen2.5-0.5B | Q4_0 SIMD | 27 | **31** | +14.8% | 1.50 | **1.306** |
| Mamba-370M | Q8_0 SIMD | 35 | **30** | −14.3% | 1.54 | **1.802** |
| Gemma3-270M-IT | Q4_0 SIMD | (not in baseline) | **56** | — | — | **1.323** |

All v2 measurements: 3 runs × 3 prompts each, CV < 1.05%.

## 2. Key statistics: baseline → v2

| Metric | Baseline | v2 | Interpretation |
|---|---|---|---|
| Modern α_eff (median) | 1.54 | **1.527** | Stable |
| Modern 95% BCa CI (median) | [1.50, 1.65] | **[1.374, 1.65]** | Widened (more variance captured) |
| LOAO MAPE, modern only | 4.1% | **8.8%** | Doubled (honest estimate) |
| LOAO max error, modern | 10.7% | **18.3%** (Mamba) | New worst offender |
| α_eff GPT-2 legacy | 1.98 | 1.98 | Stable |
| All-families median | 1.54 | 1.54 | Stable |
| All-families MAPE | 7.9% | 12.1% | Doubled |

## 3. Methodology

- Functional form fit on log-transformed data: log(tok) = log(B) − log(α) − log(2P)
- α_eff = B / (2P · tok) per data point; aggregated via median (primary) and Huber M-estimator (robustness)
- Leave-One-Architecture-Out (LOAO) cross-validation: 9 families (8 modern + GPT-2 legacy)
- Bootstrap 95% CI via BCa, 9999 resamples, seed=42
- Cook's distance threshold = 4/n

## 4. What we want the external verifier to verify

1. **Is the re-fit correct?** α_eff point estimate moves from 1.54 → 1.527 (−0.8%) while CI widens [1.50, 1.65] → [1.374, 1.65]. Is the Bca bootstrap implementation in `analyze_scaling_law.py` sound? Any reason the CI asymmetry should concern us?

2. **Is Mamba-370M's jump from 35 → 30 tok/call (α 1.54 → 1.80) a genuine signal or a build artifact?** The build was updated between measurements; other Transformers sped up while Mamba slowed down. Possible explanation: kernel optimizations favor attention over SSM. How should this be reported?

3. **Is it acceptable to publish with MAPE = 8.8% for the "modern" regime?** The baseline paper claims 4.1%. The honest answer is "both are right, with different N". Propose the phrasing.

4. **Should the paper still separate modern (α≈1.53) from legacy GPT-2 (α≈1.98)?** Sensitivity check: with v2 data, Cook's distance no longer flags GPT-2 as dominant (now Pythia-70M and DistilGPT2 are tied). Is the two-regime framing still justified?

5. **Publication strategy.** We have 13 single-run points + 5 multi-run points. Options:
   - (A) Report v2 with CI [1.37, 1.65], MAPE 8.8%, disclose weak spots
   - (B) Repeat all 13 models with 3 reps before publishing (adds 6-8 hours of canister time)
   - (C) Report v2 headline + baseline as sensitivity
   
   Which is defensible for a cs.DC/cs.LG arXiv submission?

## 5. Files provided

```
artifact/
├── results/raw/
│   ├── core_measurements.csv          # baseline (45 rows, 1 run each)
│   └── core_measurements_v2.csv       # v2 with 5 refitted rows (n_runs=3)
├── data/verified/
│   ├── variance.csv                   # raw 3-rep data for 5 models
│   ├── coherence.csv                  # output quality audit (5/5 coherent except Pythia-70M)
│   └── gguf_metadata.csv              # model metadata verified via GGUFReader
├── results/historical/scaling_law_baseline/
│   ├── alpha_bootstrap_comparison.csv
│   ├── loao_modern_families_summary.csv
│   ├── influence_diagnostics.csv
│   ├── robust_fit.csv
│   └── scaling_law_full_analysis.pdf  # 4-panel figure
├── results/current/scaling_law/       # v2 outputs (same structure)
└── results/current/extended_analysis/
    └── variance_update_log.csv        # diff baseline → v2
```

## 6. Open questions for the reviewer

- The **new** α_eff = 1.527 is almost identical to the baseline 1.54. Is this too convenient? Any methodological concern that bias-toward-prior might be at play?
- Mamba α = 1.80 (v2) is 18% above the modern median. With N=1 SSM point, can we claim generalization to non-Transformer architectures, or is this a warning sign?
- The Bca bootstrap CI moves asymmetrically (lower bound drops more than upper bound rises). Is this expected given the data distribution?

## 7. What the paper ultimately needs to show

A reader should walk away with:
1. Same configuration, more tok/call without quality loss — verified via Q4_0 SIMD = Q8_0 on coherence tests
2. α_eff ≈ 1.5 (modern archs) is a universal constant within ±15% across 11 families
3. Architecture choice (Transformer/Mamba/RNN) matters less than parameter count P
4. Per-architecture optimization (TQ2_0 SIMD, Falcon-H1 hybrid) can beat the modal α when the architecture permits

Your verdict on whether the v2 refit supports this narrative honestly is the primary ask.

---

**Contact:** Siméon Fluck (simeon.autoinfo@gmail.com), Julien Aerni. Paper draft: paper-v8.md.
