# Paper 1 Long-Form Measurement Inventory

This note preserves the additional measured rows that appeared in the long Paper 1 draft but are intentionally not all treated as scaling-law calibration anchors in the short paper.

## Included In The Short Paper Fit

- Modern local decoder calibration rows in `artifact/data/models.csv` and `artifact/results/raw/core_measurements_v2.csv`.
- Variance-verified current-build anchors for Pythia-70M, SmolLM2-135M, Gemma3-270M-IT, Qwen 2.5 0.5B, and Mamba-370M.

## Network And Deployment Measurements

- ICP mainnet, Qwen 2.5 0.5B Q8_0, same canister and same weights:
  - Our fork: 29 generation tokens/call, `gen@29 OK`, `gen@30 TRAP`, WASM `ef8f9d78...`.
  - onicai baseline: 10 generation tokens/call, `gen@10 OK`, `gen@11 TRAP`, WASM `6c77a958...`.
  - This is the production 2.9x software-gap result.
- Swiss Subnet rows:
  - SmolLM2-135M Q4_0: 97 tok/call, same-band network validation.
  - Gemma3-270M-IT Q4_0: 52 tok/call, -7% vs current local.
  - Qwen 2.5 0.5B Q4_0: 30 tok/call, -3% vs current local.
  - Falcon-H1-Tiny-90M Q8_0: 199 tok/call, SSN-only build-specific observation.
  - Qwen3.5 0.8B Q4_0: 13 tok/call, best-quality SSN decoder row, but crashes ICP mainnet.
  - Qwen3.5 ternary/pruned/IQ2 variants: throughput valid, quality invalid or not checked; kept as negative results.
  - EmbeddingGemma-300M Q4_0 encoder: 126 input tokens/call, systems demonstration.

Primary files: `artifact/data/onchain/icp_mainnet.csv`, `artifact/data/onchain/ssn_mainnet.csv`, and `artifact/results/raw/core_measurements_v2.csv`.

## Qwen 2.5 0.5B Canister Characterization

These rows explain why the optimized canister path matters, but they are not additional calibration anchors.

| Measurement | Value | Status |
| ----------- | ----: | ------ |
| Q8_0 SIMD, local canister | 30 tok/call | Supporting measurement |
| Q8_0 scalar, local canister | 6 tok/call | Supporting measurement |
| F16, local canister | ~3 tok/call | Supporting measurement |
| Decode profile total | 12,327,112,064 instructions / 10 tokens | Long-form profile |
| Decode profile `MUL_MAT` | 12,117,609,078 instructions, 98.3% | Long-form profile |
| Non-matmul work | 209,502,986 instructions, 1.7% | Long-form profile |
| α_eff SIMD | 1.23 | Derived from profile |
| α_eff scalar | ~6.75 | Estimated scalar comparison |
| Prompt ~5 tokens | 0 prefill calls, ~28-30 gen tok/call | Prefill characterization |
| Prompt ~50 tokens | 2 prefill calls, ~20 gen tok/call | Prefill characterization |
| Prompt ~120 tokens | 4 prefill calls, ~10-17 gen tok/call | Prefill characterization |

## Interpretation Rule

For review, keep four evidence classes separate:

1. Calibration rows: used to fit or validate the empirical law.
2. Network rows: confirm portability or expose deployment-specific behavior.
3. Negative rows: measured throughput that is not usable evidence for quality or architecture claims.
4. Canister-characterization rows: explain where instructions go and why prompt length changes latency.
