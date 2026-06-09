# Instruction-Bounded Inference — Auditable Artifact

Companion artifact for the paper
**On-Chain LLM Inference Under Instruction Budgets: An Instruction-Budget Cost Model, Ternary Floor Evidence, and Session Costs** — Aerni, Fluck & Becker, May 2026 (`../CURRENT.md`).

Running the same Qwen 2.5 0.5B Q8*0 model on the same Internet Computer Protocol (ICP) mainnet canister gives **10 tokens per call on the onicai baseline** and **29 tokens per call on our fork**. Everything in this repository exists to prove \_why* that gap is real, _what law_ governs it, and _how_ you can reproduce and extend the result.

## TL;DR

- 49 benchmark measurements across 10 architecture families on ICP (local dfx, ICP mainnet, Swiss Subnet)
- Cost model calibrated at **tok/call ≈ 40B / (1.53 × 2P)**, strict modern-only LOAO MAPE 7.7%
- Custom **TQ2_0 WASM SIMD kernel** (absent from upstream llama.cpp), 5.03 tok/MB record (Pythia-70M, end-to-end)
- **Encoder case study on-chain**: EmbeddingGemma-300M Q4_0 on Swiss Subnet at α_embed ≈ 0.53, 126 input tokens per call. System-boundary distinction from other "on-chain embedding" offerings (stored pre-computed vectors, off-chain workers, oracle submissions) is developed in the paper — we make no claim of priority.
- **Execution-boundary evidence note**: `notes/blockchain_ai_execution_boundaries.md` documents, with primary sources, which systems are native in-consensus inference versus worker/proof/middleware approaches.
- The main scaling-law figures regenerate from `scripts/analyze_scaling_law.py`; kernel analysis utilities are in `scripts/analyze_kernels.py`.

## Three headline results — where they live

| Result                                           | Paper §                             | Data                                                                      | Script                               | Canister / Build                                                                                                |
| ------------------------------------------------ | ----------------------------------- | ------------------------------------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| Scaling law α_eff = 1.53 [1.37, 1.65], LOAO 7.7% | §3 (Empirical Scaling Law)          | `data/onchain/local.csv`, `data/models.csv`, `data/verified/variance.csv` | `scripts/analyze_scaling_law.py`     | local dfx 0.31.0 + fork build `17806d52`                                                                        |
| Mainnet 29 vs 10 (Qwen 2.5 0.5B Q8_0)            | §4.1 (The 2.9× fork gap on mainnet) | `data/onchain/icp_mainnet.csv`                                            | `scripts/benchmark_repeated_runs.sh` | mainnet `zmm32-7yaaa-aaaad-qlqsq-cai` · WASM `ef8f9d78…` (ours) / `6c77a958…` (onicai) · GGUF SHA256 `ca59ca7…` |
| Embedding case study α_embed ≈ 0.53              | §7.1 (On-chain encoder case study)  | `data/onchain/ssn_mainnet.csv` (embedding rows)                           | `scripts/run_mteb_fra.py` (template) | Swiss Subnet `u2pva-3iaaa-aaaax-qaa7a-cai` · EmbeddingGemma-300M Q4_0 (212 MB)                                  |

## Why this repo exists

The paper is the short read (~12 pages). This artifact is the long read and the terrain of play:

- Every chiffre in the paper traces back to a CSV here. See `../CLAIMS-EVIDENCE-MATRIX.md`.
- Every figure in the paper regenerates from one script here.
- Every kernel mentioned in §4.3 (Theorem 1) and §5.3 (Throughput density) is deployed and measurable by running the benchmarks below.
- Anyone who wants to **extend the scaling law** (new architectures, new quantizations, new subnets) can do so without rebuilding the measurement stack from scratch.

## What's inside

```
artifact/
├── README.md                          ← you are here
├── notes/
│   └── blockchain_ai_execution_boundaries.md
├── data/                              ← citation-facing tables rebuilt from raw measurements
│   ├── README.md                      ← data dictionary
│   ├── models.csv                     ← master registry (rebuilt summary table)
│   ├── quality_audit.csv              ← coherent / degraded / garbage per model
│   ├── onchain/
│   │   ├── local.csv                  ← rebuilt local view from `results/raw/core_measurements_v2.csv`
│   │   ├── icp_mainnet.csv            ← ICP production (canister zmm32-7yaaa)
│   │   └── ssn_mainnet.csv            ← Swiss Subnet (canister u2pva-3iaaa)
│   ├── native/m4_max_baseline.csv     ← Apple M4 Max native reference
│   ├── kernel/matmul_bench.csv        ← 10 WASM SIMD kernel variants
│   ├── profiling/per_operation.csv    ← per-ggml-op instruction counts
│   └── verified/                      ← P0 audit (April 2026)
│       ├── gguf_metadata.csv          ← 77 GGUFs cross-checked with GGUFReader
│       ├── discrepancies.csv          ← GGUF audit report: exact matches, documented size-convention diffs, missing refs
│       ├── variance.csv               ← 3-rep variance on 5 models (CV < 1.05%)
│       ├── coherence.csv              ← output coherence audit
│       └── coherence_manual.csv
├── scripts/
│   ├── analyze_scaling_law.py         ← LOAO + bootstrap + sensitivity + figure
│   ├── analyze_kernels.py             ← kernel comparison + dimension scaling
│   ├── apply_variance_updates.py      ← produces core_measurements_v2.csv
│   ├── rebuild_data_tables.py         ← regenerates `data/models.csv` + `data/onchain/local.csv`
│   ├── validate_data_tables.py        ← schema + cross-table consistency checks
│   ├── check_paper_readiness.py       ← publication/readability/readiness checks
│   ├── verify_gguf_metadata.py        ← re-runs GGUFReader audit
│   ├── find_discrepancies.py          ← surfaces CSV vs GGUF mismatches
│   ├── run_variance.py                ← 3-rep benchmark protocol
│   ├── run_coherence.py               ← generates outputs for quality audit
│   ├── run_mteb_fra.py                ← MTEB(fra, v1) template for embedding
│   └── benchmark_repeated_runs.sh     ← canister-side repeated-trial protocol
└── results/
    ├── raw/                           ← long-form measurement CSVs
    │   ├── core_measurements.csv      ← baseline (1 run per config)
    │   ├── core_measurements_v2.csv   ← with 5 variance-verified updates
    │   ├── kernel_measurements.csv
    │   ├── embedding_measurements.csv
    │   └── profiling_measurements.csv
    ├── current/
    │   ├── scaling_law/               ← canonical current scaling-law outputs
    │   └── extended_analysis/         ← kernel plots + extended comparison tables
    ├── historical/
    │   └── scaling_law_baseline/      ← pre-variance baseline outputs
    ├── audit/
    │   └── CHATGPT-VERIFICATION-PACKAGE.md ← independent statistical review bundle
    └── README.md
```

## The five theories — where to look

Each theory in the paper maps to specific data and scripts here.

| Theory                                        | Paper §                                                      | Data                                            | Script                           | Figure                                                      |
| --------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------- | -------------------------------- | ----------------------------------------------------------- |
| 1. Scaling law tok/call ≈ 40B/(α·2P)          | §3 (Empirical Scaling Law)                                   | `data/onchain/local.csv`, `data/models.csv`     | `scripts/analyze_scaling_law.py` | `results/current/scaling_law/scaling_law_full_analysis.png` |
| 2. Two regimes (modern α≈1.53, legacy α≈2.0)  | §3 (Empirical Scaling Law, including legacy GPT-2 paragraph) | same + `data/native/m4_max_baseline.csv`        | same                             | same                                                        |
| 3. Matmul dominates at 98.7%                  | §4.2 (Matmul dominates at 98.7%)                             | `data/profiling/per_operation.csv`              | analysis in paper                | paper Fig 2                                                 |
| 4. Per-element sparsity is counter-productive | §4.2                                                         | `data/kernel/matmul_bench.csv`                  | `scripts/analyze_kernels.py`     | `results/current/extended_analysis/kernel_analysis.png`     |
| 5. Embedding runs at α ≈ 0.53                 | §7.1 (On-chain encoder case study)                           | `data/onchain/ssn_mainnet.csv` (embedding rows) | `scripts/run_mteb_fra.py`        | paper §7.1                                                  |

## Choosing a model for your canister

If you only read one table from this artifact, read this one.

| Use case                                    | Model               | Format              | tok/call       | Quality                                |
| ------------------------------------------- | ------------------- | ------------------- | -------------- | -------------------------------------- |
| Reasoning, CoT, multilingue                 | Qwen3.5 0.8B        | Q4_0 SIMD           | 13 (SSN-only)  | ✅ best manual quality, build-specific |
| Short-form generation, multilingual utility | Qwen 2.5 0.5B       | Q4_0 SIMD           | 31             | ✅ coherent                            |
| Balanced QA, classification factuelle       | Gemma3-270M-IT      | Q4_0 SIMD           | 56             | ✅ coherent                            |
| Classification légère, routing              | Falcon-H1-Tiny-90M  | Q8_0 SIMD           | 199 (SSN-only) | ⚠️ limited quality, build-specific     |
| Sweet spot decoding + quality               | SmolLM2-135M        | Q4_0 SIMD           | 97             | ✅ coherent                            |
| Max throughput density (tok/MB)             | Pythia-70M          | TQ2_0 SIMD (custom) | 201            | ⚠️ base model                          |
| Semantic embedding                          | EmbeddingGemma-300M | Q4_0 SIMD           | 126 input      | ✅ retrieval OK                        |

**Avoid:** Qwen 3.5 0.8B with TQ2_0, IQ2_XXS, or layer pruning — throughput is tempting (+92% on one pruned row), but quality is destroyed or not established (rows M42–M46 in `data/models.csv`, flagged `quality=invalid` or `quality=not_checked`).

`Falcon-H1-Tiny-90M` is kept as an SSN-only build-specific observation. The current local gian build does not reproduce it, so do not compare it as a same-build calibration row against SmolLM2, Gemma3, or Qwen.

## Reproduce the paper in 5 minutes

```bash
# 1. Install dependencies
pip install numpy pandas scipy matplotlib statsmodels

# 2. Reproduce the scaling law analysis (LOAO, bootstrap CI, sensitivity, figure)
python scripts/analyze_scaling_law.py \
    --data results/raw/core_measurements_v2.csv \
    --outdir results/current/scaling_law

# Expected output:
#   α_eff modern median = 1.527 [1.374, 1.65]  (95% BCa)
#   strict modern-only LOAO MAPE = 7.7%
#   all-families LOAO MAPE = 12.1%
#   GPT-2 legacy regime α_eff ≈ 1.98
#   results/current/scaling_law/scaling_law_full_analysis.png  (4-panel figure)

# 3. Reproduce the kernel analysis
python scripts/analyze_kernels.py \
    --data results/raw/kernel_measurements.csv \
    --outdir results/current/extended_analysis

# 4. Re-verify GGUF metadata against CSVs (detects model drift)
python scripts/verify_gguf_metadata.py
python scripts/find_discrepancies.py

# 5. Rebuild and validate the citation-facing tables
python scripts/rebuild_data_tables.py
python scripts/validate_data_tables.py

# 6. Run the publication-readiness checks for the current draft
python scripts/check_paper_readiness.py
```

To re-collect throughput measurements on-chain, you need a local or mainnet ICP replica with our fork deployed. See the companion `llama_cpp_canister` deployment repo used for this artifact and run:

```bash
python scripts/run_variance.py --network local --config C2 --reps 3
```

## Extend: add your own model

Four steps to get a new model into the scaling law.

1. Download a GGUF and record its metadata in the raw measurement sources (`results/raw/core_measurements.csv` / `core_measurements_v2.csv`, `data/onchain/*.csv`, `data/native/m4_max_baseline.csv` as applicable).
2. Deploy the GGUF in your canister and run `scripts/run_variance.py --reps 3` to get `tok_call_local`. Expect `CV < 5%`; if not, investigate.
3. Run `scripts/run_coherence.py` to confirm the model isn't degenerate (see `data/verified/coherence.csv` for examples).
4. Re-run `scripts/rebuild_data_tables.py`, `scripts/validate_data_tables.py`, then `scripts/analyze_scaling_law.py`. If the model drops outside ±15% of the modern median (1.53), you found something interesting.

## Environment

| Component                             | Version used                                     |
| ------------------------------------- | ------------------------------------------------ |
| dfx                                   | 0.31.0                                           |
| wasi-sdk                              | 22.0                                             |
| llama.cpp fork                        | `17806d52` (see `../CURRENT.md` §4.1)            |
| Python                                | 3.11+                                            |
| ICP replica                           | DTS enabled (40B instruction budget per message) |
| Canister ID (mainnet)                 | `zmm32-7yaaa-aaaad-qlqsq-cai`                    |
| Canister ID (Swiss Subnet, embedding) | `u2pva-3iaaa-aaaax-qaa7a-cai`                    |

## Known limitations — read this before citing a number

- **Variance coverage.** 5 of 16 model-level throughput points are variance-verified (3 reps each, CV < 1.05%). The remaining 11 are single-run binary-search estimates. The reported bootstrap CI on α_eff reflects inter-model variance, not within-model measurement noise.
- **Build drift.** Re-measurement between March and April 2026 shows ±15% drift on absolute throughput, non-uniform across architectures (Transformers +12–15%, Mamba −14%). The _shape_ of the scaling law holds; absolute numbers belong to a specific build. Always check the build hash before comparing.
- **Non-Transformer generalization.** N=1 SSM (Mamba-370M) and N=1 RNN (RWKV7-0.4B). Treat the claim "law generalizes to non-Transformers" as _suggestive_, not _validated_.
- **Coherence vs throughput.** Our variance protocol measures throughput. A separate quality audit (`data/quality_audit.csv`, `data/verified/coherence.csv`) confirms Q4_0 SIMD ≈ Q8_0 in output quality on modern Transformers, but we have **not** run MTEB or standard benchmarks end-to-end on-chain. `scripts/run_mteb_fra.py` is a template, not a completed evaluation.
- **Ternary — two distinct objects.** (a) Our **TQ2_0 WASM SIMD kernel** is integrated end-to-end in the inference framework and its throughputs (Pythia-70M 201, SmolLM2-135M 89, Qwen 0.5B 29) are real model-scale measurements (§5.2). (b) The **ternary-native DOT ×4 kernel projection** (71–122 tok/call for 135M, §4.1) is a kernel-level extrapolation only — no purpose-trained ternary weights have been run in a canister with this kernel. The two are frequently conflated in the ternary-LLM literature; this artifact keeps them separate.

## Statistical review

We sought independent statistical review on the scaling-law re-fit. See `results/audit/CHATGPT-VERIFICATION-PACKAGE.md` for the verification package and response summary. Key findings:

- BCa bootstrap implementation is sound (uses `scipy.stats.bootstrap`).
- Strict modern-only LOAO MAPE is 7.7%; legacy-in-train convention gives 8.8%. Both reported in paper.
- Mamba-370M build-dependent deviation is a real signal, not measurement noise, but is attributed to implementation, not to SSM architecture.

## Citing

If you use this artifact, please cite the paper:

```bibtex
@misc{aerni_fluck_2026_instruction_bounded,
  title  = {On-Chain LLM Inference Under Instruction Budgets: An Instruction-Budget Cost Model, Ternary Floor Evidence, and Session Costs},
  author = {Aerni, Julien and Fluck, Sim\'eon and Becker, Dustin},
  year   = {2026},
  month  = may,
  note   = {Preprint manuscript with companion artifact}
}
```

## License

- **Code and scripts** (`scripts/`, analysis code): MIT License. See `LICENSE`.
- **Data and derived result tables/plots** (`data/`, `results/raw/`, `results/current/`, `results/historical/`): CC BY 4.0. Reuse with attribution.
- **Paper manuscript source and rendered exports** (`../CURRENT.md`, `../drafts/`, `../exports/`): CC BY 4.0 unless superseded later by venue-specific publication terms.
- **Custom TQ2_0 WASM SIMD kernel** (maintained in the companion `llama_cpp_canister` deployment repo): MIT, same as llama.cpp.

## Contact

- Correspondence for the paper and artifact: Siméon Fluck — simeon.autoinfo@gmail.com

Issues and extension PRs welcome.
