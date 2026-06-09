# Research Data — Instruction-Bounded Inference

Citation-facing tables for the paper. The canonical local measurement log lives in `artifact/results/raw/core_measurements_v2.csv`; the summary tables in this directory are rebuilt from it plus the network/native CSVs.

## Structure

```
data/
├── models.csv                 ← MASTER REGISTRY — rebuilt summary table across environments
├── onchain/
│   ├── local.csv              ← rebuilt local view from `results/raw/core_measurements_v2.csv`
│   ├── icp_mainnet.csv        ← ICP production network (canister zmm32-7yaaa)
│   └── ssn_mainnet.csv        ← Swiss Subnet production network (canister u2pva-3iaaa)
├── native/
│   └── m4_max_baseline.csv    ← Native llama.cpp on Apple M4 Max (tok/s, not tok/call)
├── kernel/
│   └── matmul_bench.csv       ← Kernel-level microbenchmarks (WASM SIMD, ternary, etc.)
└── profiling/
    └── per_operation.csv      ← Per-ggml-operation instruction counts (Pythia-70M Q8_0)
```

## models.csv — Master Registry

**One row per unique (model, quant, layers) variant.** Contains results from ALL environments.

Key columns:

- `id`: Stable identifier (M01, M02, ...)
- `params_M`: Paper-facing model-size convention used in tables and narrative
- `params_M_gguf`: GGUF-audited tensor count reconstructed from `verified/gguf_metadata.csv`
- `gguf_match`: `exact` if the exact row GGUF is on disk, `proxy_same_model_quant` if structural metadata comes from the same model under another preserved quantization
- `status_tags`: Stable evidence-status label string with five required keys: `runs`, `quality`, `env`, `build`, `role`
- `tok_call_local`: Tokens per call on dfx local
- `tok_call_icp`: Tokens per call on ICP mainnet
- `tok_call_ssn`: Tokens per call on Swiss Subnet mainnet
- `alpha_eff`: Effective cost per FLOP (derived from scaling law)
- `native_gen_tok_s`: Generation tok/s on M4 Max native (NOT on-chain)
- `native_prefill_tok_s`: Prefill tok/s on M4 Max native
- `build`: `gian` (our fork), `upstream` (upstream fork), `onicai` (baseline)

## Regeneration

- Rebuild the derived tables with `python artifact/scripts/rebuild_data_tables.py`
- Validate schema and cross-table consistency with `python artifact/scripts/validate_data_tables.py`
- Audit registry-vs-GGUF provenance with `python artifact/scripts/find_discrepancies.py`
- `models.csv` and `onchain/local.csv` are generated artifacts. Do not hand-edit them unless you also update the underlying source CSVs.

## Row Status Tags

Workstream C uses one stable convention across `models.csv` and `onchain/*.csv`:

```text
runs=<...>; quality=<...>; env=<...>; build=<...>; role=<...>
```

The keys always appear in that order. Composite values use `+` inside the value, for example:

```text
runs=repeated+single_run; quality=verified; env=local+ssn_mainnet; build=build_sensitive; role=calibration+network_validation
```

Allowed values:

- `runs`: `repeated`, `3_reps`, `single_run`, `10_turns_multicall`, `not_applicable`
- `quality`: `verified`, `limited`, `invalid`, `not_checked`
- `env`: `local`, `icp_mainnet`, `ssn_mainnet`, `native_only`
- `build`: `current`, `historical`, `build_sensitive`, `build_specific`, `phase_2_wasm`, `unknown`
- `role`: `calibration`, `supporting_measurement`, `network_validation`, `systems_demo`, `negative_result`, `c2_cross_env_validation`, `c3_multicall_ssn_validation`, `baseline`, `native_only`

Interpretation guide:

- `role=calibration` means the row feeds the scaling-law or alpha calibration story.
- `role=network_validation` means the row is used as a public-network comparison or validation point.
- `role=systems_demo` marks artifact-facing deployment or kernel demonstrations that are real measurements but not calibration anchors.
- `role=negative_result` marks throughput-valid but quality-invalid or otherwise retracted / non-viable rows.
- `role=c2_cross_env_validation` marks the Paper 1.5 cross-environment confirmation rows used to test local-to-SSN transfer.
- `role=c3_multicall_ssn_validation` marks the Paper 1.5 stateful multi-call SSN demonstration rows.
- `quality=limited` means the row was quality-checked and is usable only under a narrower claim than "general text generation".
- `build=build_sensitive` means the repo has explicit evidence of material drift across builds for that family or row class.
- `build=build_specific` means the row is tied to a particular deployment/build and should not be treated as same-build comparable without an explicit matching reproduction.
- `build=phase_2_wasm` means the row comes from the separate Paper 1.5 Phase 2 WASM build used for cross-env and multicall validation rather than the main calibration build line.

## Parameter-Count Convention

Two notions of "model size" coexist in this project:

- `params_M` keeps the paper-facing model-size convention used in the scaling-law tables and narrative (`82M`, `124M`, `0.5B`, `0.8B`, etc.).
- `params_M_gguf` records the tensor count reconstructed from the GGUF file on disk.

These can differ materially for some families because GGUF audit counts all stored inference tensors, while model names often follow release / marketing conventions. The paper and scaling-law narrative always use `params_M` as the canonical `P` in `tok/call ≈ 40B / (α_eff × 2P)`; `params_M_gguf` is an audit field that makes those convention differences visible rather than leaving them implicit.

Typical sources of divergence:

- tied input/output embeddings counted twice in GGUF tensor totals
- instruct-vs-base preservation mismatches where the preserved GGUF variant is not the exact paper-facing deployment row

## Environment Differences

| Environment       | Instruction limit | Deterministic | Cycles cost       | Notes                                                                                                                                                                     |
| ----------------- | ----------------- | ------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **local** (dfx)   | 40B (simulated)   | Yes           | Free              | Most measurements here                                                                                                                                                    |
| **ICP mainnet**   | 40B (enforced)    | Yes           | ~5.3B cycles/call | Real production. Qwen3.5 CRASHES here                                                                                                                                     |
| **SSN mainnet**   | 40B (enforced)    | Yes           | 0 (rental model)  | Swiss Subnet. Used for cross-subnet validation and embedding runs; compare per-row build/hash context rather than assuming every SSN row is same-build with current local |
| **M4 Max native** | N/A               | No            | N/A               | Reference baseline — measures tok/s not tok/call                                                                                                                          |

## Build-Specific Rows

### SmolLM2-135M current canonical value

The canonical current-build `SmolLM2-135M Q4_0 SIMD` value is **97 tok/call** on both local and Swiss Subnet in the citation-facing tables. The earlier `103 tok/call` row survives only as a historical March single-run measurement in `results/raw/core_measurements.csv`; it is not the paper's canonical narrative value.

### Falcon-H1-Tiny-90M comparability

The `199 tok/call` Falcon-H1 row in `onchain/ssn_mainnet.csv` is an **SSN-only build-specific observation**. The current local gian build does not reproduce Falcon-H1 (`load_model` traps), so treat that row as a deployment observation, not as a same-build calibration point.

## Known Data Issues (corrected)

### Qwen 3.5-0.8B layer counts

Old memory files labeled the pruned variants as "36L/30L/24L". The **actual** GGUF block counts are:

- "original" = **24 layers** (18 DeltaNet + 6 Full Attention)
- "30L" = **21 layers** (block_count=21 in GGUF)
- "24L" = **12 layers** (block_count=12 in GGUF)

Verified via `gguf.GGUFReader` on 2026-04-16. All CSVs in this directory use corrected values.

### Measurement methodology

- `binary_search`: Incrementally increase max_tokens until TRAP at 40B instruction limit
- `tok_call` = maximum tokens generated in a single update call before hitting the limit
- Prefill calls are separate (consume prompt tokens, generate 0 output tokens)
- All tok_call values are pure generation (post-prefill)

### Embedding model

EmbeddingGemma-300M reports `126_input` as tok_call — this means 126 **input** tokens per call (embedding mode), not generated tokens. The model doesn't generate text.

## Provenance

| CSV                         | Primary source                                                                                                | Cross-checked with                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| models.csv                  | `artifact/results/raw/core_measurements_v2.csv` + `onchain/*.csv` + `native/m4_max_baseline.csv` + GGUF audit | `scripts/rebuild_data_tables.py`, paper tables                                                        |
| onchain/local.csv           | `artifact/results/raw/core_measurements_v2.csv`                                                               | `scripts/rebuild_data_tables.py`, paper §3 (Empirical Scaling Law) tables                             |
| onchain/icp_mainnet.csv     | `MAINNET-BENCHMARK-RESULTS.md`                                                                                | dfx logs, paper §4.1 (The 2.9× fork gap on mainnet)                                                   |
| onchain/ssn_mainnet.csv     | Memory file `ssn_benchmarks_apr15.md`                                                                         | paper §4.4 (Cross-subnet consistency), §6.1 (Pareto quantization), §7.1 (On-chain encoder case study) |
| native/m4_max_baseline.csv  | `artifact/native-baseline/results.csv`                                                                        | bench.cpp output                                                                                      |
| kernel/matmul_bench.csv     | `artifact/results/raw/kernel_measurements.csv`                                                                | paper §4.3 (Theorem 1) and §5.3 (Throughput density)                                                  |
| profiling/per_operation.csv | `artifact/results/raw/profiling_measurements.csv`                                                             | paper §4.2 (Matmul dominates at 98.7%)                                                                |

---

## Paper 1.5 extension

Additional data for Paper 1.5 (Phase 1) is under `paper_1_5/`:

- `models_paper_1_5.csv` — new model registry rows (IDs `P15-*`)
- `ternary_measurements.csv` — C1 ternary end-to-end results
- `q6k_kernel_bench.csv` — C4 Q6_K vectorize benchmark

See `paper_1_5/README.md` for details.
