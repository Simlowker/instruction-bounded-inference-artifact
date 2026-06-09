# Instruction-Bounded Inference — Reproducible Artifact

Companion artifact for the preprint:

> **On-Chain LLM Inference Under Instruction Budgets: An Instruction-Budget Cost Model, Ternary Floor Evidence, and Session Costs**
> Julien Aerni¹, Siméon Fluck², Dustin Becker³
> ¹ Meotis Sàrl · ² Kaizen Corp SA · ³ ORIGYN Foundation

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20607598-blue)](https://doi.org/10.5281/zenodo.20607598)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](LICENSE-DATA)

Running the same Qwen 2.5 0.5B Q8_0 model on the same Internet Computer (ICP) mainnet
canister gives **10 tokens/call on the onicai baseline and 29 tokens/call on our fork**
— a 2.9× software-path gap. This repository reproduces and verifies every load-bearing
claim of the paper: the decode cost model (α_eff ≈ 1.527, BCa CI [1.374, 1.65]), matmul
dominance (98.7%), the ternary α_eff ≈ 1 floor (TriLM TQ2_0, byte-identical under a
13-replica Swiss Subnet consensus), the mixed-precision Pareto analysis, and the
multi-call IO tax.

## Layout

| Path | Contents |
|---|---|
| `drafts/`, `CURRENT.md` | the manuscript (`instruction-bounded-inference.md`) |
| `exports/` | rendered `instruction-bounded-inference.{pdf,html}` |
| `artifact/data/` | calibration registry, on-chain rows (ICP mainnet / SSN), kernel & profiling benches, Paper 1.5 CSVs (ternary, mixed-precision, multi-call, IC0524), variance-verified anchors |
| `artifact/results/` | scaling-law outputs (BCa, LOAO), extended analysis, Paper 1.5 tables & figures, raw logs |
| `artifact/scripts/` | `analyze_scaling_law.py`, `check_paper_readiness.py`, `rebuild_data_tables.py`, `run_variance.py`, Paper 1.5 pipelines |
| `artifact/notes/` | β derivation, source pins (WASM/GGUF SHA-256), execution-boundary note |
| `CLAIMS-EVIDENCE-MATRIX.md` | every load-bearing number → its CSV / summary table |
| `REPRODUCE.md` | end-to-end local rebuild of the 2.9× gap (dfx, ~50 GB, ~30 min, **no ICP cycles**) |

## Quickstart

```bash
cd artifact

# 1. Consistency gate: registry ↔ tables ↔ claims  (expect "9 pass, 0 fail")
python3 scripts/check_paper_readiness.py --draft ../drafts/instruction-bounded-inference.md

# 2. Re-derive the scaling-law statistics
python3 scripts/analyze_scaling_law.py    # α median 1.527, BCa [1.374, 1.65], LOAO MAPE 7.7%

# 3. Reproduce the 2.9× fork gap from public source — see ../REPRODUCE.md
```

## Reproducibility notes

- The original 2026-04-09 mainnet WASM (`ef8f9d78…`) is **not byte-reproducible** (it was
  built from uncommitted working-tree state); the 2.9× claim is **functionally**
  reproducible from current public source — see `REPRODUCE.md` and
  `artifact/notes/source_pinning.md`.
- Byte-identical determinism between local `dfx` and a 13-replica Swiss Subnet is verified
  for the TQ2_0 ternary path.
- All measurements are on the ICP / SSN stack; the cost-model coefficients (β, α_eff) are
  specific to the ICP opcode cost table and are not claimed to transfer numerically.

## Citation

```bibtex
@misc{aerni_fluck_becker_2026_instruction_bounded,
  title  = {On-Chain LLM Inference Under Instruction Budgets: An Instruction-Budget Cost Model, Ternary Floor Evidence, and Session Costs},
  author = {Aerni, Julien and Fluck, Sim\'eon and Becker, Dustin},
  year   = {2026},
  doi    = {10.5281/zenodo.20607598},
  url    = {https://doi.org/10.5281/zenodo.20607598},
  note   = {Preprint with companion artifact}
}
```

## License

Code: [MIT](LICENSE) · Data: [CC BY 4.0](LICENSE-DATA).
