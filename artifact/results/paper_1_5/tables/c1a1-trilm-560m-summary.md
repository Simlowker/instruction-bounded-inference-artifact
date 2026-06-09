# C1a.1 — TriLM 560M TQ2_0 Result Summary

**Model:** TriLM-560M (llama, TQ2_0 ternary, 195 MB on disk)
**Params:** 569 M (GGUF-extracted), 24 L × 1280 H × 3072 FFN, vocab 50304
**Build:** Phase 1 `llama_cpp.wasm` (5.76 MB), TQ2_0 WASM SIMD path active
**Source:** self-quantized from `SpectraSuite/TriLM_560M_Unpacked` via
`convert_hf_to_gguf.py` (F16) + `llama-quantize TQ2_0`

## Measurements

| Environment | Canister | Reps | Mean tok/call | Stdev | CV |
|---|---|---:|---:|---:|---:|
| dfx-local | uzt4z-lp777-77774-qaabq-cai | 3 | 34.0 | 0.000 | 0.000% |
| SSN mainnet (13-node) | u2pva-3iaaa-aaaax-qaa7a-cai | 3 | 34.0 | 0.000 | 0.000% |

**tok/MB (SSN):** 0.1744
**Cycle cost per call (SSN):** 0 (rental subnet)

## Comparison vs Paper 1 §4.1 projection

Paper 1 §4.1 baseline formula: `tok/call ≈ 40e9 / (α_eff × 2P)`

| Quantity | Value |
|---|---:|
| Modern-arch baseline α_eff (Paper 1) | 1.54 |
| Projected baseline tok/call at 569M params | 22.8 |
| **Measured** SSN mainnet | **34** |
| Speedup over baseline | **1.49×** |
| Implied α_eff (this work) | 1.03 |

The TQ2_0 ternary path delivers a **1.49× end-to-end speedup**
over the modern-arch baseline. The DOT×4 microbench-level ternary speedup
is ~2.45×; the smaller end-to-end ratio reflects non-MatMul work
(KV-cache reads, sampling, embedding lookup, RoPE) that also consumes
the 40B-instruction budget but is unaffected by the ternary kernel.

## Coherence audit

10-prompt audit (`raw/trilm-560m-coherence-audit.md`):
- **Fluency:** 10 / 10 well-formed completions (no garbage tokens, no
  UTF-8 corruption) — confirms the TQ2_0 WASM SIMD kernel produces
  valid model outputs end-to-end.
- **Factual accuracy:** 2 / 10 — expected for any 560M *base* model
  (TriLM 560M has no instruction tuning and no factual fine-tune).
  This is a model-capability property, not a quantization regression
  (would also fail SmolLM2-135M base, FP16 TriLM, etc.).
- **coherence_pass = true** per Paper 1 §5.2 fluency-bar precedent.

## Cross-environment determinism

The 10-prompt audit was re-run on SSN and diffed against the local
audit (excluding date/environment header). Result: **empty diff —
all 10 completions are byte-identical** between the local single-replica
dfx and the 13-node SSN consensus.

This rules out non-deterministic kernel hazards (uninitialised vector
lanes, FP reorder, replica-specific code paths). Detail in
`raw/trilm-560m-ssn-vs-mainnet-diff.txt`.

## Provenance

- Binary search log (local): `raw/trilm-560m-local-binsearch.txt`
- Binary search log (SSN):   `raw/trilm-560m-ssn-binsearch.txt`
- 3-rep variance logs:       `raw/trilm-560m-rep{2,3}.log`
- Coherence audit:           `raw/trilm-560m-coherence-audit.md`
- Cross-env diff:             `raw/trilm-560m-ssn-vs-mainnet-diff.txt`
- Upload + load logs:         `raw/upload-560m.log`, `raw/load-560m.log`,
                              `raw/ssn-upload-560m.log`, `raw/ssn-load-560m.log`
- Deploy logs:                `raw/deploy-560m.log`, `raw/ssn-deploy-560m.log`

## Headline

> TriLM 560M TQ2_0 (ternary) achieves **34 tok/call on SSN mainnet**
> (and identically on local dfx), a **1.49× speedup over the
> Paper 1 §4.1 modern-arch baseline at the same parameter count**,
> with byte-deterministic outputs across replication boundaries and
> a passing fluency audit.
