# Multi-architecture per-op profiling — 2026-07-22

Extends `per_operation.csv` (Pythia-70M Q8_0, single profile behind the paper's
"98.7% matmul" claim, C5) to 5 additional model/quant configurations.

## Protocol

Same as the original Pythia profile: local replica (`dfx`), release wasm build of
2026-07-03, per-op `ic0.performance_counter` instrumentation at ggml op boundaries
(`ICPP-PROF` markers in `ggml-cpu.c`). Two `run_update` calls with `--prompt-cache`;
the profile is the second call (pure decode, prompt cached, 12 tokens,
prompt "The meaning of life is"). Canister reinstalled between models (a failed
`load_model` leaves the wasm heap in a corrupt state; reinstall guarantees a clean
baseline — cycles balance survives reinstall).

## Headline result — MUL_MAT share of decode instructions

| Model | Arch family | Quant | MUL_MAT % |
|---|---|---|---|
| Pythia-70M (original) | gpt_neox | Q8_0 | 98.70 |
| Qwen2.5-0.5B | qwen2 | Q4_0 | 98.50 |
| LFM2.5-230M | lfm2 (hybrid conv+GQA) | Q8_0 | 97.90 |
| TriLM-560M | ternary llama (custom TQ2_0 kernel) | TQ2_0 | 97.40 |
| SmolLM2-135M | llama | Q8_0 | 95.60 |
| **Mamba-370M** | **pure SSM** | Q8_0 | **75.20** |

## Interpretation

- Matmul dominance (≥95.6%) holds across attention-based transformers regardless
  of quant format (Q4_0 / Q8_0 / TQ2_0), including the ternary custom kernel and
  the LFM2.5 conv+GQA hybrid (its conv path lowers to matmul).
- SmolLM2's slightly lower share (95.6%) is FLASH_ATTN_EXT (3.3%) — the FA SIMD
  path now accounts separately for attention cost.
- **Pure SSM breaks the pattern**: Mamba-370M spends 20.2% in SSM_SCAN + 0.8%
  SSM_CONV, capping MUL_MAT at 75.2%. The "optimization collapses to one hotspot"
  claim is transformer-scoped, not universal.
- Falcon-H1-Tiny-90M (Mamba+attention hybrid) could not be profiled: `load_model`
  traps (`unreachable`) on this build. Not a data point either way.

## Raw logs

Session scratchpad `profiles/` (per-model `*.logs.txt`, `*.call{1,2}.log`,
upload/load logs). Note: the canister log ring buffer (~4 KiB) can evict the
dump header — the parser keys on `op[..]` lines, not on the header.
