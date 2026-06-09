# Load Layout Summary

Follow-up analysis for the `load_model + KV/session IO + layout GGUF + multi-call` thread.

Generated from `papers/instruction-bounded-inference/artifact/data/paper_1_5/load_layout_characterization.csv` using `artifact/scripts/paper_1_5/analyze_load_layout.py`.

## Cases

| Model | Format | Outcome | File MB | Tensors | Dominant type | Largest tensor | Prefix16 share | >4 MiB tensors | GGUF |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| Qwen2.5-1.5B | Q6_K | IC0524 | 1396.3 | 339 | `Q6_K` (100.0%) | 182.6 MB `Q6_K` | 29.7% | 86 | `.worktrees/paper-1.5-phase-1/llama_cpp_canister/models/Qwen2.5-1.5B/qwen2.5-1.5b-instruct-q6_k.gguf` |
| Falcon-H1-Tiny-90M | Q8_0 | OK | 93.8 | 386 | `Q8_0` (99.4%) | 17.0 MB `Q8_0` | 20.4% | 1 | `llama_cpp_canister/models/candidates/Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf` |
| TriLM-560M | TQ2_0 | OK | 195.1 | 219 | `TQ2_0` (56.0%) | 50.4 MB `Q6_K` | 46.9% | 2 | `.worktrees/paper-1.5-phase-1/llama_cpp_canister/models/trilm/TriLM_560M_Unpacked.TQ2_0.gguf` |
| Qwen2.5-0.5B | Q4_0 | OK | 335.8 | 290 | `Q4_0` (58.1%) | 137.9 MB `Q8_0` | 44.2% | 1 | `llama_cpp_canister/models/c3-qwen05-q4.gguf` |
| Qwen2.5-0.5B | Q6_K | OK | 482.3 | 290 | `Q8_0` (82.8%) | 137.9 MB `Q8_0` | 31.9% | 49 | `llama_cpp_canister/models/Qwen/qwen2.5-0.5b-Q6_K.gguf` |
| TriLM-3.9B | TQ2_0 | OK | 1112.7 | 273 | `TQ2_0` (81.5%) | 121.8 MB `Q6_K` | 21.8% | 92 | `.worktrees/paper-1.5-phase-1/llama_cpp_canister/models/trilm/TriLM_3.9B_Unpacked.TQ2_0.gguf` |
| Qwen2.5-1.5B | Q4_0 | other_error | 891.6 | 338 | `Q4_0` (79.3%) | 182.6 MB `Q6_K` | 23.5% | 85 | `llama_cpp_canister/models/qwen15-q4.gguf` |

## Immediate Read

- Raw file size still does not explain the outcomes on its own: `TriLM-3.9B TQ2_0` loads at `1112.7 MB`, while `Qwen2.5-1.5B Q4_0` traps at `891.6 MB`.
- Early-byte concentration is not a clean separator either: `Qwen2.5-0.5B Q4_0` loads despite `41.8%` of payload landing inside the first `16 MiB`, whereas the failing `Qwen2.5-1.5B Q4_0` sits at `20.6%`.
- The failing `Qwen2.5-1.5B Q4_0` case is still layout-distinct in two useful ways: a much larger tensor population (`338` tensors, `141` above `1 MiB`) and a large mixed-precision component (`Q6_K` token embedding at `182.6 MB`). That is enough to justify load-path instrumentation, but not enough to assign causality yet.
- The current evidence therefore supports a stricter framing: `IC0524` and `IC0502` are not just "big model" failures; they are loader-path failures over specific GGUF layouts.

## Next Useful Instrumentation

1. Emit stage markers around `common_init_from_params`, buffer allocation, and tensor materialization so `IC0502` is tied to an exact phase instead of a generic trap.
2. Add throttled tensor-progress logging in the loader (for example every 16 or 32 tensors) with tensor name, type, bytes, and cumulative offset.
3. Record backend-buffer allocation sizes before the trap, so we can separate heap-growth failures from stable-memory page-access failures.
