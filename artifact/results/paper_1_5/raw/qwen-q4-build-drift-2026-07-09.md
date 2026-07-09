# Qwen 2.5 0.5B Q4_0 — cross-campaign throughput reconciliation (SC-10, round 2)

**Question (R3 W4).** Table 1 (calibration, build `17806d52`, early April): 31 tok/call.
C2 mixed-precision campaign (build of 2026-04-20): BASE-Q4_0 = 20 tok/call — a −35% gap,
outside the paper's ±15% build envelope. Same model, same environment. Which is right?

**Provenance check (2026-07-09).** The calibration GGUF (`qwen2.5-0.5b-Q4_0.gguf`,
official download) and the campaign requants (`bq4q.gguf`, `c3-qwen05-q4.gguf`) are
byte-different files but share an identical tensor-type mix
({F32: 121, Q8_0: 1 (token_embd), Q4_0: 168}, 290 tensors, 335 MB) — kernel-identical
workloads. File provenance does **not** explain the gap.

**Fresh measurement (current release build `8a5ecf61`, local replica, warm,
single-call binary search, same protocol as Table 1):**

| n requested | outcome |
|---|---|
| 31 | trap `IC0522` |
| **30** | **OK (N_MAX)** |
| 29, 28 | OK |

`N_MAX = 30` → −3% vs the calibration 31, comfortably inside the ±15% envelope.

**Verdict.** The C2 value (20) was a **build-specific regression of the 2026-04-20 build
line, specific to the qwen2 path** (SmolLM2 on the same campaign build moved only
97 → 95, −2%). The regression is absent from both the calibration build and the current
release build. Within-campaign (same-build) comparisons in §6.1 are unaffected; the
campaign's *absolute* Qwen Q4_0 row should not be read as a deployment value. Deployers
should read absolute tok/call against a pinned WASM hash (Limitation 4).

**Session hygiene note.** Valid Qwen numbers required the safe model-swap protocol
documented in `variance-rootcause-2026-07-09/00-SUMMARY.md` (reinstall → upload →
load with explicit `-c 2048 -b 512` per §6.3 → fresh prompt-cache name). An earlier
probe series in this session was invalidated (the previous model was still resident
after an upgrade) and was discarded; model identity for the retained series was
verified by completion style (Qwen quiz-style continuation) and upload sha256.
