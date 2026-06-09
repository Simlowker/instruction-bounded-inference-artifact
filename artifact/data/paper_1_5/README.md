# Paper 1.5 — Artifact Extension

This directory contains data, scripts, and results for Paper 1.5 (*Pareto-Optimal LLM Inference on Instruction-Bounded Runtimes*), an extension of Paper 1.

**Scope of Paper 1.5 Phase 1 (this batch):**

- **C1 Ternary end-to-end:** first measurement of purpose-trained ternary LLMs (TriLM 560M, 3.9B) on ICP canisters via TQ2_0 WASM SIMD kernel.
- **C4 Q6_K vectorize:** WASM SIMD128 vectorization of the Q6_K dequant unpack, replacing the scalar fallback left in llama.cpp PR #11453.

**Files:**

- `models_paper_1_5.csv` — model registry extension (appended to parent `data/models.csv`).
- `ternary_measurements.csv` — C1 tok/call, cycles, tok/MB, coherence audit.
- `q6k_kernel_bench.csv` — C4 microbench + end-to-end Q6_K measurements.

**Companion scripts:** `../../scripts/paper_1_5/`

**Raw data:** `../../results/paper_1_5/raw/`

**Regenerate every figure/table:**

    make paper-1-5-reproduce

Design spec: [docs/superpowers/specs/2026-04-19-paper-1.5-design.md](../../../../../docs/superpowers/specs/2026-04-19-paper-1.5-design.md)
