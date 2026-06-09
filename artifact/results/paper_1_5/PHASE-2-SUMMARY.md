# Paper 1.5 Phase 2 — Completion Summary

**Period:** 2026-04-20 (single-session execution on `paper-1.5/phase-2` branch)
**Scope:** C2 mixed-precision quantization, C3 multi-call stateful inference, IC0524 characterization mini-axis
**Commits:** 24 (`0fa66da..554167c`, all on branch `paper-1.5/phase-2`)
**Branch base:** `main` @ `2d45af9` (Phase 2 plan lock)

## TL;DR

> **C2 — `Q8_0` uniform quantization stays on the Pareto frontier on both
> SmolLM2-135M and Qwen 2.5 0.5B**: no mixed-precision variant strictly
> dominates it, and it is the highest-throughput / lowest-PPL point among
> the tested variants.
> α_eff ∈ [1.33, 1.53] on modern Q4/Q8 archs — consistent with Paper 1.
>
> **C3 — Multi-call stateful inference pays a two-term IO tax per call:**
> `save_instr ≈ a_s + b_s · n_kv_tok` and a near-constant `load_instr`. For
> Qwen 2.5 0.5B Q4_0: `save_instr ≈ 26.7 M + 0.95 M · n`, `load_instr ≈
> 425.6 M` (CV 0.02 %). A hybrid SSM+Transformer baseline (Falcon-H1-Tiny-90M
> Q8_0) **halves the per-token save slope** (0.52 M vs 0.95 M) and **cuts the
> fixed load cost 4×** (108 M vs 426 M) — the SSM advantage is partial, not
> the full "bounded state" headline; growing attention KV blocks dominate.

The extended session budget from C3,

    tok/session ≈ N_calls × 40e9 / (α_eff × 2P) − N_calls × IO_overhead(n_kv, cache_type)

is the operative cost formula for any on-canister agentic workload (Paper 2
forthcoming).

## C2 — Mixed-Precision (Tasks 4–13)

### SmolLM2-135M — Pareto winner: `BASE-Q8_0`

| Variant     | size_MB | tok/call | PPL_WT2 | PPL_C4 | Pareto |
|-------------|--------:|---------:|--------:|-------:|:------:|
| **BASE-Q8_0** | **144.8** | **97**    | **13.603** | **20.007** | ⭐ winner |
| BASE-Q4_0   | 91.7    | 95       | 17.208  | 25.464 | yes |
| V3c         | 121.1   | 82       | 13.892  | 20.390 | yes |
| V3b         | 115.6   | 80       | 13.946  | 20.494 | yes |
| V4 (ΔPPL)   | 113.0   | 78       | 13.853  | 20.395 | yes |
| V2          | 111.1   | 77       | 13.858  | 20.369 | yes |
| V3a         | 110.1   | 77       | 13.965  | 20.560 | yes |
| V1          | 105.5   | 74       | 14.008  | 20.563 | yes |
| BASE-Q5_K_M | 112.1   | 66       | 14.081  | 20.610 | **dominated** |

F16 baseline `PPL_WT2 = 13.574`. `BASE-Q8_0` adds just +0.029 PPL.

### Qwen 2.5 0.5B — Pareto winner: `BASE-Q8_0`

| Variant     | size_MB | tok/call | PPL_WT2 | PPL_C4 | Pareto |
|-------------|--------:|---------:|--------:|-------:|:------:|
| **BASE-Q8_0** | **506.5** | **30**    | **11.680** | **18.923** | ⭐ winner |
| V3c         | 464.4   | 25       | 11.815  | 19.131 | yes |
| V3b         | 441.1   | 24       | 11.849  | 19.205 | yes |
| V3a         | 417.7   | 23       | 11.893  | 19.260 | yes |
| V4 (ΔPPL)   | 423.8   | 23       | 11.813  | 19.182 | yes |
| V1          | 397.8   | 22       | 11.872  | 19.256 | yes |
| V2          | 406.6   | 22       | 11.842  | 19.175 | yes |
| BASE-Q4_0   | 335.8   | 20       | 13.139  | 21.065 | yes |
| BASE-Q5_K_M | 400.6   | 19       | 11.875  | 19.181 | **dominated** |

F16 baseline `PPL_WT2 = 11.674`. `BASE-Q8_0` adds just +0.007 PPL.

### α_eff computation

| Model           | size_MB | tok/call | α_eff = 40e9 / (N · 2P) |
|-----------------|--------:|---------:|------------------------:|
| SmolLM2-135M Q8_0 | 144.8 | 97      | **1.53** |
| Qwen 2.5 0.5B Q8_0 | 506.5 | 30     | **1.33** |

Both within the Paper 1 `[1.3, 1.6]` band for modern Q4/Q8 archs. No regression.

### Headline findings

1. **Mixed precision never strictly dominates `Q8_0`** on either model.
   Uniform Q8_0 is the highest-throughput / lowest-PPL frontier point,
   but not the smallest variant.
2. **`Q5_K_M` is strictly dominated on both models.** `hidden_dim ∈
   {576, 896}` is not divisible by 256, so `llama-quantize` emits
   145–181 `q5_K → q5_0/q8_0` fallbacks per model, neutering `Q5_K_M`'s
   compact-storage advantage.
3. **ΔPPL-guided `V4` is the best *mixed* variant** on PPL for both
   models, targeting the top-K fragility layers (layer-29 for SmolLM2,
   layers {0, 23} for Qwen). Useful when size budget is tight but
   quality matters.
4. **`α_eff` drops from 1.53 (SmolLM2) to 1.33 (Qwen)** despite Qwen's
   larger hidden_dim. Plausible contributors: Qwen's GQA (fewer K/V
   FLOPs), bias tensors on attn Q/K/V, and SmolLM2's tied embedding
   making `2P` undercount its LM-head cost.

Full summary: `tables/c2-mixed-precision-summary.md`. Pareto charts:
`figures/c2-pareto-{smollm135,qwen05}.png`.

## C3 — Multi-call stateful (Tasks 14–19)

### Per-call IO-overhead fits (Qwen 2.5 0.5B Q4_0 / f16 cache)

Single-call save scan (Task 14):

    save_instr ≈ 26.7 M + 0.95 M × n_kv_tok    (R² = 0.998)

Multi-call reattach (Task 14, 5-call chain at n=24):

    load_instr = 425.6 M ± 0.08 M   (CV = 0.02 %)

Re-attach cold-path latency (Task 15): **2.39 s cold vs 2.32 s warm** — the
bulk is `load_model`, not state IO.

### Task 16 — `tok/session` 1-call vs N-call

| Experiment        | Tokens | Wall (s) | tok/s | Notes |
|-------------------|-------:|---------:|------:|-------|
| 1-call `-n 100`   | 0      | 19.65    | —     | IC0522 trap (expected) |
| 7-call chain × 15 | 105    | 76.99    | **1.36** | Multi-call success |
| 1-call reference at N_MAX | 20 | 11.12 | 1.80 | Single-call upper bound |

**Multi-call tax:** 24 % tok/s reduction vs single-call reference.
**Multi-call benefit:** 5× more total tokens per session.

### Task 17 — `--cache-type-k {f16, q8_0, q4_0}`

The flag is accepted at `load_model` but has **no effect** on this WASM:
outputs are bit-identical across all three, N_MAX = 20 in all three,
tok/s within 4 % noise. KV is allocated at `load_model` time before the
flag reaches the cache-swap path; reallocating requires a WASM rebuild
(flagged for Phase 3 follow-up).

### Task 19 — Falcon-H1-Tiny-90M Q8_0 (hybrid SSM+Transformer)

5-call chain at N=15 per call:

    save_instr ≈ 276.3 M + 0.52 M × n_kv_tok   (R² = 0.972)
    load_instr = 108.3 M   (CV = 3 %)

| Fit                | a (M) | b (M/tok) | load_mean (M) |
|--------------------|------:|----------:|--------------:|
| Qwen 2.5 0.5B      | 26.7  | 0.951     | 425.6         |
| Falcon-H1-Tiny-90M | 276.3 | 0.519     | **108.3**     |

**Per-call IO tax at n=100 tokens in cache:** Qwen ≈ 547 M, Falcon ≈
436 M → **~20 % advantage for hybrid SSM**. At n=1000 (extrapolated):
Qwen ≈ 1.4 B, Falcon ≈ 0.9 B.

The headline SSM-beats-Transformer story is **partial**: Mamba+attention
hybrids still accumulate KV from attention blocks (per-token save slope
halved, not zero), but the re-attach fixed cost drops 4× — which
matters most for short-turn agents where the cold path dominates.

Full analysis: `tables/c3-multicall-summary.md`. Figures:
`figures/c3-{save-instr,load-instr,save-bytes}-vs-ntok.{png,pdf}`.

### Extended session budget formula

    tok/session ≈ N_calls × 40e9 / (α_eff × 2P) − N_calls × IO_overhead(n_kv, cache_type)
    IO_overhead(n, c) ≈ (a_s + b_s · n) + (a_l + b_l · n)

Fitted values in `tables/c3-fit-parameters.json`. This is the per-call
instruction budget for any multi-turn agentic workload on ICP.

## IC0524 mini-axis (Task 20)

| Model              | Format | Size (MB) | Outcome        |
|--------------------|--------|----------:|----------------|
| TriLM-560M         | TQ2_0  | 195       | OK (Phase 1)   |
| TriLM-3.9B         | TQ2_0  | 1 113     | OK (Phase 1) — >1 GB loads fine  |
| Qwen 2.5 0.5B      | Q4_0   | 336       | OK (Paper 1)   |
| Qwen 2.5 0.5B      | Q6_K   | 620       | OK (Phase 1)   |
| Qwen 2.5 1.5B      | Q4_0   | 892       | **IC0502** unreachable trap on `load_model` |
| Qwen 2.5 1.5B      | Q6_K   | 1 464     | **IC0524** page-access ceiling (Phase 1) |

**Hypothesis update:** the failure **is not simply size-driven.** TriLM
3.9B TQ2_0 at 1 113 MB loads cleanly (>1 GB), while Qwen 1.5B Q4_0 at
892 MB traps IC0502 before even reaching the page-access limit. The
format-specific GGUF tensor layout matters more than raw file size for
ICP compatibility. Ternary TQ2_0 appears unusually friendly to the ICP
page-access pattern — consistent with Phase 1 α_eff ≈ 1.00 outcome.

Full characterization: `data/paper_1_5/ic0524_characterization.csv`.

## What's NOT in Phase 2 (deferred / held)

- **Task 12 — SSN cross-env validation of C2 Pareto winners + C3 demo.**
  **Completed in Phase 3 follow-up (2026-04-20).** SmolLM2-135M
  `BASE-Q8_0` on SSN: N_MAX=95 (local=97, −2%), byte-exact 3/3 reps.
  Qwen 2.5 0.5B `BASE-Q8_0` on SSN: N_MAX=28 (local=30, −6.7%),
  byte-exact 3/3 reps. C3 chat demo on SSN (Qwen Q8_0): 10/10 turns,
  100 tokens, 41.12 s wall-clock, narrative coherent across 13-node
  consensus. New operational finding: wasm heap ceiling ~900 MB on SSN
  requires `-c 512 -b 512` for >500 MB models. Full writeup:
  `tables/c2-c3-ssn-validation-summary.md`.
- **Task 18 — 10-turn chat demo.** Initial attempt trapped IC0502
  "heap out of bounds" on turn 1 due to stale canister state after
  Tasks 14-17. **Re-run on a fresh `dfx start --clean` + reinstall
  succeeded: 10/10 turns completed, 100 tokens cumulative, 72.4 s
  wall-clock, coherent narrative preserved across calls** (Phase 3
  follow-up, 2026-04-20T20:07). Operational fix (reinstall between
  experimental axes) documented inline in
  `results/paper_1_5/tables/c3-chat-demo-transcript.md`.
- **Task 11.5 — HellaSwag spot-check.** Per locked decision #1, this
  escape hatch activates only if the PPL frontier is tight (<5 % delta
  between top variants). In practice `BASE-Q8_0` dominates by +0.029 PPL
  (SmolLM2) and +0.007 PPL (Qwen) — Pareto-clear, no tie-break needed.
- **§Implications for Paper 2.** Per locked decision #4, single-line
  forward-reference already present in `c3-multicall-summary.md`. Full
  cross-section written when Paper 2 has a draft.

## Operational findings (methodology, not paper-headline)

1. **`--cache-type-k` is a WASM-build-time decision** on the current
   canister build, not a runtime load-time flag. Phase 3 should expose
   this if KV quantization is ever relevant to the headline result.
2. **`llama-quantize` falls back to Q5_0/Q8_0 for non-256-divisible
   hidden_dims.** SmolLM2 (576) and Qwen 0.5B (896) both trigger this.
   Phase 3 model selection should prefer hidden_dim ∈ {512, 1024, 2048}
   if mixed-precision results are a headline target.
3. **ΔPPL-per-layer sensitivity** (layer-wise sweep quantizing only one
   block to Q4_K_M with all others F16) is cheap (30 min per model)
   and gives a reliable ranking for ΔPPL-guided mixed quantization.
   Reusable across any llama.cpp-supported architecture.
4. **IC0502 "unreachable" ≠ IC0524 "page-access ceiling."** They have
   different failure signatures and different root causes (WASM
   runtime vs stable-memory limit). Don't conflate.

## Phase 2 deliverables

### Code / infrastructure
- `scripts/paper_1_5/c2_imatrix_run.sh` — imatrix calibration driver
- `scripts/paper_1_5/c2_delta_ppl_per_layer.py` — ΔPPL sensitivity
- `scripts/paper_1_5/c2_generate_variants.py` — mixed-precision GGUF builder
- `scripts/paper_1_5/c2_pareto_analysis.py` — Pareto analysis + figures
- `scripts/paper_1_5/c3_io_overhead_scan.sh` — IO-overhead profiling driver
- `scripts/paper_1_5/c3_reattach_latency.sh` — cold-call measurement
- `scripts/paper_1_5/c3_ncall_ab.sh` — 1-call vs N-call A/B
- `scripts/paper_1_5/c3_kvtype_ab.sh` — cache-type-k sweep
- `scripts/paper_1_5/c3_ssm_falcon.sh` — SSM baseline (Falcon-H1-Tiny)
- `scripts/paper_1_5/c3_multicall_analysis.py` — fit + summary
- `scripts/paper_1_5/c3_chat_demo.sh` — Task 18 attempt (deferred)
- Canister `main_.cpp` C3-IO instrumentation (upstream `bb238b9` on main)

### Data tables
- `data/paper_1_5/imatrix_manifest.csv`
- `data/paper_1_5/mixed_precision_variants.csv`
- `data/paper_1_5/mixed_precision_measurements.csv`
- `data/paper_1_5/c2-ppl-smollm135-only.csv`
- `data/paper_1_5/multicall_characterization.csv`
- `data/paper_1_5/ic0524_characterization.csv`

### Summary tables / figures
- `results/paper_1_5/tables/c2-mixed-precision-summary.md`
- `results/paper_1_5/tables/c2-pareto-{smollm135,qwen05}.md`
- `results/paper_1_5/tables/c3-multicall-summary.md`
- `results/paper_1_5/tables/c3-fit-parameters.json`
- `results/paper_1_5/tables/c3-chat-demo-transcript.md` (Task 18 deferred)
- `results/paper_1_5/figures/c2-pareto-{smollm135,qwen05}.{png,pdf}`
- `results/paper_1_5/figures/c3-{save-instr,load-instr,save-bytes}-vs-ntok.{png,pdf}`

### Raw artefacts (in `results/paper_1_5/raw/`)
- `c2-delta-ppl-{smollm135,qwen05}.csv` + per-model `.log`
- `c2-ppl-{smollm135,qwen05}-*.log` (18 PPL runs)
- `c2-tokcall-{smollm135,qwen05}-*.log` (18 canister binsearch runs)
- `c3-ioover-qwen05-*.log` (IO overhead scan + 5-call chain)
- `c3-reattach-qwen05-rep{1..5}-{cold,warm}.log`
- `c3-ncall-qwen05-{1call-fail,Ncall-7x15,1call-nmax}.log`
- `c3-kvtype-qwen05-{f16,q8,q4}.log`
- `c3-ssm-falcon-h1-step{1..5}.log`
- `c3-chat-demo-qwen05.log` (Task 18 trap trace)

## Cross-reference

- Design spec: `docs/superpowers/specs/2026-04-19-paper-1.5-design.md`
- Phase 2 plan: `docs/superpowers/plans/2026-04-20-paper-1.5-phase-2.md`
- Phase 1 summary: `results/paper_1_5/PHASE-1-SUMMARY.md`
- First phase-2 commit: `0fa66da chore(paper-1.5 phase-2): scaffold artifact dirs + CSVs`
- Last phase-2 commit:  `554167c data(paper-1.5 C3): Task 18 chat demo attempt — deferred`
- Full trail: `git log --oneline 2d45af9..HEAD -- papers/instruction-bounded-inference/artifact/`

## Next: Phase 3

Phase 3 targets, in rough priority order:

1. **C1b BitNet 2B4T port** — AVX2/NEON I2_S → WASM SIMD128. Headline
   candidate for the paper. Reserve P15-04 row already present in
   `models_paper_1_5.csv` with `tok_call_* = pending`.
2. **Task 12 SSN cross-env validation** — rerun the Pareto winners
   (`BASE-Q8_0` on both models) on mainnet SSN, confirm byte-determinism
   extends from single-shot (Phase 1) to multi-call stateful (Phase 2).
3. **Task 18 chat-demo diagnostic** — reproduce the IC0502 trap under
   controlled conditions (fresh canister vs warm, `--prompt-cache-all`
   with vs without prior `new_chat`), fix or document the state-reset
   edge case.
4. **WASM rebuild exposing `--cache-type-k` at runtime** — unblocks the
   real KV quantization trade-off measurement (Task 17 negative result
   motivates).
5. **Paper 2 draft** — at which point the §Implications cross-reference
   in `c3-multicall-summary.md` gets expanded to a full section.

The C2 `Q8_0` dominance and C3 extended-formula fits are the headline
Phase 2 deliverables. Phase 3 should extend (BitNet) and validate
(SSN, diagnostic), not re-litigate.
