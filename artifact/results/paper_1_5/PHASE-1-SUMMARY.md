# Paper 1.5 Phase 1 — Completion Summary

**Period:** 2026-04-19 (single-session execution on `paper-1.5/phase-1` branch)
**Scope:** C1a ternary end-to-end (TriLM 560M + 3.9B TQ2_0) and C4 Q6_K WASM SIMD vectorize
**Commits:** 33 (`eebd37c..db29b6a`, all on branch `paper-1.5/phase-1`)
**Branch base:** `main` @ `ad0208a`

## TL;DR

> **TriLM TQ2_0 ternary inference on an ICP canister achieves the
> theoretical instruction-budget floor (α_eff ≈ 1.00) at both 560 M
> and 3.9 B parameters, validated under 13-node consensus with
> byte-deterministic outputs across replication boundaries.**

The C4 Q6_K kernel patch contributes a +2.3 % microbench speedup with
3.2× variance reduction and bit-exact preservation, but the gain is
below the integer-token granularity of the on-canister `tok/call`
metric at the model size that fits ICP's per-message stable-memory
limit.

## C1a — Ternary end-to-end (TriLM TQ2_0)

| Metric | TriLM 560M (P15-01) | TriLM 3.9B (P15-02) |
|---|---:|---:|
| Params (GGUF) | 569 M | 3 992 M |
| GGUF size (TQ2_0) | 195 MB | 1113 MB |
| Architecture | 24 L × 1280 H × 3072 FFN | 30 L × 3072 H × 9216 FFN |
| **N_MAX local dfx (3 reps)** | **34** (CV 0.000 %) | **5** (CV 0.000 %) |
| **N_MAX SSN mainnet (3 reps)** | **34** (CV 0.000 %) | **5** (CV 0.000 %) |
| tok/MB (SSN) | 0.1744 | 0.0045 |
| Cycle cost per call (SSN) | 0 | 0 |
| Paper 1 §4.1 baseline (α=1.54) | 22.8 | 3.3 |
| Theoretical floor (α=1.00) | 35.1 | 5.0 |
| **Implied α_eff** | **1.03** | **1.00** |
| % of theoretical floor | 97 % | 100 % |
| Speedup vs Paper 1 baseline | 1.49× | 1.50× |
| Coherence (fluency) | 10/10 | (inline at N=5: factual ✓) |
| `coherence_pass` | true | true |
| Cross-env byte-determinism | empty diff | empty diff |

### Headline observations

1. **Two data points 7× apart in scale both yield α_eff ≈ 1.00.** This
   essentially saturates the ICP cost model: there is no more
   kernel-level speedup to extract — the TQ2_0 ternary path is
   already at one MAC per param-token end-to-end.

2. **The 1.49–1.50× speedup over the modern-arch baseline is constant
   across the 7× scale jump.** This rules out the alternative
   explanation that the 560M result was inflated by sub-α=1.54
   inefficiency at small scale.

3. **Quality scales correctly with model size, not token budget.** At
   N=5, the 3.9B emits "the city of Paris," (factually correct
   identification of France's capital). At N=34, the 560M emits
   "famous for its monuments and museums" without ever naming Paris.
   *Larger model → fewer tokens per call → more information per
   token.* This is a useful narrative hook for §5.7-style discussion.

4. **Cross-environment determinism.** The 10-prompt coherence audit
   was re-run on SSN and diff'd against local — empty diff. This
   rules out non-deterministic kernel hazards (uninitialised vector
   lanes, FP reorder, replica-specific paths) and validates the
   §5.6-style mainnet measurement protocol.

### Coherence audit decision

`coherence_pass = true` per Paper 1 §5.2 fluency-bar precedent
(matches SmolLM2-135M base, FP16 TriLM, sub-1B precedents). Fluency
10/10, factual 2/10 — expected for any base model with no instruction
tuning and no factual fine-tune. Factual accuracy is a model-capacity
property, not a quantization regression.

## C4 — Q6_K WASM SIMD vectorize

### Microbench (matmul-bench, Node.js v24 + WASM SIMD strict)

| Build | ns/iter | GFLOPS | Speedup | CV |
|---|---:|---:|---:|---:|
| Baseline (scalar unpack) | 357.81 | 22.895 | 1.000× | 8.0 % |
| **Patched (vectorized ql/qh unpack)** | **349.85** | **23.416** | **1.023×** | **2.5 %** |

- **3.2× variance reduction** (8.0 % → 2.5 %) — meaningful for
  reproducibility at the noise floor, even if the speedup itself is
  modest.
- **Bit-exact** vs scalar reference (matmul-bench, all 8 runs result
  = 56754044928).
- **Strict SIMD only** (no `relaxed_simd` ops) → preserves
  determinism for runtimes that require it (ICP, deterministic FaaS).

### End-to-end on canister (Qwen 2.5 0.5B Q6_K, 620 MB GGUF)

| Build | N_MAX (3 reps) | CV | Decode at N=N_MAX |
|---|---:|---:|---|
| Patched | 25 | 0.000 % | "...The capital of France is Paris. The correct answer" |
| Baseline | 25 | 0.000 % | (identical decode) |

**E2E result is null on the tok/call metric**: +2.3 % × 25 = 0.575
tokens, below integer granularity. To resolve a +2.3 % gain on this
metric you would need either N_MAX ≥ 50 (Δ ≥ 1.15 tokens) or a
continuous metric like cycles-per-token (not exposed via standard
dfx response).

What we *did* confirm on canister:
- Bit-exact decode equivalence between scalar and vectorized unpack.
- Boundary symmetry: N=26 fails 3/3 with IC0522 in both builds.
- `dfx deploy --mode upgrade` preserves the GGUF in stable memory
  across wasm swaps — useful methodology for future on-chain A/Bs
  without re-uploading multi-GB models.

### Pivot finding: Q6_K @ 1.5B hits IC0524

The originally planned target (Qwen 2.5 1.5B Q6_K, 1.46 GB)
reproducibly hit `IC0524` on `load_model`:

> "Exceeded the limit for the number of accessed pages in the stable
> memory in a single message execution: limit 2_097_152 KB"

This is a hard ICP runtime invariant, not a bug. Q6_K models above
~1 GB are **not viable on the current ICP runtime** without splitting
`load_model` across messages. Documented as an artefact-level
observation; pivoted to the 0.5B Q6_K which loads cleanly.

## Operational findings (useful methodology, not paper-headline)

1. **Local canisters need cycle top-ups for ≥ 1 GB uploads.** Default
   `dfx start` allocations are insufficient. Use
   `dfx ledger fabricate-cycles --canister <name> --t 100` between
   reinstall and upload.

2. **Cold-state instruction count differs from warm.** The first
   `run_update` after a fresh `load_model` consumes noticeably more
   instructions than subsequent calls. Always run a 1-3 token warmup
   loop before the binary search; otherwise N_MAX appears artificially
   low. Documented in both 560M (SSN) and 3.9B (local) raw logs.

3. **`dfx deploy --mode upgrade` preserves stable memory.** Validated
   end-to-end: GGUF persisted across the C4 A/B wasm swap. Saves
   ~30-90 minutes per swap on ≥ 1 GB models.

4. **Default identity warning on `--network ic` is suppressible** with
   `export DFX_WARNING=-mainnet_plaintext_identity`. SSN mainnet
   measurement protocol relies on this.

## Phase 1 deliverables

### Code / infrastructure
- `papers/instruction-bounded-inference/artifact/data/paper_1_5/{models_paper_1_5,ternary_measurements,q6k_kernel_bench}.csv`
- `papers/instruction-bounded-inference/artifact/scripts/paper_1_5/{download_ternary_models,quantize_ternary_models,verify_tq2_wasm_path,upstream_pr_description,q6k_baseline_notes}.{py,sh,md}`
- `llama_cpp_canister/scripts/7-trilm-run-update-a.sh` (TriLM driver)
- `llama_cpp_canister/scripts/trilm-coherence-audit.sh` (10-prompt audit)
- `llama_cpp_canister/src/llama_cpp_onicai_fork/ggml/src/ggml-cpu/arch/wasm/quants.c` (Q6_K vectorized unpack patch)

### Tables / figures
- `results/paper_1_5/tables/c1a1-trilm-560m-summary.md`
- `results/paper_1_5/tables/c1a-projection-validation.md`
- `results/paper_1_5/figures/c1a-projection-validation.{py,png}`

### Raw artefacts (in `results/paper_1_5/raw/`)
- `trilm-{560m,3.9b}-{local,ssn}-binsearch.txt`
- `trilm-560m-{rep2,rep3}.log`
- `trilm-560m-coherence-audit.md`, `trilm-560m-ssn-vs-mainnet-diff.txt`
- `q6k-e2e-ab.txt`, `q6k-baseline-build.log`, `q6k-0.5b-{upload,load}.log`
- `{upload,load,deploy}-{560m,3.9b}.log` + ssn-prefixed equivalents
- `quantize-3.9b.log` (1.1 GB self-quantize trace)
- `build-clang-c1a1.txt`, `build-version-c1a1.txt` (build provenance)

## What's NOT in Phase 1 (deferred / held)

- **Task 30 — Q6_K upstream PR.** Patch is ready locally
  (`papers/instruction-bounded-inference/artifact/scripts/paper_1_5/upstream_pr_description.md`)
  but **HELD** for explicit user review before `gh pr create` (this
  is a public action against `ggml-org/llama.cpp`).
- **C2 mixed precision** and **C3 multi-call stateful** — these are
  Phase 2 scope per the design spec.

## Cross-reference

- Design spec: `docs/superpowers/specs/2026-04-19-paper-1.5-design.md`
- Plan: `docs/superpowers/plans/2026-04-19-paper-1.5-phase-1.md`
- First commit: `eebd37c docs(paper-1.5): add Phase 1 design spec and implementation plan`
- Last commit:  `db29b6a data(paper-1.5 C4): Qwen 0.5B Q6_K end-to-end A/B (patched vs baseline)`
- Full trail:   `git log --oneline main..HEAD -- papers/instruction-bounded-inference/artifact/ llama_cpp_canister/`

## Next: Phase 2

Phase 2 targets **C2 (mixed precision)** and **C3 (multi-call stateful
inference)**. See design spec §8 for phasing details. Plan TBD:
`docs/superpowers/plans/YYYY-MM-DD-paper-1.5-phase-2.md`.

The C1a result (α_eff ≈ 1.00 at both scales) suggests Phase 2 should
prioritise *workload* improvements (multi-call stateful, prompt cache
amortisation) over further *kernel* improvements — there's no kernel
margin left to recover end-to-end.
