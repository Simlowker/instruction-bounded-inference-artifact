# C3 — Multi-call Stateful Inference Summary

> Task 19 consolidation. Covers Tasks 13–19 of Phase 2 (C3 multi-call
> characterization), with Falcon-H1-Tiny-90M serving as an SSM baseline
> against the Qwen 2.5 0.5B Transformer reference.

## Methodology

Paper 1 gave the per-call instruction budget as

    tok/call ≈ 40e9 / (α_eff × 2P)

where `α_eff` is the effective instructions-per-parameter overhead and
`P` is the number of live parameters (Paper 1 §4). For a multi-call
session using llama.cpp's `--prompt-cache-all` mechanism, each call pays
two extra instruction costs that the per-call formula ignores:

- **save\_instr** — serialize the growing KV cache (or Mamba state)
  back to stable storage at end-of-call;
- **load\_instr** — deserialize the cache at start-of-call (for calls
  2..N, the "reattach").

This gives the extended session budget

    tok/session ≈ N_calls × 40e9 / (α_eff × 2P)
                  − N_calls × IO_overhead(n_kv_tok, cache_type)

with

    IO_overhead(n, c) = save_instr(n, c) + load_instr(n, c)
                      ≈ (a_s + b_s · n) + (a_l + b_l · n)

fitted from the measurements below. The key question for Paper 2
(certified agents) is whether `b_s` and `b_l` are zero for bounded-state
models (SSMs) and positive for Transformers (growing KV). The data
answers: **yes**, with caveats.

Canister WASM used the C3-IO instrumentation (commit `bb238b9` on
`llama_cpp_canister` main) which emits a `C3-IO-END:` log line after
each `run_update` carrying `load_instr`, `save_terminal_instr`,
`save_terminal_tokens`, and `save_terminal_bytes`.

## Measurements

All rows below come from
`data/paper_1_5/multicall_characterization.csv`; full raw dfx canister
logs are in `results/paper_1_5/raw/c3-*.log`.

### Qwen 2.5 0.5B Q4_0 (Transformer, growing KV, f16 cache)

Source: Task 13 (`c3-ioover-qwen05-n*`, single-call sweep) + Task 14
(`c3-ioover-qwen05-mc-step*`, 5-call chain).

**Single-call save scan (fresh cache, generating N tokens):**

| sid   | save_t_tokens | save_t_bytes | save_instr     | load_instr |
|-------|--------------:|-------------:|---------------:|-----------:|
| n1    | 10            | 123 653      | 35 877 338     | 0          |
| n5    | 14            | 172 869      | 40 282 639     | 0          |
| n10   | 19            | 234 389      | 44 648 698     | 0          |
| n15   | 24            | 295 909      | 49 779 767     | 0          |
| n20   | 29            | 357 429      | 53 986 888     | 0          |

Fit: `save_instr ≈ 26.7M + 0.95M × n_tok` (R²=0.9984).

**Multi-call load (reattach, 24-token cache):**

| step | save_instr     | load_instr     | mode       |
|-----:|---------------:|---------------:|------------|
| 1    | 49 893 478     | 0              | fresh      |
| 2    | 44 773 789     | 425 546 749    | continue   |
| 3    | 43 464 626     | 425 644 580    | continue   |
| 4    | 43 541 546     | 425 744 886    | continue   |
| 5    | 43 588 402     | 425 639 466    | continue   |

`load_instr` mean = **425.6 M**, stdev = 80.9 K, **CV = 0.02 %** — i.e.
load overhead is essentially *constant* at small cache sizes, dominated
by the deserialization + tensor-allocation fixed cost. A linear term in
`n_tok` is indistinguishable from zero across this 24-token range.

**Observations:**

- Multi-call tax at 100-tok target: 24 % tok/s reduction vs single-call
  reference (Task 16, `c3-ncall-qwen05-*`).
- Re-attach latency (cold load_model + state load): ~2.39 s cold vs
  2.32 s warm — the cost is `load_model`, not state IO (Task 15).
- KV cache type `--cache-type-k f16|q8_0|q4_0` is a *no-op* on the
  current WASM: outputs are bit-identical and tok/s within 4 % noise
  (Task 17). KV is allocated at `load_model` time before the flag is
  respected by the cache-swap path; reallocating requires a WASM rebuild
  (flagged for follow-up).

### Falcon-H1-Tiny-90M Q8_0 (hybrid Mamba+Transformer, f16)

Source: Task 19 (`c3-ssm-falcon-h1-step*`, 5-call chain on local
replica). Model: `tiiuae/Falcon-H1-Tiny-90M-Instruct` → f16 GGUF →
Q8_0 (94 MB). Canister ctx fixed at `-c 2048` (model advertises 256 k,
which traps ggml heap-alloc for the full KV budget at ~3 GB).

**Single-call N_MAX:** 100 tok/call (sustained, deterministic). Seed
probe at N=150 returned OK with `generated_eog=true` in 13.9 s — the
sampler hit a natural EOG long before consuming the 40 B instruction
budget. However all three confirmation reps at N=150 failed with
IC0522 because sampling variability lets the model keep going far
enough to cross the limit. We therefore report the reliable floor
(N=100, wall 18.9 s, ~5.3 tok/s sustained) rather than the optimistic
peak. Per-memory SSN Phase-1 reference was 199 tok/call on a different
WASM configuration; we do not achieve it on this local build. Full
binsearch trace: `ok=[50 100]`, `fail=[151 153 156 162 175 200]`,
`marginal=[150 once]`.

**Multi-call IO overhead (5× run_update, N=15 per call, new_chat
fresh at step 1, `--prompt-cache-all` chain):**

| step | n_tok_in_cache | save_instr   | load_instr   | save_bytes | load_bytes | wall_s |
|-----:|---------------:|-------------:|-------------:|-----------:|-----------:|-------:|
| 1    | 52             | 301 163 543  | 0            | 5 617 665  | -          | 5.94   |
| 2    | 66             | 311 880 388  | 104 518 822  | 5 789 921  | 5 617 665  | 2.99   |
| 3    | 80             | 320 384 720  | 107 066 201  | 5 962 177  | 5 789 921  | 3.01   |
| 4    | 94             | 323 907 725  | 109 523 464  | 6 134 433  | 5 962 177  | 3.07   |
| 5    | 108            | 331 449 562  | 111 977 696  | 6 306 689  | 6 134 433  | 3.15   |

**Observations on the SSM hypothesis.**

- `save_bytes` grows by **12 306 B per token** (slope: (6 306 689 −
  5 617 665) / (108 − 52) = 12.3 kB/tok). This is *not* a bounded
  state — it is effectively the same per-token growth rate as Qwen's
  transformer KV (~12 kB/tok). Falcon-H1 is a **hybrid**: Mamba blocks
  interleaved with Transformer attention blocks, and the attention
  blocks still contribute a growing KV cache. The ~5.6 MB baseline at
  n=52 reflects the bounded Mamba state (conv buffers + SSM latents)
  *plus* initial attention KV; the growth rate is the attention KV.
- `save_instr` fit: **276 M + 0.52 M × n_tok** (R²=0.972). Compare to
  Qwen's **26.7 M + 0.95 M × n_tok** — Falcon-H1's per-token save cost
  is **45 % lower** than Qwen's (0.52 M vs 0.95 M), but its intercept
  is ~10× higher because the fixed Mamba state is ~6 MB (vs Qwen's
  ~120 kB). The crossover point where Falcon-H1's save becomes cheaper
  than Qwen's is around n_tok ≈ (276 − 27) / (0.95 − 0.52) ≈ 580 tokens
  in cache — beyond the single-call capacity on this WASM.
- `load_instr` mean = **108 M** (CV = 3 %), growing from 104.5 M at
  n=52 to 112.0 M at n=94 → slope ≈ 180 K instr/tok. This is still
  dominated by the ~100 M fixed cost but not *constant* like Qwen's
  425.6 M was. Falcon-H1 load is **4× cheaper in absolute terms** than
  Qwen's load despite the serialized state being ~18× larger in bytes
  — the load-side savings come from deserializing monolithic Mamba
  tensors rather than per-layer attention KV blocks.

**Bottom line.** Falcon-H1-Tiny-90M is a *partial* win on C3: the
per-token save growth rate is halved (0.52 M vs 0.95 M), but not
eliminated as a pure-Mamba model would predict. The big headline
is the fixed load cost: **108 M vs 426 M** — a 4× win on the re-attach
path per call, which matters more for short-turn agents than raw
per-token slope.

## Fit parameters

Run `c3_multicall_analysis.py` to regenerate. Values:

| model / cache                               | a_s (M) | b_s (M/tok) | R²    | load_mean (M) | load CV |
|---------------------------------------------|---------|-------------|-------|---------------|--------:|
| Qwen 2.5 0.5B Q4_0 / f16 (Transformer)      | 26.7    | 0.951       | 0.998 | 425.6         | 0.0002  |
| Falcon-H1-Tiny-90M Q8_0 / f16 (hybrid SSM+T) | 276.3   | 0.519       | 0.972 | 108.3         | 0.030   |

Corrected Paper-1.5 headline claim (vs the original hypothesis of zero
slope): **for the hybrid Mamba+Transformer tiny model we measured,
`b_s` drops by ~45 %** compared to the pure-Transformer baseline
(0.52 M vs 0.95 M per KV token), but it is **not zero** — the
attention blocks still contribute a growing KV. The SSM/Mamba
contribution shows up primarily as (i) a 4× reduction in the
per-call `load_instr` fixed cost (108 M vs 426 M) and (ii) a much
larger fixed state footprint (5.6 MB vs 0.12 MB baseline save_bytes),
which pushes the crossover at which Falcon-H1 becomes cheaper to save
per-token out to ~580 KV tokens — beyond the reach of a single ICP
update call on this WASM.

## Implications

- **Short multi-turn agents (cache ≪ 100 tok).** The constant term `a`
  dominates; the relevant number is the per-call IO floor. On this
  WASM that's ~452 M instr/call for Qwen (26.7 save + 425.6 load) vs
  ~384 M for Falcon-H1 (276 save + 108 load). Hybrid SSM wins by
  ~15 % on the cold-path.
- **Medium-length sessions (cache ~100 tok).** Qwen's total per-call
  IO at n=100 ≈ 26.7 + 95 (save) + 425.6 (load) ≈ 547 M. Falcon-H1 at
  n=100 ≈ 276 + 52 + 108 ≈ 436 M. Falcon still wins by ~20 %.
- **Long chat sessions (cache ≫ 500 tok).** This is where the slope
  difference matters most: at n=1000, Qwen's save alone extrapolates
  to ~980 M instr/call vs Falcon's ~795 M (~20 % cheaper save), plus
  Falcon's 4× lower load. Combined per-call IO tax at n=1000 would be
  Qwen ~1.4 B vs Falcon ~0.9 B. This is still within the 40 B budget,
  but the extra headroom lets Falcon generate more user-visible tokens
  per call.
- **Re-attach cold path dominated by `load_model`**, not state IO.
  Optimising serialization size does not help cold-path latency until
  `load_model` itself is cached or memoized. Task 15 showed the
  `load_model` step dominates the re-attach cost at ~2.3 s wall-clock
  for Qwen 0.5B; we did not repeat the measurement for Falcon but
  expect it to be proportional to model-size differences.

## Figures

Generated by `c3_multicall_figures.py` from
`data/paper_1_5/multicall_characterization.csv`:

- `results/paper_1_5/figures/c3-save-instr-vs-ntok.{png,pdf}` — per-call
  save cost vs KV size (Qwen vs Falcon-H1-Tiny).
- `results/paper_1_5/figures/c3-load-instr-vs-ntok.{png,pdf}` — per-call
  load cost across multi-call steps (Qwen ~426 M constant vs Falcon
  ~108 M).
- `results/paper_1_5/figures/c3-save-bytes-vs-ntok.{png,pdf}` — serialized
  cache size on log-y (Falcon ~50× larger fixed state footprint).

## Forward reference

These fits feed directly into Paper 2 (certified agents, forthcoming):
the session-level instruction budget is the relevant cost input for any
on-canister agentic workload requiring privacy guarantees
(on-canister state) plus responses of ≥100 tokens. Task 21 will
produce a cross-paper figure overlaying these fits with the
`tok/call ≈ 40e9 / (α_eff × 2P)` single-call curve from Paper 1.
