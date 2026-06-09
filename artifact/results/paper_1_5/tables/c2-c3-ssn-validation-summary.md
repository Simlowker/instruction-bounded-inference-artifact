# Paper 1.5 C2 + C3 SSN Cross-Env Validation (Task 12 completion)

**Date:** 2026-04-20
**Canister:** `u2pva-3iaaa-aaaax-qaa7a-cai` (SSN mainnet, 13-node consensus)
**Context:** Originally deferred from Phase 2 pending user confirmation;
run in Phase 3 follow-up.

## TL;DR

> **C2 `BASE-Q8_0` Pareto winners and C3 multi-call stateful demo both
> replicate on SSN mainnet with byte-identical outputs across 3 reps per
> config and coherent multi-turn state across 10 calls.** Cross-env
> tok/call delta is −2% (SmolLM2) and −6.7% (Qwen), within the SSN
> variance band observed on prior models.

This extends the Phase 1 SSN byte-determinism claim from single-shot
TriLM to (a) Q8_0 mixed-precision winners and (b) multi-call prompt-cache
state.

## C2 Pareto winners on SSN

| Model             | Variant     | Size (MB) | N_MAX local | N_MAX SSN | Δ     | α_eff SSN | Byte-exact |
|-------------------|-------------|----------:|------------:|----------:|------:|----------:|:----------:|
| SmolLM2-135M      | BASE-Q8_0   | 145       | 97          | **95**    | −2 %  | 1.559     | ✓ (3/3)    |
| Qwen 2.5 0.5B     | BASE-Q8_0   | 531       | 30          | **28**    | −6.7 %| 1.429     | ✓ (3/3)    |

3-rep wall-clock stats:

| Model       | rep1 (s) | rep2 (s) | rep3 (s) | mean  | stdev | CV     |
|-------------|---------:|---------:|---------:|------:|------:|-------:|
| SmolLM2     | 8.78     | 8.71     | 9.26     | 8.92  | 0.29  | 3.2 %  |
| Qwen 0.5B   | 8.89     | 8.89     | 8.71     | 8.83  | 0.10  | 1.2 %  |

## C3 multi-call chat demo on SSN

Model: Qwen 2.5 0.5B `BASE-Q8_0` (currently loaded after C2 measurement).
Opener: "Elric the blacksmith discovered that" → 10 consecutive
`run_update -n 10` calls with `--prompt-cache-all`.

| Metric                | Value       |
|-----------------------|-------------|
| Turns attempted       | 10          |
| Turns completed       | **10**      |
| Cumulative tokens     | 100         |
| Total wall-clock      | **41.12 s** |
| Mean per-turn wall    | 4.11 s      |
| State continuity      | ✓ (narrative coherent across calls) |
| Byte-exact state      | ✓ (13-node consensus) |

Local reference (Task 18 re-run on Qwen 0.5B Q4_0): 72.36 s / 10 turns.
SSN is **faster** in this run because it runs Q8_0 (less runtime dequant
overhead) — a minor wasm-mechanics artefact, not an SSN-vs-local signal.

Full transcript: `tables/c3-ssn-chat-demo-transcript.md`.

## New operational finding — wasm heap pagination ceiling

During initial Qwen Q8_0 load on SSN, `load_model` with default args
(context = n_ctx_train = 32 768) trapped **IC0502 "heap out of bounds"**
after `sched_reserve` completed. Canister logs showed:

    weights        = 531 MiB
    KV cache       = 384 MiB (32 768 ctx, f16)
    compute buffer = 300 MiB
    total          ≈ 1 215 MiB

`-c 2048` reduced KV to 24 MiB but compute buffer stayed 302 MiB →
still OOB. **Fix: `-c 512 -b 512`** → total ≈ 850 MiB, loads cleanly.

**Implication:** the practical wasm heap ceiling on SSN (and likely
other 34-node subnets) is closer to ~900 MB than the 3 GB
`wasm_memory_limit` would suggest, for any model path that does a
single large `sched_reserve` allocation. Phase 3 BitNet 2B4T planning
(expected ~1.8 GB weights alone) must budget ctx/batch accordingly.

Distinct from **IC0524 stable-memory page-access ceiling** (~1 GB
demonstrated in Phase 1 on Qwen 1.5B Q6_K): different error code,
different allocator, different fix.

## Operational receipt — canister reinstall + state management

Two IC0502 patterns observed during this task:

1. **Load-N after load-N-1 with a larger model (same file):** the
   residual heap state from the prior load is incompatible with a
   bigger allocation. Fix: `dfx canister install --mode reinstall`
   then re-upload. (Same pattern observed locally during Task 18
   diagnostic — confirms it's not SSN-specific.)
2. **First-time load of a model that needs more heap than the
   `sched_reserve` path can get:** fix above (`-c 512 -b 512`).

Canister was left with Qwen 2.5 0.5B BASE-Q8_0 loaded. `canister_ids.json`
was restored to its pre-task target (`zmm32-7yaaa` ICP mainnet) after
measurements completed.

## Cycles / cost

SSN is a rental subnet (no per-call cycles intra-subnet per
`swiss_subnet_beta.md`). Actual cycle burn observed: 0.

Total artefact cost of Task 12:
- 145 MB SmolLM2 Q8_0 upload (~5 min wall-clock)
- 531 MB Qwen Q8_0 upload (~7 min at 1.27 MB/s)
- 1× canister reinstall after TriLM 3.9B TQ2_0 heap conflict
- 1× canister upgrade (failed to clear heap — reinstall was needed)

## Artefacts

- `data/onchain/ssn_mainnet.csv` — 3 new rows (SmolLM2, Qwen tok/call, Qwen chat demo)
- `results/paper_1_5/raw/c2-ssn-smollm135-q8.log`
- `results/paper_1_5/raw/c2-ssn-qwen05-q8.log`
- `results/paper_1_5/raw/c3-ssn-chat-demo-qwen05.log`
- `results/paper_1_5/tables/c3-ssn-chat-demo-transcript.md`
- `scripts/paper_1_5/c3_ssn_chat_demo.sh` (SSN-specific chat demo driver)
