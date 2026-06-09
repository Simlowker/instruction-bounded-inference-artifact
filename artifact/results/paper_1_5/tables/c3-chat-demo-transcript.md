# C3 Multi-Call Chat Demo — 10-Turn Transcript

**Date:** 2026-04-20T20:07:25+00:00
**Model:** Qwen 2.5 0.5B Q4_0 (`models/c3-qwen05-q4.gguf`)
**KV cache type:** f16 (default; --cache-type flags are no-ops on this wasm, see Task 17)
**Network:** local dfx replica
**N per turn:** 10 tokens
**Turns attempted:** 10  (**completed successfully:** 10)
**Cumulative tokens (successful turns):** 100
**Total wall-clock (all attempts):** 72.356 s
**Mean per-turn wall:** 7.236 s

## Pivot note

Original Phase 2 plan specified **Qwen 2.5 1.5B-Instruct Q4_0** for the chat demo.
During pre-check (Task 20), Qwen 1.5B Q4_0 was found to trap IC0502 'unreachable'
on the current wasm build. Q8_0 on 1.5B is predicted to trip IC0524 (~1.7 GB > 1 GB
stable-memory ceiling, matches the paper's §5.6 observation). We therefore
**pivoted to Qwen 0.5B Q4_0** — proven stable across Tasks 14/15/16/17. This is a
smaller base model (no RLHF); the paper's C3 coherence claim is about
**bit-consistent state across canister calls**, not literary quality.

## Diagnostic note (Phase 3 re-run)

First attempt on 2026-04-20T19:22 trapped IC0502 "heap out of bounds" on turn 1
(wall 0.98 s, well before any inference could complete). Root cause: **stale
canister state** after Tasks 14–17 (many `new_chat` + `prompt-cache-all`
sequences without reinstall). Reproduction under a fresh `dfx start --clean` +
canister reinstall succeeded on the first attempt (this transcript). Operational
implication: chain-of-many-sessions tests should reinstall the canister between
experimental axes; for multi-turn demos specifically, ensure no leftover
`.canister_cache/*/sessions/` state from prior tests.

## Coherence observations

- No obvious coherence breaks detected by automated heuristics.

## Per-turn summary

| Turn | Wall (s) | Mode | load_tokens | load_bytes | save_tokens | save_bytes | save_instr | load_instr |
|------|----------|------|-------------|------------|-------------|------------|------------|------------|
| 1 | 9.878 | fresh | -1 | -1 | -1 | -1 | -1 | 0 |
| 2 | 7.390 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 3 | 7.276 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 4 | 7.170 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 5 | 7.026 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 6 | 7.072 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 7 | 6.791 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 8 | 6.683 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 9 | 6.565 | continue | -1 | -1 | -1 | -1 | -1 | 0 |
| 10 | 6.505 | continue | -1 | -1 | -1 | -1 | -1 | 0 |

## Per-turn emitted text (first ~100 chars)

**Turn 1:** ` a certain quantity of copper can be reduced to iron`

**Turn 2:** ` 15.5 grams of iron. If`

**Turn 3:** ` The blacksmith found that for every 10`

**Turn 4:** `2.5 grams of iron, 2.`

**Turn 5:** `5 grams of carbon dioxide are produced. How many`

**Turn 6:** ` many grams of copper are needed to produce 1`

**Turn 7:** `30.5 grams of iron? To determine`

**Turn 8:** ` determine how many grams of copper are needed to produce`

**Turn 9:** ` produce 30.5 grams of iron,`

**Turn 10:** `, we start by understanding the relationship between the quantities`

## Full concatenated narrative

Opener (user-provided seed):

> Elric the blacksmith discovered that

Model continuation across 10 successful turns (of 10 planned):

```
 a certain quantity of copper can be reduced to iron 15.5 grams of iron. If The blacksmith found that for every 102.5 grams of iron, 2.5 grams of carbon dioxide are produced. How many many grams of copper are needed to produce 130.5 grams of iron? To determine determine how many grams of copper are needed to produce produce 30.5 grams of iron,, we start by understanding the relationship between the quantities
```

## Final conversation buffer (turn 10)

```
Elric the blacksmith discovered that a certain quantity of copper can be reduced to 15.5 grams of iron. The blacksmith found that for every 12.5 grams of iron, 25 grams of carbon dioxide are produced. How many grams of copper are needed to produce 30.5 grams of iron? To determine how many grams of copper are needed to produce 30.5 grams of iron, we start by understanding the relationship between the
```
