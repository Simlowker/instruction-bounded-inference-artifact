# Load-cost slope sweep (b_l) — Qwen 2.5 0.5B Q4_0, current release build (2026-07-09)

**Question (round 2, R1).** Table 3's April fit reported the Qwen load cost as a
near-constant (`b_l ≈ 0` convention). Measure the load slope directly.

**Protocol.** Local replica, zero cycles. Model `c3-qwen05-q4.gguf` (the C3 campaign
file), loaded with `-c 2048 -b 512` (§6.3 recipe), fresh prompt cache. One growing
chat session: turn 1 seeds a prompt; turns 2–16 continue with an empty prompt and
`-n 15` (`--prompt-cache-all`), so the KV grows ~15 tokens/turn. After each call the
instrumented `C3-IO-END` log line exposes `load_instr` / `load_tokens` for the
state-load that opened the call. Raw per-turn logs archived here.

**Result (15 points, n_kv 21 → 217):**

    load_instr ≈ 43.0 M + 0.1769 M · n_kv    (R² = 0.9998)

The load cost is linear in cache size with a small but **non-zero** slope — about 19%
of the April save slope (b_s = 0.951 M/tok). The earlier `b_l ≈ 0` reading is an
approximation that holds at small caches; for long sessions the load growth term
belongs in the session budget alongside the save term.

**Build note.** These values are for the current release build (`8a5ecf61…`), whose
load path differs substantially from the April campaign build (Table 3's Qwen load
mean was 425.6 M on that build; the current build loads ~10× cheaper at comparable
cache sizes). Per-build pinning applies (Limitation 4); the *linearity* and the
existence of a non-zero slope are the structural findings.

**Data:** `load_sweep.csv` (call, load_tokens, load_instr) + `turn*-c3io.log` (raw).
