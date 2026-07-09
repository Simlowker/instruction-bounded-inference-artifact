# Root-cause of the anomalous SmolLM2-135M variance session [150, 150, 75] (2026-04-16 21:45)

**Question (revision round 2, R1).** `variance.csv` holds two same-evening SmolLM2-135M
Q4_0 sessions: 21:45:57 → per-prompt values [A=150, B=150, C=75] (CV ≈ 35%) and
21:55:38 → [97, 97, 97] (CV 0%, the adopted anchor). What was the first session
measuring?

**Method (2026-07-09, local replica, zero cycles).** Redeployed the *byte-original*
April GGUF (`smollm2-135m-Q4_0.gguf`, 91 726 912 bytes, sha256 `8d7a7726…`, file dated
2026-04-05) on the current canister build (`8a5ecf61…`), and probed the single-call
budget boundary and the per-prompt EOG behavior under the documented binary-search
protocol (same `run_update` args, warm state).

## Findings

1. **The anomalous values are impossible under the single-call N_MAX metric.**
   A single `-n 150` call traps `IC0522` (instruction limit) — as it must: the
   single-call boundary is 94–95 tokens on the current build (`testA-n150.log`,
   boundary probes n=96 trap / n=94–93 OK) and was 97 on the April build. A recorded
   value of 150 therefore **cannot be a single-call generation count on any measured
   build**. The 21:45 session was measuring a different quantity — consistent with
   cap-limited *cumulative multi-call* generation (cap = 150; e.g. 97 + 53 via
   prompt-cache continuation), a mechanism the multi-call machinery of §6.2 provides.
2. **The sub-cap value (C = 75) is consistent with per-prompt early termination under
   that metric, but is not reproducible on the current build**: prompt C
   ("In machine learning a") generates ≥ 90 tokens with `generated_eog = false` today
   (`testC-n90.log`). Completion trajectories on this Q4_0/llama path are
   build-sensitive (unlike the byte-stable TQ2_0/TriLM path), so the April-build
   trajectory that may have ended early at 75 cannot be replayed without the April
   binary.
3. **The adopted [97, 97, 97] session matches the documented binary-search protocol
   exactly** and its value sits at the April-build boundary (97), adjacent to today's
   94–95 — a fresh, quantified build-drift data point (−2…−3%, inside the paper's
   ±15% Limitation 4 band).

## Verdict

The anomaly is a **metric mismatch, not nondeterminism**: the 21:45 session recorded
cap-limited multi-call totals (metric later abandoned), the 21:55 session recorded
single-call N_MAX (the paper's metric). The paper's caption wording is updated from
"whose cause was not recorded" to state this identification.

## Operational finding (bonus)

Loading a **stale prompt-cache file created by a different model** traps the canister
(`IC0502 unreachable`): after swapping models on the same canister, `new_chat` with the
old cache name is not sufficient — use a fresh cache filename (or remove the file).
Reproduced twice before diagnosis; relevant to any multi-model canister workflow.

## Files

| File | Content |
|---|---|
| `testA-n150.log` | Single call `-n 150` → IC0522 (impossibility proof) |
| `testC-n97.log`, `testA-n97.log` | `-n 97` traps on current build (boundary moved 97→94/95) |
| `testC-n90.log` | Prompt C generates 90 tokens, `generated_eog=false` (EOG-today refuted) |
| `boundary probes` | n=96 trap; n=94, 93 OK (logged in session transcript) |

## Operational finding #2 (same session)

**Canister upgrade does not evict the loaded model.** After `dfx deploy -m upgrade`,
generation kept working with no `load_model` call — icpp-pro orthogonal persistence
restores the wasm heap, so the previously-loaded model (and its memory footprint)
survives upgrades. Consequence: swapping to a *larger* model on a long-lived canister
requires `-m reinstall` (wiping the stable FS, hence a full model re-upload) — an
upgrade alone leaves the old model resident and the new load traps
`IC0502 heap out of bounds`. Together with finding #1 (stale prompt-cache from a
different model also traps IC0502), this defines the safe model-swap protocol:
reinstall → upload → load with explicit `-c`/`-b` → fresh cache filename.
