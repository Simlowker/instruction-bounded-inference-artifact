# TriLM 560M TQ2_0 — Standard ICP Mainnet Validation (2026-07-09)

**Purpose.** Close the one environment gap in the paper's R3 evidence: the ternary
α_eff ≈ 1 floor and byte-exact determinism were previously demonstrated on the
local dfx replica and the Swiss Subnet (SSN) only — never on a **standard ICP
mainnet application subnet**. This campaign replicates both results on the
public network, and re-establishes the paper's "Live deployments" line
(canister had been frozen out of cycles and protocol-uninstalled since ~June).

Run as a follow-up to the 2026-07-08 pre-deposit adversarial panel review
(repairs the R7 archival gap: this time every side of every diff is committed).

## Environments

| Side | Environment | Canister | Subnet |
|---|---|---|---|
| A | local dfx replica (`dfx` 0.29.x, clean state) | `uxrrr-q7777-77774-qaaaq-cai` | local |
| B | **ICP mainnet, standard application subnet** | `zmm32-7yaaa-aaaad-qlqsq-cai` | `5kdm2-62fc6-fwnja-hutkz-ycsnm-4z33i-woh43-4cenu-ev7mi-gii6t-4ae` |

Subnet IDs verified via the public IC API (`ic-api.internetcomputer.org`).
For contrast, the April SSN runs lived on subnet `3zsyy-cnoqf-…` (Swiss Subnet)
— a different production subnet of the same network.

## Provenance (all pinned)

| Item | Value |
|---|---|
| Canister WASM (identical both sides) | sha256 `8a5ecf61031f236487d233ad4ee637c81215c4b85db5169ed176406f1cdde472` |
| WASM build | Comoto fork, built 2026-07-03, repo `Simlowker/gian` @ `a080c5f` (release line incl. `feat/lfm2-enable` merge `4a22a16`); on-chain module hash verified equal on A and B |
| Model GGUF | `TriLM_560M_Unpacked.TQ2_0.gguf`, 204 612 288 bytes, sha256 `c9d5324b3c006242e3f2564ff963c593b633538744373770daacbe9722e850c3` |
| GGUF provenance | **Rebuilt from scratch 2026-07-09** — HF `SpectraSuite/TriLM_560M_Unpacked` snapshot `5bf1f28cbac165628fe1777d56b91c6afb5c039b` → `convert_hf_to_gguf.py --outtype f16` (fork tree) → `llama-quantize … TQ2_0` (Homebrew llama.cpp 8660). This is an **independent conversion pipeline** from the April 2026 GGUF. |
| Upload integrity | Canister-side sha256 after chunked upload identical to local file on **both** A and B |
| Protocol | Identical to April (`trilm-560m-local-binsearch.txt`): prompt `"The capital of France is"`, `run_update` args `--model models/model.gguf --prompt-cache prompt.cache --prompt-cache-all -sp -p <prompt> -n N`, `set_max_tokens 10000` (non-binding), `new_chat` before each run, warm protocol (see note) |

## Result 1 — Floor replication on standard mainnet

| Measurement | Local (A) | **Mainnet standard (B)** | April local | April SSN |
|---|---|---|---|---|
| N_MAX (3 reps) | **34, 34, 34** (CV 0%) | **34, 34, 34** (CV 0%) | 34 | 34 |
| Boundary N=35 | trap `IC0522` | trap `IC0522` | trap | trap |
| Implied α_eff (P=569M, B=40e9) | 1.03 | **1.03** | 1.03 | 1.03 |

Raw logs: `local-rep{1,2,3}-n34*.log`, `local-rep1-n35-warm.log`,
`ic-rep{1,2,3}-n34.log`, `ic-rep1-n35.log`.

**Warm-up note.** The first full-budget call after `load_model` is cold (heap
page-in of the 195 MB weights) and traps `IC0522` even at N=34 (observed on A;
logged in `local-rep1-n34.log` at the campaign scratch level). One small warm-up
call (`-n 2`) restores the warm regime; all reported N_MAX values are warm,
matching the April protocol. Consistent with the paper's cold/warm operational
finding.

## Result 2 — Byte-exact determinism across three environments

10-prompt coherence audit (April prompt set, `-n 24`, deterministic sampling):

- **`diff audit-local.txt audit-ic.txt` → EMPTY.** 10/10 completions byte-identical
  between the local replica and the standard-mainnet canister, same WASM build,
  same GGUF (files committed here, both sides).
- Prompt #6 (contains escaped quotes) re-captured raw on both sides:
  `local-audit-p6-raw.log` ↔ `ic-audit-p6-raw.log` — identical.
- All 10 completions are **word-for-word equal to the April 2026 local audit**
  (`trilm-560m-coherence-audit.md`), which was itself byte-identical to the
  April SSN run (`trilm-560m-ssn-vs-mainnet-diff.txt`, empty diff).
- N=34 generation output equals April's logged N=34 string exactly
  ("… There are so many").

**Net claim supported:** identical output bytes across
(i) three execution environments — local replica, Swiss Subnet (April),
standard ICP mainnet subnet (this campaign);
(ii) two build generations three months apart (April build vs `8a5ecf61`); and
(iii) two independent GGUF conversion pipelines (April's file vs the
2026-07-09 rebuild — different tooling versions, same tensor bytes semantics).

**Scope note (honest bounds).** Local↔mainnet equality is a *direct* diff at
identical WASM+GGUF (this campaign). Equality with April SSN is *via* the
archived April audit table (textual, all 10 completions + the N=34 string);
the April SSN raw completions file itself was not archived (known R7 gap) —
the April empty-diff note plus today's word-for-word match against the April
local table is the evidence chain.

## Cost accounting (cycles)

Campaign total ≈ **1.05 TC** on the mainnet canister: WASM install + 195 MB
chunked upload (dominant, ≈ 4.4k cycles/byte all-in) + ~25 update calls
including 4 full-budget `IC0522` boundary probes. Canister left running with
the model loaded and ≈ 0.95 TC balance (live deployment restored).

## Files in this directory

| File | Content |
|---|---|
| `local-rep1-n34-warm.log` … `local-rep3-n34.log` | Local N=34 runs (full Candid responses) |
| `local-rep1-n35-warm.log` | Local N=35 boundary trap |
| `ic-rep1-n34.log` … `ic-rep3-n34.log` | Mainnet N=34 runs |
| `ic-rep1-n35.log` | Mainnet N=35 boundary trap (IC0522) |
| `audit-local.txt`, `audit-ic.txt` | 10-prompt audit outputs, both sides (diff = empty) |
| `local-audit-p6-raw.log`, `ic-audit-p6-raw.log` | Raw prompt-#6 responses (escaped-quote case) |
