# Parameter Conventions for α — `α_eff` (published-parameter) vs `α_exec` (executed-FLOPs)

**Origin.** Revision round 2 (2026-07-09) raised a definitional critique (C1): the paper
defined α_eff as "effective cost per FLOP" while computing it against `2 × P_published`,
where `P_published` includes parameters that never participate in per-token matmuls —
the input token-embedding table (a row *lookup*) in untied models, and learned position
embeddings. This note documents the corrected two-coefficient treatment; the recompute
script is `artifact/scripts/compute_alpha_exec.py`, its output
`artifact/results/current/scaling_law/alpha_exec.csv`.

## Definitions

- **α_eff** (retained symbol; a.k.a. α_param): `C_token ≈ α_eff × 2·P_published`.
  The *deployment predictor*: computable from a model card alone. This is the quantity
  fitted and LOAO-validated in §3 (median 1.527, CV 9.7%, n=11).
- **α_exec**: cost per *executed* FLOP. `P_exec = P_gguf_audited − token_embedding_table`
  (iff a separate output head exists — untied or export-duplicated) `− learned positions`.
  For tied models whose GGUF stores the table once, that table IS the executed LM head:
  `P_exec = P_gguf`. For encoder-only models the token table is always excluded (no LM head).

## Tie-status provenance (per fit model)

| Model | Tied? | Evidence |
|---|---|---|
| Pythia-70M / 14M | untied | GPT-NeoX `embed_in`/`embed_out`; separate `output.weight` in GGUF census |
| SmolLM2-135M / 360M | tied | HF `tie_word_embeddings: true`; no `output.weight` in GGUF |
| Gemma3-270M-IT | tied | Gemma family ties; no `output.weight` |
| OpenELM-270M | tied | shared input/output embedding (model card); no `output.weight` |
| Qwen2.5-0.5B | tied | HF config; fit-row Q4_0 GGUF audits at 494 M (single table). The Q8_0-instruct export duplicates the head (630 M stored) — stored duplication does not change α_exec |
| H2O-Danube3-500M | untied | separate `output.weight` in canonical GGUF (audited 513.6 M) |
| Mamba-370M | tied | no separate head in GGUF |
| Qwen3-0.6B | tied | HF config; GGUF 596 M single table |
| RWKV7-0.4B | untied | separate head (audited 450.9 M; table 67.1 M) |
| GPT-2 / DistilGPT2 | tied (wte = head) | exports duplicate the head as a stored tensor; executed once |
| TriLM 560M / 3.9B | untied | separate output table in F16 census (64.2 M / ≈154 M) |
| EmbeddingGemma-300M | encoder | no LM head; 201.3 M lookup table excluded from P_exec |

All fit models use rotary/no learned positions; GPT-2's learned positions (≈0.8 M) are
excluded from its P_exec.

## Results (2026-07-09 recompute)

- **Modern core (9 of 11):** α_exec median **1.536**, min 1.31, max 1.79, **CV 9.3%** —
  the cluster survives the corrected convention essentially unchanged (tied models are
  invariant by construction; Danube3 and RWKV7 shift within-cluster: 1.48→1.60, 1.56→1.63).
- **Small-dimension overhead regime:** Pythia-70M (d=512) α_exec **2.154**; Pythia-14M
  (d=128) α_exec **3.115**. Under the published-parameter convention these models
  *appeared* cluster-consistent (1.37, 1.70) because never-executed lookup parameters
  inflated the denominator. The executed convention reveals a genuine per-FLOP overhead
  regime at small hidden dimensions.
- **GPT-2 (legacy):** α_exec ≈ 2.05–2.07 — unchanged from the nominal convention (tied).
  The earlier observation that *audited stored* counts (+24–32%, duplicated tensors)
  compress GPT-2's α toward the cluster is itself a stored-convention artifact: duplicated
  tensors are not executed twice. The legacy code-path penalty is real under α_exec.
- **Ternary floor:** TriLM-560M α_exec = **1.166** (±3%); TriLM-3.9B ≤ **1.042**
  (N_MAX = 5, one-sided). The param-convention values (1.034 / ≤1.002) sit near exactly
  1.0 partly because TriLM's untied input table inflates `2·P_published` — a coincidence
  of convention. Under α_exec the floor decomposes consistently with Theorem 1 accounting:
  ≈0.5 units/FLOP of ternary arithmetic + unamortized per-weight load/dequant.
- **Encoder:** EmbeddingGemma-300M α_embed = 0.53 (published-parameter) → α_exec ≈ **1.61**,
  inside the modern cluster. The apparent "below-floor" coefficient was a lookup-table
  artifact (201.3 M of 308 M parameters never execute FLOPs in the forward pass); the
  encoder's real advantage is workload shape (batch amortization), not per-FLOP efficiency.

## Interpretation

α_eff (published-parameter) remains the right *planning* instrument — its input is public
and its LOAO transfer holds. α_exec is the right *physical* instrument — it unifies decode,
encode, and ternary results under one definition with no sub-floor anomalies, at the price
of requiring tensor-level knowledge of each export. The paper reports both, labeled.
