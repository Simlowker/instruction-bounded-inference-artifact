# MTEB(fra, v1) Off-Chain Baseline

Off-chain embedding quality benchmark for the on-chain inference paper.

## Quick Start

```bash
cd papers/instruction-bounded-inference/artifact

# 1. Activate venv
source mteb-venv/bin/activate

# 2. Login to HuggingFace (required for gated model)
huggingface-cli login

# 3. Run MTEB(fra, v1) — Retrieval, Reranking, STS
python scripts/run_mteb_fra.py
```

## Environment

| Package              | Version  |
|----------------------|----------|
| Python               | 3.13.12  |
| torch                | 2.11.0   |
| mteb                 | 2.12.16  |
| sentence-transformers| 5.4.1    |
| transformers         | 5.5.4    |
| numpy                | 2.4.4    |
| scipy                | 1.17.1   |
| pandas               | 3.0.2    |

## Tasks (MTEB(fra, v1) subset)

| Type      | Task                             |
|-----------|----------------------------------|
| Retrieval | AlloprofRetrieval                |
| Retrieval | BSARDRetrieval                   |
| Retrieval | MintakaRetrieval                 |
| Retrieval | SyntecRetrieval                  |
| Retrieval | XPQARetrieval                    |
| Reranking | AlloprofReranking                |
| Reranking | SyntecReranking                  |
| STS       | SICKFr                           |
| STS       | STSBenchmarkMultilingualSTS      |
| STS       | STS22                            |

## Outputs

- `mteb_fra_scores.csv` — per-task main scores
- `mteb_fra_scores.json` — full results with metadata
- `audit_set.json` — 80 sentences (40 FR + 40 EN) + 10 queries for on/off-chain fidelity test

## Audit Set Design

80 sentences across 8 domains (finance, legal, tech, science, culture, medical, daily, edge cases).
10 queries including 3 cross-lingual (FR query → EN docs, EN query → FR docs).

Edge cases: single words, numbers, very long sentences, pangrams.
