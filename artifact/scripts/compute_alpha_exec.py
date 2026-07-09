#!/usr/bin/env python3
"""Recompute alpha under the executed-FLOPs convention (C1 fix, revision round 2).

Two coefficients (see paper §2):
  alpha_param — C_token ≈ alpha_param × 2·P_published. Deployment predictor:
                computable from a model card. This is the quantity fitted in §3.
  alpha_exec  — cost per *executed* FLOP. P_exec excludes parameters that never
                participate in per-token matmuls: the input token-embedding table
                (a row lookup) when a separate output head exists, and learned
                position embeddings. For tied models whose GGUF stores the table
                once, that single table IS the executed LM head → P_exec = P_gguf.
                For encoder-only models (no LM head), the token table is always
                excluded.

Inputs : ../data/models.csv (registry; params_M_gguf = audited tensor totals of
         the exact benchmarked file; vocab_size, hidden_dim for the table size).
Output : ../results/current/scaling_law/alpha_exec.csv

Tie-status provenance (TIED dict below): determined by tensor census of the
canonical GGUFs (presence/absence of a separate `output.weight` tensor),
cross-checked against published model configs (tie_word_embeddings). Pythia
(GPT-NeoX embed_in/embed_out) and RWKV7 (separate head) are untied; positions
are rotary (no learned table) for all fit models.
"""
import csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "..", "data", "models.csv")
OUT = os.path.join(HERE, "..", "results", "current", "scaling_law", "alpha_exec.csv")
B = 40e9

# (name, quant preference for the §3 fit row, tied, learned_pos_params_M)
CONFIG = [
    ("Pythia-70M",        ["Q4_0"],        False, 0.0),
    ("SmolLM2-135M",      ["Q4_0"],        True,  0.0),
    ("Gemma3-270M-IT",    ["Q4_0"],        True,  0.0),
    ("OpenELM-270M",      ["Q4_0"],        True,  0.0),
    ("SmolLM2-360M",      ["Q4_0"],        True,  0.0),
    ("Qwen2.5-0.5B",      ["Q4_0"],        True,  0.0),
    ("H2O-Danube3-500M",  ["Q4_0"],        False, 0.0),
    ("Pythia-14M",        ["Q4_0"],        False, 0.0),
    ("Mamba-370M",        ["Q8_0", "Q4_0"], True,  0.0),
    ("Qwen3-0.6B",        ["Q8_0", "Q4_0"], True,  0.0),
    ("RWKV7-0.4B",        ["Q8_0", "Q4_0"], False, 0.0),
    # Legacy sensitivity rows (excluded from the modern fit; duplicated-head exports →
    # subtract the input table per the general rule; learned positions excluded):
    ("DistilGPT2",        ["Q4_0", "Q8_0"], False, 0.6),
    ("GPT2-124M",         ["Q4_0", "Q8_0"], False, 0.8),
]
SMALL_DIM_REGIME = {"Pythia-70M", "Pythia-14M"}  # d ≤ 512 AND gpt_neox: confounded at n=2 (observation, not a law)
LEGACY = {"DistilGPT2", "GPT2-124M"}

def fit_row(rows, name, quants):
    for q in quants:
        for r in rows:
            if r["name"] == name and r["quant"] == q and r["simd"] == "yes" and r["tok_call_local"]:
                return r
    return None

def main():
    rows = list(csv.DictReader(open(REG)))
    out = []
    for name, quants, tied, posM in CONFIG:
        r = fit_row(rows, name, quants)
        if r is None:
            raise SystemExit(f"fit row not found: {name}")
        P_nom = float(r["params_M"]); P_gguf = float(r["params_M_gguf"])
        V = float(r["vocab_size"]); d = float(r["hidden_dim"]); tok = float(r["tok_call_local"])
        table_M = V * d / 1e6
        P_exec = P_gguf - (0.0 if tied else table_M) - posM
        a_param = B / (tok * 2 * P_nom * 1e6)
        a_exec = B / (tok * 2 * P_exec * 1e6)
        out.append(dict(name=name, quant=r["quant"], tok_call=int(tok), tied=tied,
                        params_M_nominal=P_nom, params_M_gguf=P_gguf,
                        embed_table_M=round(table_M, 1), params_M_exec=round(P_exec, 1),
                        alpha_param=round(a_param, 3), alpha_exec=round(a_exec, 3),
                        regime=("legacy_code_path" if name in LEGACY else
                                "small_dim_gptneox_outlier" if name in SMALL_DIM_REGIME else
                                "modern_core")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    core = [o["alpha_exec"] for o in out if o["regime"] == "modern_core"]
    allv = [o["alpha_exec"] for o in out]
    param = [o["alpha_param"] for o in out]
    print(f"alpha_param (n={len(param)}): median {st.median(param):.3f}, CV {st.pstdev(param)/st.mean(param)*100:.1f}%")
    print(f"alpha_exec core (n={len(core)}): median {st.median(core):.3f}, "
          f"min {min(core):.2f}, max {max(core):.2f}, CV {st.pstdev(core)/st.mean(core)*100:.1f}%")
    print(f"alpha_exec all  (n={len(allv)}): median {st.median(allv):.3f}, CV {st.pstdev(allv)/st.mean(allv)*100:.1f}%")
    # Ternary + encoder under the executed convention (registry-independent constants
    # documented in the paper §5.1/§7.1; TriLM tables are untied per tensor census).
    print(f"TriLM-560M : alpha_param 1.034 -> alpha_exec {1.034*569.0/(569.0-64.2):.3f}")
    print(f"TriLM-3.9B : alpha_param <=1.002 -> alpha_exec <= {1.002*3992.0/(3992.0-154.1):.3f} (N=5, one-sided)")
    print(f"EmbeddingGemma-300M (encoder): alpha_param 0.53 -> alpha_exec {0.53*308.0/101.5:.2f}")
    print(f"written: {os.path.relpath(OUT, HERE + '/..')}")

if __name__ == "__main__":
    main()
