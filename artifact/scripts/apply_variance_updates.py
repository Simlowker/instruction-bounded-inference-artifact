#!/usr/bin/env python3
"""
Applique les mesures de variance.csv (5 modeles x 3 reps) sur core_measurements.csv
pour produire core_measurements_v2.csv.

Inputs:
  - artifact/results/raw/core_measurements.csv (baseline)
  - artifact/data/verified/variance.csv (nouveaux reps)

Output:
  - artifact/results/raw/core_measurements_v2.csv
  - stdout: tableau des changements avec delta et refit alpha_eff
"""
from pathlib import Path

import pandas as pd

B = 40e9
HERE = Path(__file__).resolve().parent.parent

CORE = HERE / "results/raw/core_measurements.csv"
VARIANCE = HERE / "data/verified/variance.csv"
OUT = HERE / "results/raw/core_measurements_v2.csv"


def recompute_alpha(params_M: float, tok_call: float) -> float:
    return round(B / (2.0 * params_M * 1e6 * tok_call), 3)


def main():
    df = pd.read_csv(CORE)
    var = pd.read_csv(VARIANCE)

    # Garder seulement la MEILLEURE serie de reps par modele (CV le plus bas)
    var = var[var["mean_tok_call"] > 0]  # exclure Falcon-H1 crash (mean=0)
    var["cv"] = var["std_tok_call"] / var["mean_tok_call"]
    var = var.sort_values("cv").drop_duplicates(subset=["model_tag", "quant", "simd"], keep="first")

    print(f"Variance reps retenues (CV < 5%): {len(var)} modeles")
    print(var[["model_tag", "quant", "simd", "mean_tok_call", "std_tok_call", "cv"]].to_string(index=False))
    print()

    changes = []
    for _, vrow in var.iterrows():
        model_tag = vrow["model_tag"]
        quant = vrow["quant"]
        simd = vrow["simd"]
        new_tok = round(vrow["mean_tok_call"])

        # Mapping des noms (variance utilise "Pythia-70M", core utilise pareil sauf certains)
        name_map = {"Qwen2.5-0.5B": "Qwen-2.5-0.5B"}
        model_core = name_map.get(model_tag, model_tag)

        mask = (
            (df["model"] == model_core)
            & (df["quant"] == quant)
            & (df["simd"] == simd)
            & (df["network"] == "local")
            & (df["build"].isin(["gian", "upstream"]))  # accepte les deux
        )
        if mask.sum() == 0:
            # Pas dans core_measurements.csv — ajouter nouvelle ligne
            # (cas Gemma3-270M Q4_0 local)
            print(f"  [NEW] {model_tag} {quant} simd={simd} local = {new_tok} tok/call (pas dans baseline)")
            template_mask = df["model"] == model_core
            template = df[template_mask].iloc[0] if template_mask.sum() > 0 else None
            new_row = {
                "model": model_tag,
                "architecture": template["architecture"] if template is not None else vrow["arch"],
                "family": template["family"] if template is not None else (vrow["arch"].replace("_", "").replace("3", "3") if vrow["arch"] != "gpt_neox" else "gpt_neox"),
                "params_M": vrow["params_M"],
                "layers": vrow["layers"],
                "hidden_dim": template["hidden_dim"] if template is not None else "",
                "d_ff": template["d_ff"] if template is not None else "",
                "n_kv": template["n_kv"] if template is not None else "",
                "flops_per_tok_M": template["flops_per_tok_M"] if template is not None else "",
                "quant": quant,
                "simd": simd,
                "build": "gian",
                "network": "local",
                "gguf_size_MB": vrow["size_MB"],
                "tok_per_call": new_tok,
                "alpha_eff": recompute_alpha(vrow["params_M"], new_tok),
                "measurement_type": "binary_search",
                "n_runs": 3,
                "notes": f"variance reps: mean={vrow['mean_tok_call']} std={vrow['std_tok_call']} (P0-1)",
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            changes.append({
                "model": model_tag,
                "quant": quant,
                "old_tok": None,
                "new_tok": new_tok,
                "delta_pct": None,
                "old_alpha": None,
                "new_alpha": recompute_alpha(vrow["params_M"], new_tok),
                "note": "NEW ROW",
            })
            continue

        if mask.sum() > 1:
            print(f"  [WARN] {model_tag} {quant}: {mask.sum()} rows match — prend la premiere")
            idx = df[mask].index[0]
            mask = df.index == idx

        old_tok = int(df.loc[mask, "tok_per_call"].values[0])
        old_alpha = df.loc[mask, "alpha_eff"].values[0]
        params = df.loc[mask, "params_M"].values[0]
        new_alpha = recompute_alpha(params, new_tok)
        delta_pct = round((new_tok - old_tok) / old_tok * 100, 1)

        df.loc[mask, "tok_per_call"] = new_tok
        df.loc[mask, "alpha_eff"] = new_alpha
        df.loc[mask, "n_runs"] = 3
        df.loc[mask, "notes"] = (
            str(df.loc[mask, "notes"].values[0])
            + f" [VARIANCE UPDATE {vrow['mean_tok_call']}±{vrow['std_tok_call']}]"
        )
        changes.append({
            "model": model_tag,
            "quant": quant,
            "old_tok": old_tok,
            "new_tok": new_tok,
            "delta_pct": delta_pct,
            "old_alpha": old_alpha,
            "new_alpha": new_alpha,
            "note": "UPDATE",
        })

    print()
    print("Changements appliques:")
    for c in changes:
        print(f"  {c['model']:<20} {c['quant']:<8} {c['note']:<10} "
              f"tok: {c['old_tok']}→{c['new_tok']} ({c['delta_pct']}%)  "
              f"alpha: {c['old_alpha']}→{c['new_alpha']}")

    df.to_csv(OUT, index=False)
    print(f"\nWrote: {OUT}")

    # Sauve aussi le log des changements pour le package ChatGPT
    pd.DataFrame(changes).to_csv(
        HERE / "results/current/extended_analysis/variance_update_log.csv",
        index=False,
    )
    print(
        "Wrote:"
        f" {HERE / 'results/current/extended_analysis/variance_update_log.csv'}"
    )


if __name__ == "__main__":
    main()
