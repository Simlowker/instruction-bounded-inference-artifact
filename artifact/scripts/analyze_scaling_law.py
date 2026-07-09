#!/usr/bin/env python3
"""
Scaling law analysis for instruction-bounded inference paper.

Implements the "Option C hybrid" strategy:
  - Primary calibration on modern decoder-only families (excluding GPT-2)
  - Full sensitivity analysis with/without GPT-2
  - Robust M-estimator on all families
  - Influence diagnostics (Cook's distance, studentized residuals)
  - LOAO cross-validation at both scopes

Produces:
  1. Leave-One-Architecture-Out (LOAO) cross-validation
  2. Bootstrap confidence intervals on α_eff
  3. Sensitivity table: modern-only vs all-families vs robust
  4. Influence diagnostics per family
  5. Calibration + residual figures

Usage:
    python analyze_scaling_law.py [--data PATH] [--outdir PATH] [--no-plot]

Requires: numpy, pandas, scipy, matplotlib, statsmodels
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

# ---------- constants ----------
B = 40e9  # ICP instruction budget per update call
LEGACY_FAMILIES = {"gpt2"}  # GPT-2 family: absolute position embeddings, no GQA/RoPE
AUXILIARY_NOTE_PATTERNS = (
    "instruct variant",
    "output degenerate",
)


# ---------- data loading ----------
def load_core_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = df["alpha_eff"].isna() & df["tok_per_call"].notna() & df["params_M"].notna()
    df.loc[mask, "alpha_eff"] = B / (2.0 * df.loc[mask, "params_M"] * 1e6 * df.loc[mask, "tok_per_call"])
    return df


def drop_auxiliary_rows(df: pd.DataFrame) -> pd.DataFrame:
    notes = df["notes"].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for pattern in AUXILIARY_NOTE_PATTERNS:
        mask |= notes.str.contains(pattern, regex=False)
    return df[~mask].copy()


def select_homogeneous_set(df: pd.DataFrame, quant: str = "Q4_0", simd: str = "yes",
                           network: str = "local", build: str = "gian") -> pd.DataFrame:
    mask = (
        (df["quant"] == quant) &
        (df["simd"] == simd) &
        (df["network"] == network) &
        (df["build"] == build)
    )
    subset = drop_auxiliary_rows(df[mask].copy())
    subset = subset.sort_values("notes").drop_duplicates(subset=["model"], keep="first")
    return subset.reset_index(drop=True)


def build_combined_set(df_all: pd.DataFrame) -> pd.DataFrame:
    """Build combined SIMD local set (Q4_0 gian + Q8_0 upstream), one entry per model."""
    df = df_all[
        (df_all["simd"] == "yes") &
        (df_all["network"] == "local") &
        (df_all["quant"].isin(["Q4_0", "Q8_0"])) &
        (df_all["alpha_eff"].notna())
    ].copy()
    df = drop_auxiliary_rows(df)
    df = df.sort_values(["model", "quant"]).drop_duplicates(subset=["model"], keep="first")
    return df.reset_index(drop=True)


# ---------- LOAO ----------
def leave_one_architecture_out(df: pd.DataFrame) -> pd.DataFrame:
    groups = df["family"].unique()
    rows = []
    for held_out in groups:
        train = df[df["family"] != held_out]
        test = df[df["family"] == held_out]
        if len(train) == 0 or len(test) == 0:
            continue
        alpha_train = np.median(train["alpha_eff"].values)
        for _, row in test.iterrows():
            P = row["params_M"] * 1e6
            y_true = row["tok_per_call"]
            y_pred = B / (alpha_train * 2.0 * P)
            rel_error = (y_pred - y_true) / y_true
            rows.append({
                "held_out_arch": held_out,
                "model": row["model"],
                "params_M": row["params_M"],
                "tok_measured": y_true,
                "tok_predicted": round(y_pred, 1),
                "alpha_train_median": round(alpha_train, 3),
                "relative_error": round(rel_error, 4),
                "absolute_relative_error": round(abs(rel_error), 4),
                "is_legacy": held_out in LEGACY_FAMILIES,
            })
    return pd.DataFrame(rows)


def loao_summary(loao_df: pd.DataFrame) -> pd.DataFrame:
    summary = loao_df.groupby("held_out_arch").agg(
        n_models=("model", "count"),
        alpha_train=("alpha_train_median", "first"),
        mean_abs_rel_error=("absolute_relative_error", "mean"),
        max_abs_rel_error=("absolute_relative_error", "max"),
        is_legacy=("is_legacy", "first"),
    ).reset_index()
    overall = pd.DataFrame([{
        "held_out_arch": "OVERALL",
        "n_models": len(loao_df),
        "alpha_train": np.nan,
        "mean_abs_rel_error": loao_df["absolute_relative_error"].mean(),
        "max_abs_rel_error": loao_df["absolute_relative_error"].max(),
        "is_legacy": False,
    }])
    modern_only = loao_df[~loao_df["is_legacy"]]
    if len(modern_only) > 0:
        overall_modern = pd.DataFrame([{
            "held_out_arch": "MODERN_ONLY",
            "n_models": len(modern_only),
            "alpha_train": np.nan,
            "mean_abs_rel_error": modern_only["absolute_relative_error"].mean(),
            "max_abs_rel_error": modern_only["absolute_relative_error"].max(),
            "is_legacy": False,
        }])
        summary = pd.concat([summary, overall, overall_modern], ignore_index=True)
    else:
        summary = pd.concat([summary, overall], ignore_index=True)
    return summary


# ---------- bootstrap CI ----------
def bootstrap_alpha(df: pd.DataFrame, n_resamples: int = 9999, label: str = "") -> dict:
    alpha_values = df["alpha_eff"].dropna().values
    result = {"label": label, "n": len(alpha_values)}

    if len(alpha_values) < 3:
        result.update({"mean": np.mean(alpha_values), "std": np.nan,
                       "median": np.median(alpha_values),
                       "ci_mean_low": np.nan, "ci_mean_high": np.nan,
                       "ci_median_low": np.nan, "ci_median_high": np.nan,
                       "method": "insufficient_data"})
        return result

    ci_mean = bootstrap((alpha_values,), np.mean, method="BCa",
                        n_resamples=n_resamples, random_state=42).confidence_interval
    ci_med = bootstrap((alpha_values,), np.median, method="BCa",
                       n_resamples=n_resamples, random_state=42).confidence_interval

    result.update({
        "mean": round(np.mean(alpha_values), 4),
        "std": round(np.std(alpha_values, ddof=1), 4),
        "median": round(np.median(alpha_values), 4),
        "ci_mean_low": round(ci_mean.low, 4),
        "ci_mean_high": round(ci_mean.high, 4),
        "ci_median_low": round(ci_med.low, 4),
        "ci_median_high": round(ci_med.high, 4),
        "method": "BCa_bootstrap_9999",
    })
    return result


# ---------- robust estimation ----------
def robust_alpha_fit(df: pd.DataFrame) -> dict:
    """Fit α_eff using Huber M-estimator on log-transformed scaling law.

    Model: log(tok/call) = log(B) - log(α) - log(2P)
    → log(tok/call) + log(2P) = log(B/α)  [constant if α is universal]
    We fit α via robust regression of log(tok) on log(2P).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels not installed"}

    log_tok = np.log(df["tok_per_call"].values)
    log_2P = np.log(2.0 * df["params_M"].values * 1e6)

    # OLS first (for comparison)
    X = sm.add_constant(-log_2P)
    ols = sm.OLS(log_tok, X).fit()

    # Robust: Huber M-estimator
    robust = sm.RLM(log_tok, X, M=sm.robust.norms.HuberT()).fit()

    # Extract α: intercept = log(B/α) → α = B/exp(intercept)
    alpha_ols = B / np.exp(ols.params[0])
    alpha_robust = B / np.exp(robust.params[0])

    return {
        "alpha_ols": round(alpha_ols, 4),
        "alpha_robust_huber": round(alpha_robust, 4),
        "slope_ols": round(ols.params[1], 4),
        "slope_robust": round(robust.params[1], 4),
        "slope_expected": -1.0,  # perfect scaling → slope = -1
        "r_squared_ols": round(ols.rsquared, 4),
    }


def influence_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Cook's distance and studentized residuals per data point."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()

    log_tok = np.log(df["tok_per_call"].values)
    log_2P = np.log(2.0 * df["params_M"].values * 1e6)

    X = sm.add_constant(-log_2P)
    ols = sm.OLS(log_tok, X).fit()
    infl = ols.get_influence()

    cooks_d, _ = infl.cooks_distance
    student_resid = infl.resid_studentized_external

    result = df[["model", "family", "params_M", "tok_per_call", "alpha_eff"]].copy()
    result["cooks_distance"] = np.round(cooks_d, 4)
    result["studentized_residual"] = np.round(student_resid, 4)
    result["is_legacy"] = result["family"].isin(LEGACY_FAMILIES)

    # Flag influential points (Cook's d > 4/n threshold)
    threshold = 4.0 / len(df)
    result["influential"] = result["cooks_distance"] > threshold

    return result.sort_values("cooks_distance", ascending=False)


# ---------- sensitivity table ----------
def sensitivity_analysis(df_all: pd.DataFrame, df_modern: pd.DataFrame) -> pd.DataFrame:
    """Compare α_eff estimates across scopes and estimators."""
    rows = []

    # 1. All families — median
    rows.append({
        "scope": "All families",
        "estimator": "Median",
        "n_models": len(df_all),
        "n_families": df_all["family"].nunique(),
        "alpha_eff": round(np.median(df_all["alpha_eff"]), 4),
    })

    # 2. All families — mean
    rows.append({
        "scope": "All families",
        "estimator": "Mean",
        "n_models": len(df_all),
        "n_families": df_all["family"].nunique(),
        "alpha_eff": round(np.mean(df_all["alpha_eff"]), 4),
    })

    # 3. All families — robust (Huber)
    robust = robust_alpha_fit(df_all)
    if "error" not in robust:
        rows.append({
            "scope": "All families",
            "estimator": "Huber M-estimator",
            "n_models": len(df_all),
            "n_families": df_all["family"].nunique(),
            "alpha_eff": robust["alpha_robust_huber"],
        })

    # 4. Modern only — median
    rows.append({
        "scope": "Modern families",
        "estimator": "Median",
        "n_models": len(df_modern),
        "n_families": df_modern["family"].nunique(),
        "alpha_eff": round(np.median(df_modern["alpha_eff"]), 4),
    })

    # 5. Modern only — mean
    rows.append({
        "scope": "Modern families",
        "estimator": "Mean",
        "n_models": len(df_modern),
        "n_families": df_modern["family"].nunique(),
        "alpha_eff": round(np.mean(df_modern["alpha_eff"]), 4),
    })

    # 6. Legacy only — median
    df_legacy = df_all[df_all["family"].isin(LEGACY_FAMILIES)]
    if len(df_legacy) > 0:
        rows.append({
            "scope": "Legacy (GPT-2)",
            "estimator": "Median",
            "n_models": len(df_legacy),
            "n_families": df_legacy["family"].nunique(),
            "alpha_eff": round(np.median(df_legacy["alpha_eff"]), 4),
        })

    return pd.DataFrame(rows)


# ---------- plots ----------
def plot_full_analysis(df_all: pd.DataFrame, df_modern: pd.DataFrame,
                       loao_all: pd.DataFrame, influence: pd.DataFrame, outdir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    families_all = sorted(df_all["family"].unique())
    cmap = plt.cm.tab10
    fam_colors = {fam: cmap(i) for i, fam in enumerate(families_all)}

    # --- Panel A: Scaling law with two regimes ---
    ax = axes[0, 0]
    alpha_modern = np.median(df_modern["alpha_eff"].values)
    alpha_all = np.median(df_all["alpha_eff"].values)

    P_range = np.logspace(np.log10(10), np.log10(1000), 200)
    ax.plot(P_range, B / (alpha_modern * 2.0 * P_range * 1e6),
            "k-", alpha=0.7, linewidth=2, label=f"Modern α = {alpha_modern:.2f}")
    ax.plot(P_range, B / (alpha_all * 2.0 * P_range * 1e6),
            "k--", alpha=0.3, label=f"All families α = {alpha_all:.2f}")

    for fam in families_all:
        sub = df_all[df_all["family"] == fam]
        marker = "x" if fam in LEGACY_FAMILIES else "o"
        edge = "red" if fam in LEGACY_FAMILIES else "k"
        ax.scatter(sub["params_M"], sub["tok_per_call"],
                   color=fam_colors[fam], s=80, zorder=3, label=fam,
                   marker=marker, edgecolors=edge, linewidth=1.0 if fam in LEGACY_FAMILIES else 0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("tok/call")
    ax.set_title("A. Scaling law with modern / legacy regimes")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel B: LOAO predicted vs measured ---
    ax = axes[0, 1]
    max_val = max(loao_all["tok_measured"].max(), loao_all["tok_predicted"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="perfect")

    for fam in sorted(loao_all["held_out_arch"].unique()):
        sub = loao_all[loao_all["held_out_arch"] == fam]
        marker = "x" if fam in LEGACY_FAMILIES else "o"
        edge = "red" if fam in LEGACY_FAMILIES else "k"
        ax.scatter(sub["tok_measured"], sub["tok_predicted"],
                   color=fam_colors.get(fam, "gray"), s=80, zorder=3,
                   label=f"{fam}{'*' if fam in LEGACY_FAMILIES else ''}",
                   marker=marker, edgecolors=edge, linewidth=1.0 if fam in LEGACY_FAMILIES else 0.5)

    ax.set_xlabel("Measured tok/call")
    ax.set_ylabel("Predicted tok/call (LOAO)")
    ax.set_title("B. LOAO validation (* = legacy)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Residuals by family ---
    ax = axes[1, 0]
    families_sorted = sorted(loao_all["held_out_arch"].unique())
    positions = {fam: i for i, fam in enumerate(families_sorted)}

    for fam in families_sorted:
        sub = loao_all[loao_all["held_out_arch"] == fam]
        x_pos = positions[fam]
        color = "red" if fam in LEGACY_FAMILIES else fam_colors.get(fam, "gray")
        ax.scatter([x_pos] * len(sub), sub["relative_error"] * 100,
                   color=color, s=60, zorder=3, edgecolors="k", linewidth=0.5)

    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axhline(y=15, color="orange", linestyle=":", alpha=0.5, label="±15% band")
    ax.axhline(y=-15, color="orange", linestyle=":", alpha=0.5)

    ax.set_xticks(range(len(families_sorted)))
    ax.set_xticklabels(families_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative error (%)")
    ax.set_title("C. Residuals by architecture family")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight legacy region
    for fam in LEGACY_FAMILIES:
        if fam in positions:
            ax.axvspan(positions[fam] - 0.4, positions[fam] + 0.4,
                       alpha=0.1, color="red", label="legacy" if fam == list(LEGACY_FAMILIES)[0] else None)

    # --- Panel D: Influence (Cook's distance) ---
    ax = axes[1, 1]
    if len(influence) > 0:
        threshold = 4.0 / len(influence)
        colors = ["red" if influence.iloc[i]["is_legacy"] else fam_colors.get(influence.iloc[i]["family"], "gray")
                  for i in range(len(influence))]
        bars = ax.barh(range(len(influence)), influence["cooks_distance"],
                       color=colors, edgecolor="k", linewidth=0.5)
        ax.set_yticks(range(len(influence)))
        ax.set_yticklabels([f"{r['model']} ({r['family']})" for _, r in influence.iterrows()],
                           fontsize=7)
        ax.axvline(x=threshold, color="red", linestyle="--", alpha=0.5, label=f"4/n = {threshold:.3f}")
        ax.set_xlabel("Cook's distance")
        ax.set_title("D. Influence diagnostics")
        ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(outdir / "scaling_law_full_analysis.png", dpi=200)
    fig.savefig(outdir / "scaling_law_full_analysis.pdf")
    plt.close()
    print(f"  Saved: {outdir / 'scaling_law_full_analysis.png'}")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Scaling law analysis (Option C hybrid)")
    parser.add_argument("--data", default="artifact/results/raw/core_measurements_v2.csv")
    parser.add_argument(
        "--outdir",
        default="artifact/results/current/scaling_law",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df_all_raw = load_core_data(args.data)
    print(f"  Total rows: {len(df_all_raw)}")

    # Build combined set (one entry per model, SIMD local)
    df_all = build_combined_set(df_all_raw)
    df_modern = df_all[~df_all["family"].isin(LEGACY_FAMILIES)].copy()
    df_legacy = df_all[df_all["family"].isin(LEGACY_FAMILIES)].copy()

    print(f"\n  Combined set: {len(df_all)} models, {df_all['family'].nunique()} families")
    print(f"  Modern: {len(df_modern)} models, {df_modern['family'].nunique()} families")
    print(f"  Legacy: {len(df_legacy)} models ({', '.join(df_legacy['model'].values)})")

    # =============================================
    # 1. SENSITIVITY TABLE
    # =============================================
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: α_eff across scopes and estimators")
    print("=" * 60)

    sens = sensitivity_analysis(df_all, df_modern)
    sens.to_csv(outdir / "sensitivity_alpha.csv", index=False)
    print(sens.to_string(index=False))

    # =============================================
    # 2. LOAO — ALL FAMILIES
    # =============================================
    print("\n" + "=" * 60)
    print("LOAO: ALL FAMILIES")
    print("=" * 60)

    loao_all = leave_one_architecture_out(df_all)
    loao_all.to_csv(outdir / "loao_all_families.csv", index=False)
    summary_all = loao_summary(loao_all)
    summary_all.to_csv(outdir / "loao_all_families_summary.csv", index=False)
    print(summary_all.to_string(index=False))

    # =============================================
    # 3. LOAO — MODERN ONLY (primary calibration)
    # =============================================
    print("\n" + "=" * 60)
    print("LOAO: MODERN FAMILIES (primary calibration)")
    print("=" * 60)

    loao_modern = leave_one_architecture_out(df_modern)
    loao_modern.to_csv(outdir / "loao_modern_families.csv", index=False)
    summary_modern = loao_summary(loao_modern)
    summary_modern.to_csv(outdir / "loao_modern_families_summary.csv", index=False)
    print(summary_modern.to_string(index=False))

    # =============================================
    # 4. BOOTSTRAP CI — both scopes
    # =============================================
    print("\n" + "=" * 60)
    print("BOOTSTRAP CI on α_eff")
    print("=" * 60)

    ci_all = bootstrap_alpha(df_all, label="all_families")
    ci_modern = bootstrap_alpha(df_modern, label="modern_only")

    ci_df = pd.DataFrame([ci_all, ci_modern])
    ci_df.to_csv(outdir / "alpha_bootstrap_comparison.csv", index=False)

    for ci in [ci_all, ci_modern]:
        print(f"\n  [{ci['label']}] n={ci['n']}")
        print(f"    mean  = {ci['mean']} [{ci.get('ci_mean_low','?')}, {ci.get('ci_mean_high','?')}]")
        print(f"    median= {ci['median']} [{ci.get('ci_median_low','?')}, {ci.get('ci_median_high','?')}]")

    # =============================================
    # 5. ROBUST FIT
    # =============================================
    print("\n" + "=" * 60)
    print("ROBUST FIT (Huber M-estimator)")
    print("=" * 60)

    robust = robust_alpha_fit(df_all)
    print(f"  α_eff (OLS, all):    {robust.get('alpha_ols', 'N/A')}")
    print(f"  α_eff (Huber, all):  {robust.get('alpha_robust_huber', 'N/A')}")
    print(f"  slope (OLS):         {robust.get('slope_ols', 'N/A')} (expected: -1.0)")
    print(f"  slope (Huber):       {robust.get('slope_robust', 'N/A')}")
    print(f"  R² (OLS):            {robust.get('r_squared_ols', 'N/A')}")

    robust_modern = robust_alpha_fit(df_modern)
    print(f"\n  α_eff (OLS, modern): {robust_modern.get('alpha_ols', 'N/A')}")
    print(f"  α_eff (Huber, mod.): {robust_modern.get('alpha_robust_huber', 'N/A')}")
    print(f"  slope (OLS, modern): {robust_modern.get('slope_ols', 'N/A')}")
    print(f"  R² (OLS, modern):    {robust_modern.get('r_squared_ols', 'N/A')}")

    pd.DataFrame([
        {"scope": "all_families", **robust},
        {"scope": "modern_only", **robust_modern},
    ]).to_csv(outdir / "robust_fit.csv", index=False)

    # =============================================
    # 6. INFLUENCE DIAGNOSTICS
    # =============================================
    print("\n" + "=" * 60)
    print("INFLUENCE DIAGNOSTICS (Cook's distance)")
    print("=" * 60)

    influence = influence_diagnostics(df_all)
    if len(influence) > 0:
        influence.to_csv(outdir / "influence_diagnostics.csv", index=False)
        print(influence[["model", "family", "cooks_distance", "studentized_residual",
                         "influential", "is_legacy"]].to_string(index=False))
    else:
        print("  (statsmodels not available)")

    # =============================================
    # 7. PLOTS
    # =============================================
    if not args.no_plot:
        print("\nGenerating figures...")
        plot_full_analysis(df_all, df_modern, loao_all, influence, outdir)

    # =============================================
    # FINAL SUMMARY FOR PAPER
    # =============================================
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER (Option C hybrid)")
    print("=" * 60)

    ov_all = summary_all[summary_all["held_out_arch"] == "OVERALL"].iloc[0]
    ov_mod = summary_modern[summary_modern["held_out_arch"] == "OVERALL"].iloc[0]
    ov_mod_legacy_train = summary_all[summary_all["held_out_arch"] == "MODERN_ONLY"].iloc[0]

    print(f"\n  PRIMARY (modern, {len(df_modern)} models, {df_modern['family'].nunique()} families):")
    print(f"    LOAO MAPE: {ov_mod['mean_abs_rel_error']:.1%}")
    print(f"    LOAO max:  {ov_mod['max_abs_rel_error']:.1%}")
    print(f"    Legacy-in-train convention: {ov_mod_legacy_train['mean_abs_rel_error']:.1%}")
    print(f"    α_eff:     {ci_modern['median']} [{ci_modern.get('ci_median_low','?')}, "
          f"{ci_modern.get('ci_median_high','?')}] 95% CI (median)")

    print(f"\n  SENSITIVITY (all, {len(df_all)} models, {df_all['family'].nunique()} families):")
    print(f"    LOAO MAPE: {ov_all['mean_abs_rel_error']:.1%}")
    print(f"    LOAO max:  {ov_all['max_abs_rel_error']:.1%} (GPT-2)")
    print(f"    α_eff:     {ci_all['median']} [{ci_all.get('ci_median_low','?')}, "
          f"{ci_all.get('ci_median_high','?')}] 95% CI (median)")

    print(f"\n  ROBUST (Huber M-estimator, all families):")
    print(f"    α_eff: {robust.get('alpha_robust_huber', 'N/A')}")

    print(f"\n  LEGACY REGIME:")
    print(f"    GPT-2 α_eff: ~{np.median(df_legacy['alpha_eff']):.2f} "
          f"(vs modern ~{np.median(df_modern['alpha_eff']):.2f})")
    print(f"    Hypothesis: absolute position embeddings + no GQA → higher α")

    print(f"\n  ⚠️  Primary calibration mixes variance-verified points with single-run binary-search estimates.")
    print(f"  ⚠️  CI reflects inter-model variance; measurement variance is quantified only for the variance-verified subset.")


if __name__ == "__main__":
    main()
