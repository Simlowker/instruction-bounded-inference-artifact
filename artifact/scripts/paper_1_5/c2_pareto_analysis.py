#!/usr/bin/env python3
"""Pareto analysis for mixed-precision GGUF variants.

Joins 3 CSVs (variants metadata, canister tok/call measurements, PPL eval),
computes the Pareto frontier over 3 objectives (min size_MB, min PPL_WT2,
max tok/call), writes a markdown table + 2-panel PNG figure.

Reusable across models via --label (e.g. smollm135, qwen05).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def robust_read_csv(path: Path) -> pd.DataFrame:
    """Read CSVs that may contain unquoted commas in the trailing `notes` column.

    Strategy: use the stdlib csv reader (which handles RFC-4180 quoting), and if
    a row has more fields than the header, merge the trailing overflow back into
    the last column.
    """
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        rows = []
        for row in reader:
            if len(row) > ncols:
                row = row[: ncols - 1] + [",".join(row[ncols - 1:])]
            elif len(row) < ncols:
                row = row + [""] * (ncols - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def strip_suffix(variant_id: str, label: str) -> str:
    suffix = f"-{label}"
    return variant_id[: -len(suffix)] if variant_id.endswith(suffix) else variant_id


def load_and_join(variants_csv: Path, measurements_csv: Path, ppl_csv: Path, label: str) -> pd.DataFrame:
    variants = robust_read_csv(variants_csv)
    measurements = robust_read_csv(measurements_csv)
    ppl = robust_read_csv(ppl_csv)

    # Coerce numeric columns
    variants["size_MB"] = pd.to_numeric(variants["size_MB"])
    measurements["tok_call_n3_mean"] = pd.to_numeric(measurements["tok_call_n3_mean"])
    ppl["perplexity_wikitext2"] = pd.to_numeric(ppl["perplexity_wikitext2"])
    ppl["perplexity_c4"] = pd.to_numeric(ppl["perplexity_c4"])

    # Normalise variant_id to short form (strip -<label> suffix where present)
    variants["variant_short"] = variants["variant_id"].apply(lambda v: strip_suffix(v, label))
    measurements["variant_short"] = measurements["variant_id"].apply(lambda v: strip_suffix(v, label))
    ppl["variant_short"] = ppl["variant_id"].apply(lambda v: strip_suffix(v, label))

    df = variants[["variant_short", "size_MB", "base_format", "mixed_strategy"]].merge(
        measurements[["variant_short", "tok_call_n3_mean"]], on="variant_short", how="inner"
    ).merge(
        ppl[["variant_short", "perplexity_wikitext2", "perplexity_c4"]], on="variant_short", how="inner"
    )
    df = df.rename(columns={
        "variant_short": "variant",
        "tok_call_n3_mean": "tok_call",
        "perplexity_wikitext2": "ppl_wt2",
        "perplexity_c4": "ppl_c4",
    })
    return df


def pareto_mask(df: pd.DataFrame) -> pd.Series:
    """A point is Pareto-optimal if no other point dominates it.

    Dominance: other is <= on all min objectives (size, ppl_wt2) AND >= on max
    objective (tok_call), with strict inequality on at least one dim.
    """
    mask = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            le_size = other["size_MB"] <= row["size_MB"]
            le_ppl = other["ppl_wt2"] <= row["ppl_wt2"]
            ge_tok = other["tok_call"] >= row["tok_call"]
            strict = (
                other["size_MB"] < row["size_MB"]
                or other["ppl_wt2"] < row["ppl_wt2"]
                or other["tok_call"] > row["tok_call"]
            )
            if le_size and le_ppl and ge_tok and strict:
                dominated = True
                break
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)


def write_markdown_table(df: pd.DataFrame, output: Path, label: str) -> None:
    df_sorted = df.sort_values("tok_call", ascending=False).reset_index(drop=True)
    lines = []
    lines.append(f"| variant | size_MB | tok/call | PPL_WT2 | PPL_C4 | Pareto |")
    lines.append(f"|---------|--------:|---------:|--------:|-------:|:------:|")
    for _, row in df_sorted.iterrows():
        flag = "yes" if row["pareto"] else "-"
        lines.append(
            f"| {row['variant']} | {row['size_MB']:.1f} | {int(row['tok_call'])} | "
            f"{row['ppl_wt2']:.3f} | {row['ppl_c4']:.3f} | {flag} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def generate_figure(df: pd.DataFrame, output: Path, label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: tok/call vs PPL_WT2 (inverted), bubble size = size_MB
    for _, row in df.iterrows():
        color = "red" if row["pareto"] else "steelblue"
        alpha = 0.85 if row["pareto"] else 0.55
        ax1.scatter(row["tok_call"], row["ppl_wt2"],
                    s=row["size_MB"] * 4, c=color, alpha=alpha,
                    edgecolors="black", linewidths=0.6)
        ax1.annotate(row["variant"], (row["tok_call"], row["ppl_wt2"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("tok/call (higher = better)")
    ax1.set_ylabel("Perplexity WT2 (lower = better, axis inverted)")
    ax1.set_title(f"{label}: tok/call vs PPL (bubble = size_MB)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: size_MB vs PPL_WT2 (inverted), bubble size = tok/call
    for _, row in df.iterrows():
        color = "red" if row["pareto"] else "steelblue"
        alpha = 0.85 if row["pareto"] else 0.55
        ax2.scatter(row["size_MB"], row["ppl_wt2"],
                    s=row["tok_call"] * 5, c=color, alpha=alpha,
                    edgecolors="black", linewidths=0.6)
        ax2.annotate(row["variant"], (row["size_MB"], row["ppl_wt2"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("size_MB (lower = better)")
    ax2.set_ylabel("Perplexity WT2 (lower = better, axis inverted)")
    ax2.set_title(f"{label}: size vs PPL (bubble = tok/call)")
    ax2.grid(True, alpha=0.3)

    # Legend proxy
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=10, label="Pareto-optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markeredgecolor="black", markersize=10, label="Dominated"),
    ]
    ax1.legend(handles=legend, loc="best")
    fig.suptitle(f"Pareto frontier: mixed-precision variants ({label})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants-csv", required=True, type=Path)
    p.add_argument("--measurements-csv", required=True, type=Path)
    p.add_argument("--ppl-csv", required=True, type=Path)
    p.add_argument("--label", required=True,
                   help="Short suffix used to strip variant_id (e.g. smollm135, qwen05).")
    p.add_argument("--output-table", required=True, type=Path)
    p.add_argument("--output-figure", required=True, type=Path)
    args = p.parse_args()

    df = load_and_join(args.variants_csv, args.measurements_csv, args.ppl_csv, args.label)
    if df.empty:
        print("ERROR: join produced 0 rows; check variant_id normalisation.", file=sys.stderr)
        return 2
    df["pareto"] = pareto_mask(df)

    # Console
    print(f"\n=== Pareto analysis: {args.label} ({len(df)} variants) ===")
    df_print = df.sort_values("tok_call", ascending=False)[
        ["variant", "size_MB", "tok_call", "ppl_wt2", "ppl_c4", "pareto"]
    ]
    print(df_print.to_string(index=False))
    winners = df[df["pareto"]]["variant"].tolist()
    print(f"\nPareto-optimal ({len(winners)}): {', '.join(winners)}")

    write_markdown_table(df, args.output_table, args.label)
    generate_figure(df, args.output_figure, args.label)
    print(f"\nTable  -> {args.output_table}")
    print(f"Figure -> {args.output_figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
