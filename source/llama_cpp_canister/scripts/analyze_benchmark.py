#!/usr/bin/env python3
"""
Analyze benchmark CSV results for the instruction-bounded inference paper.

Usage:
  python -m scripts.analyze_benchmark benchmark_results_*.csv
  python -m scripts.analyze_benchmark --latest
  python -m scripts.analyze_benchmark results1.csv results2.csv --output analysis.csv
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent


def load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = Path(p).name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def find_latest_csv() -> str:
    pattern = str(PROJECT_DIR / "benchmark_results_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No benchmark_results_*.csv found in {PROJECT_DIR}")
        sys.exit(1)
    return files[-1]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["tokens_generated", "total_instructions", "per_token_decode",
                    "per_token_total", "decode_instructions", "sample_instructions",
                    "setup_instructions", "overhead_instructions"]

    if "prefill_instructions" in df.columns:
        numeric_cols.append("prefill_instructions")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    group_cols = ["config_id", "model", "params_M", "arch", "quant", "prompt_len_target"]
    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for name, group in grouped:
        config_id, model, params_m, arch, quant, prompt_len = name
        row = {
            "config_id": config_id,
            "model": model,
            "params_M": params_m,
            "arch": arch,
            "quant": quant,
            "prompt_len": prompt_len,
            "n_measurements": len(group),
            "tok_generated_med": group["tokens_generated"].median(),
            "instr_total_med": group["total_instructions"].median(),
            "instr_per_tok_decode_med": group["per_token_decode"].median(),
            "instr_per_tok_decode_std": group["per_token_decode"].std(),
            "decode_instr_med": group["decode_instructions"].median(),
            "sample_instr_med": group["sample_instructions"].median(),
            "setup_instr_med": group["setup_instructions"].median(),
        }

        if "prefill_instructions" in df.columns and group["prefill_instructions"].notna().any():
            row["prefill_instr_med"] = group["prefill_instructions"].median()
            decode_med = group["decode_instructions"].median()
            prefill_med = group["prefill_instructions"].median()
            if decode_med and decode_med > 0:
                row["prefill_decode_ratio"] = prefill_med / decode_med
            else:
                row["prefill_decode_ratio"] = None
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(["params_M", "prompt_len"]).reset_index(drop=True)
    return result


def determinism_table(df: pd.DataFrame) -> pd.DataFrame:
    df["per_token_decode"] = pd.to_numeric(df["per_token_decode"], errors="coerce")
    df["total_instructions"] = pd.to_numeric(df["total_instructions"], errors="coerce")

    det_rows = df[df["prompt_id"].str.contains("det", case=False, na=False)]
    if det_rows.empty:
        det_rows = df.copy()

    group_cols = ["config_id", "model", "prompt_len_target"]
    grouped = det_rows.groupby(group_cols, dropna=False)

    rows = []
    for name, group in grouped:
        if len(group) < 2:
            continue
        config_id, model, prompt_len = name
        ptd = group["per_token_decode"].dropna()
        ti = group["total_instructions"].dropna()
        rows.append({
            "config_id": config_id,
            "model": model,
            "prompt_len": prompt_len,
            "n_reps": len(group),
            "per_tok_decode_mean": ptd.mean(),
            "per_tok_decode_std": ptd.std(),
            "per_tok_decode_cv": (ptd.std() / ptd.mean() * 100) if ptd.mean() > 0 else None,
            "per_tok_decode_min": ptd.min(),
            "per_tok_decode_max": ptd.max(),
            "total_instr_mean": ti.mean(),
            "total_instr_std": ti.std(),
            "total_instr_cv": (ti.std() / ti.mean() * 100) if ti.mean() > 0 else None,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["config_id", "prompt_len"]).reset_index(drop=True)
    return result


def fmt_int(v):
    if pd.isna(v):
        return "-"
    return f"{int(v):,}"


def fmt_float(v, decimals=1):
    if pd.isna(v):
        return "-"
    return f"{v:.{decimals}f}"


def print_summary(summary: pd.DataFrame):
    print("\n" + "=" * 120)
    print("  BENCHMARK SUMMARY — sorted by params_M, prompt_len")
    print("=" * 120)

    header = (
        f"{'Config':<6} {'Model':<22} {'P(M)':>5} {'Arch':<10} {'Quant':<5} "
        f"{'PL':>4} {'N':>3} {'Tok(med)':>8} {'Instr_total(med)':>18} "
        f"{'Instr/tok_dec(med)':>20} {'±std':>12}"
    )
    print(header)
    print("-" * 120)

    for _, r in summary.iterrows():
        std_str = f"±{fmt_int(r.get('instr_per_tok_decode_std'))}" if pd.notna(r.get("instr_per_tok_decode_std")) else ""
        line = (
            f"{r['config_id']:<6} {r['model']:<22} {int(r['params_M']):>5} {r['arch']:<10} {r['quant']:<5} "
            f"{int(r['prompt_len']):>4} {int(r['n_measurements']):>3} {fmt_int(r['tok_generated_med']):>8} "
            f"{fmt_int(r['instr_total_med']):>18} {fmt_int(r['instr_per_tok_decode_med']):>20} {std_str:>12}"
        )
        print(line)

    if "prefill_instr_med" in summary.columns and summary["prefill_instr_med"].notna().any():
        print("\n" + "-" * 80)
        print("  PREFILL / DECODE BREAKDOWN")
        print("-" * 80)
        header2 = f"{'Config':<6} {'Model':<22} {'PL':>4} {'Prefill(med)':>16} {'Decode(med)':>16} {'Ratio P/D':>10}"
        print(header2)
        for _, r in summary.iterrows():
            if pd.notna(r.get("prefill_instr_med")):
                print(
                    f"{r['config_id']:<6} {r['model']:<22} {int(r['prompt_len']):>4} "
                    f"{fmt_int(r.get('prefill_instr_med')):>16} {fmt_int(r.get('decode_instr_med')):>16} "
                    f"{fmt_float(r.get('prefill_decode_ratio'), 2):>10}"
                )


def print_determinism(det: pd.DataFrame):
    if det.empty:
        print("\n  No determinism data (need >=2 reps per condition).")
        return

    print("\n" + "=" * 100)
    print("  DETERMINISM ANALYSIS — variance across repetitions")
    print("=" * 100)

    header = (
        f"{'Config':<6} {'Model':<22} {'PL':>4} {'Reps':>4} "
        f"{'per_tok_dec_mean':>16} {'std':>12} {'CV%':>6} "
        f"{'min':>14} {'max':>14}"
    )
    print(header)
    print("-" * 100)

    for _, r in det.iterrows():
        print(
            f"{r['config_id']:<6} {r['model']:<22} {int(r['prompt_len']):>4} {int(r['n_reps']):>4} "
            f"{fmt_int(r['per_tok_decode_mean']):>16} {fmt_int(r['per_tok_decode_std']):>12} "
            f"{fmt_float(r.get('per_tok_decode_cv'), 2):>6} "
            f"{fmt_int(r['per_tok_decode_min']):>14} {fmt_int(r['per_tok_decode_max']):>14}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("files", nargs="*", help="CSV files to analyze")
    parser.add_argument("--latest", action="store_true", help="Use latest benchmark_results_*.csv")
    parser.add_argument("--output", default=None, help="Save summary CSV to this path")
    args = parser.parse_args()

    if args.latest:
        csv_files = [find_latest_csv()]
    elif args.files:
        csv_files = args.files
    else:
        csv_files = [find_latest_csv()]

    print(f"Loading: {', '.join(csv_files)}")
    df = load_csvs(csv_files)
    print(f"Total rows: {len(df)}")
    print(f"Configs found: {sorted(df['config_id'].unique())}")

    summary = summary_table(df)
    det = determinism_table(df)

    print_summary(summary)
    print_determinism(det)

    output_path = args.output or str(PROJECT_DIR / "benchmark_analysis.csv")
    summary.to_csv(output_path, index=False)
    print(f"\nSummary CSV saved to: {output_path}")

    if not det.empty:
        det_path = output_path.replace(".csv", "_determinism.csv")
        det.to_csv(det_path, index=False)
        print(f"Determinism CSV saved to: {det_path}")


if __name__ == "__main__":
    main()
