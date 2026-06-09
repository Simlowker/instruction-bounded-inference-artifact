#!/usr/bin/env python3
"""
Kernel-level analysis for instruction-bounded inference paper.

Produces:
  1. Clean kernel comparison table (for paper Table X)
  2. Alpha vs dimension scaling analysis + figure
  3. Speedup sensitivity analysis
  4. Metering overhead decomposition

Usage:
    python analyze_kernels.py [--outdir PATH] [--no-plot]

Requires: numpy, pandas, matplotlib (optional)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- raw data (from matmul-bench RESULTS*.txt) ----------
# These are deterministic ICP instruction counts — no stochastic variance.
# 512x512 matvec, 100 iterations, 52,428,800 FLOPs total

KERNEL_512 = {
    "F32 Scalar":                {"instr": 387_841_416, "alpha": 7.397},
    "F32 SIMD (f32x4)":         {"instr": 105_166_216, "alpha": 2.006},
    "F32 SIMD x4 unrolled":     {"instr": 115_764_616, "alpha": 2.208},
    "F32 SIMD x8 unrolled":     {"instr": 106_139_016, "alpha": 2.024},
    "Ternary LUT (f32 cast)":   {"instr": 215_911_816, "alpha": 4.118},
    "Ternary AddSub (masks)":   {"instr": 293_633_416, "alpha": 5.601},
    "Ternary Shuffle (scalar)": {"instr": 505_908_616, "alpha": 9.649},
    "Ternary INT SIMD (extmul)":{"instr":  70_708_619, "alpha": 1.349},
    "Ternary DOT (i32x4.dot)":  {"instr":  59_035_019, "alpha": 1.126},
    "Ternary DOT x4 unrolled":  {"instr":  42_855_819, "alpha": 0.817},
    "Ternary CondNeg (XOR+SUB)":{"instr":  77_262_219, "alpha": 1.474},
}

FLOPS_512 = 52_428_800

# Variable dimension data (50 iterations each)
DIM_DATA = [
    {"dim": 128,  "f32_instr": 3_431_249,   "tern_instr": 1_850_421,   "alpha_f32": 2.094, "alpha_tern": 1.129},
    {"dim": 256,  "f32_instr": 13_108_049,  "tern_instr": 6_976_821,   "alpha_f32": 2.000, "alpha_tern": 1.065},
    {"dim": 512,  "f32_instr": 51_200_849,  "tern_instr": 27_060_021,  "alpha_f32": 1.953, "alpha_tern": 1.032},
    {"dim": 576,  "f32_instr": 64_628_049,  "tern_instr": 34_128_821,  "alpha_f32": 1.948, "alpha_tern": 1.029},
    {"dim": 768,  "f32_instr": 114_279_249, "tern_instr": 60_250_421,  "alpha_f32": 1.938, "alpha_tern": 1.022},
    {"dim": 1024, "f32_instr": 202_343_249, "tern_instr": 106_548_021, "alpha_f32": 1.930, "alpha_tern": 1.016},
]

# Metering and comparison data
METERING = {
    "empty_loop_instr": 177_460_428,
    "empty_loop_iters": 26_214_400,
    "cost_per_iter": 6.770,
    "beta": 1.844,
}

COMPARISONS = {
    "rmsnorm_f32": 179_457_216,
    "rmsnorm_i32": 186_164_419,
    "rmsnorm_ratio": 0.964,
    "f32_scalar_mul_add": 301_466_823,
    "i32_scalar_mul_add": 249_038_028,
    "f32_i32_ratio": 1.211,
    "f32x4_simd": 103_220_424,
    "i32x4_simd": 98_305_230,
    "f32x4_i32x4_ratio": 1.050,
    "transformer_attn": 456_386_216,
    "gdn_layer": 461_927_016,
    "transf_gdn_ratio": 0.988,
}


def kernel_table() -> pd.DataFrame:
    """Clean kernel comparison table for paper."""
    rows = []
    ref = KERNEL_512["F32 SIMD (f32x4)"]["instr"]

    for name, d in KERNEL_512.items():
        speedup_vs_f32 = ref / d["instr"]
        rows.append({
            "kernel": name,
            "instructions_M": round(d["instr"] / 1e6, 1),
            "alpha_kernel": d["alpha"],
            "speedup_vs_f32_simd": round(speedup_vs_f32, 3),
        })

    return pd.DataFrame(rows)


def dimension_scaling() -> pd.DataFrame:
    """Alpha vs dimension analysis."""
    df = pd.DataFrame(DIM_DATA)
    df["speedup_f32_tern"] = df["f32_instr"] / df["tern_instr"]

    # Compute asymptotic alpha (metering overhead → 0 as dim → ∞)
    # alpha_theoretical_f32 = 2.0 (from ICP cost table: f32.mul=2 + f32.add=2 per MAC, /2 for FLOP def)
    # alpha_theoretical_tern ≈ 1.0 (i32x4.dot_i16x8_s = 1 per 4 MACs = 0.25/MAC + overhead)
    df["alpha_f32_excess"] = df["alpha_f32"] - 2.0  # overhead above theoretical
    df["alpha_tern_excess"] = df["alpha_tern"] - 1.0

    return df


def plot_analysis(dim_df: pd.DataFrame, outdir: Path):
    """Generate kernel analysis figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Kernel comparison bar chart
    ax = axes[0]
    kt = kernel_table()
    # Sort by alpha
    kt_sorted = kt.sort_values("alpha_kernel")
    colors = ["#2ecc71" if a < 1.5 else "#f39c12" if a < 2.5 else "#e74c3c"
              for a in kt_sorted["alpha_kernel"]]
    ax.barh(range(len(kt_sorted)), kt_sorted["alpha_kernel"], color=colors, edgecolor="k", linewidth=0.5)
    ax.set_yticks(range(len(kt_sorted)))
    ax.set_yticklabels(kt_sorted["kernel"], fontsize=7)
    ax.set_xlabel("α (ICP cost / FLOP)")
    ax.set_title("A. Kernel instruction cost")
    ax.axvline(x=2.006, color="k", linestyle="--", alpha=0.3, label="F32 SIMD baseline")
    ax.legend(fontsize=7)

    # Panel B: Alpha vs dimension
    ax = axes[1]
    ax.plot(dim_df["dim"], dim_df["alpha_f32"], "o-", color="#3498db", label="F32 SIMD")
    ax.plot(dim_df["dim"], dim_df["alpha_tern"], "s-", color="#e74c3c", label="Ternary DOT")
    ax.axhline(y=2.0, color="#3498db", linestyle=":", alpha=0.3, label="F32 theoretical (2.0)")
    ax.axhline(y=1.0, color="#e74c3c", linestyle=":", alpha=0.3, label="Ternary theoretical (~1.0)")
    ax.set_xlabel("Matrix dimension")
    ax.set_ylabel("α (ICP cost / FLOP)")
    ax.set_title("B. α converges to theoretical as dim↑")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: Speedup vs dimension
    ax = axes[2]
    ax.plot(dim_df["dim"], dim_df["speedup_f32_tern"], "D-", color="#9b59b6")
    ax.axhline(y=2.0, color="k", linestyle="--", alpha=0.3, label="2× theoretical limit")
    ax.set_xlabel("Matrix dimension")
    ax.set_ylabel("F32/Ternary speedup")
    ax.set_title("C. Ternary speedup approaches 2× at scale")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.8, 2.0)

    plt.tight_layout()
    fig.savefig(outdir / "kernel_analysis.png", dpi=200)
    fig.savefig(outdir / "kernel_analysis.pdf")
    plt.close()
    print(f"  Saved: {outdir / 'kernel_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Kernel analysis")
    parser.add_argument(
        "--outdir",
        default="artifact/results/current/extended_analysis",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Kernel comparison
    print("=== KERNEL COMPARISON (512x512, 100 iter) ===")
    kt = kernel_table()
    kt.to_csv(outdir / "kernel_comparison.csv", index=False)
    print(kt.to_string(index=False))

    # Dimension scaling
    print("\n=== ALPHA vs DIMENSION ===")
    dim_df = dimension_scaling()
    dim_df.to_csv(outdir / "alpha_vs_dimension.csv", index=False)
    print(dim_df[["dim", "alpha_f32", "alpha_tern", "speedup_f32_tern"]].to_string(index=False))

    # Metering overhead
    print(f"\n=== METERING OVERHEAD ===")
    print(f"  β = {METERING['beta']:.3f}")
    print(f"  Cost/iter = {METERING['cost_per_iter']:.3f} ICP units")
    print(f"  → {(METERING['beta'] - 1) / METERING['beta'] * 100:.1f}% of loop cost is metering overhead")

    # Key comparisons
    print(f"\n=== REFUTATION COMPARISONS ===")
    print(f"  f32/i32 scalar: {COMPARISONS['f32_i32_ratio']:.3f}x (table predicts 2.0x)")
    print(f"  f32x4/i32x4 SIMD: {COMPARISONS['f32x4_i32x4_ratio']:.3f}x (table predicts 2.0x)")
    print(f"  RMSNorm F32/INT: {COMPARISONS['rmsnorm_ratio']:.3f}x (INT is NOT faster)")
    print(f"  Transformer/GDN: {COMPARISONS['transf_gdn_ratio']:.3f}x (linear attn = no gain)")

    # Key finding for paper
    print(f"\n=== KEY FINDINGS ===")
    best = KERNEL_512["Ternary DOT x4 unrolled"]
    ref = KERNEL_512["F32 SIMD (f32x4)"]
    print(f"  Best kernel: Ternary DOT x4 (α={best['alpha']:.3f})")
    print(f"  vs F32 SIMD baseline (α={ref['alpha']:.3f}): {ref['instr']/best['instr']:.2f}x faster")
    print(f"  Unrolling gain on ternary: {(59_035_019/42_855_819 - 1)*100:.1f}%")
    print(f"  Unrolling gain on F32: {(105_166_216/115_764_616 - 1)*100:.1f}% (NEGATIVE — hurts)")

    if not args.no_plot:
        print("\nGenerating figures...")
        plot_analysis(dim_df, outdir)


if __name__ == "__main__":
    main()
