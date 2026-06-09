#!/usr/bin/env python3
"""C1a projection validation — TriLM TQ2_0 measured vs Paper 1 §4.1 projection.

Plots tok/call vs parameter count (log-log) for both TriLM 560M and 3.9B,
overlaid with Paper 1's modern-arch projection (alpha_eff = 1.54) and the
theoretical floor (alpha_eff = 1.0). Each model gets two markers: SSN
mainnet measurement and local dfx measurement (which agree by construction
since the 40B instruction limit is a protocol invariant).
"""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[3]   # artifact/
MEAS = BASE / "data/paper_1_5/ternary_measurements.csv"
REG  = BASE / "data/paper_1_5/models_paper_1_5.csv"

rows = list(csv.DictReader(MEAS.open()))
reg = {r["id"]: r for r in csv.DictReader(REG.open())}


def measured_mean(model_id, env):
    vals = [
        float(r["tok_call"])
        for r in rows
        if r["model_id"] == model_id and r["environment"] == env
    ]
    return statistics.mean(vals) if vals else None


def projection(P_M, alpha):
    """Paper 1 §4.1 formula: tok/call ≈ 40e9 / (alpha * 2 * P)."""
    return 40e9 / (alpha * 2 * P_M * 1e6)


# Per-model measured + projected values
models = [("P15-01", "TriLM 560M"), ("P15-02", "TriLM 3.9B")]
points = []
for mid, label in models:
    if mid not in reg:
        continue
    P = float(reg[mid]["params_M_gguf"] or reg[mid]["params_M"])
    local = measured_mean(mid, "dfx-local")
    ssn   = measured_mean(mid, "ssn-mainnet")
    proj154 = projection(P, 1.54)
    proj_floor = projection(P, 1.0)
    points.append({
        "id": mid, "label": label, "P": P,
        "local": local, "ssn": ssn,
        "proj154": proj154, "proj_floor": proj_floor,
    })

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5.5))

# Continuous projection lines across the parameter range
P_range = np.logspace(np.log10(100), np.log10(10000), 200)  # 100M to 10B
ax.plot(P_range, [projection(P, 1.54) for P in P_range],
        ls="--", color="tab:gray", alpha=0.7,
        label=r"Paper 1 §4.1 projection ($\alpha_\mathrm{eff}=1.54$)")
ax.plot(P_range, [projection(P, 1.0) for P in P_range],
        ls=":", color="black", alpha=0.6,
        label=r"Theoretical floor ($\alpha_\mathrm{eff}=1.0$)")

# Measured points
for p in points:
    if p["ssn"] is not None:
        ax.scatter(p["P"], p["ssn"], s=130, marker="o",
                   color="tab:blue", edgecolor="black", zorder=5,
                   label=f"{p['label']} measured (SSN, {int(p['ssn'])} tok/call)")
    if p["local"] is not None and p["local"] != p["ssn"]:
        ax.scatter(p["P"], p["local"], s=80, marker="s",
                   color="tab:cyan", edgecolor="black", zorder=4,
                   label=f"{p['label']} measured (local, {int(p['local'])} tok/call)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Model parameters (M)")
ax.set_ylabel("tok/call (max within 40 B-instruction budget)")
ax.set_title("C1a — TriLM TQ2_0 measured vs Paper 1 §4.1 projection")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best", fontsize=9)

# Annotation summarising α_eff
text_lines = []
for p in points:
    if p["ssn"] is not None:
        a = 40e9 / (p["ssn"] * 2 * p["P"] * 1e6)
        text_lines.append(f"{p['label']}: α_eff = {a:.2f}")
ax.text(0.04, 0.04, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7"))

out = Path(__file__).parent / "c1a-projection-validation.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"Saved {out}")
print()
print("--- Numbers ---")
for p in points:
    print(f"  {p['label']:<14} P={int(p['P']):>5}M  local={p['local']}  ssn={p['ssn']}  "
          f"proj(1.54)={p['proj154']:.1f}  floor(1.0)={p['proj_floor']:.1f}")
