#!/usr/bin/env python3
"""Paper 1.5 C3 — Task 19.4: derive extended tok/session formula from C3 data.

Reads multicall_characterization.csv and fits:
  save_instr = a_s + b_s * n_tok
  load_instr = a_l + b_l * n_tok (should be near-constant for small n)

Per (model, kv_cache_type) pair.

Writes summary to stdout + optional JSON dump.
"""
from __future__ import annotations
import csv, re, statistics, json, sys
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "data" / "paper_1_5" / "multicall_characterization.csv"

def parse_notes(s: str) -> dict:
    d = {}
    if not s:
        return d
    for tok in s.split(";"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k.strip()] = v.strip()
    return d

def linear_fit(xs, ys):
    """Return (a, b, R2) for y = a + b*x."""
    import numpy as np
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(xs) < 2:
        return (float(ys[0]) if len(ys) else 0.0, 0.0, 1.0)
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = a + b * xs
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2

def main():
    rows = list(csv.DictReader(open(CSV)))
    print(f"Loaded {len(rows)} rows from {CSV}")

    # Group: single-call fresh sweeps for save fit; multi-call continue for load fit
    qwen_ioover_fresh = []  # (n_tok_saved, save_instr)
    qwen_mc_loads = []       # load_instr values (constant check)
    qwen_mc_saves = []       # (n_tok_saved, save_instr) during continue

    falcon_mc_saves = []     # same shape
    falcon_mc_loads = []
    falcon_mc_points = []    # full rows for table

    for r in rows:
        sid = r["session_id"]
        notes = parse_notes(r["notes"])
        save_tok = int(notes.get("save_t_tokens", "-1"))
        save_instr = int(r["io_overhead_write_inst"] or 0)
        load_instr = int(r["io_overhead_read_inst"] or 0)

        if sid.startswith("c3-ioover-qwen05-n"):
            qwen_ioover_fresh.append((save_tok, save_instr))
        elif sid.startswith("c3-ioover-qwen05-mc-step"):
            if save_instr > 0:
                qwen_mc_saves.append((save_tok, save_instr))
            if load_instr > 0:
                qwen_mc_loads.append(load_instr)
        elif sid.startswith("c3-ssm-falcon-h1-step"):
            falcon_mc_points.append(r)
            if save_instr > 0:
                falcon_mc_saves.append((save_tok, save_instr))
            if load_instr > 0:
                falcon_mc_loads.append(load_instr)

    out = {}

    # --- Qwen fits ---
    print("\n=== Qwen 2.5 0.5B Q4_0 (f16 KV) ===")
    if qwen_ioover_fresh:
        xs, ys = zip(*qwen_ioover_fresh)
        a, b, r2 = linear_fit(xs, ys)
        print(f"save_terminal_instr = {a:,.0f} + {b:,.0f} * n_tok   (R²={r2:.4f}, n={len(xs)})")
        out["qwen05_q4_f16_save"] = {"a": a, "b": b, "R2": r2, "n": len(xs)}
    if qwen_mc_loads:
        m = statistics.mean(qwen_mc_loads)
        s = statistics.stdev(qwen_mc_loads) if len(qwen_mc_loads) > 1 else 0.0
        cv = s / m if m else 0.0
        print(f"load_instr (mc)  mean={m:,.0f}  stdev={s:,.0f}  CV={cv:.4f}  n={len(qwen_mc_loads)}")
        out["qwen05_q4_f16_load"] = {"mean": m, "stdev": s, "cv": cv, "n": len(qwen_mc_loads)}

    # --- Falcon fits ---
    print("\n=== Falcon-H1-Tiny-90M Q8_0 (bounded SSM state) ===")
    if falcon_mc_saves:
        xs, ys = zip(*falcon_mc_saves)
        a, b, r2 = linear_fit(xs, ys)
        print(f"save_terminal_instr = {a:,.0f} + {b:,.0f} * n_tok   (R²={r2:.4f}, n={len(xs)})")
        # Also compute raw stats (to detect constancy)
        m = statistics.mean(ys)
        s = statistics.stdev(ys) if len(ys) > 1 else 0.0
        cv = s / m if m else 0.0
        print(f"   raw save_instr mean={m:,.0f}  stdev={s:,.0f}  CV={cv:.4f}")
        out["falcon_ssm_q8_save"] = {"a": a, "b": b, "R2": r2, "n": len(xs), "mean": m, "cv": cv}
    if falcon_mc_loads:
        m = statistics.mean(falcon_mc_loads)
        s = statistics.stdev(falcon_mc_loads) if len(falcon_mc_loads) > 1 else 0.0
        cv = s / m if m else 0.0
        print(f"load_instr (mc)  mean={m:,.0f}  stdev={s:,.0f}  CV={cv:.4f}  n={len(falcon_mc_loads)}")
        out["falcon_ssm_q8_load"] = {"mean": m, "stdev": s, "cv": cv, "n": len(falcon_mc_loads)}

    # Print falcon table rows
    if falcon_mc_points:
        print("\nFalcon per-step detail:")
        print(f"{'sid':40s} {'n_tok':>6s} {'wall':>7s} {'save':>14s} {'load':>14s}")
        for r in falcon_mc_points:
            notes = parse_notes(r["notes"])
            n_tok = notes.get("save_t_tokens", "?")
            wall = r["wall_clock_s"]
            si = r["io_overhead_write_inst"] or "0"
            li = r["io_overhead_read_inst"] or "0"
            print(f"{r['session_id']:40s} {n_tok:>6s} {wall:>7s} {si:>14s} {li:>14s}")

    # Emit JSON
    out_path = CSV.parent.parent.parent / "results" / "paper_1_5" / "tables" / "c3-fit-parameters.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
