#!/usr/bin/env python3
"""Cross-reference models.csv registry values vs GGUF metadata. Emit discrepancies.csv."""
from pathlib import Path

from registry_utils import DATA_DIR, load_csv_lenient

ROOT = Path(__file__).resolve().parents[1]
MODELS = DATA_DIR / "models.csv"
GGUF = DATA_DIR / "verified" / "gguf_metadata.csv"
OUT = DATA_DIR / "verified" / "discrepancies.csv"


def main():
    models = load_csv_lenient(MODELS)
    gguf_rows = load_csv_lenient(GGUF)
    gguf_by_stem = {}
    for r in gguf_rows:
        stem = Path(r["file"]).stem
        gguf_by_stem[stem] = r

    rows = []
    for m in models:
        mid = m["id"]
        gguf_path = m.get("gguf_path", "").strip()
        stem = Path(gguf_path).stem if gguf_path.endswith(".gguf") else ""
        if not stem or stem not in gguf_by_stem:
            rows.append({
                "id": mid,
                "name": m["name"],
                "status": "NO_FILE",
                "issues": f"no GGUF on disk for {gguf_path or '[blank gguf_path]'}",
            })
            continue
        g = gguf_by_stem[stem]
        issues = []
        documented_diffs = []

        def cmp(field_csv, field_gguf, cast=int):
            v = m.get(field_csv, "").strip()
            gv = g.get(field_gguf, "").strip()
            if not v or not gv:
                return
            try:
                vv = cast(v)
                gvv = cast(gv)
            except (ValueError, TypeError):
                return
            if vv != gvv:
                issues.append(f"{field_csv}={v} vs GGUF={gv}")

        cmp("layers_real", "block_count")
        cmp("hidden_dim", "embedding_length")
        cmp("d_ff", "feed_forward_length")
        cmp("vocab_size", "vocab_size")
        # `params_M` is paper-facing and may intentionally differ from the GGUF tensor
        # count. Treat that as a documented convention difference only when the audited
        # `params_M_gguf` value itself matches the GGUF metadata.
        try:
            pm_csv = float(m.get("params_M", ""))
            pm_csv_gguf = float(m.get("params_M_gguf", "") or 0)
            pm_g = float(g.get("params_M_calc", 0))
            if pm_csv_gguf and pm_g and abs(pm_csv_gguf - pm_g) > 1e-6:
                issues.append(f"params_M_gguf={pm_csv_gguf} vs GGUF_calc={pm_g}M")
            if pm_csv and pm_g:
                pct = abs(pm_csv - pm_g) / pm_g * 100
                if pct > 5:
                    documented_diffs.append(f"params_M={pm_csv} vs GGUF_calc={pm_g}M ({pct:.1f}%)")
        except (ValueError, TypeError):
            pass

        if issues:
            status = "MISMATCH"
        elif documented_diffs:
            status = "PARAM_COUNT_CONVENTION"
        else:
            status = "OK"

        rows.append({
            "id": mid,
            "name": m["name"],
            "status": status,
            "gguf_file": Path(g["file"]).name,
            "gguf_match": m.get("gguf_match", ""),
            "L_csv": m.get("layers_real", ""),
            "L_gguf": g.get("block_count", ""),
            "d_csv": m.get("hidden_dim", ""),
            "d_gguf": g.get("embedding_length", ""),
            "P_csv": m.get("params_M", ""),
            "P_csv_gguf": m.get("params_M_gguf", ""),
            "P_gguf": g.get("params_M_calc", ""),
            "issues": "; ".join(issues + documented_diffs) if issues or documented_diffs else "",
        })

    import csv

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id", "name", "status", "gguf_file", "L_csv", "L_gguf",
                "gguf_match", "d_csv", "d_gguf", "P_csv", "P_csv_gguf", "P_gguf", "issues",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    mismatch = [r for r in rows if r.get("status") == "MISMATCH"]
    conventions = [r for r in rows if r.get("status") == "PARAM_COUNT_CONVENTION"]
    nofile = [r for r in rows if r.get("status") == "NO_FILE"]
    print(f"Total rows: {len(rows)}")
    print(f"OK: {len(rows) - len(mismatch) - len(nofile) - len(conventions)}")
    print(f"PARAM_COUNT_CONVENTION: {len(conventions)}")
    print(f"MISMATCH: {len(mismatch)}")
    print(f"NO_FILE: {len(nofile)}")
    print()
    print("Documented parameter-count convention differences:")
    for r in conventions:
        print(f"  {r['id']} {r['name']}: {r['issues']}")
    print()
    print("Mismatches:")
    for r in mismatch:
        print(f"  {r['id']} {r['name']}: {r['issues']}")
    print()
    print("No GGUF file found:")
    for r in nofile:
        print(f"  {r['id']} {r['name']}: {r.get('issues', '')}")
    print()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
