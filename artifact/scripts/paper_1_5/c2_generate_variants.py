#!/usr/bin/env python3
"""Paper 1.5 C2 — generate 6 mixed-precision variants (V1, V2, V3a/b/c, V4).

V1  : embedding+lm_head Q8_0, rest Q4_K_M
V2  : attn (q,k,v) Q8_0, MLP Q4_K_M
V3a : first 4 layers Q8_0, rest Q4_K_M
V3b : first 8 layers Q8_0, rest Q4_K_M
V3c : first 12 layers Q8_0, rest Q4_K_M
V4  : top-20% ΔPPL-sensitive layers Q8_0 (ranked from c2-delta-ppl-<model>.csv)

Writes one row per variant to mixed_precision_variants.csv with sha256 + size.
Appends combined quantize logs to a sibling .log file.

Implemented by Task 6 (SmolLM2-135M) / Task 10 (Qwen 0.5B).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# --------------------------------------------------------------------------- #
# GGUF metadata helpers
# --------------------------------------------------------------------------- #

def detect_block_count(model_path: Path) -> int:
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as e:
        print(
            "error: python package `gguf` not installed. Invoke via "
            ".worktree-cache/quantize-venv/bin/python",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    reader = GGUFReader(str(model_path))
    for field in reader.fields.values():
        if field.name.endswith(".block_count"):
            return int(field.parts[-1][0])
    raise RuntimeError(f"Could not find `<arch>.block_count` in {model_path}")


# --------------------------------------------------------------------------- #
# Regex builder
# --------------------------------------------------------------------------- #

def layers_regex(layer_ids: list[int]) -> str:
    """Build a regex that matches exactly the given block layer indices,
    for attn_* / ffn_* weights.

    Uses explicit alternation with `\\.<idx>\\.` boundaries to avoid prefix
    ambiguity (e.g. `blk.1.` accidentally matching `blk.11.`).
    """
    if not layer_ids:
        raise ValueError("layer_ids must be non-empty")
    # Sort and dedup
    ids = sorted(set(layer_ids))
    alt = "|".join(str(i) for i in ids)
    return rf"^blk\.({alt})\.(attn_.*|ffn_.*)\.weight$"


# --------------------------------------------------------------------------- #
# Variant spec
# --------------------------------------------------------------------------- #

def build_variants(
    *,
    label: str,
    block_count: int,
    top_layers: list[int],
) -> list[dict]:
    """Return the 6 variant descriptors."""
    variants: list[dict] = []

    # V1: embedding-lifted
    variants.append({
        "variant_id": f"V1-{label}",
        "mixed_strategy": "embedding_q8",
        "extra_args": ["--token-embedding-type", "q8_0",
                       "--output-tensor-type", "q8_0"],
        "layers_q8": "",
        "layers_q4": "ALL",
        "token_embedding_type": "q8_0",
        "output_tensor_type": "",
        "notes_prefix": "tied embedding (no separate output.weight)",
    })

    # V2: attention (q/k/v) lifted
    v2_regex = r"^blk\.\d+\.attn_(q|k|v)\.weight$"
    variants.append({
        "variant_id": f"V2-{label}",
        "mixed_strategy": "attn_qkv_q8",
        "extra_args": ["--tensor-type", f"{v2_regex}=q8_0"],
        "layers_q8": "",
        "layers_q4": "ALL",
        "token_embedding_type": "",
        "output_tensor_type": "",
        "notes_prefix": f"attn_q/k/v -> q8_0 via regex",
    })

    # V3a/b/c: first-N-layers lifted
    for tag, n in [("V3a", 4), ("V3b", 8), ("V3c", 12)]:
        ids = list(range(min(n, block_count)))
        regex = layers_regex(ids)
        variants.append({
            "variant_id": f"{tag}-{label}",
            "mixed_strategy": f"first{n}_q8",
            "extra_args": ["--tensor-type", f"{regex}=q8_0"],
            "layers_q8": ",".join(str(i) for i in ids),
            "layers_q4": ",".join(str(i) for i in range(block_count) if i not in ids),
            "token_embedding_type": "",
            "output_tensor_type": "",
            "notes_prefix": f"layers 0..{n-1} -> q8_0",
        })

    # V4: top-20% ΔPPL-sensitive
    v4_regex = layers_regex(top_layers)
    variants.append({
        "variant_id": f"V4-{label}",
        "mixed_strategy": "top_deltappl_q8",
        "extra_args": ["--tensor-type", f"{v4_regex}=q8_0"],
        "layers_q8": ",".join(str(i) for i in sorted(top_layers)),
        "layers_q4": ",".join(str(i) for i in range(block_count)
                              if i not in top_layers),
        "token_embedding_type": "",
        "output_tensor_type": "",
        "notes_prefix": f"top-{len(top_layers)} ΔPPL layers -> q8_0",
    })

    return variants


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_QUANT_SIZE_RE = re.compile(
    r"quant size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(MiB|GiB|KiB|B)", re.IGNORECASE
)
_FALLBACK_RE = re.compile(r"falling back to|fallback", re.IGNORECASE)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_quantize(
    *,
    llama_quantize: str,
    src: Path,
    dst: Path,
    imatrix: Path,
    base_type: str,
    extra_args: list[str],
    log_fh,
) -> tuple[int, str]:
    """Run llama-quantize. Returns (returncode, captured_run_section).
    Writes combined stdout/stderr to log_fh.
    """
    cmd = [
        llama_quantize,
        "--imatrix", str(imatrix),
        *extra_args,
        str(src),
        str(dst),
        base_type,
    ]
    marker = f"[quantize-start-{time.time_ns()}]"
    log_fh.write(f"\n{marker} cmd: {' '.join(cmd)}\n")
    log_fh.flush()
    start_offset = log_fh.tell()
    res = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    log_fh.flush()
    log_path = Path(log_fh.name)
    with open(log_path, "r") as rf:
        rf.seek(start_offset)
        section = rf.read()
    return res.returncode, section


def extract_warnings(section: str) -> list[str]:
    """Best-effort pick of lines mentioning fallback quantization or warnings."""
    out = []
    for line in section.splitlines():
        stripped = line.strip()
        # llama-quantize prints "[NN/MM] tensor_name ... converting to q5_0 .. fallback"
        if _FALLBACK_RE.search(stripped):
            out.append(stripped)
        elif "WARN" in stripped.upper() and "warn" in stripped.lower():
            out.append(stripped)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help="baseline F16 GGUF")
    ap.add_argument("--imatrix", required=True, type=Path,
                    help="imatrix GGUF (generated in Task 4)")
    ap.add_argument("--delta-ppl-csv", required=True, type=Path,
                    help="per-layer ΔPPL CSV (from Task 5)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="where to write *.gguf variants")
    ap.add_argument("--label", required=True, type=str,
                    help="short tag (e.g. smollm135, qwen05)")
    ap.add_argument("--model-id", type=str, default=None,
                    help="model id for CSV (default: uppercase label guess)")
    ap.add_argument("--variants-csv", required=True, type=Path,
                    help="append-target CSV for variant metadata")
    ap.add_argument("--base-format", default="Q4_K_M", type=str)
    ap.add_argument("--top-frac", type=float, default=0.2,
                    help="fraction of layers (by block_count) to lift for V4")
    ap.add_argument("--llama-quantize", type=str,
                    default=os.environ.get("LLAMA_QUANTIZE",
                                           "/opt/homebrew/bin/llama-quantize"))
    ap.add_argument("--log-file", type=Path, default=None,
                    help="combined stdout/stderr log path (default: "
                         "<variants-csv-dir>/../results/paper_1_5/raw/"
                         "c2-variants-<label>.log)")
    args = ap.parse_args()

    # --- resolve inputs --------------------------------------------------
    model = args.model.resolve()
    imatrix = args.imatrix.resolve()
    delta_csv = args.delta_ppl_csv.resolve()
    variants_csv = args.variants_csv.resolve()

    for p in (model, imatrix, delta_csv):
        if not p.is_file():
            print(f"error: missing input file: {p}", file=sys.stderr)
            return 3
    if not shutil.which(args.llama_quantize) and not Path(args.llama_quantize).is_file():
        print(f"error: llama-quantize not found: {args.llama_quantize}", file=sys.stderr)
        return 3

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not variants_csv.is_file():
        print(f"error: variants CSV not found (header must pre-exist): {variants_csv}",
              file=sys.stderr)
        return 3

    # model_id default: simple mapping for the known labels
    default_ids = {"smollm135": "SmolLM2-135M", "qwen05": "Qwen2.5-0.5B"}
    model_id = args.model_id or default_ids.get(args.label, args.label)

    # log file default
    if args.log_file is None:
        # walk to worktree root
        here = Path(__file__).resolve()
        worktree_root = None
        for cand in here.parents:
            if (cand / ".worktree-cache").exists():
                worktree_root = cand
                break
        if worktree_root is None:
            raise RuntimeError("could not locate worktree root")
        log_path = (worktree_root
                    / "papers/instruction-bounded-inference/artifact/results/paper_1_5/raw"
                    / f"c2-variants-{args.label}.log")
    else:
        log_path = args.log_file.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- detect block_count ---------------------------------------------
    block_count = detect_block_count(model)
    if block_count <= 0:
        print(f"error: invalid block_count {block_count}", file=sys.stderr)
        return 4

    # --- rank ΔPPL layers -----------------------------------------------
    rows = []
    with open(delta_csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r["layer_idx"]), float(r["delta_ppl"])))
    rows.sort(key=lambda t: t[1], reverse=True)
    n_top = max(1, round(block_count * args.top_frac))
    top_layers = sorted([layer for layer, _ in rows[:n_top]])

    print(f"[c2_variants] label={args.label}  model_id={model_id}",
          file=sys.stderr)
    print(f"[c2_variants] block_count={block_count}  top_frac={args.top_frac}  "
          f"n_top={n_top}  top_layers={top_layers}", file=sys.stderr)
    print(f"[c2_variants] output_dir={out_dir}", file=sys.stderr)
    print(f"[c2_variants] variants_csv={variants_csv}", file=sys.stderr)
    print(f"[c2_variants] log={log_path}", file=sys.stderr)

    # --- build variants -------------------------------------------------
    variants = build_variants(
        label=args.label,
        block_count=block_count,
        top_layers=top_layers,
    )

    # --- run each variant ----------------------------------------------
    csv_rows_to_append: list[dict] = []
    t_start = time.monotonic()

    with open(log_path, "a") as log_fh:
        log_fh.write(
            f"\n# ===== c2_generate_variants.py ({args.label}) =====\n"
            f"# started_utc={_dt.datetime.utcnow().isoformat()}Z\n"
            f"# model={model}\n"
            f"# imatrix={imatrix}\n"
            f"# base_format={args.base_format}\n"
            f"# block_count={block_count}\n"
            f"# top_layers={top_layers}\n"
        )
        log_fh.flush()

        for v in variants:
            dst = out_dir / f"{v['variant_id']}.gguf"
            # Remove pre-existing output for deterministic reruns
            if dst.exists():
                dst.unlink()
            log_fh.write(f"\n##### variant {v['variant_id']} #####\n")
            log_fh.flush()
            t0 = time.monotonic()
            rc, section = run_quantize(
                llama_quantize=args.llama_quantize,
                src=model,
                dst=dst,
                imatrix=imatrix,
                base_type=args.base_format,
                extra_args=v["extra_args"],
                log_fh=log_fh,
            )
            dt = time.monotonic() - t0
            if rc != 0 or not dst.exists():
                print(f"[c2_variants] BLOCKED: {v['variant_id']} failed (rc={rc})",
                      file=sys.stderr)
                return 5
            size_bytes = dst.stat().st_size
            size_mb = round(size_bytes / 1e6, 3)
            sha = sha256_file(dst)
            warnings = extract_warnings(section)
            notes = v["notes_prefix"]
            if warnings:
                notes += f"; {len(warnings)} fallback/warn lines"
            ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # relative path to worktree root. Use the *unresolved* output_dir
            # so that a symlinked .worktree-cache (e.g. shared across worktrees)
            # is preserved as a relative path under the current worktree.
            here = Path(__file__).resolve()
            worktree_root = None
            for cand in here.parents:
                if (cand / ".worktree-cache").exists():
                    worktree_root = cand
                    break
            unresolved_dst = (args.output_dir / f"{v['variant_id']}.gguf")
            try:
                rel_path = str(unresolved_dst.resolve().relative_to(worktree_root)) \
                    if worktree_root else str(unresolved_dst)
            except ValueError:
                # dst resolves outside worktree (symlinked cache) — use the
                # symlink-preserving relative form via os.path.relpath against
                # the unresolved absolute path.
                abs_unresolved = unresolved_dst if unresolved_dst.is_absolute() \
                    else (Path.cwd() / unresolved_dst)
                try:
                    rel_path = os.path.relpath(abs_unresolved, start=worktree_root)
                except Exception:
                    rel_path = str(abs_unresolved)

            csv_rows_to_append.append({
                "variant_id": v["variant_id"],
                "model_id": model_id,
                "base_format": args.base_format,
                "mixed_strategy": v["mixed_strategy"],
                "layers_q8": v["layers_q8"],
                "layers_q4": v["layers_q4"],
                "token_embedding_type": v["token_embedding_type"],
                "output_tensor_type": v["output_tensor_type"],
                "gguf_path": rel_path,
                "size_MB": f"{size_mb}",
                "sha256": sha,
                "timestamp_utc": ts,
                "notes": notes,
            })
            print(
                f"[c2_variants] {v['variant_id']:<16s} size={size_mb:7.3f} MB "
                f"sha={sha[:12]}... ({dt:.1f}s) warns={len(warnings)}",
                file=sys.stderr,
            )
            log_fh.write(
                f"[{v['variant_id']}] size={size_mb} MB sha256={sha} "
                f"elapsed={dt:.1f}s warns={len(warnings)}\n"
            )
            log_fh.flush()

        total = time.monotonic() - t_start
        log_fh.write(f"\n# done  total_elapsed={total:.1f}s\n")

    # --- append CSV rows ------------------------------------------------
    fieldnames = [
        "variant_id", "model_id", "base_format", "mixed_strategy",
        "layers_q8", "layers_q4", "token_embedding_type", "output_tensor_type",
        "gguf_path", "size_MB", "sha256", "timestamp_utc", "notes",
    ]
    with open(variants_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(csv_rows_to_append)

    total = time.monotonic() - t_start
    print(f"[c2_variants] DONE — {len(variants)} variants in {total:.1f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
