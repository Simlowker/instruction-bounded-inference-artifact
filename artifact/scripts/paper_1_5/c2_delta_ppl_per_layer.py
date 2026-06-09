#!/usr/bin/env python3
"""Paper 1.5 C2 — ΔPPL-per-layer sensitivity.

For each transformer block L in a F16 GGUF, quantize only L's attn+ffn weights
to Q4_K_M (rest F16), measure perplexity on wikitext2-test, record
ΔPPL_L = PPL_L − PPL_baseline.

Output: a CSV with columns
    layer_idx, regex, ppl, delta_ppl, ppl_uncertainty

Reusable across Task 5 (SmolLM2-135M) and Task 10 (Qwen 0.5B).

Notes on the quantize strategy
------------------------------
llama-quantize applies `--tensor-type REGEX=TYPE` overrides only when the base
ftype is a quantized type (see llama-quant.cpp: `ggml_is_quantized(default_type)`
guard). We therefore pass base ftype `Q4_K_M` and use two overrides:

  1. target layer L:  `^blk\\.L\\.(attn_.*|ffn_.*)\\.weight$`  → q4_K
  2. all other layers: `^blk\\.(?!L\\.)[0-9]+\\.(attn_.*|ffn_.*)\\.weight$` → f16

Combined with `--output-tensor-type f16` and `--token-embedding-type f16`, this
leaves everything at F16 except the weights of layer L, which get Q4_K_M-family
quantization (q4_K, with automatic fallback to q5_0/q6_K/q8_0 for tensor shapes
not divisible by 256 — same fallback policy as a plain `Q4_K_M` run on the same
model).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# GGUF metadata -> block_count
# --------------------------------------------------------------------------- #

def detect_block_count(model_path: Path) -> int:
    """Read `<arch>.block_count` from a GGUF file using the `gguf` library."""
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as e:
        print(
            "error: python package `gguf` not installed. Activate "
            ".worktree-cache/quantize-venv (or its python) before running.",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    reader = GGUFReader(str(model_path))
    for field in reader.fields.values():
        if field.name.endswith(".block_count"):
            # field.parts is a list of numpy arrays; take the scalar
            return int(field.parts[-1][0])
    raise RuntimeError(f"Could not find `<arch>.block_count` in {model_path}")


# --------------------------------------------------------------------------- #
# Subprocess helpers
# --------------------------------------------------------------------------- #

_FINAL_PPL_RE = re.compile(
    r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)"
)


def quantize_single_layer(
    *,
    llama_quantize: str,
    src_f16: Path,
    dst: Path,
    layer_idx: int,
    log_fh,
) -> str:
    """Run llama-quantize so that only layer `layer_idx` is Q4_K_M-ized.

    Returns the positive (target) regex used, so it can be recorded in the CSV.
    """
    keep_regex = rf"^blk\.{layer_idx}\.(attn_.*|ffn_.*)\.weight$"
    other_regex = rf"^blk\.(?!{layer_idx}\.)[0-9]+\.(attn_.*|ffn_.*)\.weight$"

    cmd = [
        llama_quantize,
        "--tensor-type", f"{keep_regex}=q4_K",
        "--tensor-type", f"{other_regex}=f16",
        "--output-tensor-type", "f16",
        "--token-embedding-type", "f16",
        str(src_f16),
        str(dst),
        "Q4_K_M",
    ]
    log_fh.write(f"\n[quantize layer={layer_idx}] cmd: {' '.join(cmd)}\n")
    log_fh.flush()
    res = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not dst.exists():
        raise RuntimeError(
            f"llama-quantize failed for layer {layer_idx} (rc={res.returncode})"
        )
    return keep_regex


def run_perplexity(
    *,
    llama_perplexity: str,
    model: Path,
    calib: Path,
    ctx_size: int,
    log_fh,
) -> tuple[float, float]:
    """Run llama-perplexity and extract (ppl, uncertainty).

    llama-perplexity is chatty (prompt-eval progress dots) and prints the
    final estimate to stderr. To avoid PIPE buffering / orphaning issues in
    non-interactive harnesses, we redirect stdout+stderr straight to the
    log file handle and re-read the tail for the `Final estimate` line.
    """
    cmd = [
        llama_perplexity,
        "--model", str(model),
        "--file", str(calib),
        "--ctx-size", str(ctx_size),
    ]
    # Marker so we can find this run's section in the log afterwards.
    marker = f"[perplexity-start-{time.time_ns()}]"
    log_fh.write(f"\n{marker} cmd: {' '.join(cmd)}\n")
    log_fh.flush()
    start_offset = log_fh.tell()

    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    log_fh.flush()

    if res.returncode != 0:
        raise RuntimeError(f"llama-perplexity exited with rc={res.returncode}")

    # Re-read only this run's chunk of the log file and extract final PPL.
    log_path = Path(log_fh.name)
    with open(log_path, "r") as rf:
        rf.seek(start_offset)
        chunk = rf.read()
    for line in reversed(chunk.splitlines()):
        m = _FINAL_PPL_RE.search(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise RuntimeError("Could not find 'Final estimate: PPL = X +/- Y' in output")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help="path to the baseline F16 GGUF")
    ap.add_argument("--calib", required=True, type=Path,
                    help="path to the perplexity corpus (WikiText-2-raw test split .txt)")
    ap.add_argument("--label", required=True, type=str,
                    help="short tag (e.g. smollm135, qwen05) for intermediate file names")
    ap.add_argument("--output", required=True, type=Path,
                    help="destination CSV (layer_idx,regex,ppl,delta_ppl,ppl_uncertainty)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="override auto-detected block_count (for debugging)")
    ap.add_argument("--ctx-size", type=int, default=2048,
                    help="perplexity ctx size (default 2048 to match imatrix)")
    ap.add_argument("--stage-dir", type=Path, default=None,
                    help="scratch dir for per-layer GGUFs (default: "
                         "<worktree>/.worktree-cache/deltappl-stage)")
    ap.add_argument("--llama-quantize", type=str,
                    default=os.environ.get("LLAMA_QUANTIZE", "/opt/homebrew/bin/llama-quantize"))
    ap.add_argument("--llama-perplexity", type=str,
                    default=os.environ.get("LLAMA_PERPLEXITY", "/opt/homebrew/bin/llama-perplexity"))
    args = ap.parse_args()

    # --- resolve & sanity-check inputs -----------------------------------
    model = args.model.resolve()
    calib = args.calib.resolve()
    output = args.output.resolve()

    for path in (model, calib):
        if not path.is_file():
            print(f"error: missing input file: {path}", file=sys.stderr)
            return 3
    for binary in (args.llama_quantize, args.llama_perplexity):
        if not shutil.which(binary) and not Path(binary).is_file():
            print(f"error: executable not found: {binary}", file=sys.stderr)
            return 3

    if args.stage_dir is None:
        # infer worktree root: walk up from this script until we find a dir
        # that contains a `.worktree-cache` sibling (symlink or dir). This
        # works regardless of where exactly the script is in the tree and
        # regardless of how sys.argv[0] was resolved.
        here = Path(__file__).resolve()
        worktree_root: Optional[Path] = None
        for candidate in here.parents:
            if (candidate / ".worktree-cache").exists():
                worktree_root = candidate
                break
        if worktree_root is None:
            raise RuntimeError(
                "could not locate worktree root (.worktree-cache not found "
                f"in any parent of {here}). Pass --stage-dir explicitly."
            )
        stage_dir = worktree_root / ".worktree-cache" / "deltappl-stage"
    else:
        stage_dir = args.stage_dir.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- detect layer count ---------------------------------------------
    n_layers = args.n_layers if args.n_layers is not None else detect_block_count(model)
    if n_layers <= 0:
        print(f"error: invalid block_count {n_layers}", file=sys.stderr)
        return 4

    # sibling .log next to the CSV (same stem)
    log_path = output.with_suffix(".log")
    print(f"[c2_delta_ppl] model={model}", file=sys.stderr)
    print(f"[c2_delta_ppl] calib={calib}", file=sys.stderr)
    print(f"[c2_delta_ppl] n_layers={n_layers}  ctx_size={args.ctx_size}", file=sys.stderr)
    print(f"[c2_delta_ppl] stage_dir={stage_dir}", file=sys.stderr)
    print(f"[c2_delta_ppl] output={output}", file=sys.stderr)
    print(f"[c2_delta_ppl] log={log_path}", file=sys.stderr)

    t_start = time.monotonic()
    rows: list[dict] = []

    with open(log_path, "w") as log_fh:
        log_fh.write(
            f"# c2_delta_ppl_per_layer.py\n"
            f"# label={args.label}\n"
            f"# model={model}\n"
            f"# calib={calib}\n"
            f"# n_layers={n_layers}\n"
            f"# ctx_size={args.ctx_size}\n"
        )
        log_fh.flush()

        # --- baseline PPL (F16, unmodified) ------------------------------
        log_fh.write("\n===== baseline F16 perplexity =====\n")
        log_fh.flush()
        t0 = time.monotonic()
        baseline_ppl, baseline_unc = run_perplexity(
            llama_perplexity=args.llama_perplexity,
            model=model,
            calib=calib,
            ctx_size=args.ctx_size,
            log_fh=log_fh,
        )
        t_baseline = time.monotonic() - t0
        print(f"[c2_delta_ppl] baseline PPL = {baseline_ppl:.4f} +/- {baseline_unc:.5f}  "
              f"({t_baseline:.1f}s)", file=sys.stderr)
        log_fh.write(f"\n[baseline] PPL={baseline_ppl} +/- {baseline_unc}  elapsed={t_baseline:.1f}s\n")
        log_fh.flush()

        # --- per-layer loop ---------------------------------------------
        for L in range(n_layers):
            stage_gguf = stage_dir / f"{args.label}.blkq.{L:02d}.gguf"
            log_fh.write(f"\n===== layer {L} =====\n")
            log_fh.flush()
            iter_t0 = time.monotonic()
            regex = quantize_single_layer(
                llama_quantize=args.llama_quantize,
                src_f16=model,
                dst=stage_gguf,
                layer_idx=L,
                log_fh=log_fh,
            )
            ppl, unc = run_perplexity(
                llama_perplexity=args.llama_perplexity,
                model=stage_gguf,
                calib=calib,
                ctx_size=args.ctx_size,
                log_fh=log_fh,
            )
            # delete the staged GGUF promptly (30 * ~270 MB would be ~8 GB otherwise)
            try:
                stage_gguf.unlink()
            except FileNotFoundError:
                pass

            delta = ppl - baseline_ppl
            iter_elapsed = time.monotonic() - iter_t0
            total_elapsed = time.monotonic() - t_start
            rows.append({
                "layer_idx": L,
                "regex": regex,
                "ppl": f"{ppl:.6f}",
                "delta_ppl": f"{delta:.6f}",
                "ppl_uncertainty": f"{unc:.6f}",
            })

            if L == 0 or (L + 1) % 5 == 0 or L == n_layers - 1:
                print(
                    f"[c2_delta_ppl] layer {L:2d}: PPL={ppl:.4f} "
                    f"Δ={delta:+.4f} (iter={iter_elapsed:.1f}s  total={total_elapsed:.1f}s)",
                    file=sys.stderr,
                )
            log_fh.write(
                f"[layer {L}] PPL={ppl} +/- {unc}  delta={delta}  iter_elapsed={iter_elapsed:.1f}s\n"
            )
            log_fh.flush()

        total = time.monotonic() - t_start
        log_fh.write(f"\n[done] total_elapsed={total:.1f}s  baseline_ppl={baseline_ppl}\n")

    # --- write CSV -------------------------------------------------------
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer_idx", "regex", "ppl", "delta_ppl", "ppl_uncertainty"]
        )
        writer.writeheader()
        writer.writerows(rows)

    total = time.monotonic() - t_start
    print(
        f"[c2_delta_ppl] DONE — {n_layers} layers in {total:.1f}s "
        f"({total/60:.1f} min). Baseline PPL={baseline_ppl:.4f}. CSV={output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
