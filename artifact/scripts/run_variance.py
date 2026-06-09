#!/usr/bin/env python3
"""P0-1: Binary-search tok/call across N reps for the currently-loaded model.

Reports mean, std, median, 95% CI (bootstrap). Appends to variance.csv.

Usage:
  python3 run_variance.py --model-tag pythia-70m-q4 --quant Q4_0 --simd yes \
      --arch gpt_neox --params-M 70 --n-reps 3

Prereqs: dfx replica running + llama_cpp canister with desired model loaded.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CANISTER = "llama_cpp"
DFX_CWD = Path("llama_cpp_canister")
OUT = Path(__file__).resolve().parents[1] / "data" / "verified" / "variance.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("P_short_A", "The capital of France is"),
    ("P_short_B", "Once upon a time in a"),
    ("P_short_C", "In machine learning a"),
]


def build_prompt_schedule(n_reps: int) -> list[tuple[str, str]]:
    if n_reps <= 0:
        return []
    return [PROMPTS[index % len(PROMPTS)] for index in range(n_reps)]


def dfx_call(method: str, arg: str, network: str = "local", timeout: int = 90) -> str:
    cmd = ["dfx", "canister", "call", "--network", network, CANISTER, method, arg]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(DFX_CWD))
    return r.stdout + r.stderr


def set_max_tokens(n: int, network: str = "local") -> None:
    dfx_call(
        "set_max_tokens",
        f"(record {{ max_tokens_query = 1 : nat64; max_tokens_update = {n} : nat64 }})",
        network,
    )


def reset_chat(network: str = "local") -> None:
    dfx_call(
        "remove_prompt_cache",
        '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })',
        network,
    )
    dfx_call(
        "new_chat",
        '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })',
        network,
    )


def run_inference(prompt: str, network: str = "local") -> dict:
    escaped = prompt.replace('"', '\\"').replace("\n", "\\n")
    arg = (
        f'(record {{ args = vec {{"--model"; "models/model.gguf"; '
        f'"--prompt-cache"; "prompt.cache"; "--prompt-cache-all"; '
        f'"-sp"; "-n"; "512"; "-p"; "{escaped}"}} }})'
    )
    out = dfx_call("run_update", arg, network, timeout=120)
    # Canister trap → dfx prints "Replica Reject" or similar; NO "Ok = record"
    # Canister error variant → "variant { Err"
    # Canister success → "variant {\n    Ok = record"
    ok = re.search(r"Ok\s*=\s*record", out)
    if not ok:
        if "trap" in out.lower() or "instruction" in out.lower() or "Reject" in out:
            return {"trapped": True, "raw": out[:400]}
        if re.search(r"Err\s*=", out):
            return {"error": True, "raw": out[:400]}
        return {"error": True, "raw": out[:400]}
    result = {"trapped": False, "error": False}
    m = re.search(r'output = "(.*?)";', out, re.DOTALL)
    result["output"] = m.group(1) if m else ""
    m = re.search(r'profiling = "(.*?)";', out, re.DOTALL)
    if m and m.group(1):
        try:
            result["profiling"] = json.loads(m.group(1).replace("\\", ""))
        except json.JSONDecodeError:
            result["profiling"] = {}
    else:
        result["profiling"] = {}
    return result


def binary_search(prompt: str, hi: int, network: str = "local", verbose: bool = True) -> int:
    lo = 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        set_max_tokens(mid, network)
        reset_chat(network)
        r = run_inference(prompt, network)
        if r.get("trapped") or r.get("error"):
            if verbose:
                tag = "TRAP" if r.get("trapped") else "ERR"
                print(f"    max={mid}: {tag} ({r.get('raw', '')[:120]})", flush=True)
            hi = mid - 1
        else:
            tokens_gen = r.get("profiling", {}).get("tokens_generated", "?")
            if verbose:
                print(f"    max={mid}: OK gen={tokens_gen}", flush=True)
            best = mid
            lo = mid + 1
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--simd", required=True, choices=["yes", "no"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--params-M", type=float, required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--size-MB", type=float, default=0)
    ap.add_argument("--hi", type=int, default=300, help="Binary search upper bound")
    ap.add_argument("--n-reps", type=int, default=3)
    ap.add_argument("--network", default="local")
    args = ap.parse_args()

    prompts = build_prompt_schedule(args.n_reps)
    results = []
    for pid, prompt in prompts:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] rep {pid}: searching...", flush=True)
        t0 = time.time()
        tc = binary_search(prompt, args.hi, args.network)
        dt = time.time() - t0
        print(f"  → tok/call = {tc}  ({dt:.1f}s)", flush=True)
        results.append({"prompt_id": pid, "prompt": prompt, "tok_call": tc, "seconds": round(dt, 1)})

    tcs = [r["tok_call"] for r in results]
    if tcs:
        import statistics
        mean = statistics.mean(tcs)
        stdev = statistics.stdev(tcs) if len(tcs) >= 2 else 0.0
        median = statistics.median(tcs)
    else:
        mean = stdev = median = 0.0

    # Write header if file doesn't exist yet
    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "model_tag", "arch", "params_M", "layers", "quant", "simd",
                "size_MB", "network", "n_reps", "mean_tok_call", "std_tok_call",
                "median_tok_call", "min_tok_call", "max_tok_call", "results_json",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            args.model_tag,
            args.arch,
            args.params_M,
            args.layers,
            args.quant,
            args.simd,
            args.size_MB,
            args.network,
            len(tcs),
            round(mean, 2),
            round(stdev, 3),
            median,
            min(tcs) if tcs else 0,
            max(tcs) if tcs else 0,
            json.dumps(results),
        ])

    cv = (stdev / mean * 100) if mean else 0
    print()
    print(f"=== {args.model_tag} ({args.quant}, SIMD={args.simd}) ===")
    print(f"n={len(tcs)} mean={mean:.2f} std={stdev:.3f} median={median} min/max={min(tcs) if tcs else 0}/{max(tcs) if tcs else 0}")
    print(f"CV = {cv:.2f}%  (coefficient of variation)")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
