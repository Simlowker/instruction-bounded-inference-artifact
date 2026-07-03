#!/usr/bin/env python3
"""
Benchmark script for instruction-bounded inference paper.
Runs repeated measurements with performance_counter profiling.

Usage:
  # Full benchmark (5 configs × 3 lengths × 4 prompts × 2 reps = 120 runs)
  python -m scripts.run_benchmark --network local

  # Determinism test only (2 configs × 10 identical reps = 20 runs)
  python -m scripts.run_benchmark --network local --determinism-only

  # Single config test
  python -m scripts.run_benchmark --network local --config C2

  # Binary search for tok/call on one config
  python -m scripts.run_benchmark --network local --config C1 --binary-search
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "benchmark_config.json"
CANISTER_NAME = "llama_cpp"


def dfx_call(method: str, argument: str, network: str = "local") -> str:
    cmd = ["dfx", "canister", "call", CANISTER_NAME, method, argument]
    if network != "local":
        cmd.extend(["--network", network])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout + result.stderr


def upload_model(gguf_path: str, network: str = "local"):
    abs_path = PROJECT_DIR / gguf_path
    if not abs_path.exists():
        print(f"  ERROR: GGUF file not found: {abs_path}")
        return False
    cmd = [
        sys.executable, "-m", "scripts.upload",
        "--network", network,
        "--canister", CANISTER_NAME,
        "--canister-filename", "models/model.gguf",
        "--filetype", "gguf",
        str(abs_path),
    ]
    print(f"  Uploading {gguf_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=600)
    if "Congratulations" in result.stdout or "Congratulations" in result.stderr:
        print("  Upload OK")
        return True
    print(f"  Upload output: {result.stdout[-200:]}")
    return True  # may already be uploaded


def load_model(network: str = "local") -> bool:
    r = dfx_call("load_model", '(record { args = vec {"--model"; "models/model.gguf";} })', network)
    if "Ok" in r:
        print("  Model loaded OK")
        return True
    print(f"  Load FAILED: {r[:200]}")
    return False


def set_max_tokens(max_tokens: int, network: str = "local"):
    dfx_call("set_max_tokens",
             f'(record {{ max_tokens_query = 1 : nat64; max_tokens_update = {max_tokens} : nat64 }})',
             network)


def reset_chat(network: str = "local"):
    dfx_call("remove_prompt_cache",
             '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })', network)
    dfx_call("new_chat",
             '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })', network)


def run_inference(prompt: str, network: str = "local") -> dict | None:
    escaped = prompt.replace('"', '\\"').replace('\n', '\\n')
    arg = (
        f'(record {{ args = vec {{"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"; '
        f'"--prompt-cache-all"; "-sp"; "-n"; "512"; "-p"; "{escaped}"}} }})'
    )
    r = dfx_call("run_update", arg, network)

    if "Ok" not in r:
        if "instruction" in r.lower() or "trap" in r.lower():
            return {"trapped": True, "raw": r[:300]}
        return {"error": True, "raw": r[:300]}

    result = {"trapped": False, "error": False}

    # Parse output
    m = re.search(r'output = "(.*?)";', r, re.DOTALL)
    result["output"] = m.group(1) if m else ""

    # Parse generated_eog
    result["generated_eog"] = "generated_eog = true" in r

    # Parse profiling JSON
    m = re.search(r'profiling = "(.*?)";', r, re.DOTALL)
    if m and m.group(1):
        profiling_str = m.group(1).replace("\\", "")
        try:
            result["profiling"] = json.loads(profiling_str)
        except json.JSONDecodeError:
            result["profiling"] = {"raw": profiling_str}
    else:
        result["profiling"] = {}

    # Parse prompt_remaining
    m = re.search(r'prompt_remaining = "(.*?)";', r, re.DOTALL)
    result["prompt_remaining"] = m.group(1) if m else ""

    return result


def count_output_tokens(output: str) -> int:
    """Rough token count from output text (actual count is in profiling)."""
    return len(output.split()) if output else 0


def binary_search_tok_call(prompt: str, known_tok_call: int, network: str = "local") -> dict:
    """Binary search for max tokens per call before TRAP."""
    lo, hi = 1, known_tok_call + 50
    best = 0
    best_profiling = {}

    print(f"  Binary search [{lo}, {hi}]")

    while lo <= hi:
        mid = (lo + hi) // 2
        set_max_tokens(mid, network)
        reset_chat(network)
        r = run_inference(prompt, network)

        if r is None or r.get("trapped") or r.get("error"):
            print(f"    max_tokens={mid}: TRAP")
            hi = mid - 1
        else:
            tokens = r.get("profiling", {}).get("tokens_generated", 0)
            print(f"    max_tokens={mid}: OK (generated {tokens})")
            best = mid
            best_profiling = r.get("profiling", {})
            lo = mid + 1

    return {"tok_call": best, "profiling": best_profiling}


def run_single_measurement(config: dict, prompt_id: str, prompt_len: str,
                           prompt_text: str, rep: int, network: str,
                           writer: csv.DictWriter):
    """Run a single inference measurement and log to CSV."""
    set_max_tokens(config["safe_max_tokens"], network)
    reset_chat(network)

    result = run_inference(prompt_text, network)

    if result is None or result.get("error"):
        print(f"    {config['id']}/{prompt_len}/{prompt_id}/r{rep}: ERROR")
        return

    if result.get("trapped"):
        print(f"    {config['id']}/{prompt_len}/{prompt_id}/r{rep}: TRAPPED at safe_max_tokens={config['safe_max_tokens']}")
        return

    prof = result.get("profiling", {})
    tokens_gen = prof.get("tokens_generated", 0)

    row = {
        "timestamp": datetime.now().isoformat(),
        "config_id": config["id"],
        "model": config["name"],
        "params_M": config["params_M"],
        "arch": config["arch"],
        "quant": config["quant"],
        "prompt_id": prompt_id,
        "prompt_len_target": prompt_len,
        "prompt_tokens": prof.get("prompt_tokens", ""),
        "max_tokens_set": config["safe_max_tokens"],
        "tokens_generated": tokens_gen,
        "total_instructions": prof.get("total", ""),
        "decode_instructions": prof.get("decode", ""),
        "sample_instructions": prof.get("sample", ""),
        "setup_instructions": prof.get("setup", ""),
        "overhead_instructions": prof.get("overhead", ""),
        "per_token_total": prof.get("per_token_total", ""),
        "per_token_decode": prof.get("per_token_decode", ""),
        "generated_eog": result.get("generated_eog", False),
        "prompt_remaining": "yes" if result.get("prompt_remaining") else "no",
        "env": network,
        "rep": rep,
    }
    writer.writerow(row)

    print(f"    {config['id']}/{prompt_len}/{prompt_id}/r{rep}: "
          f"{tokens_gen} tok, {prof.get('total', '?')} instr, "
          f"{prof.get('per_token_decode', '?')} instr/tok(decode)")


def main():
    parser = argparse.ArgumentParser(description="ICP LLM Benchmark")
    parser.add_argument("--network", default="local")
    parser.add_argument("--config", help="Run only this config ID (e.g., C1)")
    parser.add_argument("--determinism-only", action="store_true")
    parser.add_argument("--binary-search", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    configs = cfg["configs"]
    prompts = cfg["prompts"]
    protocol = cfg["protocol"]

    if args.config:
        configs = [c for c in configs if c["id"] == args.config]
        if not configs:
            print(f"Config {args.config} not found")
            sys.exit(1)

    # Output CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(PROJECT_DIR / f"benchmark_results_{timestamp}.csv")

    fieldnames = [
        "timestamp", "config_id", "model", "params_M", "arch", "quant",
        "prompt_id", "prompt_len_target", "prompt_tokens", "max_tokens_set",
        "tokens_generated", "total_instructions", "decode_instructions",
        "sample_instructions", "setup_instructions", "overhead_instructions",
        "per_token_total", "per_token_decode", "generated_eog",
        "prompt_remaining", "env", "rep",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for config in configs:
            print(f"\n{'='*60}")
            print(f"  CONFIG: {config['id']} — {config['name']} {config['quant']}")
            print(f"{'='*60}")

            # Upload & load model
            if not args.skip_upload:
                upload_model(config["gguf"], args.network)
            if not load_model(args.network):
                print(f"  SKIPPING {config['id']} — load failed")
                continue

            # Binary search mode
            if args.binary_search:
                print("\n  --- BINARY SEARCH ---")
                short_prompt = prompts["16"]["P1"]
                bs = binary_search_tok_call(short_prompt, config["known_tok_call"], args.network)
                print(f"  Result: tok/call = {bs['tok_call']}")
                print(f"  Profiling: {json.dumps(bs['profiling'], indent=2)}")
                continue

            # Determinism test
            if args.determinism_only or config["id"] in protocol["determinism_configs"]:
                if args.determinism_only or not args.config:
                    if config["id"] in protocol["determinism_configs"]:
                        print(f"\n  --- DETERMINISM TEST ({protocol['determinism_reps']} reps) ---")
                        prompt_text = prompts["128"]["P1"]
                        for rep in range(1, protocol["determinism_reps"] + 1):
                            run_single_measurement(
                                config, "P1_det", "128", prompt_text,
                                rep, args.network, writer
                            )
                            csvfile.flush()

            if args.determinism_only:
                continue

            # Full benchmark: 3 lengths × 4 prompts × 2 reps
            for prompt_len in ["16", "128", "512"]:
                print(f"\n  --- prompt_len={prompt_len} ---")
                for prompt_id in ["P1", "P2", "P3", "P4"]:
                    prompt_text = prompts[prompt_len][prompt_id]
                    for rep in range(1, protocol["reps_per_prompt"] + 1):
                        run_single_measurement(
                            config, prompt_id, prompt_len, prompt_text,
                            rep, args.network, writer
                        )
                        csvfile.flush()

    print(f"\n{'='*60}")
    print(f"  DONE — Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
