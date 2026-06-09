#!/usr/bin/env python3
"""P0-3: Generate one sample per model and classify coherence.

Writes data/verified/coherence.csv with:
  model, prompt, output_text, word_count, classification_auto, classification_manual

Auto-classification:
- degenerate: >50% tokens are repeats of a short pattern (<=10 chars),
              or output is empty
- degraded:   heavy repetition OR non-English tokens mixed
- coherent:   low repetition

Usage:
  python3 run_coherence.py --model-tag SmolLM2-135M --quant Q4_0 --simd yes \
     --prompt "The capital of France is"
"""
import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from collections import Counter

DFX_CWD = Path("llama_cpp_canister")
OUT = Path(__file__).resolve().parents[1] / "data" / "verified" / "coherence.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def dfx_call(method, arg, timeout=120):
    cmd = ["dfx", "canister", "call", "--network", "local", "llama_cpp", method, arg]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(DFX_CWD))
    return r.stdout + r.stderr


def set_max_tokens(n):
    dfx_call("set_max_tokens", f"(record {{ max_tokens_query = 1 : nat64; max_tokens_update = {n} : nat64 }})")


def reset_chat():
    dfx_call("remove_prompt_cache",
             '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })')
    dfx_call("new_chat",
             '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })')


def generate(prompt, max_tokens=100):
    set_max_tokens(max_tokens)
    reset_chat()
    escaped = prompt.replace('"', '\\"')
    arg = (
        f'(record {{ args = vec {{"--model"; "models/model.gguf"; '
        f'"--prompt-cache"; "prompt.cache"; "--prompt-cache-all"; '
        f'"-sp"; "-n"; "512"; "-p"; "{escaped}"}} }})'
    )
    out = dfx_call("run_update", arg, timeout=180)
    m = re.search(r'output = "(.*?)";', out, re.DOTALL)
    return m.group(1) if m else ""


def classify(text: str) -> dict:
    words = text.split()
    n = len(words)
    # empty
    if n < 3:
        return {"label": "degenerate", "reason": "empty/too_short", "n_words": n}
    # word repetition
    counts = Counter(words)
    most = counts.most_common(1)[0]
    top_word_freq = most[1] / n
    # bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(n - 1)]
    bigram_counts = Counter(bigrams)
    top_bigram_freq = (bigram_counts.most_common(1)[0][1] / len(bigrams)) if bigrams else 0
    # 3-gram repetition
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(n - 2)]
    tri_counts = Counter(trigrams)
    top_tri_freq = (tri_counts.most_common(1)[0][1] / len(trigrams)) if trigrams else 0

    label = "coherent"
    reason_parts = []
    if top_tri_freq > 0.15 or top_bigram_freq > 0.25:
        label = "degenerate"
        reason_parts.append(f"heavy_ngram_loop(bi={top_bigram_freq:.2f},tri={top_tri_freq:.2f})")
    elif top_word_freq > 0.2 or top_bigram_freq > 0.12:
        label = "degraded"
        reason_parts.append(f"mild_repetition(word={top_word_freq:.2f},bi={top_bigram_freq:.2f})")

    return {
        "label": label,
        "reason": "; ".join(reason_parts) or "ok",
        "n_words": n,
        "top_word_freq": round(top_word_freq, 3),
        "top_bigram_freq": round(top_bigram_freq, 3),
        "top_trigram_freq": round(top_tri_freq, 3),
        "most_common_word": most[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--simd", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=100)
    args = ap.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating {args.model_tag} {args.quant}/{args.simd}...")
    text = generate(args.prompt, args.max_tokens)
    if not text:
        print("  ERROR: no output")
        return
    clf = classify(text)
    print(f"  [{clf['label']}] {clf['reason']}")
    print(f"  Output: {text[:200]}...")

    write_header = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "timestamp", "model_tag", "quant", "simd", "prompt", "max_tokens",
                "output", "label", "reason", "n_words",
                "top_word_freq", "top_bigram_freq", "top_trigram_freq", "most_common_word",
            ])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            args.model_tag, args.quant, args.simd,
            args.prompt, args.max_tokens, text,
            clf["label"], clf["reason"], clf["n_words"],
            clf["top_word_freq"], clf["top_bigram_freq"], clf["top_trigram_freq"],
            clf["most_common_word"],
        ])


if __name__ == "__main__":
    main()
