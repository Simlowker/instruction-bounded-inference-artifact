#!/usr/bin/env bash
# Repeated-run benchmark protocol for instruction-bounded inference paper.
#
# This script measures tok/call for each model configuration N times
# via binary search (gen@K OK, gen@K+1 TRAP = K tok/call).
#
# Prerequisites:
#   - dfx CLI installed (tested with dfx 0.31.0)
#   - Local ICP replica running (dfx start --background)
#   - llama_cpp_canister deployed with target model
#   - Python 3.11+ with pandas, numpy, scipy
#
# Usage:
#   ./benchmark_repeated_runs.sh <model_tag> <n_runs> [network]
#   ./benchmark_repeated_runs.sh smollm2-135m-q4 20 local
#   ./benchmark_repeated_runs.sh qwen-0.5b-q8 10 ic
#
# Output:
#   artifact/results/raw/<model_tag>_runs.csv

set -euo pipefail

MODEL_TAG="${1:?Usage: $0 <model_tag> <n_runs> [network]}"
N_RUNS="${2:?Usage: $0 <model_tag> <n_runs> [network]}"
NETWORK="${3:-local}"

CANISTER="llama_cpp_canister"
OUTDIR="artifact/results/raw"
OUTFILE="${OUTDIR}/${MODEL_TAG}_runs.csv"

# Prompts for variance testing (different prompts → different token sequences)
PROMPTS=(
    "The meaning of life is"
    "In the beginning there was"
    "Once upon a time in a"
    "The future of artificial intelligence"
    "explain quantum computing in simple"
)

mkdir -p "$OUTDIR"

echo "model_tag,network,run_id,prompt_id,prompt,max_tokens_ok,max_tokens_trap,tok_per_call,timestamp,dfx_version" > "$OUTFILE"

DFX_VERSION=$(dfx --version 2>/dev/null || echo "unknown")

echo "=== Repeated Run Benchmark ==="
echo "  Model: $MODEL_TAG"
echo "  Runs: $N_RUNS"
echo "  Network: $NETWORK"
echo "  Prompts: ${#PROMPTS[@]}"
echo "  dfx: $DFX_VERSION"
echo "  Output: $OUTFILE"
echo ""

for run in $(seq 1 "$N_RUNS"); do
    # Cycle through prompts
    prompt_idx=$(( (run - 1) % ${#PROMPTS[@]} ))
    prompt="${PROMPTS[$prompt_idx]}"

    echo "--- Run $run/$N_RUNS (prompt $prompt_idx: '${prompt:0:30}...') ---"

    # Binary search for max tokens
    lo=1
    hi=200  # Adjust upper bound based on model
    best_ok=0

    while [ $lo -le $hi ]; do
        mid=$(( (lo + hi) / 2 ))

        # Call canister
        result=$(dfx canister call --network "$NETWORK" "$CANISTER" inference \
            "(record { prompt = \"$prompt\"; max_tokens = $mid })" 2>&1 || true)

        if echo "$result" | grep -q "exceeded\|trap\|error\|Error"; then
            hi=$(( mid - 1 ))
        else
            best_ok=$mid
            lo=$(( mid + 1 ))
        fi
    done

    trap_at=$(( best_ok + 1 ))
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "  → tok/call = $best_ok (gen@$best_ok OK, gen@$trap_at TRAP)"

    echo "$MODEL_TAG,$NETWORK,$run,$prompt_idx,\"$prompt\",$best_ok,$trap_at,$best_ok,$ts,$DFX_VERSION" >> "$OUTFILE"
done

echo ""
echo "=== Done: $N_RUNS runs saved to $OUTFILE ==="
echo ""

# Quick stats
python3 -c "
import pandas as pd, numpy as np
from scipy.stats import bootstrap

df = pd.read_csv('$OUTFILE')
x = df['tok_per_call'].values.astype(float)
print(f'  n = {len(x)}')
print(f'  mean = {x.mean():.2f}')
print(f'  std = {x.std(ddof=1):.2f}')
print(f'  median = {np.median(x):.1f}')
print(f'  min/max = {x.min():.0f}/{x.max():.0f}')

if len(x) >= 3:
    ci = bootstrap((x,), np.mean, method='BCa', n_resamples=9999, random_state=42).confidence_interval
    print(f'  95% CI (BCa): [{ci.low:.2f}, {ci.high:.2f}]')
"
