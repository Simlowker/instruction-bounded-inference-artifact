#!/usr/bin/env bash
# run_bench.sh — minimal automation of the paper §4.1 binary-search trap protocol.
#
# Usage:
#   bash scripts/run_bench.sh gian      # expects to find max_tokens=29 OK, max_tokens=30 TRAP
#   bash scripts/run_bench.sh onicai    # expects to find max_tokens=10 OK, max_tokens=11 TRAP
#
# Prerequisites:
# - dfx local replica running with canister 'llama_cpp' deployed and Qwen 2.5 0.5B Q8_0 uploaded
#   as 'models/model.gguf'. See REPRODUCE.md sections 3–5.

set -euo pipefail

FORK="${1:-gian}"

if [ "$FORK" = "gian" ]; then
  OK_MAX=29
  TRAP_MAX=30
  N_PREFILL=2
elif [ "$FORK" = "onicai" ]; then
  OK_MAX=10
  TRAP_MAX=11
  N_PREFILL=3
else
  echo "Unknown fork: $FORK (expected 'gian' or 'onicai')" >&2
  exit 1
fi

PROMPT='<|im_start|>user\nAnswer the following question as brief as possible. This is the question: What are the key differences between proof-of-work and proof-of-stake consensus mechanisms?<|im_end|>\n<|im_start|>assistant\n'

echo "=== Bench: $FORK (OK_MAX=$OK_MAX, TRAP_MAX=$TRAP_MAX, N_PREFILL=$N_PREFILL) ==="

echo "--- Setting max_tokens_update = $OK_MAX ---"
dfx canister call llama_cpp set_max_tokens \
  "(record { max_tokens_query = 1 : nat64; max_tokens_update = $OK_MAX : nat64 })" | tail -3

echo "--- load_model ---"
dfx canister call llama_cpp load_model \
  '(record { args = vec {"--model"; "models/model.gguf"} })' | tail -3

echo "--- new_chat ---"
dfx canister call llama_cpp new_chat \
  '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })' | tail -3

echo "--- Prefill x $N_PREFILL ---"
for i in $(seq 1 $N_PREFILL); do
  echo ">>> Prefill call $i"
  dfx canister call llama_cpp run_update \
    "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" | tail -12
done

echo "--- Pure gen @ $OK_MAX (DOIT OK) ---"
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" | tail -12

echo "--- Pure gen @ $TRAP_MAX (DOIT TRAP) ---"
dfx canister call llama_cpp set_max_tokens \
  "(record { max_tokens_query = 1 : nat64; max_tokens_update = $TRAP_MAX : nat64 })" | tail -3
dfx canister call llama_cpp run_update \
  "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" 2>&1 | tail -5 || true

echo "=== Done. Expected: $OK_MAX OK, $TRAP_MAX TRAP (IC0522). ==="
