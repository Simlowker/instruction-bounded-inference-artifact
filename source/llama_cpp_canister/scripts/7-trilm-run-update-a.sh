#!/bin/bash

#######################################################################
# run from parent folder as:
# scripts/7-trilm-run-update-a.sh --network [local|ic] [-n N_TOKENS]
#######################################################################

# Default network type is local
NETWORK_TYPE="local"

# Default token budget (binary-search target — Task 15 sweeps this)
N_TOKENS="512"

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --network)
            shift
            if [ "$1" = "local" ] || [ "$1" = "ic" ]; then
                NETWORK_TYPE=$1
            else
                echo "Invalid network type: $1. Use 'local' or 'ic'."
                exit 1
            fi
            shift
            ;;
        -n)
            shift
            N_TOKENS=$1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --network [local|ic] [-n N_TOKENS]"
            exit 1
            ;;
    esac
done

echo "Using network type: $NETWORK_TYPE"
echo "Token budget: $N_TOKENS"

# TriLM 560M is a base (non-instruction) model — use plain prompt continuation,
# no chat template tags. Prompt chosen for deterministic auditability:
# "The capital of France is" should yield "Paris" (or geographic continuation).
PROMPT="The capital of France is"

echo " "
echo "--------------------------------------------------"
echo "Calling run_update for llama_cpp (TriLM 560M)"
dfx canister call llama_cpp run_update --network "$NETWORK_TYPE" \
    "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"$PROMPT\"; \"-n\"; \"$N_TOKENS\" } })"
