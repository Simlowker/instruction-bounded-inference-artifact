#!/bin/bash
# Coherence audit: 10 diverse prompts on TriLM 560M TQ2_0
# Produces Markdown report listing prompt + model completion for human validation.
# Run from the llama_cpp_canister/ directory.
#
# Usage: bash scripts/trilm-coherence-audit.sh [--network local|ic] > audit.md

NETWORK_TYPE="local"
while [ $# -gt 0 ]; do
    case "$1" in
        --network) shift; NETWORK_TYPE="$1"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Short, factually-checkable prompts spanning geography, history, science,
# literature, and procedures. TriLM 560M is a base model (no chat template),
# so plain-text continuations are appropriate.
PROMPTS=(
    "The capital of France is"
    "Water boils at a temperature of"
    "The largest planet in our solar system is"
    "Shakespeare wrote a play called"
    "The theory of relativity was proposed by"
    "DNA stands for"
    "The Great Wall of China was built"
    "To bake bread, you need flour, water, yeast, and"
    "The Pacific Ocean is the"
    "Photosynthesis is the process by which plants"
)

echo "# TriLM 560M TQ2_0 — Coherence Audit"
echo
echo "Environment: ${NETWORK_TYPE}"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Model: models/trilm/TriLM_560M_Unpacked.TQ2_0.gguf"
echo "Decoding: default sampling (deterministic), -n 24"
echo
echo "| # | Prompt | Completion |"
echo "|---|--------|------------|"

for i in "${!PROMPTS[@]}"; do
    n=$((i + 1))
    PROMPT="${PROMPTS[$i]}"

    # Fresh chat for each prompt to avoid cache bleed
    dfx canister call llama_cpp new_chat \
        '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "prompt.cache"} })' \
        --network "$NETWORK_TYPE" > /dev/null 2>&1

    OUT=$(dfx canister call llama_cpp run_update --network "$NETWORK_TYPE" \
        "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"$PROMPT\"; \"-n\"; \"24\"} })" 2>&1)

    COMPLETION=$(echo "$OUT" | awk '/output = /{ sub(/.*output = "/, ""); sub(/";$/, ""); print; exit }')
    # Strip leading/trailing quotes and escape pipe chars for markdown
    COMPLETION=$(echo "$COMPLETION" | tr '|' '/' | tr -d '\n')

    # Escape any markdown-breaking characters in prompt column
    P_ESC=$(echo "$PROMPT" | tr '|' '/')

    echo "| $n | $P_ESC | $COMPLETION |"
done
