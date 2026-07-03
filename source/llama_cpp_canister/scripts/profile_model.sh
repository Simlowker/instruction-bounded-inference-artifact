#!/bin/bash
# Profile a model on the llama_cpp canister
# Usage: ./scripts/profile_model.sh <model_path> <model_name> <params_M>

MODEL_PATH="$1"
MODEL_NAME="$2"
PARAMS_M="$3"

echo "=== Profiling $MODEL_NAME ($PARAMS_M M params) ==="

# Upload
echo "Uploading $MODEL_PATH..."
python3 -m scripts.upload --network local --canister llama_cpp --canister-filename models/model.gguf --filetype gguf "$MODEL_PATH" 2>&1 | tail -3

# Load
echo "Loading model..."
dfx canister call llama_cpp load_model '(record { args = vec {"--model"; "models/model.gguf";} })' 2>&1 | grep -E "Ok|Err" | head -1

# Set max tokens
dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 10 : nat64 })' 2>&1 > /dev/null

# Init chat
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 | grep -E "Ok|Err" | head -1

# Enable profiling and run first call (includes prompt processing)
dfx canister call llama_cpp profile_enable 2>&1 > /dev/null

echo "Running inference (call 1 - prompt + decode)..."
dfx canister call llama_cpp run_update '(record { args = vec {"--prompt-cache"; "prompt.cache"; "--prompt-cache-all"; "-n"; "512"; "-p"; "The meaning of life is"} })' 2>&1 | grep "output" | head -1

# Get results for call 1
echo ""
echo "--- Call 1 (prompt + decode) ---"
dfx canister call llama_cpp profile_results 2>&1 | grep -v WARNING | grep -v metadata | grep -v '"name"' | sed 's/\\n/\n/g' | grep -E "Total:|MUL_MAT|C_matmul|C_non"

# Reset and run call 2 (pure decode)
dfx canister call llama_cpp profile_reset 2>&1 > /dev/null

echo ""
echo "Running inference (call 2 - pure decode)..."
dfx canister call llama_cpp run_update '(record { args = vec {"--prompt-cache"; "prompt.cache"; "--prompt-cache-all"; "-n"; "512"; "-p"; "The meaning of life is"} })' 2>&1 | grep "output" | head -1

echo ""
echo "--- Call 2 (pure decode) ---"
PROFILE=$(dfx canister call llama_cpp profile_results 2>&1 | grep -v WARNING | grep -v metadata | grep -v '"name"' | sed 's/\\n/\n/g')
echo "$PROFILE" | grep -E "Total:|MUL_MAT|C_matmul|C_non"

# Extract key numbers
TOTAL=$(echo "$PROFILE" | grep "Total:" | grep -o '[0-9]*')
MATMUL=$(echo "$PROFILE" | grep "C_matmul" | grep -o '[0-9]*' | head -1)
echo ""
echo "=== Summary for $MODEL_NAME ==="
echo "Params: ${PARAMS_M}M"
echo "C_total: $TOTAL"
echo "C_matmul: $MATMUL"
if [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    PCT=$(python3 -c "print(f'{$MATMUL/$TOTAL*100:.1f}%')" 2>/dev/null)
    ALPHA=$(python3 -c "print(f'{$MATMUL/($PARAMS_M*2*1000000*10):.2f}')" 2>/dev/null)
    echo "Matmul %: $PCT"
    echo "α_Q8_réel (approx): $ALPHA"
fi
echo ""
