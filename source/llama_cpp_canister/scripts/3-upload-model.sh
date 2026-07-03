#!/bin/bash

LLAMA_CPP_CANISTER_PATH="./"
export PYTHONPATH="${PYTHONPATH}:$(realpath $LLAMA_CPP_CANISTER_PATH)"

#######################################################################
# run from parent folder as:
# scripts/upload-model.sh --network [local|ic]
#######################################################################

# Default network type is local
NETWORK_TYPE="local"

# The gguf model file to upload (Relative to llama_cpp_canister folder)
# MODEL="models/stories260Ktok512.gguf"
# MODEL="models/stories15Mtok4096.gguf"
# MODEL="models/tensorblock/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
# MODEL="models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"
# MODEL="models/trilm/TriLM_560M_Unpacked.TQ2_0.gguf"
# MODEL="models/trilm/TriLM_3.9B_Unpacked.TQ2_0.gguf"
# MODEL="models/Qwen2.5-1.5B/qwen2.5-1.5b-instruct-q6_k.gguf"  # 1.46 GB → IC0524 stable-mem page limit
MODEL="models/Qwen2.5-0.5B-Q6_K/qwen2.5-0.5b-instruct-q6_k.gguf"
# MODEL="models/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf"
# MODEL="models/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q2_K.gguf"

# Parse command line arguments for network type
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --network [local|ic]"
            exit 1
            ;;
    esac
done

echo "Using network type: $NETWORK_TYPE"

#######################################################################
echo " "
echo "==================================================="
echo "Uploading model to llama_cpp"

echo " "
echo "--------------------------------------------------"
echo "Checking health endpoint for llama_cpp"
output=$(dfx canister call llama_cpp health --network $NETWORK_TYPE )

if [ "$output" != "(variant { Ok = record { status_code = 200 : nat16 } })" ]; then
    echo "llama_cpp health check failed."
    echo $output
    exit 1
else
    echo "llama_cpp health check succeeded."
fi

echo " "
echo "--------------------------------------------------"
echo "Upload the model ($MODEL) to llama_cpp"
python3 -m scripts.upload --network $NETWORK_TYPE --canister llama_cpp --canister-filename models/model.gguf $MODEL

if [ $? -ne 0 ]; then
    echo "scripts.upload for llama_cpp exited with an error."
    echo $?
    exit 1
fi