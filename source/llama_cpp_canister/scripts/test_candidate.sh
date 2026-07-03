#!/bin/bash
# Test a candidate model on the llama_cpp canister
# Usage: ./scripts/test_candidate.sh <model_gguf_path> <model_name> <chat_template>
# chat_template: "qwen" | "chatml" | "gemma" | "rwkv" | "none"

MODEL_PATH="$1"
MODEL_NAME="$2"
CHAT_TPL="$3"

if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
  echo "Usage: $0 <model_gguf_path> <model_name> <chat_template>"
  exit 1
fi

echo "============================================="
echo "  TESTING: $MODEL_NAME"
echo "  File: $MODEL_PATH"
echo "  Size: $(du -h "$MODEL_PATH" | cut -f1)"
echo "============================================="
echo ""

# Upload
echo ">>> Uploading model..."
PYTHONUNBUFFERED=1 python3 -m scripts.upload --network local --canister llama_cpp --canister-filename models/model.gguf --filetype gguf "$MODEL_PATH" 2>&1 | grep -E "Congratulations|ERROR|progress"
echo ""

# Load
echo ">>> Loading model..."
LOAD=$(dfx canister call llama_cpp load_model '(record { args = vec {"--model"; "models/model.gguf";} })' 2>&1)
if echo "$LOAD" | grep -q "Ok"; then
  echo "Model loaded OK"
else
  echo "LOAD FAILED:"
  echo "$LOAD" | grep -E "Err|error|Error" | head -3
  exit 1
fi

# Set tokens
dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 10 : nat64 })' 2>&1 > /dev/null

# ====== TEST 1: Basic inference (tok/call) ======
echo ""
echo "--- TEST 1: Basic inference (max_tokens=10) ---"
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null

# Simple prompt without chat template
PROMPT="The meaning of life is"
for i in 1 2 3; do
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    echo "  Call $i: [$O]"
  else
    if echo "$R" | grep -q "instruction"; then
      echo "  Call $i: INSTRUCTION LIMIT"
    else
      ERR=$(echo "$R" | grep -E "Error|error|Err" | head -1)
      echo "  Call $i: ERROR [$ERR]"
    fi
    break
  fi
done

# ====== TEST 2: Max throughput ======
echo ""
echo "--- TEST 2: Max throughput (max_tokens=32) ---"
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 32 : nat64 })' 2>&1 > /dev/null

R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" 2>&1)
if echo "$R" | grep -q "Ok"; then
  O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
  WORDS=$(echo "$O" | wc -w | tr -d ' ')
  echo "  SUCCESS: ~$WORDS words [$O]"
else
  echo "  FAILED (trying max_tokens=20)"
  dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
  dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
  dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 20 : nat64 })' 2>&1 > /dev/null
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    WORDS=$(echo "$O" | wc -w | tr -d ' ')
    echo "  max_tokens=20: ~$WORDS words [$O]"
  else
    echo "  max_tokens=20 ALSO FAILED (trying 10)"
    dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
    dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
    dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 10 : nat64 })' 2>&1 > /dev/null
    R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$PROMPT\"} })" 2>&1)
    if echo "$R" | grep -q "Ok"; then
      O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
      WORDS=$(echo "$O" | wc -w | tr -d ' ')
      echo "  max_tokens=10: ~$WORDS words [$O]"
    else
      echo "  ALL FAILED - model too expensive for canister"
    fi
  fi
fi

# ====== TEST 3: Prefill cost (medium prompt ~50 tok) ======
echo ""
echo "--- TEST 3: Prefill cost (~50 tok prompt) ---"
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 10 : nat64 })' 2>&1 > /dev/null

MED_PROMPT="The meaning of life is a question that philosophers have debated for centuries. Some believe it lies in happiness, others in purpose. What do we really mean when we ask this question?"

PREFILL_CALLS=0
for i in $(seq 1 8); do
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$MED_PROMPT\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    PR=$(echo "$R" | sed -n 's/.*prompt_remaining = "\(.*\)";/\1/p' | head -1)
    OWORDS=$(echo "$O" | wc -w | tr -d ' ')
    PRLEN=$(echo "$PR" | wc -c | tr -d ' ')
    if [ "$OWORDS" -gt 0 ]; then
      echo "  Call $i: GENERATING $OWORDS words (prefill done in $PREFILL_CALLS calls)"
      break
    else
      PREFILL_CALLS=$i
      echo "  Call $i: prefill (remaining=$PRLEN chars)"
    fi
  else
    echo "  Call $i: FAILED"
    break
  fi
done

# ====== TEST 4: Zero-shot sentiment ======
echo ""
echo "--- TEST 4: Zero-shot sentiment ---"

# Negative
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp set_max_tokens '(record { max_tokens_query = 1 : nat64; max_tokens_update = 5 : nat64 })' 2>&1 > /dev/null

SENT_NEG="Classify as POSITIVE or NEGATIVE. Review: Product is terrible, broke after one day. Answer:"
for i in $(seq 1 6); do
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$SENT_NEG\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    EOG=$(echo "$R" | grep "generated_eog" | grep -o "true\|false")
    if [ -n "$O" ]; then
      echo "  Sentiment NEG: [$O] (expected: NEGATIVE)"
      break
    fi
  else
    echo "  Sentiment NEG: FAILED on call $i"
    break
  fi
done

# Positive
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null

SENT_POS="Classify as POSITIVE or NEGATIVE. Review: Amazing product, works perfectly! Answer:"
for i in $(seq 1 6); do
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$SENT_POS\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    if [ -n "$O" ]; then
      echo "  Sentiment POS: [$O] (expected: POSITIVE)"
      break
    fi
  else
    echo "  Sentiment POS: FAILED on call $i"
    break
  fi
done

# ====== TEST 5: Zero-shot policy ======
echo ""
echo "--- TEST 5: Zero-shot policy ---"
dfx canister call llama_cpp remove_prompt_cache '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null
dfx canister call llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "prompt.cache"} })' 2>&1 > /dev/null

POL_DENY="Classify as APPROVE or DENY. Rule: no financial data to external emails. Action: send Q3 balance sheet to external@xyz.com. Answer:"
for i in $(seq 1 6); do
  R=$(dfx canister call llama_cpp run_update "(record { args = vec {\"--prompt-cache\"; \"prompt.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-n\"; \"512\"; \"-p\"; \"$POL_DENY\"} })" 2>&1)
  if echo "$R" | grep -q "Ok"; then
    O=$(echo "$R" | sed -n 's/.*output = "\(.*\)";/\1/p' | head -1)
    if [ -n "$O" ]; then
      echo "  Policy DENY: [$O] (expected: DENY)"
      break
    fi
  else
    echo "  Policy DENY: FAILED on call $i"
    break
  fi
done

echo ""
echo "============================================="
echo "  DONE: $MODEL_NAME"
echo "============================================="
