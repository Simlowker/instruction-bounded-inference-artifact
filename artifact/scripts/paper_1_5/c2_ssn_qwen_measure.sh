#!/bin/bash
# Task 12 SSN — Qwen 0.5B BASE-Q8_0 measurement protocol
set -euo pipefail
export DFX_WARNING=-mainnet_plaintext_identity

LOG="artifact/results/paper_1_5/raw/c2-ssn-qwen05-q8.log"
PROMPT="The capital of France is"

echo "=== SSN mainnet validation — Qwen 2.5 0.5B BASE-Q8_0 ===" | tee "${LOG}"
echo "Date: $(date -u +%FT%TZ)" | tee -a "${LOG}"
echo "Canister: u2pva-3iaaa-aaaax-qaa7a-cai" | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "--- load_model ---" | tee -a "${LOG}"
dfx canister call --network ic llama_cpp load_model '(record { args = vec {"--model"; "models/model.gguf"; "--no-warmup"} })' 2>&1 | tail -10 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "--- warmup ---" | tee -a "${LOG}"
dfx canister call --network ic llama_cpp new_chat '(record { args = vec {"--prompt-cache"; "ssn-qwen-warm.cache"; "--model"; "models/model.gguf"} })' 2>&1 | tail -3 | tee -a "${LOG}"
dfx canister call --network ic llama_cpp run_update '(record { args = vec {"--model"; "models/model.gguf"; "--prompt-cache"; "ssn-qwen-warm.cache"; "--prompt-cache-all"; "-sp"; "-p"; "Hi"; "-n"; "3"} })' 2>&1 | tail -5 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "--- binsearch ---" | tee -a "${LOG}"
# Try N=30 first (local N_MAX), then bracket
for N in 30 28 26 32; do
  echo ">>> N=${N}" | tee -a "${LOG}"
  dfx canister call --network ic llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"ssn-qwen-bin${N}.cache\"; \"--model\"; \"models/model.gguf\"} })" 2>&1 | tail -1 >> "${LOG}"
  t0=$(python3 -c "import time;print(time.time())")
  set +e
  OUT=$(dfx canister call --network ic llama_cpp run_update "(record { args = vec {\"--model\"; \"models/model.gguf\"; \"--prompt-cache\"; \"ssn-qwen-bin${N}.cache\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${PROMPT}\"; \"-n\"; \"${N}\"} })" 2>&1)
  rc=$?
  set -e
  t1=$(python3 -c "import time;print(time.time())")
  dt=$(python3 -c "print(round(${t1}-${t0},2))")
  if [ $rc -eq 0 ]; then
    echo "N=${N} OK wall=${dt}s" | tee -a "${LOG}"
    echo "${OUT}" | grep -E "status_code|output = " | head -2 | tee -a "${LOG}"
  else
    echo "N=${N} FAIL wall=${dt}s" | tee -a "${LOG}"
    echo "${OUT}" | grep -E "Canister trapped|IC05" | head -1 | tee -a "${LOG}"
  fi
done

echo "" | tee -a "${LOG}"
echo "--- done, review log and run 3-rep at N_MAX ---" | tee -a "${LOG}"
