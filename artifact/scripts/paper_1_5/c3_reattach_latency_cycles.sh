#!/usr/bin/env bash
# Paper 1.5 C3 — Task 15: stop/start cycles only (prefill assumed done).
# Uses existing c3t15.cache (~112 tok, 1.38MB from interrupted prefill).
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CACHE="c3t15.cache"
CSV="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-reattach-qwen05.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

call_run_update() {
  local n="$1"
  local prompt="$2"
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${prompt}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

append_csv_row() {
  local sid="$1"
  local n_calls="$2"
  local tokens_per_call="$3"
  local total_tokens="$4"
  local wall_s="$5"
  local reattach_ms="$6"
  local notes="$7"
  printf "%s,Qwen2.5-0.5B,f16,%s,%s,%s,%s,,,,%s,,%s,%s\n" \
    "${sid}" "${n_calls}" "${tokens_per_call}" "${total_tokens}" "${wall_s}" \
    "${reattach_ms}" "$(now_iso)" "${notes}" >> "${CSV}"
}

echo "" | tee -a "${LOG}"
echo "=== Task 15 cycles-only start $(now_iso) === cache_tokens≈112" | tee -a "${LOG}"

for rep in 1 2 3 4 5; do
  echo "" | tee -a "${LOG}"
  echo "=== REP ${rep} ===" | tee -a "${LOG}"

  echo "--- stopping canister ---" | tee -a "${LOG}"
  dfx canister stop llama_cpp --network "${NETWORK}" >> "${LOG}" 2>&1 || true

  echo "--- starting canister ---" | tee -a "${LOG}"
  dfx canister start llama_cpp --network "${NETWORK}" >> "${LOG}" 2>&1 || true

  # Note: not re-loading model explicitly; empirical finding is that
  # load_model trapped after stop/start in dev sandbox, yet subsequent
  # run_update works (appears to lazy-reload from stable mem).
  # --- COLD CALL (1st run_update post-restart = re-attach) ---
  echo "--- COLD run_update rep ${rep} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update 3 "" >> "${LOG}" 2>&1
  t1=$(now_ns)
  cold_ms=$(python3 -c "print(round((${t1}-${t0})/1e6, 1))")
  cold_s=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "cold wall_ms=${cold_ms}" | tee -a "${LOG}"

  append_csv_row "c3-reattach-qwen05-rep${rep}-cold" 1 3 3 "${cold_s}" "${cold_ms}" "first-run_update-after-stop-start;cache_tokens=112;includes-load_model-+-state-load"

  # --- WARM CALL ---
  echo "--- WARM run_update rep ${rep} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update 3 "" >> "${LOG}" 2>&1
  t1=$(now_ns)
  warm_ms=$(python3 -c "print(round((${t1}-${t0})/1e6, 1))")
  warm_s=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "warm wall_ms=${warm_ms}" | tee -a "${LOG}"

  append_csv_row "c3-reattach-qwen05-rep${rep}-warm" 1 3 3 "${warm_s}" "${warm_ms}" "second-run_update-after-restart;reference-warm-no-reattach;cache_tokens=115"

done

echo "" | tee -a "${LOG}"
echo "=== Task 15 cycles-only complete $(now_iso) ===" | tee -a "${LOG}"
