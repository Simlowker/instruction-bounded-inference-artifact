#!/usr/bin/env bash
# Paper 1.5 C3 — Task 15: re-attach latency (cold-call after canister stop/start)
#
# Protocol:
#   1. Pre-load prompt.cache with ~200 tok via multi-call prefill (done once).
#   2. For rep in 1..5:
#        dfx canister stop llama_cpp
#        dfx canister start llama_cpp
#        time 1st run_update  (cold re-attach: WASM page faults + state load)
#        time 2nd run_update  (warm)
#   3. Append 10 rows to multicall_characterization.csv.
#
# Cache file: c3t15.cache (Qwen 2.5 0.5B Q4_0).
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CACHE="c3t15.cache"
CSV="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-reattach-qwen05.log"

mkdir -p "${LOG_DIR}"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

call_run_update() {
  local n="$1"
  local prompt="$2"
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${prompt}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

append_csv_row() {
  # session_id,model_id,kv_cache_type,n_calls,tokens_per_call,total_tokens,wall_clock_s,total_cycles,io_overhead_write_inst,io_overhead_read_inst,reattach_latency_ms,bit_exact_vs_singlecall,timestamp_utc,notes
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

echo "=== Task 15 start $(now_iso) ===" | tee -a "${LOG}"

# Step 1: prefill ~200 tokens via multi-call using existing c3t15.cache
# Warmup already done (new_chat + N=1,2,3). Continue to ~200 tok total.
echo ">>> Prefill phase: building ~200 tok prompt.cache" | tee -a "${LOG}"

PROMPT_INIT="<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\n"

# new_chat to clear prior state, then chain N=15 × 14 = ~196 tok
dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1

echo ">>> prefill step 1 (with short prompt)" | tee -a "${LOG}"
call_run_update 15 "${PROMPT_INIT}" >> "${LOG}" 2>&1

for step in 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  echo ">>> prefill step ${step} (continuation)" | tee -a "${LOG}"
  call_run_update 15 "" >> "${LOG}" 2>&1
done

echo ">>> Prefill complete. Starting 5 stop/start cycles." | tee -a "${LOG}"

# Step 2: 5 reps of stop/start/cold/warm
for rep in 1 2 3 4 5; do
  echo "" | tee -a "${LOG}"
  echo "=== REP ${rep} ===" | tee -a "${LOG}"

  echo "--- stopping canister ---" | tee -a "${LOG}"
  dfx canister stop llama_cpp --network "${NETWORK}" >> "${LOG}" 2>&1 || true

  echo "--- starting canister ---" | tee -a "${LOG}"
  dfx canister start llama_cpp --network "${NETWORK}" >> "${LOG}" 2>&1 || true

  # --- COLD CALL (first run_update after restart) ---
  echo "--- COLD run_update rep ${rep} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update 3 "" >> "${LOG}" 2>&1
  t1=$(now_ns)
  cold_ms=$(python3 -c "print(round((${t1}-${t0})/1e6, 1))")
  cold_s=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "cold wall_ms=${cold_ms}" | tee -a "${LOG}"

  append_csv_row "c3-reattach-qwen05-rep${rep}-cold" 1 3 3 "${cold_s}" "${cold_ms}" "first-run_update-after-stop-start-restart;includes-wasm-page-faults-and-state-reload"

  # --- WARM CALL (second run_update) ---
  echo "--- WARM run_update rep ${rep} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update 3 "" >> "${LOG}" 2>&1
  t1=$(now_ns)
  warm_ms=$(python3 -c "print(round((${t1}-${t0})/1e6, 1))")
  warm_s=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "warm wall_ms=${warm_ms}" | tee -a "${LOG}"

  append_csv_row "c3-reattach-qwen05-rep${rep}-warm" 1 3 3 "${warm_s}" "${warm_ms}" "second-run_update-after-restart;reference-warm-no-reattach"

done

echo "" | tee -a "${LOG}"
echo "=== Task 15 complete $(now_iso) ===" | tee -a "${LOG}"
