#!/usr/bin/env bash
# Paper 1.5 C3 — Task 17 v2: simplified cache-type-k arms with 90s timeout.
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CSV="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-kvtype-qwen05.log"
QUALITY_LOG="${LOG_DIR}/c3-kvtype-qwen05-decode-quality.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

append_csv_row() {
  local sid="$1" kv="$2" n_calls="$3" tok_per_call="$4" total_tokens="$5" wall_s="$6" notes="$7"
  printf "%s,Qwen2.5-0.5B,%s,%s,%s,%s,%s,,,,,,%s,%s\n" \
    "${sid}" "${kv}" "${n_calls}" "${tok_per_call}" "${total_tokens}" "${wall_s}" \
    "$(now_iso)" "${notes}" >> "${CSV}"
}

PROMPT='<|im_start|>user\nHi cat.<|im_end|>\n<|im_start|>assistant\n'

echo "" | tee -a "${LOG}"
echo "=== Task 17 v2 start $(now_iso) ===" | tee -a "${LOG}"

run_probe() {
  local cache_type="$1" cache_file="$2" n="$3"
  # Build flag cleanly
  if [ "${cache_type}" = "f16" ]; then
    dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--prompt-cache\"; \"${cache_file}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${PROMPT}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
  else
    dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--cache-type-k\"; \"${cache_type}\"; \"--prompt-cache\"; \"${cache_file}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${PROMPT}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
  fi
}

run_arm() {
  local cache_type="$1" sid="$2"
  local cache_file="c3t17v2_${cache_type}.cache"

  echo "" | tee -a "${LOG}"
  echo ">>> ARM ${cache_type}" | tee -a "${LOG}"

  local last_ok_n=0 last_ok_wall_s="" last_ok_output=""

  for probe in 10 15 20; do
    dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${cache_file}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1 || true
    echo "--- ${cache_type} probe N=${probe} ---" | tee -a "${LOG}"
    t0=$(now_ns)
    set +e
    timeout 90 bash -c "$(declare -f run_probe); PROMPT='${PROMPT}'; MODEL='${MODEL}'; NETWORK='${NETWORK}'; run_probe '${cache_type}' '${cache_file}' '${probe}'" > /tmp/c3t17v2_${cache_type}_${probe}.out 2>&1
    rc=$?
    set -e
    t1=$(now_ns)
    dt_s=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
    if [ "${rc}" -eq 0 ] && ! grep -qE "Failed update call|IC0522|trapped|^Error:" /tmp/c3t17v2_${cache_type}_${probe}.out; then
      echo "    probe N=${probe} ${cache_type} OK wall=${dt_s}s" | tee -a "${LOG}"
      cat /tmp/c3t17v2_${cache_type}_${probe}.out >> "${LOG}"
      last_ok_n=${probe}
      last_ok_wall_s=${dt_s}
      last_ok_output=$(python3 -c "import re; t=open('/tmp/c3t17v2_${cache_type}_${probe}.out').read(); m=re.search(r'output = \"(.*?)\";', t, re.DOTALL); print(m.group(1) if m else '')")
    else
      echo "    probe N=${probe} ${cache_type} FAIL (rc=${rc} wall=${dt_s}s)" | tee -a "${LOG}"
      cat /tmp/c3t17v2_${cache_type}_${probe}.out >> "${LOG}" 2>/dev/null || true
      break
    fi
  done

  if [ "${last_ok_n}" -eq 0 ]; then
    append_csv_row "${sid}" "${cache_type}" 0 0 0 "" "N_MAX=0;all-probes-failed-or-timed-out"
  else
    tokps=$(python3 -c "print(round(${last_ok_n}/(${last_ok_wall_s}), 3))")
    append_csv_row "${sid}" "${cache_type}" 1 "${last_ok_n}" "${last_ok_n}" "${last_ok_wall_s}" "N_MAX=${last_ok_n};tok_per_s=${tokps};prompt=hi-cat;probes=10,15,20;90s-timeout"
    echo "" >> "${QUALITY_LOG}"
    echo "=== ${cache_type}: N_MAX=${last_ok_n}, output ===" >> "${QUALITY_LOG}"
    echo "${last_ok_output}" >> "${QUALITY_LOG}"
  fi
}

: > "${QUALITY_LOG}"
echo "Task 17 - KV cache type trade-off decode quality log" >> "${QUALITY_LOG}"
echo "Qwen 2.5 0.5B Q4_0 on local dfx, patched C3-IO wasm." >> "${QUALITY_LOG}"
echo "Prompt: \"Tell me a story about a cat.\"" >> "${QUALITY_LOG}"
echo "Started: $(now_iso)" >> "${QUALITY_LOG}"

run_arm "f16"  "c3-kvtype-qwen05-f16"
run_arm "q8_0" "c3-kvtype-qwen05-q8"
run_arm "q4_0" "c3-kvtype-qwen05-q4"

echo "" | tee -a "${LOG}"
echo "=== Task 17 v2 complete $(now_iso) ===" | tee -a "${LOG}"
