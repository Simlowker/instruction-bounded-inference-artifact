#!/usr/bin/env bash
# Paper 1.5 C3 — Task 17: cache-type-k F16 / Q8_0 / Q4_0 trade-off.
#
# Important caveat (discovered during Task 17):
#   load_model reliably traps with "heap out of bounds" when the canister's
#   context is already populated from a previous lazy-load via run_update.
#   So we drive each KV type arm via a fresh new_chat + run_update chain,
#   passing --cache-type-k to run_update. The arg parser DOES accept the
#   flag (verified by passing INVALID_XYZ -> "Unsupported cache type:
#   INVALID_XYZ" WASI exception). Whether the flag *takes effect* at
#   run_update time (vs being silently shadowed by the already-allocated
#   f16 KV cache from the initial lazy load) is checked via the C3-IO
#   save_terminal_bytes column: f16 baseline is ~12.3 KB/tok at Qwen 0.5B;
#   q8_0 should halve, q4_0 should quarter IF the flag is honored.
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CSV="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-kvtype-qwen05.log"
QUALITY_LOG="${LOG_DIR}/c3-kvtype-qwen05-decode-quality.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

call_run_update() {
  local cache_type="$1"
  local cache_file="$2"
  local n="$3"
  local prompt="$4"
  local extra=""
  if [ -n "${cache_type}" ] && [ "${cache_type}" != "f16" ]; then
    extra="\"--cache-type-k\"; \"${cache_type}\"; "
  fi
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; ${extra}\"--prompt-cache\"; \"${cache_file}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${prompt}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

append_csv_row() {
  local sid="$1"
  local kv="$2"
  local n_calls="$3"
  local tok_per_call="$4"
  local total_tokens="$5"
  local wall_s="$6"
  local notes="$7"
  printf "%s,Qwen2.5-0.5B,%s,%s,%s,%s,%s,,,,,,%s,%s\n" \
    "${sid}" "${kv}" "${n_calls}" "${tok_per_call}" "${total_tokens}" "${wall_s}" \
    "$(now_iso)" "${notes}" >> "${CSV}"
}

PROMPT_NARR="<|im_start|>user\nTell me a story about a cat.<|im_end|>\n<|im_start|>assistant\n"

echo "=== Task 17 start $(now_iso) ===" | tee -a "${LOG}"
: > "${QUALITY_LOG}"
echo "Task 17 - KV cache type trade-off decode quality log" >> "${QUALITY_LOG}"
echo "Qwen 2.5 0.5B Q4_0 on local dfx, patched C3-IO wasm." >> "${QUALITY_LOG}"
echo "Prompt: \"Tell me a story about a cat.\"" >> "${QUALITY_LOG}"
echo "" >> "${QUALITY_LOG}"

# Run a binsearch for N_MAX per arm. Seed probes: 15, 25, 40, 60.
# Qwen baseline at f16 is ~20 tok; q8 and q4 may yield higher.

probe_n_max() {
  local cache_type="$1"
  local cache_file="$2"
  local max_n=0
  local last_ok_n=0
  for probe in 15 25 40 60; do
    dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${cache_file}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1
    set +e
    call_run_update "${cache_type}" "${cache_file}" "${probe}" "${PROMPT_NARR}" > /tmp/c3t17_probe.out 2>&1
    rc=$?
    set -e
    if [ "${rc}" -eq 0 ] && ! grep -q "Failed update call\|IC0522" /tmp/c3t17_probe.out; then
      last_ok_n=${probe}
      echo "    probe N=${probe} ${cache_type} OK" | tee -a "${LOG}"
    else
      echo "    probe N=${probe} ${cache_type} FAIL (IC0522 or trap)" | tee -a "${LOG}"
      break
    fi
  done
  echo "${last_ok_n}"
}

run_arm() {
  local cache_type="$1"
  local sid="$2"

  echo "" | tee -a "${LOG}"
  echo ">>> ARM ${cache_type}" | tee -a "${LOG}"

  local cache_file="c3t17_${cache_type}.cache"

  # Step 1: binsearch for N_MAX via probe list
  echo "--- N_MAX probe for ${cache_type} ---" | tee -a "${LOG}"
  local n_max
  n_max=$(probe_n_max "${cache_type}" "${cache_file}" | tail -1)
  echo "--- N_MAX(${cache_type}) = ${n_max} ---" | tee -a "${LOG}"

  if [ "${n_max}" -eq 0 ]; then
    echo "    all probes failed; skipping reps" | tee -a "${LOG}"
    append_csv_row "${sid}" "${cache_type}" 0 0 0 "" "N_MAX=0;all-probes-failed"
    return
  fi

  # Step 2: 3 reps at N_MAX for wall-clock
  local total_ns=0
  for rep in 1 2 3; do
    dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${cache_file}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1
    t0=$(now_ns)
    call_run_update "${cache_type}" "${cache_file}" "${n_max}" "${PROMPT_NARR}" > /tmp/c3t17_rep${rep}.out 2>&1
    t1=$(now_ns)
    dt=$((t1-t0))
    total_ns=$((total_ns+dt))
    echo "    rep ${rep} N=${n_max} ${cache_type} wall=$(python3 -c "print(round(${dt}/1e9,3))")s" | tee -a "${LOG}"
    cat /tmp/c3t17_rep${rep}.out >> "${LOG}"
  done
  mean_s=$(python3 -c "print(round(${total_ns}/3/1e9, 3))")
  tokps=$(python3 -c "print(round(${n_max}/(${total_ns}/3/1e9), 3))")
  echo "    mean wall=${mean_s}s mean tok/s=${tokps}" | tee -a "${LOG}"

  # Step 3: capture the emitted text from rep3 for quality comparison
  emitted=$(python3 -c "import re; t=open('/tmp/c3t17_rep3.out').read(); m=re.search(r'output = \"(.*?)\";', t, re.DOTALL); print(m.group(1) if m else '')")
  echo "" >> "${QUALITY_LOG}"
  echo "=== ${cache_type}: N_MAX=${n_max}, rep3 output ===" >> "${QUALITY_LOG}"
  echo "${emitted}" >> "${QUALITY_LOG}"

  append_csv_row "${sid}" "${cache_type}" 1 "${n_max}" "${n_max}" "${mean_s}" "N_MAX=${n_max};mean_tok_per_s=${tokps};prompt=cat-story;3rep-mean"
}

# f16 = reference (baseline already confirmed at ~20 tok in Task 14)
run_arm "f16" "c3-kvtype-qwen05-f16"
run_arm "q8_0" "c3-kvtype-qwen05-q8"
run_arm "q4_0" "c3-kvtype-qwen05-q4"

echo "" | tee -a "${LOG}"
echo "=== Task 17 complete $(now_iso) ===" | tee -a "${LOG}"
