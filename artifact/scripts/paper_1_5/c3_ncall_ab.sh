#!/usr/bin/env bash
# Paper 1.5 C3 — Task 16: 1-call vs N-call A/B (100 tok generation target).
#
# A: single run_update -n 100 (expected IC0522 failure)
# B: 7x run_update -n 15 chained via --prompt-cache-all (~105 tok total)
# C: single run_update -n 20 (N_MAX reference, for warm baseline tok/s)
#
# All against Qwen 2.5 0.5B Q4_0, f16 KV.
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CACHE="c3t16.cache"
CSV="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-ncall-qwen05.log"

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
  local notes="$6"
  printf "%s,Qwen2.5-0.5B,f16,%s,%s,%s,%s,,,,,,%s,%s\n" \
    "${sid}" "${n_calls}" "${tokens_per_call}" "${total_tokens}" "${wall_s}" \
    "$(now_iso)" "${notes}" >> "${CSV}"
}

PROMPT="<|im_start|>user\nTell me a story about a robot who learned to paint.<|im_end|>\n<|im_start|>assistant\n"

echo "=== Task 16 start $(now_iso) ===" | tee -a "${LOG}"

# ---------- A: 1-call N=100 (expected to fail) ----------
echo "" | tee -a "${LOG}"
echo ">>> A: new_chat + 1-call N=100 (expected IC0522)" | tee -a "${LOG}"
dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1

t0=$(now_ns)
set +e
call_run_update 100 "${PROMPT}" > /tmp/c3t16_A.out 2>&1
rc_A=$?
set -e
t1=$(now_ns)
wall_A=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
echo ">>> A: exit=${rc_A} wall=${wall_A}s" | tee -a "${LOG}"
cat /tmp/c3t16_A.out >> "${LOG}"

# Check output for IC0522
if grep -q "IC0522\|instruction-limit-exceeded\|instructions for single message" /tmp/c3t16_A.out; then
  A_notes="fail=IC0522-instruction-limit;expected-failure-mode;wall_includes_ic_overhead"
else
  # Maybe it succeeded? Check output
  A_notes="unexpected-outcome-see-log"
fi

append_csv_row "c3-ncall-qwen05-1call-fail" 1 100 0 "${wall_A}" "${A_notes}"

# ---------- B: N-call 7×N=15 ----------
echo "" | tee -a "${LOG}"
echo ">>> B: new_chat + 7× run_update N=15" | tee -a "${LOG}"
dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1

total_wall_ns=0
total_gen_tokens=0
emitted_text=""

# Step 1 with prompt
t0=$(now_ns)
call_run_update 15 "${PROMPT}" > /tmp/c3t16_B1.out 2>&1
t1=$(now_ns)
dt=$((t1-t0))
total_wall_ns=$((total_wall_ns+dt))
echo ">>> B step 1 wall=$(python3 -c "print(round(${dt}/1e9,3))")s" | tee -a "${LOG}"
cat /tmp/c3t16_B1.out >> "${LOG}"
# parse output field
out1=$(python3 -c "import re,sys; t=open('/tmp/c3t16_B1.out').read(); m=re.search(r'output = \"(.*?)\";', t, re.DOTALL); print(m.group(1) if m else '')")
emitted_text="${out1}"
# count tokens roughly via word count of output; canister returns generated token strings in "output" — use approximate token count from spaces + punctuation
# More reliable: parse generated field if present; else use N=15
total_gen_tokens=$((total_gen_tokens+15))

for step in 2 3 4 5 6 7; do
  t0=$(now_ns)
  call_run_update 15 "" > /tmp/c3t16_B${step}.out 2>&1
  t1=$(now_ns)
  dt=$((t1-t0))
  total_wall_ns=$((total_wall_ns+dt))
  echo ">>> B step ${step} wall=$(python3 -c "print(round(${dt}/1e9,3))")s" | tee -a "${LOG}"
  cat /tmp/c3t16_B${step}.out >> "${LOG}"
  outk=$(python3 -c "import re,sys; t=open('/tmp/c3t16_B${step}.out').read(); m=re.search(r'output = \"(.*?)\";', t, re.DOTALL); print(m.group(1) if m else '')")
  emitted_text="${emitted_text}${outk}"
  total_gen_tokens=$((total_gen_tokens+15))
done

wall_B=$(python3 -c "print(round(${total_wall_ns}/1e9, 3))")
toks_B=${total_gen_tokens}
tokps_B=$(python3 -c "print(round(${total_gen_tokens}/(${total_wall_ns}/1e9), 3))")
echo ">>> B total: wall=${wall_B}s tokens≈${toks_B} tok/s=${tokps_B}" | tee -a "${LOG}"
echo ">>> B emitted text (concatenated outputs):" | tee -a "${LOG}"
printf '%s\n' "${emitted_text}" | tee -a "${LOG}"

append_csv_row "c3-ncall-qwen05-Ncall-7x15" 7 15 "${toks_B}" "${wall_B}" "multi-call-success;tok_per_s=${tokps_B};prompt=robot-paint-story;sum-of-per-call-wall"

# ---------- C: 1-call N=20 (warm baseline, single-call reference) ----------
echo "" | tee -a "${LOG}"
echo ">>> C: new_chat + 1-call N=20 (reference)" | tee -a "${LOG}"
dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1

t0=$(now_ns)
call_run_update 20 "${PROMPT}" > /tmp/c3t16_C.out 2>&1
t1=$(now_ns)
wall_C=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
tokps_C=$(python3 -c "print(round(20/(${wall_C}), 3))")
echo ">>> C: wall=${wall_C}s tok/s=${tokps_C}" | tee -a "${LOG}"
cat /tmp/c3t16_C.out >> "${LOG}"

append_csv_row "c3-ncall-qwen05-1call-nmax" 1 20 20 "${wall_C}" "single-call-reference-at-N_MAX;tok_per_s=${tokps_C};prompt=robot-paint-story"

echo "" | tee -a "${LOG}"
echo "=== Task 16 complete $(now_iso) ===" | tee -a "${LOG}"
echo "summary: A fail=${rc_A} wall=${wall_A}s | B tok=${toks_B} wall=${wall_B}s tok/s=${tokps_B} | C wall=${wall_C}s tok/s=${tokps_C}" | tee -a "${LOG}"
