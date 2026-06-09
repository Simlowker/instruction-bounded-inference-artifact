#!/usr/bin/env bash
# Capture C3-IO-END events per step (log buffer overflows if we batch 5 at once).
set -euo pipefail

NETWORK="local"
MODEL="models/falcon-h1-tiny.gguf"
CACHE="c3t19m.cache"
CTX="2048"
KV_TYPE="f16"
MC_N=15

OUT_MULTI="artifact/data/paper_1_5/multicall_characterization.csv"
LOG="artifact/results/paper_1_5/raw/c3-ssm-falcon-h1-local.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

PROMPT="<|im_start|>system\nYou are a creative storyteller.<|im_end|>\n<|im_start|>user\nTell me a short story about a robot who learns to paint.<|im_end|>\n<|im_start|>assistant\n"

new_chat() {
  dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1
}

call_run_update() {
  local n="$1" p="$2"
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"-c\"; \"${CTX}\"; \"--cache-type-k\"; \"${KV_TYPE}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${p}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

# Extract the single C3-IO-END event (split across log lines) → flatten, parse fields.
flatten_c3_ioend() {
  # Fetch last 300 log lines, find C3-IO-END, concatenate everything on that log timestamp.
  local all
  all=$(dfx canister logs llama_cpp --network "${NETWORK}" 2>&1 | tail -400)
  # Find line index of last "C3-IO-END:" occurrence
  local idx
  idx=$(printf '%s\n' "$all" | grep -n "C3-IO-END:" | tail -1 | cut -d: -f1)
  if [ -z "${idx}" ]; then
    echo ""; return
  fi
  # Grab 20 lines starting from that one; strip the "[ts]:" prefix, concatenate
  printf '%s\n' "$all" | sed -n "${idx},$((idx+20))p" | sed 's/^\[[^]]*\]://g' | tr -d '\n' | tr -s ' '
}

echo "=== Task 19.3 multicall fine-grained $(now_iso) ===" | tee -a "${LOG}"
new_chat

for step in 1 2 3 4 5; do
  if [ "${step}" -eq 1 ]; then
    P="${PROMPT}"
  else
    P=""
  fi
  echo "--- mc step=${step} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update "${MC_N}" "${P}" > /tmp/c3t19m_step${step}.out 2>&1
  t1=$(now_ns)
  dt=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  eval "WALL_${step}=${dt}"
  cat /tmp/c3t19m_step${step}.out >> "${LOG}"
  echo "step ${step} wall=${dt}s" | tee -a "${LOG}"
  # Give IC some time to flush log buffer
  sleep 2
  # Parse C3-IO-END right now (before next call overwrites)
  flat=$(flatten_c3_ioend)
  if [ -z "${flat}" ]; then
    echo "  (no C3-IO-END captured for step ${step})" | tee -a "${LOG}"
    eval "IO_${step}=\"MISSING\""
  else
    echo "step ${step} C3-IO: ${flat}" | tee -a "${LOG}"
    eval "IO_${step}=\$flat"
  fi
done

# Emit CSV rows
python3 <<PY | tee -a "${LOG}"
import re, os, datetime
rows = [
  ("${IO_1}", ${WALL_1}, 1),
  ("${IO_2}", ${WALL_2}, 2),
  ("${IO_3}", ${WALL_3}, 3),
  ("${IO_4}", ${WALL_4}, 4),
  ("${IO_5}", ${WALL_5}, 5),
]

def parse_kv(s):
    d={}
    for tok in s.split():
        if '=' in tok:
            k,v=tok.split('=',1)
            d[k]=v
    return d

csv_path = r"${OUT_MULTI}"
now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
with open(csv_path, 'a') as cf:
    for io, wall, step_idx in rows:
        d = parse_kv(io or "")
        save_instr = d.get('save_terminal_instr','-1')
        save_bytes = d.get('save_terminal_bytes','-1')
        save_tokens = d.get('save_terminal_tokens','-1')
        load_instr = d.get('load_instr','0')
        load_bytes = d.get('load_bytes','-1')
        load_tokens = d.get('load_tokens','-1')
        si_instr = d.get('save_intermediate_instr','-1')
        si_bytes = d.get('save_intermediate_bytes','-1')
        si_tokens = d.get('save_intermediate_tokens','-1')
        mode = 'fresh' if step_idx==1 else 'continue'
        notes = f"save_i_bytes={si_bytes};save_t_bytes={save_bytes};save_i_tokens={si_tokens};save_t_tokens={save_tokens};load_tokens={load_tokens};load_bytes={load_bytes};mode={mode};ssm=bounded-state"
        sid = f"c3-ssm-falcon-h1-step{step_idx}"
        row = f"{sid},Falcon-H1-Tiny-90M,constant-state-ssm,1,{${MC_N}},{${MC_N}},{wall},,{save_instr},{load_instr},,,{now_iso},{notes}"
        print("APPEND:", row)
        cf.write(row + "\n")
PY

echo "=== Task 19.3 complete $(now_iso) ===" | tee -a "${LOG}"
