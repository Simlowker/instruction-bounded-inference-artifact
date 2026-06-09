#!/usr/bin/env bash
# Paper 1.5 C3 — Task 19: Falcon-H1-Tiny 90M SSM baseline on local canister.
#
# Sub-task 19.2: binsearch N_MAX + 3 reps → mixed_precision_measurements.csv
# Sub-task 19.3: 5-call multi-call session → multicall_characterization.csv
set -euo pipefail

NETWORK="local"
MODEL="models/falcon-h1-tiny.gguf"
CACHE="c3t19.cache"
VARIANT_ID="Falcon-H1-Tiny-Instruct-Q8_0"
# ctx-size kept same as load_model to avoid ggml alloc mismatch
CTX="2048"
KV_TYPE="f16"

OUT_MIXED="artifact/data/paper_1_5/mixed_precision_measurements.csv"
OUT_MULTI="artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-ssm-falcon-h1-local.log"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

# Prompt: narrative, similar to Qwen multi-call prompt, ensures non-trivial generation
PROMPT="<|im_start|>system\nYou are a creative storyteller.<|im_end|>\n<|im_start|>user\nTell me a short story about a robot who learns to paint.<|im_end|>\n<|im_start|>assistant\n"

call_run_update() {
  local n="$1"
  local prompt="$2"
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"-c\"; \"${CTX}\"; \"--cache-type-k\"; \"${KV_TYPE}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${prompt}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

new_chat() {
  dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1
}

probe_n() {
  # Fresh cache then one call of N tokens; returns 0 if Ok, 1 if Err/trap
  local n="$1"
  new_chat
  local t0 t1 wall
  t0=$(now_ns)
  set +e
  call_run_update "${n}" "${PROMPT}" > /tmp/c3t19_probe.out 2>&1
  local rc=$?
  set -e
  t1=$(now_ns)
  wall=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  if [ "${rc}" -ne 0 ]; then
    echo "  N=${n} FAIL(rc=${rc}) wall=${wall}s $(head -c 200 /tmp/c3t19_probe.out | tr -d '\n')" | tee -a "${LOG}"
    PROBE_WALL="${wall}"
    return 1
  fi
  if grep -q 'variant { Err' /tmp/c3t19_probe.out; then
    echo "  N=${n} FAIL(err) wall=${wall}s $(head -c 200 /tmp/c3t19_probe.out | tr -d '\n')" | tee -a "${LOG}"
    PROBE_WALL="${wall}"
    return 1
  fi
  # Ok — but treat "early EOG" (eog=true AND wall implies many fewer than N tokens) as INCONCLUSIVE.
  # We use wall as a cheap proxy for "did we actually run ~N tokens worth of inference?"
  # For this model on local, sustained gen is ~10 tok/s, so a call that asks for N tokens
  # but returns in < 0.5 * N / 10 = N/20 seconds probably hit EOG very early.
  local eog
  eog=$(python3 -c "import re; t=open('/tmp/c3t19_probe.out').read(); m=re.search(r'generated_eog = (true|false)', t); print(m.group(1) if m else 'unk')")
  local is_early_eog
  is_early_eog=$(python3 -c "
wall=float(${wall}); n=int(${n}); eog='${eog}'
# 'early EOG' heuristic: eog=true AND wall < 0.4 * n / 10  (i.e. we generated <40% of the budget)
# If eog=false, we ran through all N tokens, so it's a genuine OK.
print('yes' if (eog=='true' and wall < (n / 25.0)) else 'no')
")
  if [ "${is_early_eog}" = "yes" ]; then
    echo "  N=${n} SKIP-EOG wall=${wall}s eog=${eog} (hit EOG before budget; not a valid N_MAX datapoint)" | tee -a "${LOG}"
    PROBE_WALL="${wall}"
    return 1
  fi
  echo "  N=${n} OK wall=${wall}s eog=${eog}" | tee -a "${LOG}"
  PROBE_WALL="${wall}"
  return 0
}

echo "=== Task 19 start $(now_iso) ===" | tee -a "${LOG}"
echo "Model: ${MODEL} | Ctx: ${CTX} | KV: ${KV_TYPE}" | tee -a "${LOG}"

# ---------------- Warmup N=1,2,3 ----------------
echo "" | tee -a "${LOG}"
echo ">>> Warmup N=1,2,3" | tee -a "${LOG}"
for N in 1 2 3; do
  probe_n "${N}" || true
done

# ---------------- Seed probes + binsearch ----------------
echo "" | tee -a "${LOG}"
echo ">>> Seed probes 50/100/150/200" | tee -a "${LOG}"
declare -a OK_LIST=()
declare -a FAIL_LIST=()
for N in 50 100 150 200; do
  if probe_n "${N}"; then
    OK_LIST+=("${N}")
  else
    FAIL_LIST+=("${N}")
  fi
done

# Determine binsearch bounds
get_max() { printf '%s\n' "$@" | sort -n | tail -1; }
get_min() { printf '%s\n' "$@" | sort -n | head -1; }

if [ ${#OK_LIST[@]} -eq 0 ]; then
  echo "ABORT: all seed probes failed; trying lower probes 10/25" | tee -a "${LOG}"
  for N in 10 25; do
    if probe_n "${N}"; then OK_LIST+=("${N}"); else FAIL_LIST+=("${N}"); fi
  done
fi

if [ ${#OK_LIST[@]} -eq 0 ]; then
  echo "FATAL: no OK probe found; abort." | tee -a "${LOG}"
  exit 2
fi
if [ ${#FAIL_LIST[@]} -eq 0 ]; then
  # all succeeded — push higher
  for N in 250 300 400 500; do
    if probe_n "${N}"; then OK_LIST+=("${N}"); else FAIL_LIST+=("${N}"); break; fi
  done
fi

LOW=$(get_max "${OK_LIST[@]}")
HIGH=$(get_min "${FAIL_LIST[@]}")

if [ -z "${HIGH:-}" ]; then
  echo "WARN: still no FAIL found at 500; treating LOW=${LOW} as N_MAX" | tee -a "${LOG}"
  N_MAX="${LOW}"
else
  echo "" | tee -a "${LOG}"
  echo ">>> Bisect [${LOW}, ${HIGH}]" | tee -a "${LOG}"
  while [ $((HIGH - LOW)) -gt 1 ]; do
    MID=$(( (LOW + HIGH) / 2 ))
    if probe_n "${MID}"; then
      LOW="${MID}"
    else
      HIGH="${MID}"
    fi
  done
  N_MAX="${LOW}"
fi

echo "" | tee -a "${LOG}"
echo "=== N_MAX = ${N_MAX} ===" | tee -a "${LOG}"
echo "binsearch: ok=[${OK_LIST[*]}]; fail=[${FAIL_LIST[*]}]; N_MAX=${N_MAX}" | tee -a "${LOG}"

# ---------------- 3 reps at N_MAX ----------------
echo "" | tee -a "${LOG}"
echo ">>> 3 reps at N_MAX=${N_MAX}" | tee -a "${LOG}"
declare -a REP_WALLS=()
for R in 1 2 3; do
  if probe_n "${N_MAX}"; then
    REP_WALLS+=("${PROBE_WALL}")
  fi
done
REP_STR="${REP_WALLS[*]:-}"

# Append row to mixed_precision CSV
BINSEARCH_NOTE="binsearch-ssm: ok=[${OK_LIST[*]}]; fail=[${FAIL_LIST[*]}]; N_MAX=${N_MAX}; 3 reps wall=[${REP_STR}]"
{
  printf "%s,local,uxrrr-q7777-77774-qaaaq-cai,%s,0.000,,,,,,,pending,%s,\"%s\"\n" \
    "${VARIANT_ID}" "${N_MAX}" "$(now_iso)" "${BINSEARCH_NOTE}"
} >> "${OUT_MIXED}"

# ---------------- Multi-call session: 5 consecutive run_update ----------------
# Use N=15 per call (matches qwen multicall pattern), new_chat first
echo "" | tee -a "${LOG}"
echo ">>> Multi-call: 5× run_update N=15 (continuous)" | tee -a "${LOG}"
new_chat

MC_N=15

# Snapshot log line count for parsing C3-IO-END markers post-hoc
LOG_LINE_BEFORE=$(dfx canister logs llama_cpp --network "${NETWORK}" 2>/dev/null | wc -l | tr -d ' ')

# Step 1 (fresh, with prompt)
for step in 1 2 3 4 5; do
  if [ "${step}" -eq 1 ]; then
    PROMPT_USE="${PROMPT}"
  else
    PROMPT_USE=""
  fi
  echo "--- mc step=${step} ---" | tee -a "${LOG}"
  t0=$(now_ns)
  call_run_update "${MC_N}" "${PROMPT_USE}" > /tmp/c3t19_mc${step}.out 2>&1
  t1=$(now_ns)
  dt=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "step ${step} wall=${dt}s" | tee -a "${LOG}"
  cat /tmp/c3t19_mc${step}.out >> "${LOG}"
  eval "WALL_${step}=${dt}"
done

# Fetch logs and pick up C3-IO-END lines emitted during the 5 calls
sleep 1
ALL_LOGS=$(dfx canister logs llama_cpp --network "${NETWORK}" 2>/dev/null)
NEW_LOG_SECTION=$(printf '%s\n' "${ALL_LOGS}" | tail -n +$((LOG_LINE_BEFORE + 1)))
echo "=== C3-IO-END log extract ===" | tee -a "${LOG}"
printf '%s\n' "${NEW_LOG_SECTION}" | grep "C3-IO-END" | tee -a "${LOG}" > /tmp/c3t19_ioend.txt || true

# Parse + append CSV rows
python3 <<PY | tee -a "${LOG}"
import re, os, datetime
lines = []
try:
    with open('/tmp/c3t19_ioend.txt') as f:
        lines = [l.rstrip() for l in f if l.strip()]
except FileNotFoundError:
    pass

walls = [${WALL_1}, ${WALL_2}, ${WALL_3}, ${WALL_4}, ${WALL_5}]
csv_path = r"${OUT_MULTI}"
now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
print(f"parsed {len(lines)} C3-IO-END lines, wall={walls}")

def parse_kv(s):
    d={}
    for tok in s.split():
        if '=' in tok:
            k,v=tok.split('=',1)
            d[k]=v
    return d

for i,line in enumerate(lines[:5]):
    d = parse_kv(line)
    save_instr = d.get('save_terminal_instr','-1')
    save_bytes = d.get('save_terminal_bytes','-1')
    save_tokens = d.get('save_terminal_tokens','-1')
    load_instr = d.get('load_instr','0')
    load_bytes = d.get('load_bytes','-1')
    load_tokens = d.get('load_tokens','-1')
    si_instr = d.get('save_intermediate_instr','-1')
    si_bytes = d.get('save_intermediate_bytes','-1')
    si_tokens = d.get('save_intermediate_tokens','-1')
    mode = 'fresh' if i==0 else 'continue'
    notes = f"save_i_bytes={si_bytes};save_t_bytes={save_bytes};save_i_tokens={si_tokens};save_t_tokens={save_tokens};load_tokens={load_tokens};load_bytes={load_bytes};mode={mode};ssm=bounded-state"
    sid = f"c3-ssm-falcon-h1-step{i+1}"
    wall = walls[i]
    # session_id,model_id,kv_cache_type,n_calls,tokens_per_call,total_tokens,wall_clock_s,total_cycles,io_overhead_write_inst,io_overhead_read_inst,reattach_latency_ms,bit_exact_vs_singlecall,timestamp_utc,notes
    row = f"{sid},Falcon-H1-Tiny-90M,constant-state-ssm,1,{${MC_N}},{${MC_N}},{wall},,{save_instr},{load_instr},,,{now_iso},{notes}"
    print("APPEND:", row)
    with open(csv_path, 'a') as cf:
        cf.write(row + "\n")
PY

echo "" | tee -a "${LOG}"
echo "=== Task 19 complete $(now_iso) ===" | tee -a "${LOG}"
echo "N_MAX=${N_MAX}  | reps=[${REP_STR}]" | tee -a "${LOG}"
