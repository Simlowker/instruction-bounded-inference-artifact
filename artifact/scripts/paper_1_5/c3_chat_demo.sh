#!/usr/bin/env bash
# Paper 1.5 C3 — Task 18: 10-turn multi-call stateful chat demo.
#
# Pivot: uses Qwen 2.5 0.5B Q4_0 instead of Qwen 1.5B Q4_0.
# Rationale: Qwen 1.5B Q4_0 traps IC0502 "unreachable" on this wasm (Task 20
# pre-check). Q8_0 on 1.5B is predicted to trip IC0524 (~1.7 GB > 1 GB ceiling).
# Qwen 0.5B Q4_0 is already loaded on the canister and proven stable.
#
# Demo: 10 consecutive run_update calls at N=20 each, with --prompt-cache-all,
# continuing narrative. Cumulative ~200 tokens. Demonstrates bit-consistent
# multi-turn state across canister calls.
set -euo pipefail

NETWORK="local"
MODEL="models/c3-qwen05-q4.gguf"
CACHE="c3t18chat-$(date +%s).cache"
N_PER_TURN=10
N_TURNS=10

CSV="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/papers/instruction-bounded-inference/artifact/data/paper_1_5/multicall_characterization.csv"
LOG_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/papers/instruction-bounded-inference/artifact/results/paper_1_5/raw"
LOG="${LOG_DIR}/c3-chat-demo-qwen05.log"
TRANSCRIPT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/papers/instruction-bounded-inference/artifact/results/paper_1_5/tables/c3-chat-demo-transcript.md"
DONE_FLAG="/tmp/c3_chat_demo_done"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${TRANSCRIPT}")"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

call_run_update() {
  local n="$1"
  local prompt="$2"
  dfx canister call llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${prompt}\"; \"-n\"; \"${n}\"} })" --network "${NETWORK}"
}

new_chat() {
  dfx canister call llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" --network "${NETWORK}" >> "${LOG}" 2>&1
}

echo "=== Task 18 chat demo start $(now_iso) ===" | tee "${LOG}"
echo "Model: ${MODEL} | Cache: ${CACHE} | Turns: ${N_TURNS} | N/turn: ${N_PER_TURN}" | tee -a "${LOG}"

# Fresh session
new_chat

OPENER="Elric the blacksmith discovered that"

# Snapshot logs BEFORE any turn
LOG_LINE_BEFORE=$(dfx canister logs llama_cpp --network "${NETWORK}" 2>/dev/null | wc -l | tr -d ' ')

declare -a WALL_LIST=()
declare -a OUTPUT_LIST=()
declare -a CONV_LIST=()
declare -a STATUS_LIST=()
COMPLETED_TURNS=0

# Collect C3-IO-END per-turn into a sidecar file (one JSON blob per line)
IOEND_SIDECAR=/tmp/c3t18_ioend_per_turn.txt
: > "${IOEND_SIDECAR}"

for turn in $(seq 1 ${N_TURNS}); do
  echo "" | tee -a "${LOG}"
  echo "--- Turn ${turn}/${N_TURNS} ---" | tee -a "${LOG}"
  if [ "${turn}" -eq 1 ]; then
    PROMPT_USE="${OPENER}"
  else
    PROMPT_USE=""
  fi

  t0=$(now_ns)
  set +e
  call_run_update "${N_PER_TURN}" "${PROMPT_USE}" > "/tmp/c3t18_t${turn}.out" 2>&1
  rc=$?
  set -e
  t1=$(now_ns)
  dt=$(python3 -c "print(round((${t1}-${t0})/1e9, 3))")
  echo "turn ${turn} wall=${dt}s rc=${rc}" | tee -a "${LOG}"
  cat "/tmp/c3t18_t${turn}.out" >> "${LOG}"

  # Detect trap / error in output
  if [ "${rc}" -ne 0 ] || grep -qE "IC0502|IC0522|IC0524|TRAP|Canister trapped|variant \{ Err" "/tmp/c3t18_t${turn}.out"; then
    STATUS_LIST+=("err")
    WALL_LIST+=("${dt}")
    OUTPUT_LIST+=("<ERROR>")
    CONV_LIST+=("<ERROR>")
    echo "turn ${turn} ERROR — stopping demo (observed context limit)" | tee -a "${LOG}"
    break
  else
    STATUS_LIST+=("ok")
    COMPLETED_TURNS=$((COMPLETED_TURNS+1))
  fi

  # Parse output / conversation
  out_text=$(python3 -c "
import re, sys
t = open('/tmp/c3t18_t${turn}.out').read()
m = re.search(r'output = \"(.*?)\";', t, re.DOTALL)
print(m.group(1) if m else '')
")
  conv_text=$(python3 -c "
import re, sys
t = open('/tmp/c3t18_t${turn}.out').read()
m = re.search(r'conversation = \"(.*?)\";', t, re.DOTALL)
print(m.group(1) if m else '')
")

  WALL_LIST+=("${dt}")
  OUTPUT_LIST+=("${out_text}")
  CONV_LIST+=("${conv_text}")

  echo "out: ${out_text}" | tee -a "${LOG}"

  # Immediately pull logs for this turn's C3-IO-END; concat into sidecar.
  # Reconstruct the markers by joining fragments between "C3-IO-END:" occurrences.
  LOG_LINE_NOW=$(dfx canister logs llama_cpp --network "${NETWORK}" 2>/dev/null | wc -l | tr -d ' ')
  dfx canister logs llama_cpp --network "${NETWORK}" 2>/dev/null \
    | tail -n +$((LOG_LINE_BEFORE + 1)) \
    > /tmp/c3t18_logsnap_${turn}.txt
  export TURN="${turn}"
  python3 <<'PYSNAP' >> "${IOEND_SIDECAR}"
import re, sys, os, json
turn = int(os.environ["TURN"])
path = f"/tmp/c3t18_logsnap_{turn}.txt"
try:
    with open(path) as f:
        buf = f.read()
except FileNotFoundError:
    buf = ""
# Strip timestamps like "[12345. 2026-04-20T...]: " prefixes
buf = re.sub(r'^\[\d+\.\s+[0-9T:\.Z\-+]+\]:\s*', '', buf, flags=re.MULTILINE)
# Find all C3-IO-END entries; each entry runs until next blank line after "C3-IO-END:"
# Simplest: split on "C3-IO-END:" marker.
chunks = buf.split("C3-IO-END:")
if len(chunks) > 1:
    # take the LAST chunk — that's this turn's marker
    last = chunks[-1]
    # collapse whitespace
    flat = re.sub(r'\s+', ' ', last).strip()
    # Trim at next obvious non-kv section (next "ICPP-PROF" etc.)
    cut = re.split(r'(?:ICPP-PROF|===|\[TRAP\])', flat, maxsplit=1)[0]
    kv = {}
    for tok in cut.split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            kv[k] = v.rstrip(';,')
    print(json.dumps({"turn": turn, "kv": kv}))
else:
    print(json.dumps({"turn": turn, "kv": {}}))
PYSNAP
  LOG_LINE_BEFORE=${LOG_LINE_NOW}
done

# Per-turn C3-IO-END captured via sidecar ${IOEND_SIDECAR}; dump for log visibility
echo "" | tee -a "${LOG}"
echo "=== C3-IO-END per-turn sidecar ===" | tee -a "${LOG}"
cat "${IOEND_SIDECAR}" | tee -a "${LOG}" || true

# Dump bash state to JSON so python heredoc can read without shell expansion.
python3 - "${CSV}" "${TRANSCRIPT}" "${LOG}" "${N_TURNS}" "${COMPLETED_TURNS}" "${N_PER_TURN}" "${WALL_LIST[@]-}" <<'PY_DUMP' > /tmp/c3t18_state_paths.json
import sys, json
# argv: csv, transcript, log, N_PLANNED, N_COMPLETED, N_PER, walls...
csv_path, transcript, logp, np, nc, nper, *walls = sys.argv[1:]
json.dump({
    "csv_path": csv_path,
    "transcript_path": transcript,
    "log_path": logp,
    "N_PLANNED": int(np),
    "N_COMPLETED": int(nc),
    "N_PER": int(nper),
    "walls": [float(w) for w in walls if w.strip()],
}, sys.stdout)
PY_DUMP

# Dump statuses separately
python3 -c "import sys, json; json.dump(sys.argv[1:], open('/tmp/c3t18_statuses.json','w'))" "${STATUS_LIST[@]-}"

# Parse IO lines and append CSV rows + build transcript data
python3 - <<'PY'
import os, re, datetime, json, pathlib

with open('/tmp/c3t18_state_paths.json') as f:
    state = json.load(f)
csv_path = state["csv_path"]
transcript_path = state["transcript_path"]
log_path = state["log_path"]
N_PLANNED = state["N_PLANNED"]
N_COMPLETED = state["N_COMPLETED"]
N_PER = state["N_PER"]
walls = state["walls"]

with open('/tmp/c3t18_statuses.json') as f:
    statuses = json.load(f)

# We record N = number of actual attempts (may include 1 trailing error)
N = len(walls)

# Read per-turn outputs (skip if missing or error)
outputs = []
convs = []
for i in range(1, N+1):
    p = f"/tmp/c3t18_t{i}.out"
    try:
        with open(p) as f:
            t = f.read()
    except FileNotFoundError:
        outputs.append("")
        convs.append("")
        continue
    status = statuses[i-1] if i-1 < len(statuses) else "ok"
    if status == "err":
        outputs.append("<ERROR: canister trap>")
        convs.append("<ERROR: canister trap>")
        continue
    m = re.search(r'output = "(.*?)";', t, re.DOTALL)
    outputs.append(m.group(1) if m else "")
    m = re.search(r'conversation = "(.*?)";', t, re.DOTALL)
    convs.append(m.group(1) if m else "")

# Parse C3-IO-END per-turn sidecar (one JSON blob per line)
per_turn_kv = {}
try:
    with open('/tmp/c3t18_ioend_per_turn.txt') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                per_turn_kv[int(obj["turn"])] = obj.get("kv", {})
            except Exception:
                pass
except FileNotFoundError:
    pass

print(f"parsed C3-IO-END for turns: {sorted(per_turn_kv.keys())}; walls={walls}")

now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

csv_rows = []
io_records = []
# Only emit CSV rows for successful turns (exclude trailing error, if any)
for i in range(N_COMPLETED):
    turn_num = i + 1
    wall = walls[i] if i < len(walls) else 0.0
    d = per_turn_kv.get(turn_num, {})
    save_instr   = d.get('save_terminal_instr','-1')
    save_bytes   = d.get('save_terminal_bytes','-1')
    save_tokens  = d.get('save_terminal_tokens','-1')
    load_instr   = d.get('load_instr','0')
    load_bytes   = d.get('load_bytes','-1')
    load_tokens  = d.get('load_tokens','-1')
    si_instr     = d.get('save_intermediate_instr','-1')
    si_bytes     = d.get('save_intermediate_bytes','-1')
    si_tokens    = d.get('save_intermediate_tokens','-1')
    mode = 'fresh' if i==0 else 'continue'
    notes = (
        f"save_i_bytes={si_bytes};save_t_bytes={save_bytes};"
        f"save_i_tokens={si_tokens};save_t_tokens={save_tokens};"
        f"load_tokens={load_tokens};load_bytes={load_bytes};mode={mode};"
        f"turn={i+1}/{N};chat-demo-pivot-qwen05"
    )
    sid = f"c3-chat-qwen05-turn{i+1}"
    row = f"{sid},Qwen2.5-0.5B,f16,1,{N_PER},{N_PER},{wall},,{save_instr},{load_instr},,,{now_iso},{notes}"
    csv_rows.append(row)
    io_records.append({
        "turn": i+1,
        "wall_s": wall,
        "save_instr": save_instr,
        "save_bytes": save_bytes,
        "save_tokens": save_tokens,
        "load_instr": load_instr,
        "load_bytes": load_bytes,
        "load_tokens": load_tokens,
        "mode": mode,
    })

# Append CSV
with open(csv_path, 'a') as f:
    for r in csv_rows:
        f.write(r + "\n")
print(f"Appended {len(csv_rows)} rows to {csv_path}")

# Build transcript
total_wall = sum(walls)
cumulative_tokens = N_COMPLETED * N_PER

# Coherence heuristics
flags = []
for i, out in enumerate(outputs, start=1):
    # Repetition flag: same word repeated ≥3 times consecutively (case-insensitive)
    words = re.findall(r'\b\w+\b', out.lower())
    if len(words) >= 3:
        for j in range(len(words)-2):
            if words[j] == words[j+1] == words[j+2]:
                flags.append(f"- Turn {i} repeats '{words[j]}' 3 times consecutively — minor artifact")
                break
    # Short/empty output
    if len(out.strip()) < 5:
        flags.append(f"- Turn {i} emitted very short output ({len(out)} chars)")

coherence_section = "\n".join(flags) if flags else "- No obvious coherence breaks detected by automated heuristics."

lines = []
lines.append(f"# C3 Multi-Call Chat Demo — 10-Turn Transcript")
lines.append("")
lines.append(f"**Date:** {now_iso}")
lines.append(f"**Model:** Qwen 2.5 0.5B Q4_0 (`models/c3-qwen05-q4.gguf`)")
lines.append(f"**KV cache type:** f16 (default; --cache-type flags are no-ops on this wasm, see Task 17)")
lines.append(f"**Network:** local dfx replica")
lines.append(f"**N per turn:** {N_PER} tokens")
lines.append(f"**Turns attempted:** {N}  (**completed successfully:** {N_COMPLETED})")
lines.append(f"**Cumulative tokens (successful turns):** {cumulative_tokens}")
lines.append(f"**Total wall-clock (all attempts):** {total_wall:.3f} s")
lines.append(f"**Mean per-turn wall:** {total_wall/max(N,1):.3f} s")
lines.append("")
lines.append("## Pivot note")
lines.append("")
lines.append("Original Phase 2 plan specified **Qwen 2.5 1.5B-Instruct Q4_0** for the chat demo.")
lines.append("During pre-check (Task 20), Qwen 1.5B Q4_0 was found to trap IC0502 'unreachable'")
lines.append("on the current wasm build. Q8_0 on 1.5B is predicted to trip IC0524 (~1.7 GB > 1 GB")
lines.append("stable-memory ceiling, matches the paper's §5.6 observation). We therefore")
lines.append("**pivoted to Qwen 0.5B Q4_0** — already loaded on the canister and proven stable")
lines.append("across Tasks 14/15/16/17. This is a smaller base model (no RLHF); the paper's C3")
lines.append("coherence claim is about **bit-consistent state across canister calls**, not")
lines.append("literary quality.")
lines.append("")
lines.append("## Coherence observations")
lines.append("")
lines.append(coherence_section)
lines.append("")
lines.append("## Per-turn summary")
lines.append("")
lines.append("| Turn | Wall (s) | Mode | load_tokens | load_bytes | save_tokens | save_bytes | save_instr | load_instr |")
lines.append("|------|----------|------|-------------|------------|-------------|------------|------------|------------|")
for r in io_records:
    lines.append(
        f"| {r['turn']} | {r['wall_s']:.3f} | {r['mode']} | "
        f"{r['load_tokens']} | {r['load_bytes']} | "
        f"{r['save_tokens']} | {r['save_bytes']} | "
        f"{r['save_instr']} | {r['load_instr']} |"
    )
lines.append("")
lines.append("## Per-turn emitted text (first ~100 chars)")
lines.append("")
for i, out in enumerate(outputs, start=1):
    snippet = out[:100].replace("\n", " ")
    lines.append(f"**Turn {i}:** `{snippet}`")
    lines.append("")

lines.append("## Full concatenated narrative")
lines.append("")
lines.append("Opener (user-provided seed):")
lines.append("")
lines.append(f"> Elric the blacksmith discovered that")
lines.append("")
lines.append(f"Model continuation across {N_COMPLETED} successful turns (of {N_PLANNED} planned):")
lines.append("")
lines.append("```")
full = "".join(outputs)
lines.append(full)
lines.append("```")
lines.append("")
lines.append(f"## Final conversation buffer (turn {N_COMPLETED if N_COMPLETED>0 else 1})")
lines.append("")
lines.append("```")
lines.append(convs[-1] if convs else "")
lines.append("```")
lines.append("")

with open(transcript_path, 'w') as f:
    f.write("\n".join(lines))

print(f"Wrote transcript to {transcript_path}")
print(f"Cumulative tokens: {cumulative_tokens}")
print(f"Total wall: {total_wall:.3f}s")
PY

echo "" | tee -a "${LOG}"
echo "=== Task 18 chat demo complete $(now_iso) ===" | tee -a "${LOG}"
touch "${DONE_FLAG}"
