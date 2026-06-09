#!/bin/bash
# Task 12 SSN — C3 chat demo on SSN using currently loaded model
set -euo pipefail
export DFX_WARNING=-mainnet_plaintext_identity

NETWORK="ic"
MODEL="models/model.gguf"
CACHE="ssn-c3demo-$(date +%s).cache"
N_PER_TURN=10
N_TURNS=10
LOG="artifact/results/paper_1_5/raw/c3-ssn-chat-demo-qwen05.log"
TRANSCRIPT="artifact/results/paper_1_5/tables/c3-ssn-chat-demo-transcript.md"

now_iso() { date -u +"%Y-%m-%dT%H:%M:%S+00:00"; }
now_ns()  { python3 -c "import time; print(int(time.time()*1e9))"; }

echo "=== SSN chat demo start $(now_iso) ===" | tee "${LOG}"
echo "Canister: u2pva-3iaaa | Model: ${MODEL} | Cache: ${CACHE} | Turns: ${N_TURNS} | N/turn: ${N_PER_TURN}" | tee -a "${LOG}"

dfx canister call --network ${NETWORK} llama_cpp new_chat "(record { args = vec {\"--prompt-cache\"; \"${CACHE}\"; \"--model\"; \"${MODEL}\"} })" 2>&1 | tail -3 >> "${LOG}"

OPENER="Elric the blacksmith discovered that"
declare -a WALLS=()
declare -a OUTPUTS=()
COMPLETED=0
LAST_CONV=""

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
  OUT=$(dfx canister call --network ${NETWORK} llama_cpp run_update "(record { args = vec {\"--model\"; \"${MODEL}\"; \"--prompt-cache\"; \"${CACHE}\"; \"--prompt-cache-all\"; \"-sp\"; \"-p\"; \"${PROMPT_USE}\"; \"-n\"; \"${N_PER_TURN}\"} })" 2>&1)
  rc=$?
  set -e
  t1=$(now_ns)
  dt=$(python3 -c "print(round((${t1}-${t0})/1e9,3))")
  echo "turn ${turn} wall=${dt}s rc=${rc}" | tee -a "${LOG}"
  echo "${OUT}" >> "${LOG}"
  if [ ${rc} -ne 0 ] || echo "${OUT}" | grep -qE "IC05|Canister trapped"; then
    echo "turn ${turn} ERROR — stopping" | tee -a "${LOG}"
    break
  fi
  out_text=$(echo "${OUT}" | python3 -c "import sys,re;t=sys.stdin.read();m=re.search(r'output = \"(.*?)\";',t,re.DOTALL);print(m.group(1) if m else '')")
  conv_text=$(echo "${OUT}" | python3 -c "import sys,re;t=sys.stdin.read();m=re.search(r'conversation = \"(.*?)\";',t,re.DOTALL);print(m.group(1) if m else '')")
  WALLS+=("${dt}")
  OUTPUTS+=("${out_text}")
  LAST_CONV="${conv_text}"
  COMPLETED=$((COMPLETED+1))
  echo "out: ${out_text}" | tee -a "${LOG}"
done

# Emit transcript
TOTAL_WALL=$(python3 -c "print(round(sum([$(IFS=,; echo "${WALLS[*]}")]),3))" 2>/dev/null || echo "0")
MEAN_WALL=$(python3 -c "xs=[$(IFS=,; echo "${WALLS[*]}")]; print(round(sum(xs)/len(xs),3) if xs else 0)" 2>/dev/null || echo "0")

{
  echo "# C3 SSN Multi-Call Chat Demo — 10-Turn Transcript (SSN MAINNET)"
  echo ""
  echo "**Date:** $(now_iso)"
  echo "**Canister:** u2pva-3iaaa-aaaax-qaa7a-cai (SSN mainnet, 13-node consensus)"
  echo "**Model:** Qwen 2.5 0.5B BASE-Q8_0 (C2 Pareto winner)"
  echo "**Network:** \`--network ic\`"
  echo "**Turns attempted:** ${N_TURNS}  (**completed:** ${COMPLETED})"
  echo "**Cumulative tokens:** $((COMPLETED * N_PER_TURN))"
  echo "**Total wall:** ${TOTAL_WALL}s | Mean per-turn: ${MEAN_WALL}s"
  echo ""
  echo "## Per-turn output"
  echo ""
  for i in "${!OUTPUTS[@]}"; do
    t=$((i+1))
    echo "**Turn ${t}** (wall ${WALLS[$i]}s): \`${OUTPUTS[$i]}\`"
    echo ""
  done
  echo "## Final conversation buffer"
  echo ""
  echo '```'
  echo "${LAST_CONV}"
  echo '```'
} > "${TRANSCRIPT}"

echo "" | tee -a "${LOG}"
echo "=== SSN chat demo complete $(now_iso) ===" | tee -a "${LOG}"
echo "Transcript: ${TRANSCRIPT}" | tee -a "${LOG}"
echo "Total: ${COMPLETED}/${N_TURNS} turns, $((COMPLETED * N_PER_TURN)) tokens, ${TOTAL_WALL}s wall" | tee -a "${LOG}"
