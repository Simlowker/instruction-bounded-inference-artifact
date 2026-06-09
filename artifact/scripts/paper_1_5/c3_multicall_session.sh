#!/usr/bin/env bash
# Paper 1.5 C3 — multi-call stateful chat session driver.
#
# Drives an N-turn chat session against llama_cpp canister with
# --prompt-cache-all, recording per-call cycles + wall-clock + emitted tokens
# to multicall_characterization.csv. Used for:
#   - Task 16: 1-call vs N-call A/B (Qwen 0.5B Q4_0)
#   - Task 18: 10-turn chat demo (Qwen 1.5B Q4_0 + Q8 KV)
#   - Task 19: SSM baseline (Falcon-H1-Tiny 90M)
set -euo pipefail

echo "stub: implement in Task 16" >&2
exit 1
