#!/usr/bin/env bash
# Paper 1.5 C3 — prompt-cache save/load IO overhead profile (minimal).
#
# Per locked decision #2: aggregate save/load instruction count + KV byte size
# only (no per-tensor profile). Requires main_.cpp patched with
# icpp_op_profile_reset / icpp_op_profile_get brackets around
# llama_state_save_file / llama_state_load_file.
#
# Sweeps (N_tokens_in_kv, cache_type) cells and appends rows to
# multicall_characterization.csv. Runs against Qwen 0.5B Q4_0 baseline on
# local dfx replica.
#
# Implemented by Task 14.
set -euo pipefail

echo "stub: implement in Task 14" >&2
exit 1
