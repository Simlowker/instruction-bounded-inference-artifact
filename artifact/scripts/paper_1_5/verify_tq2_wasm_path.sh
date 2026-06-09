#!/usr/bin/env bash
# verify_tq2_wasm_path.sh
#
# Sanity check: confirm TQ2_0 WASM SIMD128 path is compiled into llama_cpp_canister.
# Returns 0 on success, non-zero on failure with a diagnostic message.
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
CANISTER_DIR="$REPO_ROOT/llama_cpp_canister"
FORK_DIR="$CANISTER_DIR/src/llama_cpp_onicai_fork"
TYPE_FILE="$FORK_DIR/ggml/src/ggml-quants.c"
WASM_QUANTS_FILE="$FORK_DIR/ggml/src/ggml-cpu/arch/wasm/quants.c"

if [[ ! -f "$TYPE_FILE" ]]; then
    echo "ERROR: $TYPE_FILE not found. Is the submodule initialized?"
    exit 2
fi
if [[ ! -f "$WASM_QUANTS_FILE" ]]; then
    echo "ERROR: $WASM_QUANTS_FILE not found. Expected arch/wasm/ kernels missing from submodule."
    exit 2
fi

# 1) Confirm TQ2_0 type exists in type registry
if ! grep -q "GGML_TYPE_TQ2_0" "$TYPE_FILE"; then
    echo "ERROR: GGML_TYPE_TQ2_0 not found in $TYPE_FILE"
    exit 3
fi
echo "[OK] GGML_TYPE_TQ2_0 present in ggml-quants.c"

# 2) Confirm WASM SIMD implementation of ggml_vec_dot_tq2_0_q8_K exists
# The WASM-specific kernel lives in ggml-cpu/arch/wasm/quants.c and must guard
# its body with __wasm_simd128__.
if ! grep -q "^void ggml_vec_dot_tq2_0_q8_K" "$WASM_QUANTS_FILE"; then
    echo "ERROR: ggml_vec_dot_tq2_0_q8_K not defined in $WASM_QUANTS_FILE"
    echo "       TQ2_0 on WASM will fall back to the generic scalar kernel."
    exit 4
fi
# The __wasm_simd128__ guard should appear within ~30 lines after the definition.
if ! awk '/^void ggml_vec_dot_tq2_0_q8_K/{flag=1; n=0} flag && /__wasm_simd128__/{found=1; exit} flag{n++; if(n>60) exit} END{exit !found}' "$WASM_QUANTS_FILE"; then
    echo "WARNING: No __wasm_simd128__ guard near ggml_vec_dot_tq2_0_q8_K body."
    echo "         The function exists but may fall back to scalar on WASM."
    exit 4
fi
echo "[OK] ggml_vec_dot_tq2_0_q8_K with __wasm_simd128__ guard in arch/wasm/quants.c"

# 3) Confirm icpp.toml includes WASM SIMD flag
ICPP_TOML="$CANISTER_DIR/icpp.toml"
if ! grep -q -- "-msimd128" "$ICPP_TOML"; then
    echo "WARNING: -msimd128 not found in $ICPP_TOML. WASM SIMD may not be enabled at build time."
    exit 5
fi
echo "[OK] -msimd128 found in icpp.toml"

echo ""
echo "===> TQ2_0 WASM SIMD path appears active. Safe to proceed with ternary measurements."
