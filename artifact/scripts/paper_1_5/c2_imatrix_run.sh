#!/usr/bin/env bash
# Paper 1.5 C2 — imatrix generation for mixed-precision calibration.
# Reusable across Task 4 (SmolLM2-135M) and Task 10 (Qwen 0.5B).
#
# Usage:
#   c2_imatrix_run.sh <model_f16.gguf> <output_imatrix.gguf> [label]
#
# - model_f16.gguf:  F16 GGUF of the base model to calibrate.
# - output_imatrix.gguf:  destination path for the imatrix GGUF.
# - label:  optional short tag (e.g. "smollm135", "qwen05") used in log filenames.
#           Defaults to the output basename sans extension.
#
# Calibration corpora (must exist under .worktree-cache/calibration/):
#   - calibration_wikitext2.txt  (WikiText-2-raw test split)
#   - calibration_c4.txt         (allenai/c4 en validation, ~4 MB)
#
# Runs llama-imatrix with 512 chunks × 2048 ctx over a concatenated mix of the
# two corpora (llama-imatrix's -f takes a single file; we concat to avoid
# single-corpus calibration bias per Paper 1.5 spec §4.3).
#
# Outputs:
#   - <output_imatrix.gguf>
#   - results/paper_1_5/raw/c2-imatrix-<label>-stdout.log
#   - results/paper_1_5/raw/c2-imatrix-<label>-stderr.log
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $(basename "$0") <model_f16.gguf> <output_imatrix.gguf> [label]" >&2
    exit 2
fi

MODEL_F16="$1"
OUT_IMATRIX="$2"
LABEL="${3:-$(basename "${OUT_IMATRIX%.gguf}")}"

# --- locate worktree root (this script is at .../artifact/scripts/paper_1_5/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$ARTIFACT_DIR/../../.." && pwd)"

CALIB_DIR="$REPO_ROOT/.worktree-cache/calibration"
WT_FILE="$CALIB_DIR/calibration_wikitext2.txt"
C4_FILE="$CALIB_DIR/calibration_c4.txt"

LOG_DIR="$ARTIFACT_DIR/results/paper_1_5/raw"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/c2-imatrix-${LABEL}-stdout.log"
STDERR_LOG="$LOG_DIR/c2-imatrix-${LABEL}-stderr.log"

# --- sanity checks ---
for f in "$MODEL_F16" "$WT_FILE" "$C4_FILE"; do
    if [[ ! -f "$f" ]]; then
        echo "error: required input missing: $f" >&2
        exit 3
    fi
done

if ! command -v llama-imatrix >/dev/null 2>&1; then
    echo "error: llama-imatrix not found on PATH" >&2
    exit 4
fi

mkdir -p "$(dirname "$OUT_IMATRIX")"

# --- truncate per-run logs so reruns don't accumulate ambiguous mixed output ---
: > "$STDOUT_LOG"
: > "$STDERR_LOG"

# --- build a mixed calibration corpus (concat) for llama-imatrix -f ---
# Keep the mix next to the logs so reruns are reproducible and the file
# survives if the script is killed mid-run (no EXIT trap cleanup).
MIXED_PATH="$LOG_DIR/c2-imatrix-${LABEL}-mixed-calib.txt"
cat "$WT_FILE" > "$MIXED_PATH"
printf '\n\n' >> "$MIXED_PATH"
cat "$C4_FILE" >> "$MIXED_PATH"
MIX_BYTES=$(wc -c < "$MIXED_PATH")

{
    echo "[c2_imatrix_run] label=$LABEL"
    echo "[c2_imatrix_run] model=$MODEL_F16"
    echo "[c2_imatrix_run] out=$OUT_IMATRIX"
    echo "[c2_imatrix_run] mix=$MIXED_PATH ($MIX_BYTES bytes)"
    echo "[c2_imatrix_run] chunks=512 ctx=2048 output-format=gguf"
    echo "[c2_imatrix_run] started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee -a "$STDERR_LOG" >&2

# --- run llama-imatrix (direct redirect; avoid tee-subshells that can be
#     orphaned when the parent session is torn down) ---
START_EPOCH=$(date +%s)
set +e
llama-imatrix \
    --model "$MODEL_F16" \
    --output "$OUT_IMATRIX" \
    --output-format gguf \
    --file "$MIXED_PATH" \
    --chunks 512 \
    --ctx-size 2048 \
    >>"$STDOUT_LOG" 2>>"$STDERR_LOG"
RC=$?
set -e
END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
{
    echo "[c2_imatrix_run] ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) elapsed_sec=$ELAPSED rc=$RC"
} | tee -a "$STDERR_LOG" >&2

if [[ $RC -ne 0 ]]; then
    echo "error: llama-imatrix exited with status $RC" >&2
    exit $RC
fi

# --- output sanity ---
if [[ ! -f "$OUT_IMATRIX" ]]; then
    echo "error: expected output not produced: $OUT_IMATRIX" >&2
    exit 5
fi
SIZE_BYTES=$(stat -f%z "$OUT_IMATRIX" 2>/dev/null || stat -c%s "$OUT_IMATRIX")
if [[ $SIZE_BYTES -lt 512000 ]]; then
    echo "error: imatrix suspiciously small ($SIZE_BYTES bytes < 500 KB): $OUT_IMATRIX" >&2
    exit 6
fi
if (( ELAPSED > 5400 )); then
    echo "error: wall-clock ${ELAPSED}s exceeds 90 min sanity threshold" >&2
    exit 7
fi
echo "[c2_imatrix_run] ok: $OUT_IMATRIX ($SIZE_BYTES bytes)" >&2
