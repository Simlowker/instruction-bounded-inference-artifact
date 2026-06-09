#!/usr/bin/env python3
"""
download_ternary_models.py

Downloads TriLM 560M and 3.9B GGUF files (TQ2_0 quantization) from HuggingFace
into llama_cpp_canister/models/trilm/ for Paper 1.5 C1a measurements.

Prerequisites:
    pip install huggingface_hub

Usage:
    python3 download_ternary_models.py           # downloads both
    python3 download_ternary_models.py 560M      # downloads only 560M
    python3 download_ternary_models.py 3.9B      # downloads only 3.9B
"""
from pathlib import Path
from huggingface_hub import hf_hub_download
import sys
import os


REPO_ROOT = Path(__file__).resolve().parents[5]  # worktree root (repo parent of papers/)
MODELS_DIR = REPO_ROOT / "llama_cpp_canister" / "models" / "trilm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace repos for TriLM TQ2_0 GGUFs.
# NOTE: Verify these repo IDs are current before running — HF repos change.
CANDIDATES = {
    "560M": {
        "repo_id": "QuantFactory/TriLM_560M_Unpacked-GGUF",
        "filename": "TriLM_560M_Unpacked.TQ2_0.gguf",
    },
    "3.9B": {
        "repo_id": "QuantFactory/TriLM_3.9B_Unpacked-GGUF",
        "filename": "TriLM_3.9B_Unpacked.TQ2_0.gguf",
    },
}


def download(model_key: str) -> Path:
    if model_key not in CANDIDATES:
        raise ValueError(f"Unknown model {model_key}. Choices: {list(CANDIDATES.keys())}")
    entry = CANDIDATES[model_key]
    print(f"[download] {model_key}: {entry['repo_id']} / {entry['filename']}")
    local_path = hf_hub_download(
        repo_id=entry["repo_id"],
        filename=entry["filename"],
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"[download] saved to {local_path} ({size_mb:.1f} MB)")
    return Path(local_path)


def main():
    args = sys.argv[1:]
    targets = args if args else ["560M", "3.9B"]
    for t in targets:
        try:
            download(t)
        except Exception as e:
            print(f"[ERROR] {t}: {e}", file=sys.stderr)
            print("         Check the repo_id above — HF may have renamed or removed the repo.")
            print("         Alternative: search https://huggingface.co/models?search=trilm+tq2_0")
            sys.exit(1)


if __name__ == "__main__":
    main()
