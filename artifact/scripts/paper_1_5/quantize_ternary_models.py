#!/usr/bin/env python3
"""
quantize_ternary_models.py

Self-quantize TriLM base models from HuggingFace into TQ2_0 GGUF files for
Paper 1.5 C1a measurements. Used when no pre-quantized TQ2_0 GGUF is publicly
available on HuggingFace.

Pipeline per model:
  1. snapshot_download the base HF repo (safetensors + config) from SpectraSuite
  2. convert_hf_to_gguf.py → intermediate F16 GGUF
  3. llama-quantize → final TQ2_0 GGUF

Prerequisites (must already be installed on the host):
  - huggingface_hub (pip)
  - llama.cpp source checkout under .worktree-cache/llama_cpp_src (provides
    convert_hf_to_gguf.py + matching bundled gguf-py). Brew's convert script
    references MODEL_ARCH.GEMMA4 at module-load time, which the released
    pypi `gguf` library does not yet expose, so we source from upstream.
  - /opt/homebrew/bin/llama-quantize (llama.cpp via brew) for TQ2_0 kernel
  - .worktree-cache/quantize-venv (python venv with torch, transformers,
    safetensors, sentencepiece, protobuf, numpy, gguf, huggingface_hub)

Usage:
    python3 quantize_ternary_models.py 560M      # quantize 560M only
    python3 quantize_ternary_models.py 3.9B      # quantize 3.9B only
    python3 quantize_ternary_models.py           # all (560M + 3.9B)
"""
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
import subprocess
import sys
import os


REPO_ROOT = Path(__file__).resolve().parents[5]  # worktree root
STAGING_DIR = REPO_ROOT / ".worktree-cache" / "trilm-hf-staging"
OUTPUT_DIR = REPO_ROOT / "llama_cpp_canister" / "models" / "trilm"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VENV_PY = REPO_ROOT / ".worktree-cache" / "quantize-venv" / "bin" / "python3"
LLAMA_SRC = REPO_ROOT / ".worktree-cache" / "llama_cpp_src"
CONVERT_SCRIPT = LLAMA_SRC / "convert_hf_to_gguf.py"
QUANTIZE_BIN = Path("/opt/homebrew/bin/llama-quantize")

CANDIDATES = {
    "560M": {
        "repo_id": "SpectraSuite/TriLM_560M_Unpacked",
        "output_name": "TriLM_560M_Unpacked.TQ2_0.gguf",
    },
    "3.9B": {
        "repo_id": "SpectraSuite/TriLM_3.9B_Unpacked",
        "output_name": "TriLM_3.9B_Unpacked.TQ2_0.gguf",
    },
}


def check_tooling() -> None:
    """Fail fast if required tools are missing."""
    missing = []
    if not VENV_PY.is_file():
        missing.append(
            f"{VENV_PY} — create via:\n"
            "      python3 -m venv .worktree-cache/quantize-venv\n"
            "      .worktree-cache/quantize-venv/bin/pip install torch transformers "
            "safetensors sentencepiece protobuf numpy gguf huggingface_hub"
        )
    if not CONVERT_SCRIPT.is_file():
        missing.append(
            f"{CONVERT_SCRIPT} — clone upstream via:\n"
            "      git clone --depth=1 https://github.com/ggml-org/llama.cpp.git "
            ".worktree-cache/llama_cpp_src"
        )
    if not QUANTIZE_BIN.is_file():
        missing.append(f"{QUANTIZE_BIN} (install via: brew install llama.cpp)")
    if missing:
        print("[ERROR] Missing tooling:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)


def download_base(entry: dict, model_key: str) -> Path:
    """snapshot_download the HF base model into staging."""
    target = STAGING_DIR / model_key
    print(f"[download] {entry['repo_id']} → {target}")
    snapshot_download(
        repo_id=entry["repo_id"],
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return target


def convert_to_f16_gguf(hf_dir: Path, model_key: str) -> Path:
    """Run convert_hf_to_gguf.py → intermediate F16 GGUF in staging."""
    f16_out = STAGING_DIR / f"{model_key}.F16.gguf"
    cmd = [
        str(VENV_PY), str(CONVERT_SCRIPT),
        str(hf_dir),
        "--outfile", str(f16_out),
        "--outtype", "f16",
    ]
    print(f"[convert] {' '.join(cmd)}")
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid libomp duplicate-link crash on macOS
    subprocess.run(cmd, check=True, env=env)
    size_mb = os.path.getsize(f16_out) / (1024 * 1024)
    print(f"[convert] F16 GGUF → {f16_out} ({size_mb:.1f} MB)")
    return f16_out


def quantize_to_tq2_0(f16_gguf: Path, output_name: str) -> Path:
    """Run llama-quantize F16 → TQ2_0 into the final output dir."""
    out = OUTPUT_DIR / output_name
    cmd = [str(QUANTIZE_BIN), str(f16_gguf), str(out), "TQ2_0"]
    print(f"[quantize] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"[quantize] TQ2_0 GGUF → {out} ({size_mb:.1f} MB)")
    return out


def process(model_key: str) -> Path:
    if model_key not in CANDIDATES:
        raise ValueError(f"Unknown model {model_key}. Choices: {list(CANDIDATES.keys())}")
    entry = CANDIDATES[model_key]

    # Skip if already present
    final = OUTPUT_DIR / entry["output_name"]
    if final.is_file():
        size_mb = os.path.getsize(final) / (1024 * 1024)
        print(f"[skip] {final} already exists ({size_mb:.1f} MB)")
        return final

    hf_dir = download_base(entry, model_key)
    f16_gguf = convert_to_f16_gguf(hf_dir, model_key)
    try:
        return quantize_to_tq2_0(f16_gguf, entry["output_name"])
    finally:
        # Keep F16 artifact around? No — it's 2× the size of TQ2_0. Delete.
        if f16_gguf.exists():
            f16_gguf.unlink()
            print(f"[cleanup] removed intermediate {f16_gguf}")


def main():
    check_tooling()
    targets = sys.argv[1:] or list(CANDIDATES.keys())
    for t in targets:
        try:
            process(t)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {t}: external tool failed with exit {e.returncode}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] {t}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
