#!/usr/bin/env python3
"""Shared helpers for the paper data registry."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_ROOT = Path("llama_cpp_canister/models")

# Keep the registry-facing names stable even when raw logs use a different style.
MODEL_NAME_ALIASES = {
    "GPT-2-124M": "GPT2-124M",
    "GPT2-124M": "GPT2-124M",
    "Gemma3-270M": "Gemma3-270M-IT",
    "Gemma3-270M-IT": "Gemma3-270M-IT",
    "Gemma3-1B": "Gemma3-1B-IT",
    "Gemma3-1B-IT": "Gemma3-1B-IT",
    "Llama-3.2-1B": "Llama-3.2-1B",
    "Qwen-2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen2.5-0.5B": "Qwen2.5-0.5B",
}

# Only include stems that are known-good. Rows without a confident GGUF match stay blank.
GGUF_STEM_BY_ID = {
    "M02": "pythia-14m-Q4_0",
    "M03": "IPythia-70m.Q8_0",
    "M04": "pythia-70m-Q4_0",
    "M05": "pythia-70m-Q4_K_M",
    "M06": "pythia-70m-Q6_K",
    "M07": "IPythia-70m.Q8_0",
    "M08": "pythia-70m-TQ2_0",
    "M09": "pythia-70m-TQ2_0",
    "M10": "distilgpt2.Q8_0",
    "M11": "distilgpt2.Q4_0",
    "M12": "Falcon-H1-Tiny-90M-Instruct-Q8_0",
    "M13": "gpt2.Q8_0",
    "M14": "gpt2.Q4_0",
    "M16": "smollm2-135m-Q4_0",
    "M17": "smollm2-135m-Q4_K_M",
    "M19": "smollm2-135m-TQ2_0",
    "M20": "smollm2-135m-TQ2_0",
    "M21": "smollm2-135m-instruct-q8_0",
    "M22": "amd-llama-135m-q8_0",
    "M23": "gemma3-270m-it-q8_0",
    "M24": "gemma3-270m-it-q4_0",
    "M25": "OpenELM-270M.Q4_0",
    "M27": "SmolLM2-360M.Q4_0",
    "M28": "mamba-370m-q8_0",
    "M29": "rwkv7-0.4b-world-q8_0",
    "M30": "qwen2.5-0.5b-instruct-q8_0",
    "M31": "qwen2.5-0.5b-Q4_0",
    "M32": "qwen2.5-0.5b-Q4_K_M",
    "M33": "qwen2.5-0.5b-instruct-q8_0",
    "M34": "qwen2.5-0.5b-instruct-q8_0",
    "M35": "qwen2.5-0.5b-TQ2_0",
    "M36": "qwen2.5-0.5b-TQ2_0",
    "M37": "h2o-danube3-500m-chat.Q4_0",
    "M38": "h2o-danube3-500m-chat.Q8_0",
    "M39": "bloomz-560m-q3_k_m",
    "M40": "qwen3-0.6b-q8_0",
    "M41": "qwen3.5-0.8b-q4_0",
    "M42": "qwen3.5-0.8b-tq2_0",
    "M43": "qwen3.5-0.8b-q4_0-30L",
    "M44": "qwen3.5-0.8b-q4_0-24L",
    "M45": "qwen3.5-0.8b-tq2_0-24L",
    "M46": "qwen3.5-0.8b-iq2_xxs",
    "M47": "embeddinggemma-300m-q4_0",
    "M48": "gemma3-1b-it-q2_k",
    "M49": "llama-3.2-1b-instruct-q4_0",
}

STATUS_TAG_KEYS = ["runs", "quality", "env", "build", "role"]
STATUS_TAG_ATOMS = {
    "runs": ["repeated", "3_reps", "single_run", "10_turns_multicall", "not_applicable"],
    "quality": ["verified", "limited", "invalid", "not_checked"],
    "env": ["local", "icp_mainnet", "ssn_mainnet", "native_only"],
    "build": ["current", "historical", "build_sensitive", "build_specific", "phase_2_wasm", "unknown"],
    "role": [
        "calibration",
        "supporting_measurement",
        "network_validation",
        "systems_demo",
        "negative_result",
        "c2_cross_env_validation",
        "c3_multicall_ssn_validation",
        "baseline",
        "native_only",
    ],
}


def normalize_model_name(name: str) -> str:
    return MODEL_NAME_ALIASES.get(name, name)


def load_csv_lenient(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for raw_row in reader:
            row = list(raw_row)
            if len(row) > len(header):
                row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
            elif len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append(dict(zip(header, row)))
    return rows


def scan_row_lengths(path: Path) -> tuple[int, list[tuple[int, int]]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        width = len(header)
        mismatches = []
        for lineno, row in enumerate(reader, start=2):
            if len(row) != width:
                mismatches.append((lineno, len(row)))
    return width, mismatches


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_number(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.3f}".rstrip("0").rstrip(".")


def join_note_parts(*parts: str) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = clean_text(part).strip(" .")
        if not text:
            continue
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return ". ".join(cleaned)


def join_status_atoms(values: list[str], key: str) -> str:
    ordered = [value for value in STATUS_TAG_ATOMS[key] if value in values]
    if ordered:
        return "+".join(ordered)

    # Fall back to insertion order if a caller passes an unknown-yet-useful value.
    seen: set[str] = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return "+".join(ordered)


def format_status_tags(**values: str) -> str:
    parts = []
    for key in STATUS_TAG_KEYS:
        value = clean_text(values.get(key))
        if not value:
            continue
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def validate_status_tags(text: str) -> str | None:
    cleaned = clean_text(text)
    if not cleaned:
        return "missing status_tags"

    parts = [part.strip() for part in cleaned.split(";") if part.strip()]
    seen: set[str] = set()
    parsed_keys: list[str] = []

    for part in parts:
        if "=" not in part:
            return f"invalid status tag fragment '{part}'"
        key, value = [piece.strip() for piece in part.split("=", 1)]
        if key not in STATUS_TAG_KEYS:
            return f"unknown status tag key '{key}'"
        if key in seen:
            return f"duplicate status tag key '{key}'"
        if not value:
            return f"empty value for status tag key '{key}'"
        atoms = [atom.strip() for atom in value.split("+") if atom.strip()]
        if not atoms:
            return f"empty composite value for status tag key '{key}'"
        unknown = [atom for atom in atoms if atom not in STATUS_TAG_ATOMS[key]]
        if unknown:
            return f"unknown value(s) for {key}: {', '.join(unknown)}"
        seen.add(key)
        parsed_keys.append(key)

    if parsed_keys != STATUS_TAG_KEYS:
        return f"expected keys {', '.join(STATUS_TAG_KEYS)} in order"

    return None


def parse_registry_id(value: str) -> int:
    return int(value.lstrip("M"))


def rel_gguf_path(full_path: str) -> str:
    path = Path(full_path)
    try:
        return str(path.relative_to(MODELS_ROOT))
    except ValueError:
        return str(path)


def load_gguf_index() -> dict[str, dict[str, str]]:
    rows = load_csv_lenient(DATA_DIR / "verified" / "gguf_metadata.csv")
    return {Path(row["file"]).stem: row for row in rows if row.get("file")}
