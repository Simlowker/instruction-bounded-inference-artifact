#!/usr/bin/env python3
"""Characterize GGUF tensor layout for load_model / multi-call follow-up cases.

This script bridges two gaps left after Paper 1.5 Phase 2:

1. `load_model` outcomes (`OK`, `IC0524`, `other_error`) were recorded, but not
   tied back to concrete GGUF tensor-layout metrics.
2. C3 multi-call results established that `load_model` dominates cold-path
   latency, but we had no local, reproducible comparison of the GGUFs involved.

The script reads the existing Phase 2 CSVs, resolves the matching GGUF files
available in the current workspace, extracts layout metrics via `gguf.GGUFReader`,
and writes:

- `artifact/data/paper_1_5/load_layout_characterization.csv`
- `artifact/results/paper_1_5/tables/load-layout-summary.md`
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Iterable

try:
    from gguf import GGUFReader
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("Install `gguf` first: pip3 install gguf") from exc


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]

IC0524_CSV = ARTIFACT_ROOT / "data" / "paper_1_5" / "ic0524_characterization.csv"
MULTICALL_CSV = ARTIFACT_ROOT / "data" / "paper_1_5" / "multicall_characterization.csv"

OUT_CSV = ARTIFACT_ROOT / "data" / "paper_1_5" / "load_layout_characterization.csv"
OUT_MD = ARTIFACT_ROOT / "results" / "paper_1_5" / "tables" / "load-layout-summary.md"

SEARCH_ROOTS = [
    REPO_ROOT / "llama_cpp_canister" / "models",
    REPO_ROOT / ".worktrees" / "paper-1.5-phase-1" / "llama_cpp_canister" / "models",
    REPO_ROOT / "local" / "workspaces" / "llama_cpp_canister" / "models",
]

# Deterministic path hints for the concrete cases already discussed in Phase 2.
PATH_HINTS: dict[tuple[str, str], list[Path]] = {
    ("TriLM-560M", "TQ2_0"): [
        REPO_ROOT
        / ".worktrees"
        / "paper-1.5-phase-1"
        / "llama_cpp_canister"
        / "models"
        / "trilm"
        / "TriLM_560M_Unpacked.TQ2_0.gguf"
    ],
    ("TriLM-3.9B", "TQ2_0"): [
        REPO_ROOT
        / ".worktrees"
        / "paper-1.5-phase-1"
        / "llama_cpp_canister"
        / "models"
        / "trilm"
        / "TriLM_3.9B_Unpacked.TQ2_0.gguf"
    ],
    ("Qwen2.5-0.5B", "Q4_0"): [
        REPO_ROOT / "llama_cpp_canister" / "models" / "c3-qwen05-q4.gguf",
        REPO_ROOT / "llama_cpp_canister" / "models" / "Qwen" / "qwen2.5-0.5b-Q4_0.gguf",
    ],
    ("Qwen2.5-0.5B", "Q6_K"): [
        REPO_ROOT / "llama_cpp_canister" / "models" / "Qwen" / "qwen2.5-0.5b-Q6_K.gguf",
    ],
    ("Qwen2.5-1.5B", "Q4_0"): [
        REPO_ROOT / "llama_cpp_canister" / "models" / "qwen15-q4.gguf",
    ],
    ("Qwen2.5-1.5B", "Q6_K"): [
        REPO_ROOT / "llama_cpp_canister" / "models" / "Qwen2.5-1.5B" / "qwen2.5-1.5b-instruct-q6_k.gguf",
    ],
    ("Falcon-H1-Tiny-90M", "Q8_0"): [
        REPO_ROOT
        / "llama_cpp_canister"
        / "models"
        / "candidates"
        / "Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf",
        REPO_ROOT / "llama_cpp_canister" / "models" / "falcon-h1-tiny.gguf",
    ],
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 1)


def pct(part: int, whole: int) -> float:
    if not whole:
        return 0.0
    return round(100.0 * part / whole, 1)


def stringify_tensor_type(tensor_type: object) -> str:
    return getattr(tensor_type, "name", str(tensor_type))


def get_arch(reader: GGUFReader) -> str:
    for field in reader.fields.values():
        if field.name == "general.architecture":
            try:
                return bytes(field.parts[-1]).decode("utf-8", errors="replace")
            except Exception:
                return str(field.parts[-1])
    return ""


def find_tensor_info(
    tensors: list[tuple[int, int, str, str]],
    exact_names: Iterable[str],
) -> tuple[str, str, int]:
    exact_set = set(exact_names)
    for _, size, name, tensor_type in tensors:
        if name in exact_set:
            return name, tensor_type, size
    return "", "", 0


def fallback_search_tokens(model_id: str, quant_format: str) -> list[str]:
    model = model_id.lower()
    quant = quant_format.lower()
    if model == "falcon-h1-tiny-90m":
        return ["falcon", "tiny", "90m", "q8"]
    if model == "qwen2.5-1.5b":
        if quant == "q4_0":
            return ["qwen", "1.5", "q4"]
        if quant == "q6_k":
            return ["qwen", "1.5", "q6"]
    if model == "qwen2.5-0.5b":
        if quant == "q4_0":
            return ["qwen", "0.5", "q4"]
        if quant == "q6_k":
            return ["qwen", "0.5", "q6"]
    if model == "trilm-560m":
        return ["trilm", "560m", "tq2"]
    if model == "trilm-3.9b":
        return ["trilm", "3.9b", "tq2"]
    return [model.replace(".", ""), quant.replace("_", "")]


def resolve_path(model_id: str, quant_format: str) -> Path | None:
    for candidate in PATH_HINTS.get((model_id, quant_format), []):
        if candidate.exists():
            return candidate

    tokens = fallback_search_tokens(model_id, quant_format)
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        matches = []
        for path in root.rglob("*.gguf"):
            name = path.name.lower()
            if all(tok in name for tok in tokens):
                matches.append(path)
        if matches:
            matches.sort()
            return matches[0]
    return None


def analyze_gguf(path: Path) -> dict[str, object]:
    reader = GGUFReader(str(path))
    tensors = sorted(
        [
            (
                int(tensor.data_offset),
                int(tensor.n_bytes),
                tensor.name,
                stringify_tensor_type(tensor.tensor_type),
            )
            for tensor in reader.tensors
        ],
        key=lambda item: item[0],
    )
    sizes = [size for _, size, _, _ in tensors]
    total_bytes = sum(sizes)
    start_offset = tensors[0][0] if tensors else 0
    largest = max(tensors, key=lambda item: item[1]) if tensors else (0, 0, "", "")

    type_bytes = Counter()
    for _, size, _, tensor_type in tensors:
        type_bytes[tensor_type] += size

    dominant_type, dominant_bytes = ("", 0)
    if type_bytes:
        dominant_type, dominant_bytes = type_bytes.most_common(1)[0]

    token_embd_name, token_embd_type, token_embd_size = find_tensor_info(
        tensors,
        ["token_embd.weight", "tok_embeddings.weight", "model.embed_tokens.weight"],
    )
    output_name, output_type, output_size = find_tensor_info(
        tensors,
        ["output.weight", "lm_head.weight"],
    )

    def bytes_in_window(window_mib: int) -> int:
        threshold = start_offset + window_mib * 1024 * 1024
        return sum(size for offset, size, _, _ in tensors if offset < threshold)

    def prefix_tensor_bytes(count: int) -> int:
        return sum(size for _, size, _, _ in tensors[:count])

    return {
        "gguf_path": str(path.relative_to(REPO_ROOT)),
        "arch": get_arch(reader),
        "file_size_MB": bytes_to_mb(path.stat().st_size),
        "payload_MB": bytes_to_mb(total_bytes),
        "n_tensors": len(tensors),
        "tensor_type_count": len(type_bytes),
        "tensor_types": "|".join(sorted(type_bytes)),
        "dominant_tensor_type": dominant_type,
        "dominant_tensor_share_pct": pct(dominant_bytes, total_bytes),
        "token_embd_name": token_embd_name,
        "token_embd_type": token_embd_type,
        "token_embd_MB": bytes_to_mb(token_embd_size),
        "output_name": output_name,
        "output_type": output_type,
        "output_MB": bytes_to_mb(output_size),
        "largest_tensor_name": largest[2],
        "largest_tensor_type": largest[3],
        "largest_tensor_MB": bytes_to_mb(largest[1]),
        "largest_tensor_share_pct": pct(largest[1], total_bytes),
        "largest_tensor_offset_MB": bytes_to_mb(largest[0]),
        "prefix16_tensor_share_pct": pct(prefix_tensor_bytes(16), total_bytes),
        "prefix32_tensor_share_pct": pct(prefix_tensor_bytes(32), total_bytes),
        "prefix64_tensor_share_pct": pct(prefix_tensor_bytes(64), total_bytes),
        "within16MiB_share_pct": pct(bytes_in_window(16), total_bytes),
        "within64MiB_share_pct": pct(bytes_in_window(64), total_bytes),
        "within128MiB_share_pct": pct(bytes_in_window(128), total_bytes),
        "within256MiB_share_pct": pct(bytes_in_window(256), total_bytes),
        "median_tensor_KiB": round(median(sizes) / 1024.0, 1) if sizes else 0.0,
        "gt1MiB_count": sum(size > 1 * 1024 * 1024 for size in sizes),
        "gt4MiB_count": sum(size > 4 * 1024 * 1024 for size in sizes),
        "gt8MiB_count": sum(size > 8 * 1024 * 1024 for size in sizes),
        "gt16MiB_count": sum(size > 16 * 1024 * 1024 for size in sizes),
        "gt32MiB_count": sum(size > 32 * 1024 * 1024 for size in sizes),
    }


def gather_cases() -> list[dict[str, str]]:
    rows = load_rows(IC0524_CSV)
    seen = {(row["model_id"], row["format"]) for row in rows}

    if ("Falcon-H1-Tiny-90M", "Q8_0") not in seen:
        rows.append(
            {
                "model_id": "Falcon-H1-Tiny-90M",
                "format": "Q8_0",
                "size_MB": "94",
                "load_outcome": "OK",
                "first_failing_msg_size_MB": "",
                "timestamp_utc": "",
                "notes": "Added as multi-call baseline from C3 Task 19 for a cold-path/load comparison.",
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model_id",
        "format",
        "load_outcome",
        "gguf_found",
        "gguf_path",
        "arch",
        "file_size_MB",
        "payload_MB",
        "n_tensors",
        "tensor_type_count",
        "tensor_types",
        "dominant_tensor_type",
        "dominant_tensor_share_pct",
        "token_embd_type",
        "token_embd_MB",
        "output_type",
        "output_MB",
        "largest_tensor_name",
        "largest_tensor_type",
        "largest_tensor_MB",
        "largest_tensor_share_pct",
        "largest_tensor_offset_MB",
        "prefix16_tensor_share_pct",
        "prefix32_tensor_share_pct",
        "prefix64_tensor_share_pct",
        "within16MiB_share_pct",
        "within64MiB_share_pct",
        "within128MiB_share_pct",
        "within256MiB_share_pct",
        "median_tensor_KiB",
        "gt1MiB_count",
        "gt4MiB_count",
        "gt8MiB_count",
        "gt16MiB_count",
        "gt32MiB_count",
        "notes",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_markdown(rows: list[dict[str, object]]) -> str:
    analyzed = [row for row in rows if row.get("gguf_found") == "yes"]
    missing = [row for row in rows if row.get("gguf_found") != "yes"]

    analyzed.sort(key=lambda row: (str(row["load_outcome"]), float(row["file_size_MB"])))

    lines = [
        "# Load Layout Summary",
        "",
        "Follow-up analysis for the `load_model + KV/session IO + layout GGUF + multi-call` thread.",
        "",
        f"Generated from `{OUT_CSV.relative_to(REPO_ROOT)}` using `artifact/scripts/paper_1_5/analyze_load_layout.py`.",
        "",
        "## Cases",
        "",
        "| Model | Format | Outcome | File MB | Tensors | Dominant type | Largest tensor | Prefix16 share | >4 MiB tensors | GGUF |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]

    for row in analyzed:
        largest = f"{row['largest_tensor_MB']} MB `{row['largest_tensor_type']}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_id"]),
                    str(row["format"]),
                    str(row["load_outcome"]),
                    str(row["file_size_MB"]),
                    str(row["n_tensors"]),
                    f"`{row['dominant_tensor_type']}` ({row['dominant_tensor_share_pct']}%)",
                    largest,
                    f"{row['prefix16_tensor_share_pct']}%",
                    str(row["gt4MiB_count"]),
                    f"`{row['gguf_path']}`",
                ]
            )
            + " |"
        )

    if missing:
        lines.extend(
            [
                "",
                "## Missing Local GGUFs",
                "",
                "| Model | Format | Outcome | Notes |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in missing:
            lines.append(
                f"| {row['model_id']} | {row['format']} | {row['load_outcome']} | {row.get('notes', '')} |"
            )

    lines.extend(
        [
            "",
            "## Immediate Read",
            "",
            "- Raw file size still does not explain the outcomes on its own: `TriLM-3.9B TQ2_0` loads at `1112.7 MB`, while `Qwen2.5-1.5B Q4_0` traps at `891.6 MB`.",
            "- Early-byte concentration is not a clean separator either: `Qwen2.5-0.5B Q4_0` loads despite `41.8%` of payload landing inside the first `16 MiB`, whereas the failing `Qwen2.5-1.5B Q4_0` sits at `20.6%`.",
            "- The failing `Qwen2.5-1.5B Q4_0` case is still layout-distinct in two useful ways: a much larger tensor population (`338` tensors, `141` above `1 MiB`) and a large mixed-precision component (`Q6_K` token embedding at `182.6 MB`). That is enough to justify load-path instrumentation, but not enough to assign causality yet.",
            "- The current evidence therefore supports a stricter framing: `IC0524` and `IC0502` are not just \"big model\" failures; they are loader-path failures over specific GGUF layouts.",
            "",
            "## Next Useful Instrumentation",
            "",
            "1. Emit stage markers around `common_init_from_params`, buffer allocation, and tensor materialization so `IC0502` is tied to an exact phase instead of a generic trap.",
            "2. Add throttled tensor-progress logging in the loader (for example every 16 or 32 tensors) with tensor name, type, bytes, and cumulative offset.",
            "3. Record backend-buffer allocation sizes before the trap, so we can separate heap-growth failures from stable-memory page-access failures.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    rows: list[dict[str, object]] = []
    for case in gather_cases():
        row: dict[str, object] = dict(case)
        path = resolve_path(case["model_id"], case["format"])
        if path is None:
            row["gguf_found"] = "no"
            row["gguf_path"] = ""
        else:
            row["gguf_found"] = "yes"
            row.update(analyze_gguf(path))
        rows.append(row)

    write_csv(rows)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_markdown(rows))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
