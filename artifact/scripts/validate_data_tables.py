#!/usr/bin/env python3
"""Validate the paper data tables for schema and cross-table coherence."""

from __future__ import annotations

from pathlib import Path

from registry_utils import (
    DATA_DIR,
    GGUF_STEM_BY_ID,
    RESULTS_DIR,
    clean_text,
    load_csv_lenient,
    load_gguf_index,
    normalize_model_name,
    parse_registry_id,
    scan_row_lengths,
    validate_status_tags,
)

MODELS_CSV = DATA_DIR / "models.csv"
LOCAL_CSV = DATA_DIR / "onchain" / "local.csv"
ICP_CSV = DATA_DIR / "onchain" / "icp_mainnet.csv"
SSN_CSV = DATA_DIR / "onchain" / "ssn_mainnet.csv"
NATIVE_CSV = DATA_DIR / "native" / "m4_max_baseline.csv"
QUALITY_CSV = DATA_DIR / "quality_audit.csv"
CORE_V2_CSV = RESULTS_DIR / "raw" / "core_measurements_v2.csv"

SCHEMA_PATHS = [
    MODELS_CSV,
    LOCAL_CSV,
    ICP_CSV,
    SSN_CSV,
    NATIVE_CSV,
    QUALITY_CSV,
    CORE_V2_CSV,
]


def make_key(name: str, quant: str, simd: str = "", build: str = "") -> tuple[str, str, str, str]:
    return (normalize_model_name(name), quant, simd, build)


def maybe_equal(left: str, right: str) -> bool:
    if clean_text(left) == clean_text(right):
        return True
    return False


def maybe_equal_numeric(left: str, right: str) -> bool:
    if maybe_equal(left, right):
        return True
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for path in SCHEMA_PATHS:
        _, mismatches = scan_row_lengths(path)
        if mismatches:
            errors.append(
                f"{path.name}: inconsistent CSV widths at "
                + ", ".join(f"line {lineno} ({width} cols)" for lineno, width in mismatches[:5])
            )

    models_rows = sorted(load_csv_lenient(MODELS_CSV), key=lambda row: parse_registry_id(row["id"]))
    local_rows = load_csv_lenient(LOCAL_CSV)
    icp_rows = load_csv_lenient(ICP_CSV)
    ssn_rows = load_csv_lenient(SSN_CSV)
    core_rows = load_csv_lenient(CORE_V2_CSV)
    gguf_index = load_gguf_index()

    ids = [row["id"] for row in models_rows]
    if len(ids) != len(set(ids)):
        errors.append("models.csv: duplicate registry ids detected")
    expected = [f"M{index:02d}" for index in range(1, len(ids) + 1)]
    if ids != expected:
        warnings.append("models.csv: ids are not a contiguous M01..Mn sequence")

    local_index = {make_key(row["model"], row["quant"], row["simd"], row["build"]): row for row in local_rows}

    icp_index = {
        make_key(row["model"], row["quant"], build=row["build"]): row
        for row in icp_rows
    }

    ssn_index = {}
    for row in ssn_rows:
        key = (normalize_model_name(row["model"]), row["quant"], clean_text(row.get("layers_real")))
        ssn_index[key] = row

    core_local_rows = [row for row in core_rows if row.get("network") == "local"]
    core_local_index = {
        make_key(row["model"], row["quant"], row["simd"], row["build"]): row
        for row in core_local_rows
    }

    for row in models_rows:
        status_error = validate_status_tags(row.get("status_tags", ""))
        if status_error:
            errors.append(f"{row['id']}: {status_error}")

    for index, row in enumerate(local_rows, start=1):
        status_error = validate_status_tags(row.get("status_tags", ""))
        if status_error:
            errors.append(f"local.csv row {index}: {status_error}")
        elif "env=local" not in row["status_tags"]:
            errors.append(f"local.csv row {index}: env tag must be local")

    for index, row in enumerate(icp_rows, start=1):
        status_error = validate_status_tags(row.get("status_tags", ""))
        if status_error:
            errors.append(f"icp_mainnet.csv row {index}: {status_error}")
        elif "env=icp_mainnet" not in row["status_tags"]:
            errors.append(f"icp_mainnet.csv row {index}: env tag must be icp_mainnet")

    for index, row in enumerate(ssn_rows, start=1):
        status_error = validate_status_tags(row.get("status_tags", ""))
        if status_error:
            errors.append(f"ssn_mainnet.csv row {index}: {status_error}")
        elif "env=ssn_mainnet" not in row["status_tags"]:
            errors.append(f"ssn_mainnet.csv row {index}: env tag must be ssn_mainnet")

    for row in models_rows:
        row_id = row["id"]
        local_key = make_key(row["name"], row["quant"], row["simd"], row["build"])
        weak_local_key = make_key(row["name"], row["quant"], row["simd"], "")
        model_local = clean_text(row["tok_call_local"])
        model_icp = clean_text(row["tok_call_icp"])
        model_ssn = clean_text(row["tok_call_ssn"])

        if model_local:
            local_row = local_index.get(local_key)
            if not local_row:
                local_row = next((candidate for key, candidate in local_index.items() if key[:3] == weak_local_key[:3]), None)
            if not local_row:
                errors.append(f"{row_id}: models.csv local throughput has no matching onchain/local.csv row")
            else:
                if not maybe_equal(model_local, local_row["tok_call"]):
                    errors.append(f"{row_id}: local tok/call mismatch models={model_local} local.csv={local_row['tok_call']}")
                if clean_text(row["alpha_eff"]) and clean_text(local_row["alpha_eff"]) and not maybe_equal(row["alpha_eff"], local_row["alpha_eff"]):
                    errors.append(f"{row_id}: alpha mismatch models={row['alpha_eff']} local.csv={local_row['alpha_eff']}")

            core_row = core_local_index.get(local_key)
            if not core_row:
                core_row = next((candidate for key, candidate in core_local_index.items() if key[:3] == weak_local_key[:3]), None)
            if not core_row:
                errors.append(f"{row_id}: models.csv local throughput has no matching core_measurements_v2 row")
            else:
                if not maybe_equal(model_local, core_row["tok_per_call"]):
                    errors.append(f"{row_id}: local tok/call mismatch models={model_local} core_measurements_v2={core_row['tok_per_call']}")

        if model_icp:
            icp_row = icp_index.get(make_key(row["name"], row["quant"], build=row["build"]))
            if not icp_row:
                icp_row = next(
                    (
                        candidate
                        for key, candidate in icp_index.items()
                        if key[0] == normalize_model_name(row["name"]) and key[1] == row["quant"]
                    ),
                    None,
                )
            if not icp_row:
                errors.append(f"{row_id}: models.csv ICP throughput has no matching icp_mainnet.csv row")
            elif not maybe_equal(model_icp, icp_row["tok_call_gen"]):
                errors.append(f"{row_id}: ICP tok/call mismatch models={model_icp} icp_mainnet.csv={icp_row['tok_call_gen']}")

        if model_ssn:
            ssn_key = (normalize_model_name(row["name"]), row["quant"], clean_text(row["layers_real"]))
            ssn_row = ssn_index.get(ssn_key)
            if not ssn_row:
                ssn_row = next(
                    (
                        candidate
                        for key, candidate in ssn_index.items()
                        if key[0] == normalize_model_name(row["name"]) and key[1] == row["quant"]
                    ),
                    None,
                )
            if not ssn_row:
                errors.append(f"{row_id}: models.csv SSN throughput has no matching ssn_mainnet.csv row")
            elif not maybe_equal(model_ssn, ssn_row["tok_call"]):
                errors.append(f"{row_id}: SSN tok/call mismatch models={model_ssn} ssn_mainnet.csv={ssn_row['tok_call']}")

        gguf_path = clean_text(row["gguf_path"])
        if gguf_path.endswith(".gguf"):
            stem = Path(gguf_path).stem
            meta = gguf_index.get(stem)
            if not meta:
                errors.append(f"{row_id}: gguf_path does not resolve in verified/gguf_metadata.csv ({gguf_path})")
            else:
                for field, meta_field in [
                    ("layers_real", "block_count"),
                    ("hidden_dim", "embedding_length"),
                    ("d_ff", "feed_forward_length"),
                    ("vocab_size", "vocab_size"),
                    ("params_M_gguf", "params_M_calc"),
                ]:
                    row_value = clean_text(row.get(field))
                    meta_value = clean_text(meta[meta_field])
                    if row_value and meta_value and not maybe_equal_numeric(row_value, meta_value):
                        errors.append(f"{row_id}: {field}={row_value} but GGUF metadata says {meta_value}")
        elif row_id in GGUF_STEM_BY_ID:
            warnings.append(f"{row_id}: expected a GGUF-backed row but gguf_path is blank")

        if normalize_model_name(row["name"]) == "EmbeddingGemma-300M" and clean_text(row["native_gen_tok_s"]):
            errors.append(f"{row_id}: embedding row should not have native generation tok/s populated")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  - {warning}")
        return 1

    print("Validation passed.")
    print(f"Checked {len(models_rows)} registry rows and {len(core_local_rows)} local core rows.")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
