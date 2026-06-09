#!/usr/bin/env python3
"""Rebuild citation-facing data tables from canonical measurement sources."""

from __future__ import annotations

from pathlib import Path

from registry_utils import (
    DATA_DIR,
    GGUF_STEM_BY_ID,
    MODELS_ROOT,
    RESULTS_DIR,
    clean_text,
    format_status_tags,
    format_number,
    join_note_parts,
    join_status_atoms,
    load_csv_lenient,
    load_gguf_index,
    normalize_model_name,
    parse_registry_id,
    rel_gguf_path,
    write_csv,
)

MODELS_CSV = DATA_DIR / "models.csv"
LOCAL_CSV = DATA_DIR / "onchain" / "local.csv"
ICP_CSV = DATA_DIR / "onchain" / "icp_mainnet.csv"
SSN_CSV = DATA_DIR / "onchain" / "ssn_mainnet.csv"
NATIVE_CSV = DATA_DIR / "native" / "m4_max_baseline.csv"
QUALITY_AUDIT_CSV = DATA_DIR / "quality_audit.csv"
COHERENCE_MANUAL_CSV = DATA_DIR / "verified" / "coherence_manual.csv"
CORE_V2_CSV = RESULTS_DIR / "raw" / "core_measurements_v2.csv"

MODELS_FIELDS = [
    "id",
    "name",
    "arch",
    "family",
    "params_M",
    "params_M_gguf",
    "layers_real",
    "hidden_dim",
    "d_ff",
    "vocab_size",
    "quant",
    "simd",
    "size_MB",
    "gguf_path",
    "gguf_match",
    "tok_call_local",
    "tok_call_icp",
    "tok_call_ssn",
    "alpha_eff",
    "native_gen_tok_s",
    "native_prefill_tok_s",
    "build",
    "status_tags",
    "notes",
]

LOCAL_FIELDS = [
    "date",
    "model",
    "quant",
    "simd",
    "layers_real",
    "size_MB",
    "tok_call",
    "alpha_eff",
    "build",
    "measurement_type",
    "status_tags",
    "notes",
]

# Preserve legacy notes where they carry paper-facing caveats or manual audit context.
LEGACY_NOTE_IDS = {
    "M12",
    "M23",
    "M24",
    "M37",
    "M41",
    "M42",
    "M43",
    "M44",
    "M45",
    "M46",
    "M47",
    "M49",
}

VALID_BUILDS = {"gian", "upstream", "onicai"}

# Historical rows whose exact quantized GGUF is no longer on disk. For these,
# we keep a verifiable same-model metadata reference rather than leaving the
# registry provenance blank.
PROXY_GGUF_STEM_BY_ID = {
    "M01": "pythia-14m-Q4_0",
    "M15": "smollm2-135m-Q4_0",
    "M18": "smollm2-135m-Q4_0",
    "M26": "SmolLM2-360M.Q4_0",
}

ROW_FIELD_OVERRIDES = {
    "M12": {
        "notes": "SSN only — build-specific row not reproduced on current local gian build. Quality limited. Native 470 gen tok/s 1671 prefill tok/s (prompt=16)",
    },
    "M34": {
        "build": "onicai",
        "notes": "MAINNET baseline — gen@10 OK gen@11 TRAP",
    },
    "M41": {
        "build": "gian",
        "notes": "SSN only — crashes ICP mainnet. 18DN+6FA hybrid DeltaNet. BEST QUALITY (Rayleigh+Bern+CoT). Native 131 gen tok/s",
    },
}

SYSTEMS_DEMO_IDS = {"M08", "M09", "M19", "M20", "M35", "M36", "M47"}
NEGATIVE_RESULT_IDS = {"M42", "M43", "M44", "M45", "M46", "M48"}
BASELINE_IDS = {"M34"}
NATIVE_ONLY_IDS = {"M49"}
BUILD_SENSITIVE_IDS = {"M04", "M16", "M24", "M28", "M31"}
QUALITY_OVERRIDE_BY_ID = {
    "M43": "invalid",
    "M44": "invalid",
    "M45": "invalid",
    "M47": "verified",
}


def first_non_empty(*values: object) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def valid_build(value: object) -> str:
    build = clean_text(value)
    return build if build in VALID_BUILDS else ""


def find_core_record(core_rows: list[dict[str, str]], name: str, quant: str, simd: str, build: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in core_rows
        if normalize_model_name(row.get("model", "")) == name
        and row.get("quant") == quant
        and row.get("simd") == simd
    ]
    if not candidates:
        return None
    for row in candidates:
        if row.get("build") == build:
            return row
    if build == "onicai":
        return None
    preferred_build_order = ["gian", "upstream", "onicai"]
    for preferred in preferred_build_order:
        for row in candidates:
            if row.get("build") == preferred:
                return row
    return candidates[0]


def find_network_record(rows: list[dict[str, str]], name: str, quant: str, layers_real: str = "", build: str = "") -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if normalize_model_name(row.get("model", "")) == name and row.get("quant") == quant
    ]
    if build:
        exact_build = [row for row in candidates if row.get("build", build) == build]
        if exact_build:
            candidates = exact_build
    if layers_real:
        exact_layers = [row for row in candidates if clean_text(row.get("layers_real")) == layers_real]
        if exact_layers:
            candidates = exact_layers
    if len(candidates) == 1:
        return candidates[0]
    return candidates[0] if candidates else None


def find_native_record(rows: list[dict[str, str]], name: str, quant: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if normalize_model_name(row.get("model", "")) == name
        and row.get("quant") == quant
        and row.get("prompt_len") == "16"
    ]
    return candidates[0] if candidates else None


def find_existing_local_record(rows: list[dict[str, str]], name: str, quant: str, simd: str, build: str) -> dict[str, str] | None:
    candidates = [
        row
        for row in rows
        if normalize_model_name(row.get("model", "")) == name
        and row.get("quant") == quant
        and row.get("simd") == simd
    ]
    if not candidates:
        return None
    for row in candidates:
        if row.get("build") == build:
            return row
    if build == "onicai":
        return None
    return candidates[0]


def resolve_gguf(
    legacy_row: dict[str, str],
    gguf_index: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str] | None, str]:
    stem = GGUF_STEM_BY_ID.get(legacy_row["id"])
    if stem and stem in gguf_index:
        meta = gguf_index[stem]
        return rel_gguf_path(meta["file"]), meta, "exact"
    gguf_path = clean_text(legacy_row.get("gguf_path"))
    if gguf_path.endswith(".gguf"):
        stem = Path(gguf_path).stem
        meta = gguf_index.get(stem)
        if meta:
            return rel_gguf_path(meta["file"]), meta, "exact"
    proxy_stem = PROXY_GGUF_STEM_BY_ID.get(legacy_row["id"])
    if proxy_stem and proxy_stem in gguf_index:
        meta = gguf_index[proxy_stem]
        return rel_gguf_path(meta["file"]), meta, "proxy_same_model_quant"
    return gguf_path, None, ""


def build_model_notes(
    legacy_row: dict[str, str],
    local_row: dict[str, str] | None,
    icp_row: dict[str, str] | None,
    ssn_row: dict[str, str] | None,
    native_row: dict[str, str] | None,
    gguf_match: str,
) -> str:
    override_note = ROW_FIELD_OVERRIDES.get(legacy_row["id"], {}).get("notes", "")
    if override_note:
        base = override_note
    else:
        legacy_note = clean_text(legacy_row.get("notes"))
        if legacy_row["id"] in LEGACY_NOTE_IDS or local_row is None:
            base = legacy_note or first_non_empty(
                ssn_row.get("notes") if ssn_row else "",
                icp_row.get("notes") if icp_row else "",
                local_row.get("notes") if local_row else "",
            )
        else:
            base = first_non_empty(local_row.get("notes") if local_row else "", legacy_note)

    extras = []
    if ssn_row and "SSN=" not in base and normalize_model_name(legacy_row["name"]) != "EmbeddingGemma-300M":
        extras.append(f"SSN={ssn_row['tok_call']}")
    if icp_row and "ICP=" not in base and "input tok/call" not in base:
        extras.append(f"ICP={icp_row['tok_call_gen']}")
    if native_row and "native bench done" not in base.lower():
        extras.append("Native bench done")
    if gguf_match == "proxy_same_model_quant":
        extras.append("GGUF metadata proxied from same-model alternate quant; exact file not preserved")
    return join_note_parts(base, *extras)


def build_local_note(local_row: dict[str, str]) -> str:
    return clean_text(local_row.get("notes"))


def build_local_date(legacy_row: dict[str, str], local_row: dict[str, str], existing_row: dict[str, str] | None) -> str:
    if clean_text(local_row.get("n_runs")) not in {"", "1"}:
        return "2026-04"
    if existing_row:
        return clean_text(existing_row.get("date"))
    if legacy_row["id"] in {"M23", "M24", "M28", "M29", "M40"}:
        return "2026-04"
    return "2026-03"


def build_quality_indices(
    audit_rows: list[dict[str, str]],
    coherence_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[tuple[str, str], list[dict[str, str]]], dict[tuple[str, str, str], dict[str, str]]]:
    audit_exact: dict[tuple[str, str, str], dict[str, str]] = {}
    audit_loose: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in audit_rows:
        key_exact = (
            normalize_model_name(row.get("model", "")),
            row.get("quant", ""),
            clean_text(row.get("layers")),
        )
        audit_exact[key_exact] = row
        key_loose = (normalize_model_name(row.get("model", "")), row.get("quant", ""))
        audit_loose.setdefault(key_loose, []).append(row)

    coherence_exact: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in coherence_rows:
        coherence_exact[
            (
                normalize_model_name(row.get("model_tag", "")),
                row.get("quant", ""),
                row.get("simd", ""),
            )
        ] = row

    return audit_exact, audit_loose, coherence_exact


def find_quality_record(
    audit_exact: dict[tuple[str, str, str], dict[str, str]],
    audit_loose: dict[tuple[str, str], list[dict[str, str]]],
    name: str,
    quant: str,
    layers_real: str,
) -> dict[str, str] | None:
    exact = audit_exact.get((name, quant, clean_text(layers_real)))
    if exact:
        return exact

    candidates = audit_loose.get((name, quant), [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def classify_quality_from_audit(row: dict[str, str]) -> str:
    score = clean_text(row.get("quality_score"))
    notes = clean_text(row.get("quality_notes")).lower()
    task_blob = " ".join(
        clean_text(row.get(field)).lower()
        for field in [
            "qa_capital_france",
            "reasoning_sky_blue",
            "sentiment_negative",
            "fr_capitale_suisse",
        ]
    )

    if "not_tested" in task_blob or "not tested" in notes:
        return "not_checked"
    if "invalid" in notes or "garbage" in notes:
        return "invalid"
    if score.startswith("0/"):
        return "invalid"
    if score.startswith("1/") or score.startswith("2/") or "classification+qa only" in notes:
        return "limited"
    if score:
        return "verified"
    return "not_checked"


def classify_quality_from_coherence(row: dict[str, str]) -> str:
    label = clean_text(row.get("manual_label")).lower()
    if label == "coherent":
        return "verified"
    if label == "degraded":
        return "invalid"
    return "not_checked"


def quality_status_for_row(
    legacy_row: dict[str, str],
    layers_real: str,
    audit_exact: dict[tuple[str, str, str], dict[str, str]],
    audit_loose: dict[tuple[str, str], list[dict[str, str]]],
    coherence_exact: dict[tuple[str, str, str], dict[str, str]],
) -> str:
    override = QUALITY_OVERRIDE_BY_ID.get(legacy_row["id"])
    if override:
        return override

    audit_row = find_quality_record(
        audit_exact,
        audit_loose,
        normalize_model_name(legacy_row["name"]),
        legacy_row["quant"],
        layers_real,
    )
    if audit_row:
        return classify_quality_from_audit(audit_row)

    coherence_row = coherence_exact.get(
        (
            normalize_model_name(legacy_row["name"]),
            legacy_row["quant"],
            legacy_row["simd"],
        )
    )
    if coherence_row:
        return classify_quality_from_coherence(coherence_row)

    return "not_checked"


def model_runs_tag(local_source: dict[str, str] | None, icp_row: dict[str, str] | None, ssn_row: dict[str, str] | None, native_row: dict[str, str] | None) -> str:
    atoms: list[str] = []
    if local_source:
        if clean_text(local_source.get("n_runs")) not in {"", "1"}:
            atoms.append("repeated")
        else:
            atoms.append("single_run")
    if icp_row or ssn_row:
        atoms.append("single_run")
    if not atoms and native_row:
        atoms.append("not_applicable")
    return join_status_atoms(atoms or ["not_applicable"], "runs")


def local_runs_tag(local_source: dict[str, str]) -> str:
    if clean_text(local_source.get("n_runs")) not in {"", "1"}:
        return "repeated"
    return "single_run"


def model_env_tag(local_tok: str, icp_tok: str, ssn_tok: str, native_row: dict[str, str] | None) -> str:
    atoms: list[str] = []
    if clean_text(local_tok):
        atoms.append("local")
    if clean_text(icp_tok):
        atoms.append("icp_mainnet")
    if clean_text(ssn_tok):
        atoms.append("ssn_mainnet")
    if not atoms and native_row:
        atoms.append("native_only")
    return join_status_atoms(atoms or ["native_only"], "env")


def build_tag_for_row(
    row_id: str,
    local_date: str,
    has_local: bool,
    has_icp: bool,
    has_ssn: bool,
    native_only: bool,
) -> str:
    if row_id in BUILD_SENSITIVE_IDS:
        return "build_sensitive"
    if row_id in BASELINE_IDS:
        return "build_specific"
    if not has_local and (has_icp or has_ssn):
        return "build_specific"
    if has_local and local_date == "2026-04":
        return "current"
    if has_local:
        return "historical"
    if native_only:
        return "unknown"
    return "unknown"


def model_role_tag(
    row_id: str,
    has_local: bool,
    has_icp: bool,
    has_ssn: bool,
    alpha_eff: str,
    native_row: dict[str, str] | None,
) -> str:
    roles: list[str] = []

    if row_id in BASELINE_IDS:
        roles.append("baseline")
    if row_id in NATIVE_ONLY_IDS:
        roles.append("native_only")
    if row_id in NEGATIVE_RESULT_IDS:
        roles.append("negative_result")
    if row_id in SYSTEMS_DEMO_IDS:
        roles.append("systems_demo")

    if row_id not in BASELINE_IDS | NEGATIVE_RESULT_IDS | SYSTEMS_DEMO_IDS | NATIVE_ONLY_IDS:
        if has_local and clean_text(alpha_eff):
            roles.append("calibration")
        elif has_local:
            roles.append("supporting_measurement")

    if (has_icp or has_ssn) and row_id not in BASELINE_IDS | NATIVE_ONLY_IDS:
        roles.append("network_validation")

    if not roles and native_row:
        roles.append("native_only")

    return join_status_atoms(roles or ["supporting_measurement"], "role")


def local_role_tag(row_id: str, alpha_eff: str) -> str:
    if row_id in NEGATIVE_RESULT_IDS:
        return "negative_result"
    if row_id in SYSTEMS_DEMO_IDS:
        return "systems_demo"
    if clean_text(alpha_eff):
        return "calibration"
    return "supporting_measurement"


def main() -> None:
    legacy_rows = sorted(load_csv_lenient(MODELS_CSV), key=lambda row: parse_registry_id(row["id"]))
    local_existing_rows = load_csv_lenient(LOCAL_CSV)
    icp_rows = load_csv_lenient(ICP_CSV)
    ssn_rows = load_csv_lenient(SSN_CSV)
    native_rows = load_csv_lenient(NATIVE_CSV)
    audit_rows = load_csv_lenient(QUALITY_AUDIT_CSV)
    coherence_rows = load_csv_lenient(COHERENCE_MANUAL_CSV)
    core_rows = load_csv_lenient(CORE_V2_CSV)
    gguf_index = load_gguf_index()
    audit_exact, audit_loose, coherence_exact = build_quality_indices(audit_rows, coherence_rows)

    core_local_rows = [row for row in core_rows if row.get("network") == "local"]
    rebuilt_models: list[dict[str, str]] = []
    rebuilt_local: list[dict[str, str]] = []

    for legacy in legacy_rows:
        name = legacy["name"]
        quant = legacy["quant"]
        simd = legacy["simd"]
        legacy_build = first_non_empty(
            ROW_FIELD_OVERRIDES.get(legacy["id"], {}).get("build", ""),
            valid_build(legacy["build"]),
        )

        local_row = find_core_record(core_local_rows, name, quant, simd, legacy_build)
        existing_local_row = find_existing_local_record(local_existing_rows, name, quant, simd, legacy_build)
        local_source = local_row or existing_local_row
        gguf_path, gguf_meta, gguf_match = resolve_gguf(legacy, gguf_index)
        layers_real = first_non_empty(
            gguf_meta.get("block_count") if gguf_meta else "",
            local_row.get("layers") if local_row else "",
            existing_local_row.get("layers_real") if existing_local_row else "",
            legacy.get("layers_real"),
        )

        icp_row = find_network_record(icp_rows, name, quant, build=legacy_build)
        ssn_row = find_network_record(ssn_rows, name, quant, layers_real=layers_real)
        native_row = find_native_record(native_rows, name, quant)
        local_date = build_local_date(legacy, local_source, existing_local_row) if local_source else ""
        tok_call_local = first_non_empty(
            local_row.get("tok_per_call") if local_row else "",
            existing_local_row.get("tok_call") if existing_local_row else "",
        )
        tok_call_icp = first_non_empty(icp_row.get("tok_call_gen") if icp_row else "")
        tok_call_ssn = first_non_empty(ssn_row.get("tok_call") if ssn_row else "")
        alpha_eff = format_number(
            first_non_empty(
                local_row.get("alpha_eff") if local_row else "",
                existing_local_row.get("alpha_eff") if existing_local_row else "",
                ssn_row.get("alpha_eff") if ssn_row else "",
            )
        )
        quality_status = quality_status_for_row(legacy, layers_real, audit_exact, audit_loose, coherence_exact)
        model_status_tags = format_status_tags(
            runs=model_runs_tag(local_source, icp_row, ssn_row, native_row),
            quality=quality_status,
            env=model_env_tag(tok_call_local, tok_call_icp, tok_call_ssn, native_row),
            build=build_tag_for_row(
                legacy["id"],
                local_date,
                has_local=bool(local_source),
                has_icp=bool(icp_row),
                has_ssn=bool(ssn_row),
                native_only=not any([tok_call_local, tok_call_icp, tok_call_ssn]) and bool(native_row),
            ),
            role=model_role_tag(
                legacy["id"],
                has_local=bool(local_source),
                has_icp=bool(icp_row),
                has_ssn=bool(ssn_row),
                alpha_eff=alpha_eff,
                native_row=native_row,
            ),
        )

        size_mb = first_non_empty(
            local_row.get("gguf_size_MB") if local_row else "",
            existing_local_row.get("size_MB") if existing_local_row else "",
            ssn_row.get("size_MB") if ssn_row else "",
            icp_row.get("size_MB") if icp_row else "",
            gguf_meta.get("size_MB") if gguf_meta and gguf_match == "exact" else "",
            legacy.get("size_MB"),
        )
        build = first_non_empty(
            valid_build(local_source.get("build") if local_source else ""),
            valid_build(icp_row.get("build") if icp_row else ""),
            legacy_build,
            "gian" if ssn_row else "",
        )

        model_row = {
            "id": legacy["id"],
            "name": name,
            "arch": first_non_empty(legacy.get("arch"), local_row.get("architecture") if local_row else ""),
            "family": first_non_empty(legacy.get("family"), local_row.get("family") if local_row else ""),
            "params_M": format_number(first_non_empty(local_row.get("params_M") if local_row else "", legacy.get("params_M"))),
            "params_M_gguf": format_number(gguf_meta.get("params_M_calc") if gguf_meta else ""),
            "layers_real": format_number(layers_real),
            "hidden_dim": format_number(
                first_non_empty(
                    gguf_meta.get("embedding_length") if gguf_meta else "",
                    legacy.get("hidden_dim"),
                    local_row.get("hidden_dim") if local_row else "",
                )
            ),
            "d_ff": format_number(
                first_non_empty(
                    gguf_meta.get("feed_forward_length") if gguf_meta else "",
                    legacy.get("d_ff"),
                    local_row.get("d_ff") if local_row else "",
                )
            ),
            "vocab_size": format_number(
                first_non_empty(
                    gguf_meta.get("vocab_size") if gguf_meta else "",
                    legacy.get("vocab_size"),
                )
            ),
            "quant": quant,
            "simd": simd,
            "size_MB": format_number(size_mb),
            "gguf_path": gguf_path,
            "gguf_match": gguf_match,
            "tok_call_local": tok_call_local,
            "tok_call_icp": tok_call_icp,
            "tok_call_ssn": tok_call_ssn,
            "alpha_eff": alpha_eff,
            "native_gen_tok_s": format_number(native_row.get("tokens_per_sec") if native_row else ""),
            "native_prefill_tok_s": format_number(native_row.get("prefill_tok_per_sec") if native_row else ""),
            "build": build,
            "status_tags": model_status_tags,
            "notes": build_model_notes(legacy, local_source, icp_row, ssn_row, native_row, gguf_match),
        }
        rebuilt_models.append(model_row)

        if local_source:
            rebuilt_local.append(
                {
                    "date": local_date,
                    "model": name,
                    "quant": quant,
                    "simd": simd,
                    "layers_real": format_number(layers_real),
                    "size_MB": format_number(size_mb),
                    "tok_call": tok_call_local,
                    "alpha_eff": format_number(first_non_empty(local_row.get("alpha_eff") if local_row else "", existing_local_row.get("alpha_eff") if existing_local_row else "")),
                    "build": first_non_empty(local_source.get("build"), build),
                    "measurement_type": first_non_empty(local_source.get("measurement_type"), "binary_search"),
                    "status_tags": format_status_tags(
                        runs=local_runs_tag(local_source),
                        quality=quality_status,
                        env="local",
                        build=build_tag_for_row(
                            legacy["id"],
                            local_date,
                            has_local=True,
                            has_icp=False,
                            has_ssn=False,
                            native_only=False,
                        ),
                        role=local_role_tag(legacy["id"], model_row["alpha_eff"]),
                    ),
                    "notes": build_local_note(local_source),
                }
            )

    write_csv(MODELS_CSV, MODELS_FIELDS, rebuilt_models)
    write_csv(LOCAL_CSV, LOCAL_FIELDS, rebuilt_local)

    print(f"Rebuilt {MODELS_CSV}")
    print(f"Rebuilt {LOCAL_CSV}")
    print(f"models.csv rows: {len(rebuilt_models)}")
    print(f"local.csv rows: {len(rebuilt_local)}")
    print(f"GGUF root: {MODELS_ROOT}")


if __name__ == "__main__":
    main()
