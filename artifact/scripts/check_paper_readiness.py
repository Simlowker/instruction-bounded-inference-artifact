#!/usr/bin/env python3
"""Publication-oriented readiness checks for the current paper draft."""

from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import validate_data_tables
from registry_utils import DATA_DIR, ROOT, load_csv_lenient

ARTIFACT_ROOT = ROOT
PAPER_ROOT = ARTIFACT_ROOT.parent
CURRENT_LINK = PAPER_ROOT / "CURRENT.md"
CLAIMS_MATRIX = PAPER_ROOT / "CLAIMS-EVIDENCE-MATRIX.md"
DISCREPANCIES_CSV = DATA_DIR / "verified" / "discrepancies.csv"
SCALING_SUMMARY_CSV = ARTIFACT_ROOT / "results" / "current" / "scaling_law" / "loao_modern_families_summary.csv"
BOOTSTRAP_CSV = ARTIFACT_ROOT / "results" / "current" / "scaling_law" / "alpha_bootstrap_comparison.csv"
MARKER_RE = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|\?\?\?")
BENCHMARK_RE = re.compile(r"(\d+)\s+benchmark(?:\s+measurements?)?", re.IGNORECASE)
ARTIFACT_RE = re.compile(r"artifact/[A-Za-z0-9_./-]+")


@dataclass
class CheckResult:
    status: str
    name: str
    detail: str


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def resolve_draft(explicit_draft: str | None) -> Path:
    if explicit_draft:
        draft = Path(explicit_draft)
        if not draft.is_absolute():
            draft = (Path.cwd() / draft).resolve()
        return draft
    return CURRENT_LINK.resolve()


def run_table_validation() -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = validate_data_tables.main()
    return exit_code, buffer.getvalue().strip()


def summarize_scaling_outputs() -> str | None:
    if not SCALING_SUMMARY_CSV.exists() or not BOOTSTRAP_CSV.exists():
        return None

    summary_rows = load_csv_lenient(SCALING_SUMMARY_CSV)
    bootstrap_rows = load_csv_lenient(BOOTSTRAP_CSV)
    overall = next((row for row in summary_rows if row["held_out_arch"] == "OVERALL"), None)
    modern_ci = next((row for row in bootstrap_rows if row["label"] == "modern_only"), None)
    if not overall or not modern_ci:
        return None

    mape = float(overall["mean_abs_rel_error"]) * 100.0
    max_error = float(overall["max_abs_rel_error"]) * 100.0
    return (
        f"modern LOAO MAPE {mape:.1f}%, max {max_error:.1f}%, "
        f"alpha median {modern_ci['median']} [{modern_ci['ci_median_low']}, {modern_ci['ci_median_high']}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft",
        help="Paper draft to check. Defaults to CURRENT.md target.",
    )
    args = parser.parse_args()

    results: list[CheckResult] = []
    benchmark_rows = load_csv_lenient(DATA_DIR / "models.csv")
    benchmark_count = len(benchmark_rows)

    draft_path = resolve_draft(args.draft)
    if not CURRENT_LINK.exists():
        results.append(CheckResult("FAIL", "current draft pointer", "CURRENT.md is missing"))
        draft_path = PAPER_ROOT / "drafts" / "paper-v10.1.md"
    elif not draft_path.exists():
        results.append(CheckResult("FAIL", "current draft pointer", f"{draft_path} does not exist"))
    else:
        relative = draft_path.relative_to(PAPER_ROOT)
        results.append(CheckResult("PASS", "current draft pointer", f"CURRENT.md -> {relative}"))

    validation_code, validation_output = run_table_validation()
    if validation_code == 0:
        lines = validation_output.splitlines()
        detail = lines[-1] if lines else "table validation passed"
        if lines and len(lines) >= 2:
            detail = f"{lines[0]} {lines[1]}"
        results.append(CheckResult("PASS", "data tables", detail))
    else:
        results.append(CheckResult("FAIL", "data tables", validation_output or "validation failed"))

    if draft_path.exists():
        paper_text = draft_path.read_text()

        marker_hits = [(match.group(0), line_number(paper_text, match.start())) for match in MARKER_RE.finditer(paper_text)]
        if marker_hits:
            preview = ", ".join(f"{token}@L{lineno}" for token, lineno in marker_hits[:5])
            results.append(CheckResult("FAIL", "editorial markers", preview))
        else:
            results.append(CheckResult("PASS", "editorial markers", "no TODO/TBD/FIXME markers in current draft"))

        bad_benchmark_claims: list[str] = []
        for match in BENCHMARK_RE.finditer(paper_text):
            claimed = int(match.group(1))
            if claimed != benchmark_count:
                bad_benchmark_claims.append(f"L{line_number(paper_text, match.start())}: claims {claimed}, registry has {benchmark_count}")
        if bad_benchmark_claims:
            results.append(CheckResult("FAIL", "benchmark counts", "; ".join(bad_benchmark_claims[:5])))
        else:
            results.append(CheckResult("PASS", "benchmark counts", f"all benchmark-count claims match {benchmark_count} registry rows"))

        refs = sorted(set(ARTIFACT_RE.findall(paper_text)))
        missing_refs = [ref for ref in refs if not (PAPER_ROOT / ref).exists()]
        if missing_refs:
            results.append(CheckResult("FAIL", "artifact references", ", ".join(missing_refs[:5])))
        else:
            results.append(CheckResult("PASS", "artifact references", f"{len(refs)} cited artifact paths resolve on disk"))

        base = draft_path.stem
        missing_exports = [suffix for suffix in (".pdf", ".html") if not (PAPER_ROOT / "exports" / f"{base}{suffix}").exists()]
        if missing_exports:
            results.append(CheckResult("WARN", "rendered exports", f"missing {', '.join(f'exports/{base}{suffix}' for suffix in missing_exports)}"))
        else:
            results.append(CheckResult("PASS", "rendered exports", f"exports/{base}.pdf and exports/{base}.html are present"))

    if CLAIMS_MATRIX.exists():
        matrix_text = CLAIMS_MATRIX.read_text()
        if draft_path.exists() and draft_path.name not in matrix_text:
            results.append(CheckResult("WARN", "claims matrix", "current draft name is not mentioned in CLAIMS-EVIDENCE-MATRIX.md"))
        elif "originally synchronized with `paper-v10.md`" in matrix_text or "v10-era numbers" in matrix_text:
            results.append(CheckResult("WARN", "claims matrix", "matrix is usable but still carries v10-era labels / wording"))
        else:
            results.append(CheckResult("PASS", "claims matrix", "claims matrix appears aligned with the current draft"))
    else:
        results.append(CheckResult("FAIL", "claims matrix", "CLAIMS-EVIDENCE-MATRIX.md is missing"))

    if DISCREPANCIES_CSV.exists():
        discrepancy_rows = load_csv_lenient(DISCREPANCIES_CSV)
        counts = Counter(row["status"] for row in discrepancy_rows)
        mismatches = counts.get("MISMATCH", 0)
        no_file = counts.get("NO_FILE", 0)
        param_conventions = counts.get("PARAM_COUNT_CONVENTION", 0)
        if mismatches or no_file:
            results.append(
                CheckResult(
                    "WARN",
                    "GGUF provenance audit",
                    f"{mismatches} param mismatches, {no_file} rows without GGUF file; see data/verified/discrepancies.csv",
                )
            )
        else:
            detail = "no structural registry-vs-GGUF discrepancies remain"
            if param_conventions:
                detail += f"; {param_conventions} documented param-count convention differences"
            results.append(CheckResult("PASS", "GGUF provenance audit", detail))
    else:
        results.append(CheckResult("WARN", "GGUF provenance audit", "data/verified/discrepancies.csv has not been generated"))

    scaling_summary = summarize_scaling_outputs()
    if scaling_summary:
        results.append(CheckResult("PASS", "scaling-law outputs", scaling_summary))
    else:
        results.append(CheckResult("WARN", "scaling-law outputs", "current scaling-law summary CSVs are missing"))

    status_order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    results.sort(key=lambda item: (status_order[item.status], item.name))

    counts = Counter(item.status for item in results)
    for item in results:
        print(f"{item.status:4} {item.name}: {item.detail}")

    print(
        f"\nSummary: {counts.get('PASS', 0)} pass, "
        f"{counts.get('WARN', 0)} warn, {counts.get('FAIL', 0)} fail"
    )
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
