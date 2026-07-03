# Artifact Results

## Canonical Layout

- `raw/`
  Raw measurement captures used as analysis inputs.
- `current/scaling_law/`
  Current post-variance scaling-law outputs used by the active paper draft.
- `current/extended_analysis/`
  Current kernel plots and extended comparison tables.
- `historical/scaling_law_baseline/`
  Baseline pre-variance outputs retained for auditability.
- `audit/`
  Statistical review package and any audit-side logs.

## Compatibility Aliases

The following historical names are preserved as symlinks:

- `summary_v2` → `current/scaling_law`
- `summary` → `current/extended_analysis`
- `summary_baseline` → `historical/scaling_law_baseline`
- `INDEPENDENT-VERIFICATION-PACKAGE.md` → `audit/INDEPENDENT-VERIFICATION-PACKAGE.md`
- `logs` → `audit/logs`

## Write Targets

- New scaling-law outputs should target `current/scaling_law/`.
- New kernel and auxiliary comparison outputs should target
  `current/extended_analysis/`.
- New audit artifacts should target `audit/`.
- Historical baselines should never be overwritten.
