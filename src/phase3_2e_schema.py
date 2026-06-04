from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    load_phase3_2e_config,
    optional_schema_columns,
    required_schema_columns,
    sync_mode_map,
)


# =========================================================
# Phase III.2E — Schema Validator
# =========================================================
#
# Purpose:
#
#   Validate the file-level data contract for synchronized EEG–QRNG data.
#
# This module is intentionally strict about schema structure but tolerant of
# missing data during protocol-only development.
#
# Behavior:
#
#   If synchronized input CSV does not exist:
#       status = pending_synchronized_data
#       no exception is raised
#       protocol-only report is generated
#
#   If synchronized input CSV exists:
#       required columns are validated
#       timestamps are parsed
#       numeric fields are checked
#       sync_quality and clock_drift_ms are checked
#       subject/session/window identity is checked
#       sync_mode is audited when available
#
# Important:
#
#   Schema validation does not prove synchronization. It only verifies that the
#   data table has the required contract. True synchronization is validated in:
#
#       src/phase3_2e_sync_validator.py
#


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SchemaFailure:
    severity: str
    failure_type: str
    column: str
    row_index: Optional[int]
    subject_id: Optional[str]
    session_id: Optional[str]
    window_id: Optional[str]
    value: Optional[str]
    message: str


@dataclass
class SchemaValidationResult:
    status: str
    schema_valid: bool
    protocol_only: bool
    input_file: str
    input_exists: bool
    n_rows: int
    n_columns: int
    n_required_columns: int
    n_missing_required_columns: int
    n_optional_columns_present: int
    n_failures: int
    n_errors: int
    n_warnings: int
    can_proceed_to_sync_validation: bool
    can_proceed_to_empirical_cdr: bool
    interpretation: str


# =========================================================
# Generic helpers
# =========================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if pd.isna(obj):
        return None

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _row_identity(row: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        _safe_str(row.get("subject_id")),
        _safe_str(row.get("session_id")),
        _safe_str(row.get("window_id")),
    )


def _failure(
    severity: str,
    failure_type: str,
    column: str,
    message: str,
    row_index: Optional[int] = None,
    row: Optional[Mapping[str, Any]] = None,
    value: Optional[Any] = None,
) -> SchemaFailure:
    subject_id = None
    session_id = None
    window_id = None

    if row is not None:
        subject_id, session_id, window_id = _row_identity(row)

    return SchemaFailure(
        severity=str(severity),
        failure_type=str(failure_type),
        column=str(column),
        row_index=row_index,
        subject_id=subject_id,
        session_id=session_id,
        window_id=window_id,
        value=_safe_str(value),
        message=str(message),
    )


def _failure_records(failures: Sequence[SchemaFailure]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in failures]


def _write_failures_csv(path: Path, failures: Sequence[SchemaFailure]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "severity",
        "failure_type",
        "column",
        "row_index",
        "subject_id",
        "session_id",
        "window_id",
        "value",
        "message",
    ]

    if failures:
        df = pd.DataFrame(_failure_records(failures))
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(path, index=False)


def _write_summary_txt(path: Path, result: SchemaValidationResult, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Schema Validation Summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"schema_valid: {result.schema_valid}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append(f"input_file: {result.input_file}")
    lines.append(f"input_exists: {result.input_exists}")
    lines.append(f"n_rows: {result.n_rows}")
    lines.append(f"n_columns: {result.n_columns}")
    lines.append(f"n_required_columns: {result.n_required_columns}")
    lines.append(f"n_missing_required_columns: {result.n_missing_required_columns}")
    lines.append(f"n_optional_columns_present: {result.n_optional_columns_present}")
    lines.append(f"n_failures: {result.n_failures}")
    lines.append(f"n_errors: {result.n_errors}")
    lines.append(f"n_warnings: {result.n_warnings}")
    lines.append(f"can_proceed_to_sync_validation: {result.can_proceed_to_sync_validation}")
    lines.append(f"can_proceed_to_empirical_cdr: {result.can_proceed_to_empirical_cdr}")
    lines.append("")
    lines.append("interpretation:")
    lines.append(result.interpretation)
    lines.append("")

    missing = report.get("missing_required_columns", [])
    if missing:
        lines.append("Missing required columns:")
        for col in missing:
            lines.append(f"- {col}")
        lines.append("")

    warnings = [
        item for item in report.get("failures", [])
        if str(item.get("severity", "")).lower() == "warning"
    ]

    errors = [
        item for item in report.get("failures", [])
        if str(item.get("severity", "")).lower() == "error"
    ]

    lines.append(f"error_count: {len(errors)}")
    lines.append(f"warning_count: {len(warnings)}")
    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Data loading
# =========================================================

def load_synchronized_input(cfg: Phase32EConfig) -> Tuple[pd.DataFrame, bool]:
    path = Path(cfg.synchronized_input_csv)

    if not path.exists():
        return pd.DataFrame(), False

    df = pd.read_csv(path)

    return df, True


# =========================================================
# Schema checks
# =========================================================

def check_required_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[List[str], List[SchemaFailure]]:
    failures: List[SchemaFailure] = []

    required = list(required_schema_columns(cfg))
    existing = set(df.columns)

    missing = [col for col in required if col not in existing]

    for col in missing:
        failures.append(
            _failure(
                severity="error",
                failure_type="missing_required_column",
                column=col,
                message=f"Required column is missing: {col}",
            )
        )

    return missing, failures


def check_optional_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    optional = list(optional_schema_columns(cfg))
    present = [col for col in optional if col in df.columns]
    missing = [col for col in optional if col not in df.columns]

    return {
        "optional_columns_total": len(optional),
        "optional_columns_present": present,
        "optional_columns_missing": missing,
        "n_optional_columns_present": len(present),
    }


def check_empty_dataframe(df: pd.DataFrame) -> List[SchemaFailure]:
    if len(df) > 0:
        return []

    return [
        _failure(
            severity="error",
            failure_type="empty_input",
            column="*",
            message="Synchronized input CSV exists but contains zero rows.",
        )
    ]


def check_identity_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    required_identity = list(cfg.identity_columns)

    missing_identity = [col for col in required_identity if col not in df.columns]
    if missing_identity:
        return failures

    for col in required_identity:
        missing_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")

        for idx in df.index[missing_mask]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="missing_identity_value",
                    column=col,
                    row_index=int(idx),
                    row=row,
                    value=df.loc[idx, col],
                    message=f"Missing identity value in column: {col}",
                )
            )

    duplicate_mask = df.duplicated(subset=required_identity, keep=False)

    for idx in df.index[duplicate_mask]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="duplicate_identity",
                column=",".join(required_identity),
                row_index=int(idx),
                row=row,
                value="|".join(str(row.get(col)) for col in required_identity),
                message="Duplicate subject_id/session_id/window_id combination.",
            )
        )

    return failures


def parse_timestamp_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, List[SchemaFailure]]:
    failures: List[SchemaFailure] = []
    out = df.copy()

    timestamp_cols = [
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
    ]

    missing = [col for col in timestamp_cols if col not in out.columns]
    if missing:
        return out, failures

    for col in timestamp_cols:
        parsed_col = f"__parsed_{col}"

        out[parsed_col] = pd.to_datetime(
            out[col],
            errors="coerce",
            utc=True,
        )

        invalid_mask = out[parsed_col].isna()

        for idx in out.index[invalid_mask]:
            row = out.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="invalid_timestamp",
                    column=col,
                    row_index=int(idx),
                    row=row,
                    value=out.loc[idx, col],
                    message=f"Timestamp column could not be parsed as UTC: {col}",
                )
            )

    return out, failures


def check_timestamp_logic(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    start_col = "__parsed_window_start_utc"
    end_col = "__parsed_window_end_utc"
    eeg_col = "__parsed_eeg_timestamp_utc"
    qrng_col = "__parsed_qrng_timestamp_utc"

    required = [start_col, end_col, eeg_col, qrng_col]

    if any(col not in df.columns for col in required):
        return failures

    valid_time_mask = df[required].notna().all(axis=1)

    # -----------------------------------------------------
    # Window end must be after window start
    # -----------------------------------------------------
    invalid_window = valid_time_mask & (df[end_col] <= df[start_col])

    for idx in df.index[invalid_window]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="invalid_window_interval",
                column="window_start_utc/window_end_utc",
                row_index=int(idx),
                row=row,
                value=f"{row.get('window_start_utc')} -> {row.get('window_end_utc')}",
                message="window_end_utc must be greater than window_start_utc.",
            )
        )

    # -----------------------------------------------------
    # EEG/QRNG timestamps should be inside the window
    # -----------------------------------------------------
    eeg_outside = valid_time_mask & (
        (df[eeg_col] < df[start_col]) | (df[eeg_col] > df[end_col])
    )

    for idx in df.index[eeg_outside]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="eeg_timestamp_outside_window",
                column="eeg_timestamp_utc",
                row_index=int(idx),
                row=row,
                value=row.get("eeg_timestamp_utc"),
                message="eeg_timestamp_utc must fall inside the window interval.",
            )
        )

    qrng_outside = valid_time_mask & (
        (df[qrng_col] < df[start_col]) | (df[qrng_col] > df[end_col])
    )

    for idx in df.index[qrng_outside]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="qrng_timestamp_outside_window",
                column="qrng_timestamp_utc",
                row_index=int(idx),
                row=row,
                value=row.get("qrng_timestamp_utc"),
                message="qrng_timestamp_utc must fall inside the window interval.",
            )
        )

    # -----------------------------------------------------
    # Timestamp gap between EEG and QRNG should not exceed configured bound
    # -----------------------------------------------------
    gap_ms = (df[eeg_col] - df[qrng_col]).abs().dt.total_seconds() * 1000.0
    gap_invalid = valid_time_mask & (gap_ms > float(cfg.max_timestamp_gap_ms))

    for idx in df.index[gap_invalid]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="timestamp_gap_exceeds_threshold",
                column="eeg_timestamp_utc/qrng_timestamp_utc",
                row_index=int(idx),
                row=row,
                value=str(float(gap_ms.loc[idx])),
                message=(
                    "Absolute EEG–QRNG timestamp gap exceeds "
                    f"max_timestamp_gap_ms={cfg.max_timestamp_gap_ms}."
                ),
            )
        )

    return failures


def check_monotonic_timestamps(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    required = [
        "subject_id",
        "session_id",
        "__parsed_window_start_utc",
        "__parsed_window_end_utc",
    ]

    if any(col not in df.columns for col in required):
        return failures

    if not cfg.require_monotonic_timestamps:
        return failures

    work = df.copy()
    work["__original_index"] = work.index

    for (subject_id, session_id), group in work.groupby(["subject_id", "session_id"], sort=False):
        g = group.sort_values("__parsed_window_start_utc").copy()

        start_diff = g["__parsed_window_start_utc"].diff()
        invalid_monotonic = start_diff.dt.total_seconds().fillna(0.0) < 0.0

        for idx in g.index[invalid_monotonic]:
            row = g.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="non_monotonic_window_start",
                    column="window_start_utc",
                    row_index=int(row["__original_index"]),
                    row=row,
                    value=row.get("window_start_utc"),
                    message=(
                        "window_start_utc is non-monotonic within "
                        f"subject={subject_id}, session={session_id}."
                    ),
                )
            )

        if cfg.require_non_overlapping_windows:
            previous_end = None

            for _, row_series in g.iterrows():
                row = row_series.to_dict()
                current_start = row.get("__parsed_window_start_utc")
                current_end = row.get("__parsed_window_end_utc")

                if pd.isna(current_start) or pd.isna(current_end):
                    continue

                if previous_end is not None and current_start < previous_end:
                    failures.append(
                        _failure(
                            severity="error",
                            failure_type="overlapping_windows",
                            column="window_start_utc/window_end_utc",
                            row_index=int(row["__original_index"]),
                            row=row,
                            value=f"{row.get('window_start_utc')} -> {row.get('window_end_utc')}",
                            message=(
                                "Window overlaps previous window within "
                                f"subject={subject_id}, session={session_id}."
                            ),
                        )
                    )

                previous_end = current_end

    return failures


def numeric_columns_to_validate(cfg: Phase32EConfig) -> Tuple[str, ...]:
    columns: List[str] = [
        "sync_quality",
        "clock_drift_ms",
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_entropy",
        "qrng_state",
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
    ]

    columns.extend(
        [
            "eeg_primary_delta_power",
            "eeg_primary_alpha_power",
            "eeg_secondary_delta_power",
            "eeg_secondary_alpha_power",
            "interchannel_delta_ratio",
            "interchannel_alpha_ratio",
            "fronto_parietal_delta_shift",
            "fronto_parietal_alpha_shift",
            "cross_channel_corr",
        ]
    )

    columns.extend(list(cfg.optional_eeg_columns))
    columns.extend(list(cfg.optional_qrng_columns))

    return tuple(dict.fromkeys(columns))


def check_numeric_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    for col in numeric_columns_to_validate(cfg):
        if col not in df.columns:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = numeric.isna() & df[col].notna()

        for idx in df.index[invalid_mask]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="invalid_numeric_value",
                    column=col,
                    row_index=int(idx),
                    row=row,
                    value=df.loc[idx, col],
                    message=f"Column must be numeric: {col}",
                )
            )

    return failures


def check_sync_thresholds(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    if "sync_quality" in df.columns:
        sync_quality = pd.to_numeric(df["sync_quality"], errors="coerce")

        below = sync_quality < float(cfg.min_sync_quality)

        for idx in df.index[below.fillna(False)]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="sync_quality_below_threshold",
                    column="sync_quality",
                    row_index=int(idx),
                    row=row,
                    value=df.loc[idx, "sync_quality"],
                    message=f"sync_quality must be >= {cfg.min_sync_quality}.",
                )
            )

    if "clock_drift_ms" in df.columns:
        drift = pd.to_numeric(df["clock_drift_ms"], errors="coerce").abs()

        above = drift > float(cfg.max_clock_drift_ms)

        for idx in df.index[above.fillna(False)]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="clock_drift_exceeds_threshold",
                    column="clock_drift_ms",
                    row_index=int(idx),
                    row=row,
                    value=df.loc[idx, "clock_drift_ms"],
                    message=f"abs(clock_drift_ms) must be <= {cfg.max_clock_drift_ms}.",
                )
            )

    return failures


def check_sync_mode_column(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    mode_specs = sync_mode_map(cfg)

    if cfg.sync_mode_column not in df.columns:
        failures.append(
            _failure(
                severity="warning",
                failure_type="sync_mode_column_missing",
                column=cfg.sync_mode_column,
                message=(
                    f"sync_mode column is missing. Default mode is treated as "
                    f"{cfg.default_sync_mode_if_missing}. Sync validator may block confirmatory analysis."
                ),
            )
        )
        return failures

    modes = df[cfg.sync_mode_column].fillna(cfg.default_sync_mode_if_missing).astype(str)

    for idx, mode in modes.items():
        row = df.loc[idx].to_dict()

        if mode not in mode_specs:
            failures.append(
                _failure(
                    severity="error",
                    failure_type="unknown_sync_mode",
                    column=cfg.sync_mode_column,
                    row_index=int(idx),
                    row=row,
                    value=mode,
                    message=f"Unknown sync_mode: {mode}",
                )
            )
            continue

        spec = mode_specs[mode]

        if not spec.accepted_for_confirmatory:
            failures.append(
                _failure(
                    severity="warning",
                    failure_type="sync_mode_not_confirmatory",
                    column=cfg.sync_mode_column,
                    row_index=int(idx),
                    row=row,
                    value=mode,
                    message=(
                        f"sync_mode={mode} is not accepted for confirmatory analysis. "
                        "Sync validator may block empirical CDR."
                    ),
                )
            )

    return failures


def check_basic_state_ranges(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SchemaFailure]:
    failures: List[SchemaFailure] = []

    range_checks = [
        ("qrng_state", 0, cfg.qrng_n_bins - 1),
        ("sync_quality", 0.0, 1.0),
    ]

    for col, lower, upper in range_checks:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        invalid = values.notna() & ((values < lower) | (values > upper))

        for idx in df.index[invalid]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="value_out_of_range",
                    column=col,
                    row_index=int(idx),
                    row=row,
                    value=df.loc[idx, col],
                    message=f"{col} must be within [{lower}, {upper}].",
                )
            )

    return failures


# =========================================================
# Main validation
# =========================================================

def validate_phase3_2e_schema(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    input_path = Path(cfg.synchronized_input_csv)

    df, input_exists = load_synchronized_input(cfg)

    failures: List[SchemaFailure] = []

    required_cols = list(required_schema_columns(cfg))
    optional_info = {
        "optional_columns_total": len(optional_schema_columns(cfg)),
        "optional_columns_present": [],
        "optional_columns_missing": list(optional_schema_columns(cfg)),
        "n_optional_columns_present": 0,
    }

    if not input_exists:
        result = SchemaValidationResult(
            status=cfg.status_pending_synchronized_data,
            schema_valid=False,
            protocol_only=True,
            input_file=str(input_path),
            input_exists=False,
            n_rows=0,
            n_columns=0,
            n_required_columns=len(required_cols),
            n_missing_required_columns=len(required_cols),
            n_optional_columns_present=0,
            n_failures=0,
            n_errors=0,
            n_warnings=0,
            can_proceed_to_sync_validation=False,
            can_proceed_to_empirical_cdr=False,
            interpretation=(
                "No synchronized EEG–QRNG input CSV was found. This is valid for "
                "protocol-only development. Empirical CDR cannot run until a real "
                "synchronized dataset is provided."
            ),
        )

        report = {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": "phase3_2e_schema_validator",
            "status": result.status,
            "result": asdict(result),
            "input_file": str(input_path),
            "required_columns": required_cols,
            "missing_required_columns": required_cols,
            "optional_columns": optional_info,
            "failures": [],
        }

        save_schema_outputs(cfg, report, failures, result)

        return report

    failures.extend(check_empty_dataframe(df))

    missing_required, missing_failures = check_required_columns(df, cfg)
    failures.extend(missing_failures)

    optional_info = check_optional_columns(df, cfg)

    # If required structural columns are missing, deeper row-level checks may
    # create noisy errors. Continue only with checks that can run safely.
    if len(missing_required) == 0 and len(df) > 0:
        failures.extend(check_identity_columns(df, cfg))

        parsed_df, timestamp_failures = parse_timestamp_columns(df, cfg)
        failures.extend(timestamp_failures)

        failures.extend(check_timestamp_logic(parsed_df, cfg))
        failures.extend(check_monotonic_timestamps(parsed_df, cfg))
        failures.extend(check_numeric_columns(df, cfg))
        failures.extend(check_sync_thresholds(df, cfg))
        failures.extend(check_sync_mode_column(df, cfg))
        failures.extend(check_basic_state_ranges(df, cfg))

    else:
        if len(df) > 0:
            failures.append(
                _failure(
                    severity="warning",
                    failure_type="deep_validation_skipped",
                    column="*",
                    message=(
                        "Deep row-level validation was skipped because required "
                        "schema columns are missing."
                    ),
                )
            )

    n_errors = sum(1 for item in failures if item.severity.lower() == "error")
    n_warnings = sum(1 for item in failures if item.severity.lower() == "warning")

    schema_valid = n_errors == 0

    if schema_valid:
        status = "schema_valid"
        interpretation = (
            "The synchronized input file satisfies the Phase III.2E schema contract. "
            "The next step is synchronization validation."
        )
    else:
        status = "schema_invalid"
        interpretation = (
            "The synchronized input file does not satisfy the Phase III.2E schema contract. "
            "Fix schema errors before running synchronization validation or empirical CDR."
        )

    result = SchemaValidationResult(
        status=status,
        schema_valid=bool(schema_valid),
        protocol_only=False,
        input_file=str(input_path),
        input_exists=True,
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        n_required_columns=len(required_cols),
        n_missing_required_columns=len(missing_required),
        n_optional_columns_present=int(optional_info.get("n_optional_columns_present", 0)),
        n_failures=int(len(failures)),
        n_errors=int(n_errors),
        n_warnings=int(n_warnings),
        can_proceed_to_sync_validation=bool(schema_valid),
        can_proceed_to_empirical_cdr=False,
        interpretation=interpretation,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": "phase3_2e_schema_validator",
        "status": result.status,
        "result": asdict(result),
        "input_file": str(input_path),
        "input_exists": True,
        "columns_found": list(df.columns),
        "required_columns": required_cols,
        "missing_required_columns": missing_required,
        "optional_columns": optional_info,
        "failures": _failure_records(failures),
    }

    save_schema_outputs(cfg, report, failures, result)

    return report


def save_schema_outputs(
    cfg: Phase32EConfig,
    report: Dict[str, Any],
    failures: Sequence[SchemaFailure],
    result: SchemaValidationResult,
) -> None:
    summary_txt = Path(cfg.results_dir) / "phase3_2e_schema_summary.txt"

    save_json(Path(cfg.schema_report_json), report)
    _write_failures_csv(Path(cfg.schema_failures_csv), failures)
    _write_summary_txt(summary_txt, result, report)

    print(f"[Phase3.2E Schema] Saved report: {cfg.schema_report_json}")
    print(f"[Phase3.2E Schema] Saved failures: {cfg.schema_failures_csv}")
    print(f"[Phase3.2E Schema] Saved summary: {summary_txt}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2e_config()

    report = validate_phase3_2e_schema(cfg)
    result = report.get("result", {})

    print("\n" + "=" * 78)
    print("Phase III.2E schema validation completed")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"schema_valid: {result.get('schema_valid')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"input_exists: {result.get('input_exists')}")
    print(f"n_rows: {result.get('n_rows')}")
    print(f"n_errors: {result.get('n_errors')}")
    print(f"n_warnings: {result.get('n_warnings')}")
    print(f"can_proceed_to_sync_validation: {result.get('can_proceed_to_sync_validation')}")
    print(f"can_proceed_to_empirical_cdr: {result.get('can_proceed_to_empirical_cdr')}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()