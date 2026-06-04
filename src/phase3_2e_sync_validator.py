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
    sync_mode_map,
)

from src.phase3_2e_schema import (
    validate_phase3_2e_schema,
)


# =========================================================
# Phase III.2E — Synchronization Validator
# =========================================================
#
# Purpose:
#
#   Validate whether a Phase III.2E synchronized EEG–QRNG input table is
#   eligible for empirical synchronized CDR.
#
# This module runs after schema validation.
#
# It verifies:
#
#   - synchronized input exists;
#   - schema is valid;
#   - timestamps are parseable as UTC;
#   - timestamps are monotonic within subject/session;
#   - windows are non-overlapping within subject/session;
#   - EEG and QRNG timestamps fall inside the window;
#   - EEG–QRNG timestamp gap is within threshold;
#   - sync_quality passes threshold;
#   - clock_drift_ms passes threshold;
#   - sync_mode is accepted for confirmatory analysis;
#   - QRNG sample reuse is checked when qrng_sample_id exists;
#   - synchronization can safely proceed to empirical CDR.
#
# If synchronized input is absent:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   no exception is raised
#
# If schema is invalid:
#
#   status = sync_blocked_by_schema
#
# If synchronization fails:
#
#   status = sync_invalid
#
# If synchronization passes:
#
#   status = sync_valid
#


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class SyncFailure:
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
class SyncValidationResult:
    status: str
    sync_valid: bool
    protocol_only: bool
    input_file: str
    input_exists: bool

    schema_status: str
    schema_valid: bool

    n_rows: int
    n_subjects: int
    n_sessions: int
    n_windows: int

    min_sync_quality: Optional[float]
    mean_sync_quality: Optional[float]
    max_clock_drift_ms_abs: Optional[float]
    max_eeg_qrng_gap_ms: Optional[float]
    median_eeg_qrng_gap_ms: Optional[float]

    n_failures: int
    n_errors: int
    n_warnings: int

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

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

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
) -> SyncFailure:
    subject_id = None
    session_id = None
    window_id = None

    if row is not None:
        subject_id, session_id, window_id = _row_identity(row)

    return SyncFailure(
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


def _failure_records(failures: Sequence[SyncFailure]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in failures]


def _write_failures_csv(path: Path, failures: Sequence[SyncFailure]) -> None:
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


def _write_summary_txt(path: Path, result: SyncValidationResult, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Synchronization Validation Summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"sync_valid: {result.sync_valid}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append(f"input_file: {result.input_file}")
    lines.append(f"input_exists: {result.input_exists}")
    lines.append("")
    lines.append(f"schema_status: {result.schema_status}")
    lines.append(f"schema_valid: {result.schema_valid}")
    lines.append("")
    lines.append(f"n_rows: {result.n_rows}")
    lines.append(f"n_subjects: {result.n_subjects}")
    lines.append(f"n_sessions: {result.n_sessions}")
    lines.append(f"n_windows: {result.n_windows}")
    lines.append("")
    lines.append(f"min_sync_quality: {result.min_sync_quality}")
    lines.append(f"mean_sync_quality: {result.mean_sync_quality}")
    lines.append(f"max_clock_drift_ms_abs: {result.max_clock_drift_ms_abs}")
    lines.append(f"max_eeg_qrng_gap_ms: {result.max_eeg_qrng_gap_ms}")
    lines.append(f"median_eeg_qrng_gap_ms: {result.median_eeg_qrng_gap_ms}")
    lines.append("")
    lines.append(f"n_failures: {result.n_failures}")
    lines.append(f"n_errors: {result.n_errors}")
    lines.append(f"n_warnings: {result.n_warnings}")
    lines.append("")
    lines.append(f"can_proceed_to_empirical_cdr: {result.can_proceed_to_empirical_cdr}")
    lines.append("")
    lines.append("interpretation:")
    lines.append(result.interpretation)
    lines.append("")

    errors = [
        item for item in report.get("failures", [])
        if str(item.get("severity", "")).lower() == "error"
    ]

    warnings = [
        item for item in report.get("failures", [])
        if str(item.get("severity", "")).lower() == "warning"
    ]

    lines.append(f"error_count: {len(errors)}")
    lines.append(f"warning_count: {len(warnings)}")
    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _load_input_csv(cfg: Phase32EConfig) -> Tuple[pd.DataFrame, bool]:
    path = Path(cfg.synchronized_input_csv)

    if not path.exists():
        return pd.DataFrame(), False

    return pd.read_csv(path), True


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    timestamp_cols = [
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
    ]

    for col in timestamp_cols:
        if col in out.columns:
            out[f"__parsed_{col}"] = pd.to_datetime(
                out[col],
                errors="coerce",
                utc=True,
            )

    return out


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype=float)

    return pd.to_numeric(df[col], errors="coerce")


# =========================================================
# Synchronization checks
# =========================================================

def check_schema_precondition(
    schema_report: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Tuple[bool, bool, str, str]:
    result = schema_report.get("result", {})

    input_exists = bool(result.get("input_exists", False))
    protocol_only = bool(result.get("protocol_only", False))
    schema_status = str(result.get("status", "unknown"))
    schema_valid = bool(result.get("schema_valid", False))

    if not input_exists and protocol_only:
        return False, True, schema_status, "pending_synchronized_data"

    if not schema_valid:
        return False, False, schema_status, "sync_blocked_by_schema"

    return True, False, schema_status, "schema_ready"


def check_timestamp_presence_and_parsing(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    timestamp_cols = [
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
    ]

    for col in timestamp_cols:
        parsed_col = f"__parsed_{col}"

        if col not in df.columns:
            failures.append(
                _failure(
                    severity="error",
                    failure_type="missing_timestamp_column",
                    column=col,
                    message=f"Required timestamp column is missing: {col}",
                )
            )
            continue

        if parsed_col not in df.columns:
            failures.append(
                _failure(
                    severity="error",
                    failure_type="timestamp_not_parsed",
                    column=col,
                    message=f"Parsed timestamp column missing: {parsed_col}",
                )
            )
            continue

        invalid_mask = df[parsed_col].isna()

        for idx in df.index[invalid_mask]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="invalid_timestamp",
                    column=col,
                    row_index=int(idx),
                    row=row,
                    value=row.get(col),
                    message=f"Timestamp could not be parsed as UTC: {col}",
                )
            )

    return failures


def check_window_intervals(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    start_col = "__parsed_window_start_utc"
    end_col = "__parsed_window_end_utc"

    if start_col not in df.columns or end_col not in df.columns:
        return failures

    valid = df[start_col].notna() & df[end_col].notna()
    invalid_interval = valid & (df[end_col] <= df[start_col])

    for idx in df.index[invalid_interval]:
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

    return failures


def check_timestamps_inside_windows(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    required = [
        "__parsed_window_start_utc",
        "__parsed_window_end_utc",
        "__parsed_eeg_timestamp_utc",
        "__parsed_qrng_timestamp_utc",
    ]

    if any(col not in df.columns for col in required):
        return failures

    start = df["__parsed_window_start_utc"]
    end = df["__parsed_window_end_utc"]
    eeg_ts = df["__parsed_eeg_timestamp_utc"]
    qrng_ts = df["__parsed_qrng_timestamp_utc"]

    valid = start.notna() & end.notna() & eeg_ts.notna() & qrng_ts.notna()

    eeg_outside = valid & ((eeg_ts < start) | (eeg_ts > end))
    qrng_outside = valid & ((qrng_ts < start) | (qrng_ts > end))

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
                message="EEG timestamp must be inside the synchronized window.",
            )
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
                message="QRNG timestamp must be inside the synchronized window.",
            )
        )

    return failures


def compute_eeg_qrng_gap_ms(df: pd.DataFrame) -> pd.Series:
    if "__parsed_eeg_timestamp_utc" not in df.columns:
        return pd.Series([], dtype=float)

    if "__parsed_qrng_timestamp_utc" not in df.columns:
        return pd.Series([], dtype=float)

    eeg_ts = df["__parsed_eeg_timestamp_utc"]
    qrng_ts = df["__parsed_qrng_timestamp_utc"]

    gap_ms = (eeg_ts - qrng_ts).abs().dt.total_seconds() * 1000.0

    return gap_ms


def check_eeg_qrng_gap(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    gap_ms = compute_eeg_qrng_gap_ms(df)

    if gap_ms.empty:
        return failures

    invalid = gap_ms.notna() & (gap_ms > float(cfg.max_timestamp_gap_ms))

    for idx in df.index[invalid]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="eeg_qrng_timestamp_gap_exceeds_threshold",
                column="eeg_timestamp_utc/qrng_timestamp_utc",
                row_index=int(idx),
                row=row,
                value=str(float(gap_ms.loc[idx])),
                message=(
                    "EEG–QRNG timestamp gap exceeds "
                    f"max_timestamp_gap_ms={cfg.max_timestamp_gap_ms}."
                ),
            )
        )

    return failures


def check_monotonic_and_non_overlapping_windows(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    required = [
        "subject_id",
        "session_id",
        "__parsed_window_start_utc",
        "__parsed_window_end_utc",
    ]

    if any(col not in df.columns for col in required):
        return failures

    work = df.copy()
    work["__original_index"] = work.index

    for (subject_id, session_id), group in work.groupby(["subject_id", "session_id"], sort=False):
        g = group.sort_values("__parsed_window_start_utc").copy()

        previous_start = None
        previous_end = None

        for _, row_series in g.iterrows():
            row = row_series.to_dict()
            idx = int(row["__original_index"])

            current_start = row.get("__parsed_window_start_utc")
            current_end = row.get("__parsed_window_end_utc")

            if pd.isna(current_start) or pd.isna(current_end):
                continue

            if previous_start is not None and cfg.require_monotonic_timestamps:
                if current_start < previous_start:
                    failures.append(
                        _failure(
                            severity="error",
                            failure_type="non_monotonic_window_start",
                            column="window_start_utc",
                            row_index=idx,
                            row=row,
                            value=row.get("window_start_utc"),
                            message=(
                                "window_start_utc is non-monotonic within "
                                f"subject={subject_id}, session={session_id}."
                            ),
                        )
                    )

            if previous_end is not None and cfg.require_non_overlapping_windows:
                if current_start < previous_end:
                    overlap_seconds = (previous_end - current_start).total_seconds()

                    failures.append(
                        _failure(
                            severity="error",
                            failure_type="overlapping_windows",
                            column="window_start_utc/window_end_utc",
                            row_index=idx,
                            row=row,
                            value=str(overlap_seconds),
                            message=(
                                "Synchronized windows overlap within "
                                f"subject={subject_id}, session={session_id}."
                            ),
                        )
                    )

            previous_start = current_start
            previous_end = current_end

    return failures


def check_sync_quality_and_drift(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    if "sync_quality" in df.columns:
        sync_quality = pd.to_numeric(df["sync_quality"], errors="coerce")

        invalid_numeric = sync_quality.isna()

        for idx in df.index[invalid_numeric]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="invalid_sync_quality",
                    column="sync_quality",
                    row_index=int(idx),
                    row=row,
                    value=row.get("sync_quality"),
                    message="sync_quality must be numeric.",
                )
            )

        below = sync_quality.notna() & (sync_quality < float(cfg.min_sync_quality))

        for idx in df.index[below]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="sync_quality_below_threshold",
                    column="sync_quality",
                    row_index=int(idx),
                    row=row,
                    value=row.get("sync_quality"),
                    message=f"sync_quality must be >= {cfg.min_sync_quality}.",
                )
            )

        out_of_range = sync_quality.notna() & ((sync_quality < 0.0) | (sync_quality > 1.0))

        for idx in df.index[out_of_range]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="sync_quality_out_of_range",
                    column="sync_quality",
                    row_index=int(idx),
                    row=row,
                    value=row.get("sync_quality"),
                    message="sync_quality must be within [0, 1].",
                )
            )

    if "clock_drift_ms" in df.columns:
        drift = pd.to_numeric(df["clock_drift_ms"], errors="coerce")

        invalid_numeric = drift.isna()

        for idx in df.index[invalid_numeric]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="invalid_clock_drift_ms",
                    column="clock_drift_ms",
                    row_index=int(idx),
                    row=row,
                    value=row.get("clock_drift_ms"),
                    message="clock_drift_ms must be numeric.",
                )
            )

        drift_abs = drift.abs()
        above = drift_abs.notna() & (drift_abs > float(cfg.max_clock_drift_ms))

        for idx in df.index[above]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="clock_drift_exceeds_threshold",
                    column="clock_drift_ms",
                    row_index=int(idx),
                    row=row,
                    value=row.get("clock_drift_ms"),
                    message=f"abs(clock_drift_ms) must be <= {cfg.max_clock_drift_ms}.",
                )
            )

    return failures


def check_sync_modes(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []
    specs = sync_mode_map(cfg)

    if cfg.sync_mode_column not in df.columns:
        failures.append(
            _failure(
                severity="error",
                failure_type="sync_mode_missing_for_sync_validation",
                column=cfg.sync_mode_column,
                message=(
                    "sync_mode column is required for synchronization validation. "
                    "Schema allows it as optional, but sync validation blocks confirmatory "
                    "analysis without an auditable sync mode."
                ),
            )
        )
        return failures

    modes = df[cfg.sync_mode_column].fillna(cfg.default_sync_mode_if_missing).astype(str)

    for idx, mode in modes.items():
        row = df.loc[idx].to_dict()

        if mode not in specs:
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

        spec = specs[mode]

        if not spec.accepted_for_confirmatory:
            failures.append(
                _failure(
                    severity="error",
                    failure_type="sync_mode_not_accepted_for_confirmatory",
                    column=cfg.sync_mode_column,
                    row_index=int(idx),
                    row=row,
                    value=mode,
                    message=(
                        f"sync_mode={mode} is not accepted for confirmatory synchronized CDR."
                    ),
                )
            )

    return failures


def check_window_durations(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    start_col = "__parsed_window_start_utc"
    end_col = "__parsed_window_end_utc"

    if start_col not in df.columns or end_col not in df.columns:
        return failures

    valid_window_seconds = set(int(x) for x in cfg.all_window_seconds)

    durations = (df[end_col] - df[start_col]).dt.total_seconds()

    invalid = durations.notna() & ~durations.round().astype("Int64").isin(valid_window_seconds)

    for idx in df.index[invalid]:
        row = df.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="warning",
                failure_type="window_duration_not_registered",
                column="window_start_utc/window_end_utc",
                row_index=int(idx),
                row=row,
                value=str(float(durations.loc[idx])),
                message=(
                    "Window duration is not one of the registered Phase III.2E "
                    f"window sizes: {sorted(valid_window_seconds)} seconds."
                ),
            )
        )

    if cfg.window_size_column in df.columns:
        declared = pd.to_numeric(df[cfg.window_size_column], errors="coerce")
        mismatch = declared.notna() & durations.notna() & (declared.round().astype("Int64") != durations.round().astype("Int64"))

        for idx in df.index[mismatch]:
            row = df.loc[idx].to_dict()
            failures.append(
                _failure(
                    severity="error",
                    failure_type="window_seconds_mismatch",
                    column=cfg.window_size_column,
                    row_index=int(idx),
                    row=row,
                    value=row.get(cfg.window_size_column),
                    message="Declared window_seconds does not match timestamp-derived duration.",
                )
            )

    return failures


def check_qrng_sample_reuse(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[SyncFailure]:
    failures: List[SyncFailure] = []

    # Optional but useful. If the future synchronized dataset provides a
    # qrng_sample_id column, repeated sample IDs across subject/session are blocked.
    if "qrng_sample_id" not in df.columns:
        failures.append(
            _failure(
                severity="warning",
                failure_type="qrng_sample_id_missing",
                column="qrng_sample_id",
                message=(
                    "qrng_sample_id column is missing. Cross-subject QRNG sample reuse "
                    "cannot be audited directly. This does not block protocol-only work, "
                    "but real confirmatory datasets should provide this column if possible."
                ),
            )
        )
        return failures

    work = df.copy()
    work["__subject_session"] = (
        work["subject_id"].astype(str) + "::" + work["session_id"].astype(str)
    )

    repeated = work.groupby("qrng_sample_id")["__subject_session"].nunique()
    reused_ids = set(repeated[repeated > 1].index.astype(str).tolist())

    if not reused_ids:
        return failures

    mask = work["qrng_sample_id"].astype(str).isin(reused_ids)

    for idx in work.index[mask]:
        row = work.loc[idx].to_dict()
        failures.append(
            _failure(
                severity="error",
                failure_type="qrng_sample_reused_across_subject_or_session",
                column="qrng_sample_id",
                row_index=int(idx),
                row=row,
                value=row.get("qrng_sample_id"),
                message="Same qrng_sample_id appears across more than one subject/session.",
            )
        )

    return failures


def build_sync_quality_summary(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    sync_quality = _numeric(df, "sync_quality")
    drift = _numeric(df, "clock_drift_ms").abs()
    gap_ms = compute_eeg_qrng_gap_ms(df)

    window_duration_seconds = pd.Series([], dtype=float)

    if "__parsed_window_start_utc" in df.columns and "__parsed_window_end_utc" in df.columns:
        window_duration_seconds = (
            df["__parsed_window_end_utc"] - df["__parsed_window_start_utc"]
        ).dt.total_seconds()

    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "n_sessions": int(
            df[["subject_id", "session_id"]].drop_duplicates().shape[0]
        ) if {"subject_id", "session_id"}.issubset(df.columns) else 0,
        "n_windows": int(df["window_id"].nunique()) if "window_id" in df.columns else 0,

        "sync_quality": {
            "min": float(sync_quality.min()) if len(sync_quality) and sync_quality.notna().any() else None,
            "mean": float(sync_quality.mean()) if len(sync_quality) and sync_quality.notna().any() else None,
            "median": float(sync_quality.median()) if len(sync_quality) and sync_quality.notna().any() else None,
            "max": float(sync_quality.max()) if len(sync_quality) and sync_quality.notna().any() else None,
            "threshold": float(cfg.min_sync_quality),
        },

        "clock_drift_ms_abs": {
            "max": float(drift.max()) if len(drift) and drift.notna().any() else None,
            "mean": float(drift.mean()) if len(drift) and drift.notna().any() else None,
            "median": float(drift.median()) if len(drift) and drift.notna().any() else None,
            "threshold": float(cfg.max_clock_drift_ms),
        },

        "eeg_qrng_gap_ms": {
            "max": float(gap_ms.max()) if len(gap_ms) and gap_ms.notna().any() else None,
            "mean": float(gap_ms.mean()) if len(gap_ms) and gap_ms.notna().any() else None,
            "median": float(gap_ms.median()) if len(gap_ms) and gap_ms.notna().any() else None,
            "threshold": float(cfg.max_timestamp_gap_ms),
        },

        "window_duration_seconds": {
            "unique_rounded": (
                sorted(window_duration_seconds.round().dropna().astype(int).unique().tolist())
                if len(window_duration_seconds) and window_duration_seconds.notna().any()
                else []
            ),
            "registered": list(cfg.all_window_seconds),
        },
    }


# =========================================================
# Main validation
# =========================================================

def validate_phase3_2e_synchronization(
    cfg: Optional[Phase32EConfig] = None,
    run_schema_first: bool = True,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    input_path = Path(cfg.synchronized_input_csv)

    schema_report: Dict[str, Any] = {}

    if run_schema_first:
        schema_report = validate_phase3_2e_schema(cfg)
    else:
        schema_report_path = Path(cfg.schema_report_json)

        if schema_report_path.exists():
            with open(schema_report_path, "r", encoding="utf-8") as f:
                schema_report = json.load(f)

    schema_can_continue, protocol_only, schema_status, precondition_status = check_schema_precondition(
        schema_report=schema_report,
        cfg=cfg,
    )

    # -----------------------------------------------------
    # Protocol-only / pending data path
    # -----------------------------------------------------
    if protocol_only:
        result = SyncValidationResult(
            status=cfg.status_pending_synchronized_data,
            sync_valid=False,
            protocol_only=True,
            input_file=str(input_path),
            input_exists=False,
            schema_status=schema_status,
            schema_valid=False,
            n_rows=0,
            n_subjects=0,
            n_sessions=0,
            n_windows=0,
            min_sync_quality=None,
            mean_sync_quality=None,
            max_clock_drift_ms_abs=None,
            max_eeg_qrng_gap_ms=None,
            median_eeg_qrng_gap_ms=None,
            n_failures=0,
            n_errors=0,
            n_warnings=0,
            can_proceed_to_empirical_cdr=False,
            interpretation=(
                "No synchronized EEG–QRNG dataset is available. Synchronization validation "
                "is in protocol-only mode. Empirical CDR cannot proceed until a real "
                "synchronized dataset is provided."
            ),
        )

        report = {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": "phase3_2e_sync_validator",
            "status": result.status,
            "result": asdict(result),
            "schema_report_status": schema_status,
            "input_file": str(input_path),
            "sync_quality_summary": {},
            "failures": [],
        }

        save_sync_outputs(cfg, report, [], result)

        return report

    # -----------------------------------------------------
    # Schema-blocked path
    # -----------------------------------------------------
    if not schema_can_continue:
        result = SyncValidationResult(
            status="sync_blocked_by_schema",
            sync_valid=False,
            protocol_only=False,
            input_file=str(input_path),
            input_exists=input_path.exists(),
            schema_status=schema_status,
            schema_valid=False,
            n_rows=0,
            n_subjects=0,
            n_sessions=0,
            n_windows=0,
            min_sync_quality=None,
            mean_sync_quality=None,
            max_clock_drift_ms_abs=None,
            max_eeg_qrng_gap_ms=None,
            median_eeg_qrng_gap_ms=None,
            n_failures=0,
            n_errors=0,
            n_warnings=0,
            can_proceed_to_empirical_cdr=False,
            interpretation=(
                "Synchronization validation was blocked because the schema validator "
                "did not approve the synchronized input file."
            ),
        )

        report = {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": "phase3_2e_sync_validator",
            "status": result.status,
            "result": asdict(result),
            "schema_report_status": schema_status,
            "input_file": str(input_path),
            "sync_quality_summary": {},
            "failures": [],
        }

        save_sync_outputs(cfg, report, [], result)

        return report

    # -----------------------------------------------------
    # Full sync validation path
    # -----------------------------------------------------
    df, input_exists = _load_input_csv(cfg)

    if not input_exists:
        raise FileNotFoundError(
            f"Schema reported an input file, but file was not found: {input_path}"
        )

    df = _parse_timestamps(df)

    failures: List[SyncFailure] = []

    failures.extend(check_timestamp_presence_and_parsing(df, cfg))
    failures.extend(check_window_intervals(df, cfg))
    failures.extend(check_timestamps_inside_windows(df, cfg))
    failures.extend(check_eeg_qrng_gap(df, cfg))
    failures.extend(check_monotonic_and_non_overlapping_windows(df, cfg))
    failures.extend(check_sync_quality_and_drift(df, cfg))
    failures.extend(check_sync_modes(df, cfg))
    failures.extend(check_window_durations(df, cfg))
    failures.extend(check_qrng_sample_reuse(df, cfg))

    n_errors = sum(1 for item in failures if item.severity.lower() == "error")
    n_warnings = sum(1 for item in failures if item.severity.lower() == "warning")

    sync_valid = n_errors == 0

    quality_summary = build_sync_quality_summary(df, cfg)

    min_sync_quality = quality_summary.get("sync_quality", {}).get("min")
    mean_sync_quality = quality_summary.get("sync_quality", {}).get("mean")
    max_clock_drift = quality_summary.get("clock_drift_ms_abs", {}).get("max")
    max_gap = quality_summary.get("eeg_qrng_gap_ms", {}).get("max")
    median_gap = quality_summary.get("eeg_qrng_gap_ms", {}).get("median")

    if sync_valid:
        status = "sync_valid"
        interpretation = (
            "The synchronized EEG–QRNG input passed synchronization validation. "
            "The dataset can proceed to empirical synchronized CDR, subject to later "
            "windowing, feature, control and gate checks."
        )
    else:
        status = cfg.status_sync_invalid
        interpretation = (
            "The synchronized EEG–QRNG input failed synchronization validation. "
            "Empirical synchronized CDR is blocked until synchronization errors are fixed."
        )

    result = SyncValidationResult(
        status=status,
        sync_valid=bool(sync_valid),
        protocol_only=False,
        input_file=str(input_path),
        input_exists=True,
        schema_status=schema_status,
        schema_valid=True,
        n_rows=int(quality_summary.get("n_rows", len(df))),
        n_subjects=int(quality_summary.get("n_subjects", 0)),
        n_sessions=int(quality_summary.get("n_sessions", 0)),
        n_windows=int(quality_summary.get("n_windows", 0)),
        min_sync_quality=min_sync_quality,
        mean_sync_quality=mean_sync_quality,
        max_clock_drift_ms_abs=max_clock_drift,
        max_eeg_qrng_gap_ms=max_gap,
        median_eeg_qrng_gap_ms=median_gap,
        n_failures=int(len(failures)),
        n_errors=int(n_errors),
        n_warnings=int(n_warnings),
        can_proceed_to_empirical_cdr=bool(sync_valid),
        interpretation=interpretation,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": "phase3_2e_sync_validator",
        "status": result.status,
        "result": asdict(result),
        "schema_report_status": schema_status,
        "input_file": str(input_path),
        "thresholds": {
            "min_sync_quality": float(cfg.min_sync_quality),
            "max_clock_drift_ms": float(cfg.max_clock_drift_ms),
            "max_timestamp_gap_ms": float(cfg.max_timestamp_gap_ms),
            "accepted_sync_modes": list(cfg.accepted_sync_modes),
        },
        "sync_quality_summary": quality_summary,
        "failures": _failure_records(failures),
    }

    save_sync_outputs(cfg, report, failures, result)

    return report


def save_sync_outputs(
    cfg: Phase32EConfig,
    report: Dict[str, Any],
    failures: Sequence[SyncFailure],
    result: SyncValidationResult,
) -> None:
    save_json(Path(cfg.sync_report_json), report)
    _write_failures_csv(Path(cfg.sync_failures_csv), failures)
    _write_summary_txt(Path(cfg.sync_summary_txt), result, report)

    print(f"[Phase3.2E Sync] Saved report: {cfg.sync_report_json}")
    print(f"[Phase3.2E Sync] Saved failures: {cfg.sync_failures_csv}")
    print(f"[Phase3.2E Sync] Saved summary: {cfg.sync_summary_txt}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2e_config()

    report = validate_phase3_2e_synchronization(
        cfg=cfg,
        run_schema_first=True,
    )

    result = report.get("result", {})

    print("\n" + "=" * 78)
    print("Phase III.2E synchronization validation completed")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"sync_valid: {result.get('sync_valid')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"input_exists: {result.get('input_exists')}")
    print(f"schema_status: {result.get('schema_status')}")
    print(f"schema_valid: {result.get('schema_valid')}")
    print(f"n_rows: {result.get('n_rows')}")
    print(f"n_subjects: {result.get('n_subjects')}")
    print(f"n_sessions: {result.get('n_sessions')}")
    print(f"n_errors: {result.get('n_errors')}")
    print(f"n_warnings: {result.get('n_warnings')}")
    print(f"can_proceed_to_empirical_cdr: {result.get('can_proceed_to_empirical_cdr')}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()