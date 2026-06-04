from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    describe_config,
    load_phase3_2e_config,
    optional_schema_columns,
    required_schema_columns,
)

from src.phase3_2e_schema import (
    validate_phase3_2e_schema,
)

from src.phase3_2e_sync_validator import (
    validate_phase3_2e_synchronization,
)


# =========================================================
# Phase III.2E — Loader
# =========================================================
#
# Purpose:
#
#   Load the synchronized EEG–QRNG input table after schema and synchronization
#   validation.
#
# Phase III.2E is different from Phase III.2A/B/C/D:
#
#   - It does not construct EEG/RNG alignment from Sleep-EDF + ANU QRNG.
#   - It does not reuse non-synchronized public EEG/RNG data.
#   - It expects a future synchronized window-level CSV:
#
#       data/raw/phase3_2e/synchronized/phase3_2e_synchronized_input.csv
#
#   - If that CSV is absent, the loader remains in protocol-only mode and does
#     not raise an exception.
#
# Loader behavior:
#
#   If synchronized CSV is missing:
#       status = pending_synchronized_data
#       returns empty dataframe
#       writes protocol-only metadata/report
#
#   If schema fails:
#       status = loader_blocked_by_schema
#       returns empty dataframe
#
#   If sync validation fails:
#       status = loader_blocked_by_sync
#       returns empty dataframe
#
#   If sync validation passes:
#       status = loaded_synchronized_data
#       loads synchronized windows
#       adds audit columns
#       saves canonical loader output
#
# Outputs:
#
#   data/interim/phase3_2e/phase3_2e_loaded_synchronized_windows.csv
#   data/interim/phase3_2e/phase3_2e_loader_metadata.json
#   results/phase3_2e/phase3_2e_loader_report.json
#   results/phase3_2e/phase3_2e_loader_summary.txt
#


# =========================================================
# Constants
# =========================================================

LOADER_MODULE = "phase3_2e_loader"
LOADER_VERSION = "phase3_2e_loader_v1_protocol_sync_aware"


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

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def _path_size_mtime(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "mtime_human": None,
        }

    stat = path.stat()

    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
    }


def loader_output_csv(cfg: Phase32EConfig) -> Path:
    return Path(cfg.interim_dir) / "phase3_2e_loaded_synchronized_windows.csv"


def loader_metadata_json(cfg: Phase32EConfig) -> Path:
    return Path(cfg.interim_dir) / "phase3_2e_loader_metadata.json"


def loader_report_json(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_loader_report.json"


def loader_summary_txt(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_loader_summary.txt"


def _empty_loader_frame(cfg: Phase32EConfig) -> pd.DataFrame:
    columns: List[str] = []

    columns.extend(list(required_schema_columns(cfg)))
    columns.extend(list(optional_schema_columns(cfg)))

    audit_columns = [
        "loader_row_id",
        "phase3_2e_window_key",
        "window_seconds",
        "window_midpoint_utc",
        "eeg_qrng_gap_ms",
        "clock_drift_abs_ms",
        "sync_row_valid",
        "sync_mode_effective",
        "loader_status",
    ]

    columns.extend(audit_columns)

    return pd.DataFrame(columns=list(dict.fromkeys(columns)))


# =========================================================
# Loading and normalization
# =========================================================

def load_raw_synchronized_csv(cfg: Phase32EConfig) -> Tuple[pd.DataFrame, bool]:
    path = Path(cfg.synchronized_input_csv)

    if not path.exists():
        return pd.DataFrame(), False

    df = pd.read_csv(path)

    return df, True


def parse_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    timestamp_columns = [
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
    ]

    for col in timestamp_columns:
        if col in out.columns:
            out[f"__parsed_{col}"] = pd.to_datetime(
                out[col],
                errors="coerce",
                utc=True,
            )

    return out


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    return series.apply(
        lambda x: x.isoformat() if isinstance(x, pd.Timestamp) and not pd.isna(x) else ""
    )


def add_loader_audit_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    out = df.copy()
    out = parse_timestamp_columns(out)

    out["loader_row_id"] = np.arange(len(out), dtype=int)

    if cfg.sync_mode_column in out.columns:
        out["sync_mode_effective"] = (
            out[cfg.sync_mode_column]
            .fillna(cfg.default_sync_mode_if_missing)
            .astype(str)
        )
    else:
        out["sync_mode_effective"] = str(cfg.default_sync_mode_if_missing)

    # -----------------------------------------------------
    # Window duration and midpoint
    # -----------------------------------------------------
    start_col = "__parsed_window_start_utc"
    end_col = "__parsed_window_end_utc"

    if start_col in out.columns and end_col in out.columns:
        duration = (out[end_col] - out[start_col]).dt.total_seconds()
        out["window_seconds"] = duration.round().astype("Int64")

        midpoint = out[start_col] + (out[end_col] - out[start_col]) / 2
        out["window_midpoint_utc"] = _format_timestamp_series(midpoint)

    elif cfg.window_size_column in out.columns:
        out["window_seconds"] = pd.to_numeric(out[cfg.window_size_column], errors="coerce").astype("Int64")
        out["window_midpoint_utc"] = ""

    else:
        out["window_seconds"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["window_midpoint_utc"] = ""

    # -----------------------------------------------------
    # EEG–QRNG timestamp gap
    # -----------------------------------------------------
    eeg_col = "__parsed_eeg_timestamp_utc"
    qrng_col = "__parsed_qrng_timestamp_utc"

    if eeg_col in out.columns and qrng_col in out.columns:
        out["eeg_qrng_gap_ms"] = (
            (out[eeg_col] - out[qrng_col]).abs().dt.total_seconds() * 1000.0
        )
    else:
        out["eeg_qrng_gap_ms"] = np.nan

    # -----------------------------------------------------
    # Drift and sync row validity
    # -----------------------------------------------------
    if "clock_drift_ms" in out.columns:
        out["clock_drift_abs_ms"] = pd.to_numeric(
            out["clock_drift_ms"],
            errors="coerce",
        ).abs()
    else:
        out["clock_drift_abs_ms"] = np.nan

    if "sync_quality" in out.columns:
        sync_quality = pd.to_numeric(out["sync_quality"], errors="coerce")
    else:
        sync_quality = pd.Series([np.nan] * len(out), index=out.index)

    out["sync_row_valid"] = (
        sync_quality.ge(float(cfg.min_sync_quality))
        & out["clock_drift_abs_ms"].le(float(cfg.max_clock_drift_ms))
        & pd.to_numeric(out["eeg_qrng_gap_ms"], errors="coerce").le(float(cfg.max_timestamp_gap_ms))
    ).astype(int)

    # -----------------------------------------------------
    # Stable key
    # -----------------------------------------------------
    required_key_cols = ["subject_id", "session_id", "window_id"]

    if all(col in out.columns for col in required_key_cols):
        out["phase3_2e_window_key"] = (
            out["subject_id"].astype(str)
            + "::"
            + out["session_id"].astype(str)
            + "::"
            + out["window_id"].astype(str)
        )
    else:
        out["phase3_2e_window_key"] = out["loader_row_id"].astype(str)

    out["loader_status"] = "loaded_synchronized_data"

    # -----------------------------------------------------
    # Sort deterministically
    # -----------------------------------------------------
    sort_cols: List[str] = []

    for col in [
        "subject_id",
        "session_id",
        "__parsed_window_start_utc",
        "window_id",
        "loader_row_id",
    ]:
        if col in out.columns:
            sort_cols.append(col)

    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
        out["loader_row_id"] = np.arange(len(out), dtype=int)

    # Keep parsed columns internal only.
    parsed_cols = [col for col in out.columns if col.startswith("__parsed_")]
    out = out.drop(columns=parsed_cols, errors="ignore")

    return out


def enforce_loader_column_order(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    if df.empty:
        return df

    preferred: List[str] = []

    preferred.extend([
        "loader_row_id",
        "phase3_2e_window_key",
    ])

    preferred.extend(list(cfg.identity_columns))
    preferred.extend(list(cfg.timing_columns))

    preferred.extend([
        "window_seconds",
        "window_midpoint_utc",
        "eeg_qrng_gap_ms",
        "clock_drift_abs_ms",
        "sync_row_valid",
        "sync_mode_effective",
    ])

    if cfg.sync_mode_column in df.columns:
        preferred.append(cfg.sync_mode_column)

    preferred.extend(list(cfg.required_eeg_columns))
    preferred.extend(list(cfg.required_multichannel_columns))
    preferred.extend(list(cfg.required_qrng_columns))
    preferred.extend(list(cfg.optional_eeg_columns))
    preferred.extend(list(cfg.optional_qrng_columns))

    preferred.append("loader_status")

    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]

    return df[ordered + remaining].copy()


# =========================================================
# Summaries
# =========================================================

def _value_counts_dict(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}

    return {
        str(k): int(v)
        for k, v in df[col].fillna("").astype(str).value_counts().to_dict().items()
    }


def _numeric_summary(df: pd.DataFrame, col: str) -> Dict[str, Optional[float]]:
    if df.empty or col not in df.columns:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    x = pd.to_numeric(df[col], errors="coerce")

    if not x.notna().any():
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def summarize_loaded_synchronized_frame(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    if df.empty:
        return {
            "n_rows": 0,
            "n_subjects": 0,
            "n_sessions": 0,
            "n_windows": 0,
            "window_seconds_counts": {},
            "sync_mode_counts": {},
            "sync_row_valid_counts": {},
            "sync_quality": _numeric_summary(df, "sync_quality"),
            "clock_drift_abs_ms": _numeric_summary(df, "clock_drift_abs_ms"),
            "eeg_qrng_gap_ms": _numeric_summary(df, "eeg_qrng_gap_ms"),
            "qrng_state_counts": {},
            "eeg_channel_counts": {},
            "primary_channel_counts": {},
            "secondary_channel_counts": {},
        }

    n_sessions = 0

    if {"subject_id", "session_id"}.issubset(df.columns):
        n_sessions = int(df[["subject_id", "session_id"]].drop_duplicates().shape[0])

    return {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "n_sessions": n_sessions,
        "n_windows": int(df["window_id"].nunique()) if "window_id" in df.columns else 0,

        "window_seconds_counts": _value_counts_dict(df, "window_seconds"),
        "sync_mode_counts": _value_counts_dict(df, "sync_mode_effective"),
        "sync_row_valid_counts": _value_counts_dict(df, "sync_row_valid"),

        "sync_quality": _numeric_summary(df, "sync_quality"),
        "clock_drift_abs_ms": _numeric_summary(df, "clock_drift_abs_ms"),
        "eeg_qrng_gap_ms": _numeric_summary(df, "eeg_qrng_gap_ms"),

        "qrng_state_counts": _value_counts_dict(df, "qrng_state"),
        "eeg_channel_counts": _value_counts_dict(df, "eeg_channel"),
        "primary_channel_counts": _value_counts_dict(df, "eeg_primary_channel"),
        "secondary_channel_counts": _value_counts_dict(df, "eeg_secondary_channel"),
    }


def build_loader_metadata(
    cfg: Phase32EConfig,
    status: str,
    df: pd.DataFrame,
    schema_report: Dict[str, Any],
    sync_report: Dict[str, Any],
    interpretation: str,
) -> Dict[str, Any]:
    input_path = Path(cfg.synchronized_input_csv)
    summary = summarize_loaded_synchronized_frame(df, cfg)

    return {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "phase_title": cfg.phase_title,
        "config_version": cfg.version,
        "loader_module": LOADER_MODULE,
        "loader_version": LOADER_VERSION,
        "created_at": _now_str(),

        "status": status,
        "interpretation": interpretation,

        "protocol_status": cfg.protocol_status,
        "empirical_execution_status": cfg.empirical_execution_status,

        "input_file": str(input_path),
        "input_file_info": _path_size_mtime(input_path),

        "output_csv": str(loader_output_csv(cfg)),
        "metadata_json": str(loader_metadata_json(cfg)),
        "loader_report_json": str(loader_report_json(cfg)),
        "loader_summary_txt": str(loader_summary_txt(cfg)),

        "schema_status": schema_report.get("status"),
        "schema_result": schema_report.get("result", {}),
        "sync_status": sync_report.get("status"),
        "sync_result": sync_report.get("result", {}),

        "can_proceed_to_windowing": bool(status == "loaded_synchronized_data"),
        "can_proceed_to_feature_construction": bool(status == "loaded_synchronized_data"),
        "can_proceed_to_empirical_cdr": bool(status == "loaded_synchronized_data"),

        "config": describe_config(cfg),
        "summary": summary,
    }


def build_loader_report(
    cfg: Phase32EConfig,
    metadata: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": LOADER_MODULE,
        "loader_version": LOADER_VERSION,
        "status": metadata.get("status"),
        "interpretation": metadata.get("interpretation"),
        "created_at": metadata.get("created_at"),
        "input_file": metadata.get("input_file"),
        "output_csv": metadata.get("output_csv"),
        "metadata_json": metadata.get("metadata_json"),
        "schema_status": metadata.get("schema_status"),
        "sync_status": metadata.get("sync_status"),
        "can_proceed_to_windowing": metadata.get("can_proceed_to_windowing"),
        "can_proceed_to_feature_construction": metadata.get("can_proceed_to_feature_construction"),
        "can_proceed_to_empirical_cdr": metadata.get("can_proceed_to_empirical_cdr"),
        "summary": metadata.get("summary", {}),
        "columns": list(df.columns),
    }


def write_loader_summary_txt(
    path: Path,
    metadata: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = metadata.get("summary", {})

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Loader Summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {metadata.get('status')}")
    lines.append(f"interpretation: {metadata.get('interpretation')}")
    lines.append("")
    lines.append(f"input_file: {metadata.get('input_file')}")
    lines.append(f"output_csv: {metadata.get('output_csv')}")
    lines.append(f"metadata_json: {metadata.get('metadata_json')}")
    lines.append("")
    lines.append(f"schema_status: {metadata.get('schema_status')}")
    lines.append(f"sync_status: {metadata.get('sync_status')}")
    lines.append("")
    lines.append(f"can_proceed_to_windowing: {metadata.get('can_proceed_to_windowing')}")
    lines.append(f"can_proceed_to_feature_construction: {metadata.get('can_proceed_to_feature_construction')}")
    lines.append(f"can_proceed_to_empirical_cdr: {metadata.get('can_proceed_to_empirical_cdr')}")
    lines.append("")
    lines.append(f"n_rows: {summary.get('n_rows')}")
    lines.append(f"n_subjects: {summary.get('n_subjects')}")
    lines.append(f"n_sessions: {summary.get('n_sessions')}")
    lines.append(f"n_windows: {summary.get('n_windows')}")
    lines.append("")
    lines.append(f"window_seconds_counts: {summary.get('window_seconds_counts')}")
    lines.append(f"sync_mode_counts: {summary.get('sync_mode_counts')}")
    lines.append(f"sync_row_valid_counts: {summary.get('sync_row_valid_counts')}")
    lines.append("")
    lines.append(f"sync_quality: {summary.get('sync_quality')}")
    lines.append(f"clock_drift_abs_ms: {summary.get('clock_drift_abs_ms')}")
    lines.append(f"eeg_qrng_gap_ms: {summary.get('eeg_qrng_gap_ms')}")
    lines.append("")
    lines.append(f"qrng_state_counts: {summary.get('qrng_state_counts')}")
    lines.append(f"eeg_channel_counts: {summary.get('eeg_channel_counts')}")
    lines.append(f"primary_channel_counts: {summary.get('primary_channel_counts')}")
    lines.append(f"secondary_channel_counts: {summary.get('secondary_channel_counts')}")
    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Cache compatibility
# =========================================================

def _cached_loader_is_compatible(
    cfg: Phase32EConfig,
    cached_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> Tuple[bool, str]:
    if cached_df.empty:
        return False, "cached_dataframe_empty"

    if str(metadata.get("loader_version")) != LOADER_VERSION:
        return False, "loader_version_changed"

    if str(metadata.get("config_version")) != str(cfg.version):
        return False, "config_version_changed"

    input_path = Path(cfg.synchronized_input_csv)
    current_input_info = _path_size_mtime(input_path)
    cached_input_info = metadata.get("input_file_info", {})

    if bool(current_input_info.get("exists")) != bool(cached_input_info.get("exists")):
        return False, "input_existence_changed"

    if current_input_info.get("size_bytes") != cached_input_info.get("size_bytes"):
        return False, "input_size_changed"

    if current_input_info.get("mtime") != cached_input_info.get("mtime"):
        return False, "input_mtime_changed"

    if metadata.get("status") != "loaded_synchronized_data":
        return False, "cached_status_not_loaded_synchronized_data"

    required_columns = [
        "loader_row_id",
        "phase3_2e_window_key",
        "sync_row_valid",
        "eeg_qrng_gap_ms",
        "clock_drift_abs_ms",
    ]

    missing = [col for col in required_columns if col not in cached_df.columns]

    if missing:
        return False, f"cached_missing_loader_columns={missing}"

    return True, "compatible"


# =========================================================
# Save / load
# =========================================================

def save_phase3_2e_loader_outputs(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    cfg: Phase32EConfig,
) -> None:
    Path(cfg.interim_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    output_csv = loader_output_csv(cfg)
    metadata_json = loader_metadata_json(cfg)
    report_json = loader_report_json(cfg)
    summary_txt = loader_summary_txt(cfg)

    # Even in protocol-only mode, save an empty CSV with expected columns.
    df.to_csv(output_csv, index=False)

    save_json(metadata_json, metadata)

    report = build_loader_report(cfg, metadata, df)
    save_json(report_json, report)

    write_loader_summary_txt(summary_txt, metadata)

    print(f"[Phase3.2E Loader] Saved loaded windows: {output_csv}")
    print(f"[Phase3.2E Loader] Saved metadata: {metadata_json}")
    print(f"[Phase3.2E Loader] Saved report: {report_json}")
    print(f"[Phase3.2E Loader] Saved summary: {summary_txt}")


def load_cached_phase3_2e_loader_outputs(
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    output_csv = loader_output_csv(cfg)
    metadata_json = loader_metadata_json(cfg)

    df = pd.read_csv(output_csv)

    with open(metadata_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return df, metadata


# =========================================================
# Main build/load functions
# =========================================================

def build_phase3_2e_loader_dataset(
    cfg: Optional[Phase32EConfig] = None,
    run_schema_first: bool = True,
    run_sync_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print("[Phase3.2E Loader] Starting synchronized loader")

    schema_report: Dict[str, Any] = {}
    sync_report: Dict[str, Any] = {}

    if run_sync_first:
        sync_report = validate_phase3_2e_synchronization(
            cfg=cfg,
            run_schema_first=run_schema_first,
        )

        # The sync validator already runs schema when run_schema_first=True.
        schema_report_path = Path(cfg.schema_report_json)

        if schema_report_path.exists():
            with open(schema_report_path, "r", encoding="utf-8") as f:
                schema_report = json.load(f)

    elif run_schema_first:
        schema_report = validate_phase3_2e_schema(cfg)

    input_path = Path(cfg.synchronized_input_csv)
    input_exists = input_path.exists()

    # -----------------------------------------------------
    # Protocol-only path: no synchronized data yet.
    # -----------------------------------------------------
    if not input_exists:
        df = _empty_loader_frame(cfg)

        status = cfg.status_pending_synchronized_data
        interpretation = (
            "No synchronized EEG–QRNG input CSV was found. Loader completed in "
            "protocol-only mode and saved empty canonical outputs. Empirical "
            "windowing, feature construction and CDR remain blocked until real "
            "synchronized data are provided."
        )

        metadata = build_loader_metadata(
            cfg=cfg,
            status=status,
            df=df,
            schema_report=schema_report,
            sync_report=sync_report,
            interpretation=interpretation,
        )

        save_phase3_2e_loader_outputs(df, metadata, cfg)

        return df, metadata

    # -----------------------------------------------------
    # Schema/sync blocked path.
    # -----------------------------------------------------
    sync_result = sync_report.get("result", {})
    sync_status = str(sync_result.get("status", sync_report.get("status", "")))
    sync_valid = bool(sync_result.get("sync_valid", False))

    schema_result = schema_report.get("result", {})
    schema_valid = bool(schema_result.get("schema_valid", False))

    if not schema_valid:
        df = _empty_loader_frame(cfg)

        status = "loader_blocked_by_schema"
        interpretation = (
            "Synchronized input CSV exists, but schema validation failed. Loader "
            "did not load empirical rows. Fix schema errors before windowing or CDR."
        )

        metadata = build_loader_metadata(
            cfg=cfg,
            status=status,
            df=df,
            schema_report=schema_report,
            sync_report=sync_report,
            interpretation=interpretation,
        )

        save_phase3_2e_loader_outputs(df, metadata, cfg)

        return df, metadata

    if run_sync_first and not sync_valid:
        df = _empty_loader_frame(cfg)

        status = "loader_blocked_by_sync"
        interpretation = (
            "Synchronized input CSV exists and schema validation passed, but "
            "synchronization validation failed. Loader did not expose empirical "
            "rows for CDR. Fix synchronization errors first."
        )

        metadata = build_loader_metadata(
            cfg=cfg,
            status=status,
            df=df,
            schema_report=schema_report,
            sync_report=sync_report,
            interpretation=interpretation,
        )

        save_phase3_2e_loader_outputs(df, metadata, cfg)

        return df, metadata

    # -----------------------------------------------------
    # Full loading path.
    # -----------------------------------------------------
    raw_df, _ = load_raw_synchronized_csv(cfg)

    loaded_df = add_loader_audit_columns(raw_df, cfg)
    loaded_df = enforce_loader_column_order(loaded_df, cfg)

    status = "loaded_synchronized_data"
    interpretation = (
        "Synchronized EEG–QRNG input CSV was loaded successfully after schema "
        "and synchronization validation. The dataset can proceed to windowing, "
        "feature construction, controls and empirical synchronized CDR."
    )

    metadata = build_loader_metadata(
        cfg=cfg,
        status=status,
        df=loaded_df,
        schema_report=schema_report,
        sync_report=sync_report,
        interpretation=interpretation,
    )

    save_phase3_2e_loader_outputs(loaded_df, metadata, cfg)

    return loaded_df, metadata


def load_or_build_phase3_2e_dataset(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    run_schema_first: bool = True,
    run_sync_first: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    output_csv = loader_output_csv(cfg)
    metadata_path = loader_metadata_json(cfg)

    if output_csv.exists() and metadata_path.exists() and not force_rebuild:
        print(f"[Phase3.2E Loader] Loading cached output: {output_csv}")

        try:
            cached_df, metadata = load_cached_phase3_2e_loader_outputs(cfg)
            compatible, reason = _cached_loader_is_compatible(cfg, cached_df, metadata)

            if compatible:
                print(f"[Phase3.2E Loader] Cached output compatible: {reason}")
                return cached_df, metadata

            print(
                "[Phase3.2E Loader] Cached output is not compatible. "
                f"Reason: {reason}. Rebuilding..."
            )

        except Exception as exc:
            print(f"[Phase3.2E Loader] Failed to load cached output: {exc}. Rebuilding...")

    df, metadata = build_phase3_2e_loader_dataset(
        cfg=cfg,
        run_schema_first=run_schema_first,
        run_sync_first=run_sync_first,
    )

    return df, metadata


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2e_config()

    df, metadata = load_or_build_phase3_2e_dataset(
        cfg=cfg,
        force_rebuild=True,
        run_schema_first=True,
        run_sync_first=True,
    )

    summary = metadata.get("summary", {})

    print("\n" + "=" * 78)
    print("Phase III.2E loader completed")
    print("=" * 78)
    print(f"status: {metadata.get('status')}")
    print(f"schema_status: {metadata.get('schema_status')}")
    print(f"sync_status: {metadata.get('sync_status')}")
    print(f"n_rows: {summary.get('n_rows')}")
    print(f"n_subjects: {summary.get('n_subjects')}")
    print(f"n_sessions: {summary.get('n_sessions')}")
    print(f"n_windows: {summary.get('n_windows')}")
    print(f"can_proceed_to_windowing: {metadata.get('can_proceed_to_windowing')}")
    print(f"can_proceed_to_feature_construction: {metadata.get('can_proceed_to_feature_construction')}")
    print(f"can_proceed_to_empirical_cdr: {metadata.get('can_proceed_to_empirical_cdr')}")
    print(f"output_csv: {metadata.get('output_csv')}")
    print(f"metadata_json: {metadata.get('metadata_json')}")
    print(f"interpretation: {metadata.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()