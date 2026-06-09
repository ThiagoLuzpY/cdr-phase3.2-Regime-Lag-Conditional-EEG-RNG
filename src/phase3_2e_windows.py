from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    WindowSpec,
    describe_config,
    load_phase3_2e_config,
    window_seconds_map,
)

from src.phase3_2e_loader import (
    load_or_build_phase3_2e_dataset,
)


# =========================================================
# Phase III.2E — Window Builder
# =========================================================
#
# Purpose:
#
#   Build and validate the canonical synchronized window table for Phase III.2E.
#
# Phase III.2E consumes a synchronized EEG–QRNG table whose rows represent the
# smallest auditable windows available in the current source. This module:
#
#   - consumes the loader output;
#   - preserves native registered window sizes;
#   - derives longer registered windows from complete, contiguous and
#     non-overlapping source windows;
#   - never fabricates a shorter temporal resolution from coarser input;
#   - verifies registered window sizes;
#   - classifies primary vs exploratory windows;
#   - enforces sync-valid row requirements;
#   - checks minimum total windows;
#   - checks minimum windows per subject;
#   - builds a window inventory;
#   - blocks empirical feature construction if synchronized data are absent or
#     window validity is insufficient.
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   empty canonical outputs are saved
#
# Outputs:
#
#   data/interim/phase3_2e/phase3_2e_windows.csv
#   data/interim/phase3_2e/phase3_2e_window_inventory.csv
#   results/phase3_2e/phase3_2e_windows_report.json
#   results/phase3_2e/phase3_2e_windows_summary.txt
#


# =========================================================
# Constants
# =========================================================

WINDOW_MODULE = "phase3_2e_windows"
WINDOW_VERSION = "phase3_2e_windows_v2_multiresolution_registered_windows"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class WindowBuildResult:
    status: str
    protocol_only: bool
    input_rows: int
    output_rows: int
    n_subjects: int
    n_sessions: int
    n_valid_windows: int
    n_primary_valid_windows: int
    n_exploratory_valid_windows: int
    n_registered_window_sizes: int
    n_unregistered_window_rows: int
    can_proceed_to_feature_construction: bool
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

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except Exception:
        return default


def _value_counts_dict(
    df: pd.DataFrame,
    col: str,
) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}

    return {
        str(k): int(v)
        for k, v in (
            df[col]
            .fillna("")
            .astype(str)
            .value_counts()
            .to_dict()
            .items()
        )
    }


def _numeric_summary(
    df: pd.DataFrame,
    col: str,
) -> Dict[str, Optional[float]]:
    if df.empty or col not in df.columns:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    x = pd.to_numeric(
        df[col],
        errors="coerce",
    )

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


# =========================================================
# Output paths
# =========================================================

def windows_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(cfg.windows_csv)


def window_inventory_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.interim_dir)
        / "phase3_2e_window_inventory.csv"
    )


def windows_report_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_windows_report.json"
    )


def windows_summary_txt_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_windows_summary.txt"
    )


def windows_metadata_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.interim_dir)
        / "phase3_2e_windows_metadata.json"
    )


# =========================================================
# Empty outputs
# =========================================================

def empty_windows_frame(
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    columns = [
        "window_row_id",
        "phase3_2e_window_key",
        "subject_id",
        "session_id",
        "window_id",
        "window_seconds",
        "window_derivation",
        "source_window_seconds",
        "source_window_count",
        "source_window_ids",
        "window_spec_name",
        "window_scope",
        "registered_window_size",
        "primary_window",
        "exploratory_window",
        "window_size_ready",
        "window_total_count_ready",
        "window_subject_count_ready",
        "sync_row_valid",
        "window_valid",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
        "window_status",
    ]

    return pd.DataFrame(columns=columns)


def empty_window_inventory(
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for spec in cfg.window_specs:
        rows.append(
            {
                "window_spec_name": spec.name,
                "window_seconds": int(spec.seconds),
                "primary": bool(spec.primary),
                "min_windows_total": int(
                    spec.min_windows_total
                ),
                "min_windows_per_subject": int(
                    spec.min_windows_per_subject
                ),
                "n_rows": 0,
                "n_valid_windows": 0,
                "n_subjects": 0,
                "min_windows_per_subject_observed": 0,
                "median_windows_per_subject_observed": 0.0,
                "max_windows_per_subject_observed": 0,
                "meets_total_minimum": False,
                "meets_subject_minimum": False,
                "window_size_ready": False,
                "description": spec.description,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# Window derivation and validation
# =========================================================

def derive_window_seconds_if_needed(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    out = df.copy()

    if "window_seconds" in out.columns:
        out["window_seconds"] = (
            pd.to_numeric(
                out["window_seconds"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

        return out

    if (
        "window_start_utc" not in out.columns
        or "window_end_utc" not in out.columns
    ):
        out["window_seconds"] = pd.Series(
            [pd.NA] * len(out),
            dtype="Int64",
        )

        return out

    start = pd.to_datetime(
        out["window_start_utc"],
        errors="coerce",
        utc=True,
    )

    end = pd.to_datetime(
        out["window_end_utc"],
        errors="coerce",
        utc=True,
    )

    duration = (
        end - start
    ).dt.total_seconds()

    out["window_seconds"] = (
        duration
        .round()
        .astype("Int64")
    )

    return out


def _timestamp_to_iso_z(
    value: Any,
) -> Optional[str]:
    timestamp = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(timestamp):
        return None

    return timestamp.isoformat().replace(
        "+00:00",
        "Z",
    )


def _first_valid(
    series: pd.Series,
    default: Any = None,
) -> Any:
    valid = series[
        series.notna()
    ]

    if valid.empty:
        return default

    return valid.iloc[0]


def _last_valid(
    series: pd.Series,
    default: Any = None,
) -> Any:
    valid = series[
        series.notna()
    ]

    if valid.empty:
        return default

    return valid.iloc[-1]


def _mean_timestamp_iso(
    series: pd.Series,
) -> Optional[str]:
    timestamps = pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    ).dropna()

    if timestamps.empty:
        return None

    mean_ns = int(
        np.mean(
            timestamps
            .astype("int64")
            .to_numpy(dtype=np.int64)
        )
    )

    return _timestamp_to_iso_z(
        pd.to_datetime(
            mean_ns,
            unit="ns",
            utc=True,
        )
    )


def _signed_max_abs(
    series: pd.Series,
    default: float = 0.0,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return float(default)

    return float(
        values.loc[
            values.abs().idxmax()
        ]
    )


def _numeric_fraction(
    series: pd.Series,
) -> float:
    if series.empty:
        return 0.0

    return float(
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .notna()
        .mean()
    )


def _aggregate_column(
    column: str,
    series: pd.Series,
) -> Any:
    """
    Aggregate one loader column across a longer non-overlapping window.

    Continuous features are averaged. State/bin columns retain the last
    observed value. Validity columns require all source rows to be valid.
    Quality columns use the minimum observed quality.
    """
    lower = column.lower()

    if (
        lower.endswith("_state")
        or lower.endswith("_bin")
    ):
        value = _last_valid(series)

        if value is None:
            return np.nan

        numeric = pd.to_numeric(
            pd.Series([value]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(numeric):
            return int(numeric)

        return value

    if lower.endswith("_valid"):
        values = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .fillna(0)
        )

        return int(
            values.eq(1).all()
        )

    if column == "empirical_claim_allowed":
        normalized = (
            series.astype(str)
            .str.strip()
            .str.lower()
        )

        return bool(
            normalized.isin(
                {
                    "1",
                    "true",
                    "t",
                    "yes",
                    "y",
                    "sim",
                }
            ).all()
        )

    if column in {
        "sync_quality",
        "eeg_quality_score",
        "qrng_quality_score",
    }:
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return float(
            values.min()
        )

    if column == "clock_drift_ms":
        return _signed_max_abs(series)

    if column == "clock_drift_abs_ms":
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return float(
            values.abs().max()
        )

    if (
        lower.endswith("_count")
        or lower.endswith("_samples")
        or lower.endswith("_bits")
        or lower.startswith("n_")
    ):
        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        if values.notna().any():
            return float(
                values.sum()
            )

    if (
        pd.api.types.is_numeric_dtype(
            series
        )
        or _numeric_fraction(series) >= 0.80
    ):
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return float(
            values.mean()
        )

    return _first_valid(series)


def _rows_are_contiguous(
    previous_row: pd.Series,
    current_row: pd.Series,
    source_seconds: int,
    cfg: Phase32EConfig,
) -> bool:
    previous_start = pd.to_datetime(
        previous_row.get(
            "window_start_utc"
        ),
        errors="coerce",
        utc=True,
    )

    previous_end = pd.to_datetime(
        previous_row.get(
            "window_end_utc"
        ),
        errors="coerce",
        utc=True,
    )

    current_start = pd.to_datetime(
        current_row.get(
            "window_start_utc"
        ),
        errors="coerce",
        utc=True,
    )

    current_end = pd.to_datetime(
        current_row.get(
            "window_end_utc"
        ),
        errors="coerce",
        utc=True,
    )

    timestamps = [
        previous_start,
        previous_end,
        current_start,
        current_end,
    ]

    if any(
        pd.isna(value)
        for value in timestamps
    ):
        return False

    tolerance_seconds = max(
        float(
            cfg.max_timestamp_gap_ms
        ) / 1000.0,
        0.001,
    )

    previous_duration = (
        previous_end
        - previous_start
    ).total_seconds()

    current_duration = (
        current_end
        - current_start
    ).total_seconds()

    gap_seconds = (
        current_start
        - previous_end
    ).total_seconds()

    return bool(
        abs(
            previous_duration
            - float(source_seconds)
        )
        <= tolerance_seconds
        and abs(
            current_duration
            - float(source_seconds)
        )
        <= tolerance_seconds
        and abs(gap_seconds)
        <= tolerance_seconds
    )


def _split_contiguous_runs(
    group: pd.DataFrame,
    source_seconds: int,
    cfg: Phase32EConfig,
) -> List[pd.DataFrame]:
    if group.empty:
        return []

    ordered = (
        group
        .sort_values(
            [
                "window_start_utc",
                "window_end_utc",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    runs: List[pd.DataFrame] = []
    run_start = 0

    for position in range(
        1,
        len(ordered),
    ):
        previous_row = ordered.iloc[
            position - 1
        ]

        current_row = ordered.iloc[
            position
        ]

        if not _rows_are_contiguous(
            previous_row=previous_row,
            current_row=current_row,
            source_seconds=source_seconds,
            cfg=cfg,
        ):
            runs.append(
                ordered.iloc[
                    run_start:position
                ].copy()
            )

            run_start = position

    runs.append(
        ordered.iloc[
            run_start:
        ].copy()
    )

    return runs


def _aggregate_source_block(
    block: pd.DataFrame,
    target_seconds: int,
    source_seconds: int,
    derived_window_id: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {}

    handled_columns = {
        "window_row_id",
        "phase3_2e_window_key",
        "window_id",
        "window_seconds",
        "window_start_utc",
        "window_end_utc",
        "window_midpoint_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_qrng_gap_ms",
        "window_derivation",
        "source_window_seconds",
        "source_window_count",
        "source_window_ids",
        "window_spec_name",
        "window_scope",
        "registered_window_size",
        "primary_window",
        "exploratory_window",
        "window_size_ready",
        "window_total_count_ready",
        "window_subject_count_ready",
        "window_valid",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
        "window_status",
    }

    for column in block.columns:
        if column not in handled_columns:
            row[column] = _aggregate_column(
                column,
                block[column],
            )

    start = pd.to_datetime(
        block["window_start_utc"],
        errors="coerce",
        utc=True,
    ).min()

    end = pd.to_datetime(
        block["window_end_utc"],
        errors="coerce",
        utc=True,
    ).max()

    if (
        pd.notna(start)
        and pd.notna(end)
    ):
        midpoint = (
            start
            + (end - start) / 2
        )
    else:
        midpoint = None

    if "eeg_timestamp_utc" in block.columns:
        eeg_timestamp = _mean_timestamp_iso(
            block["eeg_timestamp_utc"]
        )
    else:
        eeg_timestamp = None

    if "qrng_timestamp_utc" in block.columns:
        qrng_timestamp = _mean_timestamp_iso(
            block["qrng_timestamp_utc"]
        )
    else:
        qrng_timestamp = None

    row.update(
        {
            "subject_id": str(
                _first_valid(
                    block["subject_id"],
                    "",
                )
            ),
            "session_id": str(
                _first_valid(
                    block["session_id"],
                    "",
                )
            ),
            "window_id": int(
                derived_window_id
            ),
            "window_seconds": int(
                target_seconds
            ),
            "window_start_utc": (
                _timestamp_to_iso_z(start)
            ),
            "window_end_utc": (
                _timestamp_to_iso_z(end)
            ),
            "window_midpoint_utc": (
                _timestamp_to_iso_z(
                    midpoint
                )
            ),
            "eeg_timestamp_utc": (
                eeg_timestamp
            ),
            "qrng_timestamp_utc": (
                qrng_timestamp
            ),
            "window_derivation": (
                "aggregated_nonoverlapping"
            ),
            "source_window_seconds": int(
                source_seconds
            ),
            "source_window_count": int(
                len(block)
            ),
            "source_window_ids": (
                "|".join(
                    block[
                        "window_id"
                    ]
                    .astype(str)
                    .tolist()
                )
                if "window_id"
                in block.columns
                else ""
            ),
        }
    )

    eeg_ts = pd.to_datetime(
        eeg_timestamp,
        errors="coerce",
        utc=True,
    )

    qrng_ts = pd.to_datetime(
        qrng_timestamp,
        errors="coerce",
        utc=True,
    )

    if (
        pd.notna(eeg_ts)
        and pd.notna(qrng_ts)
    ):
        row["eeg_qrng_gap_ms"] = float(
            abs(
                (
                    eeg_ts
                    - qrng_ts
                ).total_seconds()
            )
            * 1000.0
        )

    if "clock_drift_ms" in row:
        drift = _safe_float(
            row["clock_drift_ms"],
            default=None,
        )

        if drift is not None:
            row[
                "clock_drift_abs_ms"
            ] = abs(
                float(drift)
            )

    return row


def _build_target_windows(
    source_df: pd.DataFrame,
    source_seconds: int,
    target_seconds: int,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    if (
        target_seconds <= source_seconds
        or target_seconds
        % source_seconds != 0
    ):
        return pd.DataFrame()

    factor = int(
        target_seconds
        // source_seconds
    )

    required_columns = {
        "subject_id",
        "session_id",
        "window_start_utc",
        "window_end_utc",
    }

    if (
        factor <= 1
        or not required_columns.issubset(
            source_df.columns
        )
    ):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    grouped = source_df.groupby(
        [
            "subject_id",
            "session_id",
        ],
        sort=False,
        dropna=False,
    )

    for (
        _subject_id,
        _session_id,
    ), group in grouped:
        runs = _split_contiguous_runs(
            group=group,
            source_seconds=source_seconds,
            cfg=cfg,
        )

        derived_window_id = 0

        for run in runs:
            n_complete_blocks = int(
                len(run)
                // factor
            )

            for block_index in range(
                n_complete_blocks
            ):
                start_position = (
                    block_index
                    * factor
                )

                block = run.iloc[
                    start_position:
                    start_position + factor
                ].copy()

                if len(block) != factor:
                    continue

                rows.append(
                    _aggregate_source_block(
                        block=block,
                        target_seconds=(
                            target_seconds
                        ),
                        source_seconds=(
                            source_seconds
                        ),
                        derived_window_id=(
                            derived_window_id
                        ),
                    )
                )

                derived_window_id += 1

    return pd.DataFrame(rows)


def build_multiresolution_windows(
    loaded_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    """
    Preserve native registered windows and derive missing longer windows.

    A target duration is generated from the largest available native duration
    that divides it exactly. Shorter windows are never inferred from coarser
    input. With a native 30-second surrogate, 60-second and 90-second windows
    are generated, while 5-second windows remain unavailable.
    """
    prepared = (
        derive_window_seconds_if_needed(
            loaded_df,
            cfg,
        )
    )

    registered_seconds = tuple(
        sorted(
            {
                int(value)
                for value in (
                    cfg.all_window_seconds
                )
            }
        )
    )

    native_seconds = tuple(
        sorted(
            {
                int(value)
                for value in (
                    pd.to_numeric(
                        prepared[
                            "window_seconds"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .tolist()
                )
                if int(value)
                in registered_seconds
            }
        )
    )

    native_df = prepared[
        pd.to_numeric(
            prepared["window_seconds"],
            errors="coerce",
        ).isin(
            registered_seconds
        )
    ].copy()

    if native_df.empty:
        return native_df, {
            "source_rows": int(
                len(loaded_df)
            ),
            "native_registered_rows": 0,
            "output_rows": 0,
            "registered_seconds": list(
                registered_seconds
            ),
            "native_seconds": [],
            "generated_seconds": [],
            "available_seconds": [],
            "unavailable_seconds": list(
                registered_seconds
            ),
            "generated_rows_by_seconds": {},
            "non_overlapping_aggregation": True,
            "shorter_windows_fabricated": False,
        }

    native_df["window_derivation"] = (
        "native"
    )

    native_df[
        "source_window_seconds"
    ] = (
        pd.to_numeric(
            native_df["window_seconds"],
            errors="coerce",
        )
        .astype("Int64")
    )

    native_df[
        "source_window_count"
    ] = 1

    if "window_id" in native_df.columns:
        native_df[
            "source_window_ids"
        ] = (
            native_df[
                "window_id"
            ]
            .astype(str)
        )
    else:
        native_df[
            "source_window_ids"
        ] = ""

    frames: List[
        pd.DataFrame
    ] = [
        native_df
    ]

    generated_seconds: List[
        int
    ] = []

    generated_rows_by_seconds: Dict[
        str,
        int,
    ] = {}

    for target_seconds in registered_seconds:
        if target_seconds in native_seconds:
            continue

        candidate_sources = [
            seconds
            for seconds in native_seconds
            if (
                seconds
                < target_seconds
                and target_seconds
                % seconds == 0
            )
        ]

        if not candidate_sources:
            continue

        source_seconds = max(
            candidate_sources
        )

        source_df = native_df[
            pd.to_numeric(
                native_df[
                    "window_seconds"
                ],
                errors="coerce",
            ).eq(
                source_seconds
            )
        ].copy()

        generated = _build_target_windows(
            source_df=source_df,
            source_seconds=source_seconds,
            target_seconds=target_seconds,
            cfg=cfg,
        )

        if generated.empty:
            continue

        frames.append(
            generated
        )

        generated_seconds.append(
            int(target_seconds)
        )

        generated_rows_by_seconds[
            str(target_seconds)
        ] = int(
            len(generated)
        )

    output = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    sort_columns = [
        column
        for column in [
            "subject_id",
            "session_id",
            "window_start_utc",
            "window_seconds",
        ]
        if column in output.columns
    ]

    if sort_columns:
        output = (
            output
            .sort_values(
                sort_columns,
                kind="stable",
            )
            .reset_index(
                drop=True
            )
        )

    available_seconds = tuple(
        sorted(
            {
                int(value)
                for value in (
                    pd.to_numeric(
                        output[
                            "window_seconds"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            }
        )
    )

    unavailable_seconds = [
        seconds
        for seconds in registered_seconds
        if seconds
        not in available_seconds
    ]

    metadata = {
        "source_rows": int(
            len(loaded_df)
        ),
        "native_registered_rows": int(
            len(native_df)
        ),
        "output_rows": int(
            len(output)
        ),
        "registered_seconds": list(
            registered_seconds
        ),
        "native_seconds": list(
            native_seconds
        ),
        "generated_seconds": (
            generated_seconds
        ),
        "available_seconds": list(
            available_seconds
        ),
        "unavailable_seconds": (
            unavailable_seconds
        ),
        "generated_rows_by_seconds": (
            generated_rows_by_seconds
        ),
        "non_overlapping_aggregation": True,
        "shorter_windows_fabricated": False,
        "note": (
            "Longer windows were generated only from complete, "
            "contiguous and non-overlapping native blocks. "
            "Shorter unavailable durations were not fabricated."
        ),
    }

    return output, metadata


def classify_window_rows(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    out = df.copy()

    spec_by_seconds = (
        window_seconds_map(cfg)
    )

    registered_seconds = {
        int(value)
        for value in (
            cfg.all_window_seconds
        )
    }

    primary_seconds = {
        int(value)
        for value in (
            cfg.primary_window_seconds
        )
    }

    exploratory_seconds = {
        int(value)
        for value in (
            cfg.secondary_window_seconds
        )
    }

    secondary_only_seconds = (
        exploratory_seconds
        - primary_seconds
    )

    out["window_row_id"] = np.arange(
        len(out),
        dtype=int,
    )

    out = (
        derive_window_seconds_if_needed(
            out,
            cfg,
        )
    )

    if {
        "subject_id",
        "session_id",
        "window_id",
    }.issubset(
        out.columns
    ):
        out[
            "phase3_2e_window_key"
        ] = (
            out[
                "subject_id"
            ].astype(str)
            + "::"
            + out[
                "session_id"
            ].astype(str)
            + "::w"
            + out[
                "window_seconds"
            ].astype(str)
            + "::"
            + out[
                "window_id"
            ].astype(str)
        )
    else:
        out[
            "phase3_2e_window_key"
        ] = (
            out[
                "window_row_id"
            ].astype(str)
            + "::w"
            + out[
                "window_seconds"
            ].astype(str)
        )

    window_seconds_numeric = (
        pd.to_numeric(
            out["window_seconds"],
            errors="coerce",
        )
    )

    out[
        "registered_window_size"
    ] = (
        window_seconds_numeric
        .isin(
            registered_seconds
        )
        .astype(int)
    )

    out[
        "primary_window"
    ] = (
        window_seconds_numeric
        .isin(
            primary_seconds
        )
        .astype(int)
    )

    out[
        "exploratory_window"
    ] = (
        window_seconds_numeric
        .isin(
            secondary_only_seconds
        )
        .astype(int)
    )

    def _spec_name(
        value: Any,
    ) -> str:
        seconds = _safe_int(
            value,
            default=-1,
        )

        spec = spec_by_seconds.get(
            seconds
        )

        if spec is None:
            return "unregistered"

        return spec.name

    def _scope(
        value: Any,
    ) -> str:
        seconds = _safe_int(
            value,
            default=-1,
        )

        if seconds in primary_seconds:
            return "primary"

        if seconds in exploratory_seconds:
            return "exploratory"

        return "unregistered"

    out[
        "window_spec_name"
    ] = (
        out[
            "window_seconds"
        ]
        .apply(_spec_name)
    )

    out[
        "window_scope"
    ] = (
        out[
            "window_seconds"
        ]
        .apply(_scope)
    )

    if "sync_row_valid" in out.columns:
        sync_valid = (
            pd.to_numeric(
                out[
                    "sync_row_valid"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        sync_valid = pd.Series(
            np.zeros(
                len(out),
                dtype=int,
            ),
            index=out.index,
        )

    identity_valid = pd.Series(
        np.ones(
            len(out),
            dtype=bool,
        ),
        index=out.index,
    )

    for column in [
        "subject_id",
        "session_id",
        "window_id",
    ]:
        if column in out.columns:
            identity_valid &= (
                out[column].notna()
                & (
                    out[column]
                    .astype(str)
                    .str.strip()
                    != ""
                )
            )
        else:
            identity_valid &= False

    out[
        "window_valid"
    ] = (
        sync_valid.eq(1)
        & out[
            "registered_window_size"
        ].eq(1)
        & identity_valid
    ).astype(int)

    out[
        "valid_for_primary_analysis"
    ] = (
        out[
            "window_valid"
        ].eq(1)
        & out[
            "primary_window"
        ].eq(1)
    ).astype(int)

    out[
        "valid_for_exploratory_analysis"
    ] = (
        out[
            "window_valid"
        ].eq(1)
        & (
            out[
                "primary_window"
            ].eq(1)
            | out[
                "exploratory_window"
            ].eq(1)
        )
    ).astype(int)

    out[
        "window_status"
    ] = np.where(
        out[
            "window_valid"
        ].eq(1),
        "valid_window",
        "invalid_or_unregistered_window",
    )

    return out


def build_window_inventory(
    windows_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    rows: List[
        Dict[str, Any]
    ] = []

    if windows_df.empty:
        return empty_window_inventory(
            cfg
        )

    for spec in cfg.window_specs:
        subset = windows_df[
            pd.to_numeric(
                windows_df[
                    "window_seconds"
                ],
                errors="coerce",
            ).eq(
                int(
                    spec.seconds
                )
            )
        ].copy()

        if subset.empty:
            rows.append(
                {
                    "window_spec_name": (
                        spec.name
                    ),
                    "window_seconds": int(
                        spec.seconds
                    ),
                    "primary": bool(
                        spec.primary
                    ),
                    "min_windows_total": int(
                        spec.min_windows_total
                    ),
                    "min_windows_per_subject": int(
                        spec.min_windows_per_subject
                    ),
                    "n_rows": 0,
                    "n_valid_windows": 0,
                    "n_subjects": 0,
                    "min_windows_per_subject_observed": 0,
                    "median_windows_per_subject_observed": 0.0,
                    "max_windows_per_subject_observed": 0,
                    "meets_total_minimum": False,
                    "meets_subject_minimum": False,
                    "window_size_ready": False,
                    "description": (
                        spec.description
                    ),
                }
            )

            continue

        valid_subset = subset[
            subset[
                "window_valid"
            ].astype(int).eq(1)
        ].copy()

        n_valid = int(
            len(valid_subset)
        )

        if (
            "subject_id"
            in valid_subset.columns
            and not valid_subset.empty
        ):
            n_subjects = int(
                valid_subset[
                    "subject_id"
                ].nunique()
            )
        else:
            n_subjects = 0

        if (
            "subject_id"
            in valid_subset.columns
            and not valid_subset.empty
        ):
            subject_counts = (
                valid_subset
                .groupby(
                    "subject_id"
                )
                .size()
            )

            min_per_subject = int(
                subject_counts.min()
            )

            median_per_subject = float(
                subject_counts.median()
            )

            max_per_subject = int(
                subject_counts.max()
            )
        else:
            min_per_subject = 0
            median_per_subject = 0.0
            max_per_subject = 0

        meets_total = (
            n_valid
            >= int(
                spec.min_windows_total
            )
        )

        meets_subject = (
            n_subjects > 0
            and min_per_subject
            >= int(
                spec.min_windows_per_subject
            )
        )

        rows.append(
            {
                "window_spec_name": (
                    spec.name
                ),
                "window_seconds": int(
                    spec.seconds
                ),
                "primary": bool(
                    spec.primary
                ),
                "min_windows_total": int(
                    spec.min_windows_total
                ),
                "min_windows_per_subject": int(
                    spec.min_windows_per_subject
                ),
                "n_rows": int(
                    len(subset)
                ),
                "n_valid_windows": (
                    n_valid
                ),
                "n_subjects": (
                    n_subjects
                ),
                "min_windows_per_subject_observed": (
                    min_per_subject
                ),
                "median_windows_per_subject_observed": (
                    median_per_subject
                ),
                "max_windows_per_subject_observed": (
                    max_per_subject
                ),
                "meets_total_minimum": bool(
                    meets_total
                ),
                "meets_subject_minimum": bool(
                    meets_subject
                ),
                "window_size_ready": bool(
                    meets_total
                    and meets_subject
                ),
                "description": (
                    spec.description
                ),
            }
        )

    return pd.DataFrame(rows)


def attach_inventory_flags(
    windows_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    if windows_df.empty:
        return windows_df

    out = windows_df.copy()

    ready_map: Dict[
        int,
        bool,
    ] = {}

    total_map: Dict[
        int,
        bool,
    ] = {}

    subject_map: Dict[
        int,
        bool,
    ] = {}

    if not inventory_df.empty:
        for _, row in inventory_df.iterrows():
            seconds = _safe_int(
                row.get(
                    "window_seconds"
                ),
                default=-1,
            )

            ready_map[
                seconds
            ] = bool(
                row.get(
                    "window_size_ready",
                    False,
                )
            )

            total_map[
                seconds
            ] = bool(
                row.get(
                    "meets_total_minimum",
                    False,
                )
            )

            subject_map[
                seconds
            ] = bool(
                row.get(
                    "meets_subject_minimum",
                    False,
                )
            )

    out[
        "window_size_ready"
    ] = (
        out[
            "window_seconds"
        ]
        .apply(
            lambda value: int(
                bool(
                    ready_map.get(
                        _safe_int(
                            value,
                            -1,
                        ),
                        False,
                    )
                )
            )
        )
    )

    out[
        "window_total_count_ready"
    ] = (
        out[
            "window_seconds"
        ]
        .apply(
            lambda value: int(
                bool(
                    total_map.get(
                        _safe_int(
                            value,
                            -1,
                        ),
                        False,
                    )
                )
            )
        )
    )

    out[
        "window_subject_count_ready"
    ] = (
        out[
            "window_seconds"
        ]
        .apply(
            lambda value: int(
                bool(
                    subject_map.get(
                        _safe_int(
                            value,
                            -1,
                        ),
                        False,
                    )
                )
            )
        )
    )

    out[
        "valid_for_primary_analysis"
    ] = (
        out[
            "valid_for_primary_analysis"
        ].astype(int).eq(1)
        & out[
            "window_size_ready"
        ].astype(int).eq(1)
    ).astype(int)

    out[
        "valid_for_exploratory_analysis"
    ] = (
        out[
            "valid_for_exploratory_analysis"
        ].astype(int).eq(1)
        & out[
            "window_size_ready"
        ].astype(int).eq(1)
    ).astype(int)

    out[
        "window_status"
    ] = np.where(
        out[
            "window_valid"
        ].astype(int).eq(0),
        "invalid_or_unregistered_window",
        np.where(
            out[
                "window_size_ready"
            ].astype(int).eq(1),
            "ready_window",
            "valid_but_underpowered_window_size",
        ),
    )

    return out


def enforce_window_column_order(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = [
        "window_row_id",
        "phase3_2e_window_key",
        "subject_id",
        "session_id",
        "window_id",
        "window_seconds",
        "window_derivation",
        "source_window_seconds",
        "source_window_count",
        "source_window_ids",
        "window_spec_name",
        "window_scope",
        "registered_window_size",
        "primary_window",
        "exploratory_window",
        "window_size_ready",
        "window_total_count_ready",
        "window_subject_count_ready",
        "sync_row_valid",
        "window_valid",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
        "window_status",
        "window_start_utc",
        "window_end_utc",
        "window_midpoint_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_qrng_gap_ms",
        "sync_quality",
        "clock_drift_ms",
        "clock_drift_abs_ms",
        "sync_mode_effective",
        "sync_mode",
    ]

    ordered = [
        column
        for column in preferred
        if column in df.columns
    ]

    remaining = [
        column
        for column in df.columns
        if column not in ordered
    ]

    return df[
        ordered + remaining
    ].copy()


# =========================================================
# Summaries and reports
# =========================================================

def summarize_windows(
    windows_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    if windows_df.empty:
        return {
            "n_rows": 0,
            "n_subjects": 0,
            "n_sessions": 0,
            "n_valid_windows": 0,
            "n_primary_valid_windows": 0,
            "n_exploratory_valid_windows": 0,
            "n_unregistered_window_rows": 0,
            "window_seconds_counts": {},
            "window_status_counts": {},
            "window_scope_counts": {},
            "window_derivation_counts": {},
            "native_window_seconds_counts": {},
            "generated_window_seconds_counts": {},
            "sync_quality": (
                _numeric_summary(
                    windows_df,
                    "sync_quality",
                )
            ),
            "eeg_qrng_gap_ms": (
                _numeric_summary(
                    windows_df,
                    "eeg_qrng_gap_ms",
                )
            ),
            "clock_drift_abs_ms": (
                _numeric_summary(
                    windows_df,
                    "clock_drift_abs_ms",
                )
            ),
            "ready_window_specs": [],
            "primary_ready_window_specs": [],
        }

    n_sessions = 0

    if {
        "subject_id",
        "session_id",
    }.issubset(
        windows_df.columns
    ):
        n_sessions = int(
            windows_df[
                [
                    "subject_id",
                    "session_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

    ready_specs: List[str] = []
    primary_ready_specs: List[str] = []

    if not inventory_df.empty:
        ready = inventory_df[
            inventory_df[
                "window_size_ready"
            ].astype(bool)
        ].copy()

        ready_specs = (
            ready[
                "window_spec_name"
            ]
            .astype(str)
            .tolist()
        )

        primary_ready = ready[
            ready[
                "primary"
            ].astype(bool)
        ].copy()

        primary_ready_specs = (
            primary_ready[
                "window_spec_name"
            ]
            .astype(str)
            .tolist()
        )

    if "window_derivation" in windows_df.columns:
        native_windows = windows_df[
            windows_df[
                "window_derivation"
            ].astype(str).eq(
                "native"
            )
        ]

        generated_windows = windows_df[
            windows_df[
                "window_derivation"
            ]
            .astype(str)
            .eq(
                "aggregated_nonoverlapping"
            )
        ]
    else:
        native_windows = (
            windows_df.iloc[
                0:0
            ]
        )

        generated_windows = (
            windows_df.iloc[
                0:0
            ]
        )

    return {
        "n_rows": int(
            len(windows_df)
        ),
        "n_subjects": (
            int(
                windows_df[
                    "subject_id"
                ].nunique()
            )
            if "subject_id"
            in windows_df.columns
            else 0
        ),
        "n_sessions": n_sessions,
        "n_valid_windows": (
            int(
                pd.to_numeric(
                    windows_df[
                        "window_valid"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
            if "window_valid"
            in windows_df.columns
            else 0
        ),
        "n_primary_valid_windows": (
            int(
                pd.to_numeric(
                    windows_df[
                        "valid_for_primary_analysis"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
            if "valid_for_primary_analysis"
            in windows_df.columns
            else 0
        ),
        "n_exploratory_valid_windows": (
            int(
                pd.to_numeric(
                    windows_df[
                        "valid_for_exploratory_analysis"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
            if "valid_for_exploratory_analysis"
            in windows_df.columns
            else 0
        ),
        "n_unregistered_window_rows": int(
            (
                pd.to_numeric(
                    windows_df.get(
                        "registered_window_size",
                        0,
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                == 0
            ).sum()
        ),
        "window_seconds_counts": (
            _value_counts_dict(
                windows_df,
                "window_seconds",
            )
        ),
        "window_status_counts": (
            _value_counts_dict(
                windows_df,
                "window_status",
            )
        ),
        "window_scope_counts": (
            _value_counts_dict(
                windows_df,
                "window_scope",
            )
        ),
        "window_derivation_counts": (
            _value_counts_dict(
                windows_df,
                "window_derivation",
            )
        ),
        "native_window_seconds_counts": (
            _value_counts_dict(
                native_windows,
                "window_seconds",
            )
        ),
        "generated_window_seconds_counts": (
            _value_counts_dict(
                generated_windows,
                "window_seconds",
            )
        ),
        "sync_quality": (
            _numeric_summary(
                windows_df,
                "sync_quality",
            )
        ),
        "eeg_qrng_gap_ms": (
            _numeric_summary(
                windows_df,
                "eeg_qrng_gap_ms",
            )
        ),
        "clock_drift_abs_ms": (
            _numeric_summary(
                windows_df,
                "clock_drift_abs_ms",
            )
        ),
        "ready_window_specs": (
            ready_specs
        ),
        "primary_ready_window_specs": (
            primary_ready_specs
        ),
    }


def build_result_from_summary(
    status: str,
    protocol_only: bool,
    summary: Dict[str, Any],
    interpretation: str,
) -> WindowBuildResult:
    return WindowBuildResult(
        status=status,
        protocol_only=bool(
            protocol_only
        ),
        input_rows=int(
            summary.get(
                "n_rows",
                0,
            )
        ),
        output_rows=int(
            summary.get(
                "n_rows",
                0,
            )
        ),
        n_subjects=int(
            summary.get(
                "n_subjects",
                0,
            )
        ),
        n_sessions=int(
            summary.get(
                "n_sessions",
                0,
            )
        ),
        n_valid_windows=int(
            summary.get(
                "n_valid_windows",
                0,
            )
        ),
        n_primary_valid_windows=int(
            summary.get(
                "n_primary_valid_windows",
                0,
            )
        ),
        n_exploratory_valid_windows=int(
            summary.get(
                "n_exploratory_valid_windows",
                0,
            )
        ),
        n_registered_window_sizes=len(
            summary.get(
                "ready_window_specs",
                [],
            )
        ),
        n_unregistered_window_rows=int(
            summary.get(
                "n_unregistered_window_rows",
                0,
            )
        ),
        can_proceed_to_feature_construction=(
            int(
                summary.get(
                    "n_exploratory_valid_windows",
                    0,
                )
            )
            > 0
        ),
        can_proceed_to_empirical_cdr=(
            int(
                summary.get(
                    "n_primary_valid_windows",
                    0,
                )
            )
            > 0
        ),
        interpretation=interpretation,
    )


def write_windows_summary_txt(
    path: Path,
    result: WindowBuildResult,
    summary: Dict[str, Any],
    cfg: Phase32EConfig,
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append(
        "Phase III.2E — Window Summary"
    )
    lines.append("=" * 78)
    lines.append("")

    lines.append(
        f"status: {result.status}"
    )
    lines.append(
        f"protocol_only: {result.protocol_only}"
    )
    lines.append(
        f"input_rows: {result.input_rows}"
    )
    lines.append(
        f"output_rows: {result.output_rows}"
    )
    lines.append(
        f"n_subjects: {result.n_subjects}"
    )
    lines.append(
        f"n_sessions: {result.n_sessions}"
    )
    lines.append("")

    lines.append(
        f"n_valid_windows: "
        f"{result.n_valid_windows}"
    )
    lines.append(
        f"n_primary_valid_windows: "
        f"{result.n_primary_valid_windows}"
    )
    lines.append(
        f"n_exploratory_valid_windows: "
        f"{result.n_exploratory_valid_windows}"
    )
    lines.append(
        f"n_registered_window_sizes: "
        f"{result.n_registered_window_sizes}"
    )
    lines.append(
        f"n_unregistered_window_rows: "
        f"{result.n_unregistered_window_rows}"
    )
    lines.append("")

    lines.append(
        "can_proceed_to_feature_construction: "
        f"{result.can_proceed_to_feature_construction}"
    )
    lines.append(
        "can_proceed_to_empirical_cdr: "
        f"{result.can_proceed_to_empirical_cdr}"
    )
    lines.append("")

    lines.append(
        "window_seconds_counts: "
        f"{summary.get('window_seconds_counts')}"
    )
    lines.append(
        "window_status_counts: "
        f"{summary.get('window_status_counts')}"
    )
    lines.append(
        "window_scope_counts: "
        f"{summary.get('window_scope_counts')}"
    )
    lines.append(
        "window_derivation_counts: "
        f"{summary.get('window_derivation_counts')}"
    )
    lines.append(
        "native_window_seconds_counts: "
        f"{summary.get('native_window_seconds_counts')}"
    )
    lines.append(
        "generated_window_seconds_counts: "
        f"{summary.get('generated_window_seconds_counts')}"
    )
    lines.append("")

    lines.append(
        "sync_quality: "
        f"{summary.get('sync_quality')}"
    )
    lines.append(
        "eeg_qrng_gap_ms: "
        f"{summary.get('eeg_qrng_gap_ms')}"
    )
    lines.append(
        "clock_drift_abs_ms: "
        f"{summary.get('clock_drift_abs_ms')}"
    )
    lines.append("")

    lines.append(
        "ready_window_specs: "
        f"{summary.get('ready_window_specs')}"
    )
    lines.append(
        "primary_ready_window_specs: "
        f"{summary.get('primary_ready_window_specs')}"
    )
    lines.append("")

    lines.append("interpretation:")
    lines.append(
        result.interpretation
    )
    lines.append("")
    lines.append("=" * 78)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
        )


def save_window_outputs(
    windows_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    report: Dict[str, Any],
    result: WindowBuildResult,
    cfg: Phase32EConfig,
) -> None:
    Path(
        cfg.interim_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        cfg.results_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    windows_df.to_csv(
        windows_csv_path(cfg),
        index=False,
    )

    inventory_df.to_csv(
        window_inventory_csv_path(
            cfg
        ),
        index=False,
    )

    metadata = {
        "phase": cfg.phase_name,
        "project_name": (
            cfg.project_name
        ),
        "module": WINDOW_MODULE,
        "window_version": (
            WINDOW_VERSION
        ),
        "created_at": _now_str(),
        "result": asdict(result),
        "windows_csv": str(
            windows_csv_path(cfg)
        ),
        "window_inventory_csv": str(
            window_inventory_csv_path(
                cfg
            )
        ),
        "windows_report_json": str(
            windows_report_json_path(
                cfg
            )
        ),
        "windows_summary_txt": str(
            windows_summary_txt_path(
                cfg
            )
        ),
        "config": describe_config(cfg),
    }

    save_json(
        windows_metadata_json_path(
            cfg
        ),
        metadata,
    )

    save_json(
        windows_report_json_path(
            cfg
        ),
        report,
    )

    write_windows_summary_txt(
        path=windows_summary_txt_path(
            cfg
        ),
        result=result,
        summary=report.get(
            "summary",
            {},
        ),
        cfg=cfg,
    )

    print(
        "[Phase3.2E Windows] "
        f"Saved windows: "
        f"{windows_csv_path(cfg)}"
    )

    print(
        "[Phase3.2E Windows] "
        f"Saved inventory: "
        f"{window_inventory_csv_path(cfg)}"
    )

    print(
        "[Phase3.2E Windows] "
        f"Saved metadata: "
        f"{windows_metadata_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Windows] "
        f"Saved report: "
        f"{windows_report_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Windows] "
        f"Saved summary: "
        f"{windows_summary_txt_path(cfg)}"
    )


# =========================================================
# Main builder
# =========================================================

def build_phase3_2e_windows(
    cfg: Optional[
        Phase32EConfig
    ] = None,
    force_rebuild_loader: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print(
        "[Phase3.2E Windows] "
        "Loading Phase III.2E synchronized loader output"
    )

    loaded_df, loader_metadata = (
        load_or_build_phase3_2e_dataset(
            cfg=cfg,
            force_rebuild=bool(
                force_rebuild_loader
            ),
            run_schema_first=True,
            run_sync_first=True,
        )
    )

    loader_status = str(
        loader_metadata.get(
            "status",
            "",
        )
    )

    # -----------------------------------------------------
    # Protocol-only or blocked path
    # -----------------------------------------------------

    if (
        loader_status
        != "loaded_synchronized_data"
        or loaded_df.empty
    ):
        windows_df = (
            empty_windows_frame(cfg)
        )

        inventory_df = (
            empty_window_inventory(
                cfg
            )
        )

        summary = summarize_windows(
            windows_df,
            inventory_df,
            cfg,
        )

        if (
            loader_status
            == cfg.status_pending_synchronized_data
        ):
            status = (
                cfg.status_pending_synchronized_data
            )

            protocol_only = True

            interpretation = (
                "No synchronized EEG–QRNG dataset is available. "
                "Window building completed in protocol-only mode "
                "with empty canonical outputs."
            )

        elif (
            loader_status
            == "loader_blocked_by_schema"
        ):
            status = (
                "windows_blocked_by_schema"
            )

            protocol_only = False

            interpretation = (
                "Window building was blocked because the "
                "synchronized input failed schema validation."
            )

        elif (
            loader_status
            == "loader_blocked_by_sync"
        ):
            status = (
                "windows_blocked_by_sync"
            )

            protocol_only = False

            interpretation = (
                "Window building was blocked because the "
                "synchronized input failed synchronization validation."
            )

        else:
            status = (
                "windows_blocked_by_loader"
            )

            protocol_only = False

            interpretation = (
                "Window building was blocked by loader status: "
                f"{loader_status}."
            )

        result = (
            build_result_from_summary(
                status=status,
                protocol_only=(
                    protocol_only
                ),
                summary=summary,
                interpretation=(
                    interpretation
                ),
            )
        )

        report = {
            "phase": cfg.phase_name,
            "project_name": (
                cfg.project_name
            ),
            "module": WINDOW_MODULE,
            "window_version": (
                WINDOW_VERSION
            ),
            "status": result.status,
            "created_at": _now_str(),
            "loader_status": (
                loader_status
            ),
            "loader_summary": (
                loader_metadata.get(
                    "summary",
                    {},
                )
            ),
            "result": asdict(result),
            "summary": summary,
            "inventory": (
                inventory_df.to_dict(
                    orient="records"
                )
            ),
        }

        save_window_outputs(
            windows_df,
            inventory_df,
            report,
            result,
            cfg,
        )

        return (
            windows_df,
            inventory_df,
            report,
        )

    # -----------------------------------------------------
    # Full synchronized-data path
    # -----------------------------------------------------

    (
        multiresolution_df,
        multiresolution_metadata,
    ) = build_multiresolution_windows(
        loaded_df=loaded_df,
        cfg=cfg,
    )

    windows_df = (
        classify_window_rows(
            multiresolution_df,
            cfg,
        )
    )

    inventory_df = (
        build_window_inventory(
            windows_df,
            cfg,
        )
    )

    windows_df = (
        attach_inventory_flags(
            windows_df,
            inventory_df,
        )
    )

    windows_df = (
        enforce_window_column_order(
            windows_df,
            cfg,
        )
    )

    summary = summarize_windows(
        windows_df,
        inventory_df,
        cfg,
    )

    n_primary_ready = int(
        summary.get(
            "n_primary_valid_windows",
            0,
        )
    )

    n_exploratory_ready = int(
        summary.get(
            "n_exploratory_valid_windows",
            0,
        )
    )

    if n_primary_ready > 0:
        status = "windows_ready"

        interpretation = (
            "Synchronized windows were built successfully. "
            "At least one primary registered window configuration "
            "is ready for feature construction and synchronized "
            "conditional CDR. Empirical interpretation remains "
            "controlled by source provenance and the real-data guardrail."
        )

    elif n_exploratory_ready > 0:
        status = (
            "windows_ready_exploratory_only"
        )

        interpretation = (
            "Synchronized windows were built and exploratory "
            "windows are available, but no primary registered "
            "window configuration satisfies the readiness criteria."
        )

    else:
        status = (
            "windows_invalid_or_underpowered"
        )

        interpretation = (
            "Synchronized rows were loaded, but no registered "
            "window configuration satisfies validity and "
            "minimum-count requirements. Feature construction "
            "and conditional CDR remain blocked."
        )

    result = build_result_from_summary(
        status=status,
        protocol_only=False,
        summary=summary,
        interpretation=interpretation,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": (
            cfg.project_name
        ),
        "module": WINDOW_MODULE,
        "window_version": (
            WINDOW_VERSION
        ),
        "status": result.status,
        "created_at": _now_str(),
        "loader_status": (
            loader_status
        ),
        "loader_summary": (
            loader_metadata.get(
                "summary",
                {},
            )
        ),
        "multiresolution_build": (
            multiresolution_metadata
        ),
        "result": asdict(result),
        "summary": summary,
        "inventory": (
            inventory_df.to_dict(
                orient="records"
            )
        ),
        "config_window_specs": [
            {
                "name": spec.name,
                "seconds": int(
                    spec.seconds
                ),
                "primary": bool(
                    spec.primary
                ),
                "min_windows_total": int(
                    spec.min_windows_total
                ),
                "min_windows_per_subject": int(
                    spec.min_windows_per_subject
                ),
                "description": (
                    spec.description
                ),
            }
            for spec in cfg.window_specs
        ],
    }

    save_window_outputs(
        windows_df,
        inventory_df,
        report,
        result,
        cfg,
    )

    return (
        windows_df,
        inventory_df,
        report,
    )


def load_or_build_phase3_2e_windows(
    cfg: Optional[
        Phase32EConfig
    ] = None,
    force_rebuild: bool = False,
    force_rebuild_loader: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    windows_path = (
        windows_csv_path(cfg)
    )

    inventory_path = (
        window_inventory_csv_path(
            cfg
        )
    )

    report_path = (
        windows_report_json_path(
            cfg
        )
    )

    if (
        windows_path.exists()
        and inventory_path.exists()
        and report_path.exists()
        and not force_rebuild
    ):
        print(
            "[Phase3.2E Windows] "
            f"Loading cached windows: "
            f"{windows_path}"
        )

        try:
            windows_df = pd.read_csv(
                windows_path
            )

            inventory_df = pd.read_csv(
                inventory_path
            )

            with open(
                report_path,
                "r",
                encoding="utf-8",
            ) as f:
                report = json.load(f)

            if (
                report.get(
                    "window_version"
                )
                == WINDOW_VERSION
            ):
                print(
                    "[Phase3.2E Windows] "
                    "Cached windows compatible."
                )

                return (
                    windows_df,
                    inventory_df,
                    report,
                )

            print(
                "[Phase3.2E Windows] "
                "Cached version changed. Rebuilding..."
            )

        except Exception as exc:
            print(
                "[Phase3.2E Windows] "
                f"Failed to load cache: {exc}. "
                "Rebuilding..."
            )

    return build_phase3_2e_windows(
        cfg=cfg,
        force_rebuild_loader=bool(
            force_rebuild_loader
        ),
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Phase III.2E synchronized window table."
        )
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Force rebuilding window outputs."
        ),
    )

    parser.add_argument(
        "--force-rebuild-loader",
        action="store_true",
        help=(
            "Force rebuilding loader outputs before building windows."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    (
        windows_df,
        inventory_df,
        report,
    ) = load_or_build_phase3_2e_windows(
        cfg=cfg,
        force_rebuild=bool(
            args.force_rebuild
        ),
        force_rebuild_loader=bool(
            args.force_rebuild_loader
        ),
    )

    result = report.get(
        "result",
        {},
    )

    summary = report.get(
        "summary",
        {},
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Phase III.2E window builder completed"
    )

    print("=" * 78)

    print(
        f"status: "
        f"{result.get('status')}"
    )

    print(
        f"protocol_only: "
        f"{result.get('protocol_only')}"
    )

    print(
        f"input_rows: "
        f"{result.get('input_rows')}"
    )

    print(
        f"output_rows: "
        f"{result.get('output_rows')}"
    )

    print(
        f"n_valid_windows: "
        f"{result.get('n_valid_windows')}"
    )

    print(
        f"n_primary_valid_windows: "
        f"{result.get('n_primary_valid_windows')}"
    )

    print(
        f"n_exploratory_valid_windows: "
        f"{result.get('n_exploratory_valid_windows')}"
    )

    print(
        "can_proceed_to_feature_construction: "
        f"{result.get('can_proceed_to_feature_construction')}"
    )

    print(
        "can_proceed_to_empirical_cdr: "
        f"{result.get('can_proceed_to_empirical_cdr')}"
    )

    print(
        "ready_window_specs: "
        f"{summary.get('ready_window_specs')}"
    )

    print(
        "primary_ready_window_specs: "
        f"{summary.get('primary_ready_window_specs')}"
    )

    print(
        "window_derivation_counts: "
        f"{summary.get('window_derivation_counts')}"
    )

    print(
        "native_window_seconds_counts: "
        f"{summary.get('native_window_seconds_counts')}"
    )

    print(
        "generated_window_seconds_counts: "
        f"{summary.get('generated_window_seconds_counts')}"
    )

    print(
        f"windows_csv: "
        f"{windows_csv_path(cfg)}"
    )

    print(
        f"inventory_csv: "
        f"{window_inventory_csv_path(cfg)}"
    )

    print(
        "interpretation: "
        f"{result.get('interpretation')}"
    )

    print(
        "=" * 78
        + "\n"
    )


if __name__ == "__main__":
    main()