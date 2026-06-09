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
    active_conditional_pairs,
    describe_config,
    load_phase3_2e_config,
)

from src.phase3_2e_features import (
    FeatureFrame,
    load_or_build_phase3_2e_features,
)


# =========================================================
# Phase III.2E — Lagged Alignment
# =========================================================
#
# Purpose:
#
#   Build synchronized lagged alignment frames for Phase III.2E.
#
# This module consumes:
#
#   data/interim/phase3_2e/phase3_2e_features.csv
#
# and produces one lagged frame per:
#
#   window_seconds × lag_windows
#
# The alignment convention is:
#
#   lag = 0:
#       target state is the same synchronized window.
#
#   lag > 0:
#       current row at t predicts target row at t + lag.
#
#   lag < 0:
#       current row at t predicts target row at t - |lag|.
#
# Example:
#
#   window_seconds = 30
#   lag_windows = +1
#
# means:
#
#   QRNG_t → EEG_{t+30s}
#   EEG_t  → QRNG_{t+30s}
#
# The module does not create evidence by itself. It only prepares leakage-safe,
# subject/session-bounded lagged frames for later conditional CDR.
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   empty canonical inventory/report outputs are saved
#
# Outputs:
#
#   data/interim/phase3_2e/lagged_frames/*.csv
#   data/interim/phase3_2e/phase3_2e_alignment_inventory.csv
#   data/interim/phase3_2e/phase3_2e_alignment_metadata.json
#   results/phase3_2e/phase3_2e_alignment_report.json
#   results/phase3_2e/phase3_2e_alignment_summary.txt
#


# =========================================================
# Constants
# =========================================================

ALIGNMENT_MODULE = "phase3_2e_alignment"
ALIGNMENT_VERSION = "phase3_2e_alignment_v2_evidence_guided_windows_lags"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class AlignmentBuildResult:
    status: str
    protocol_only: bool
    input_rows: int
    n_frames: int
    n_window_sizes: int
    n_lags: int
    n_total_aligned_rows: int
    n_valid_alignment_rows: int
    n_valid_primary_alignment_rows: int
    n_valid_exploratory_alignment_rows: int
    n_valid_conditional_rows: int
    n_valid_mc_conditional_rows: int
    can_proceed_to_conditional_cdr: bool
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


def _lag_label(lag_windows: int) -> str:
    lag_windows = int(lag_windows)

    if lag_windows < 0:
        return f"m{abs(lag_windows)}"

    if lag_windows > 0:
        return f"p{lag_windows}"

    return "0"


def _window_label(window_seconds: int) -> str:
    return f"w{int(window_seconds):02d}s"


def _lag_direction(lag_windows: int) -> str:
    lag_windows = int(lag_windows)

    if lag_windows > 0:
        return "future_target"

    if lag_windows < 0:
        return "past_target"

    return "concurrent"


# =========================================================
# Output paths
# =========================================================

def lagged_frame_path(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> Path:
    filename = (
        f"phase3_2e_"
        f"{_window_label(window_seconds)}_"
        f"lag_{_lag_label(lag_windows)}.csv"
    )

    return Path(cfg.lagged_frames_dir) / filename


def alignment_inventory_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(cfg.alignment_inventory_csv)


def alignment_metadata_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.interim_dir)
        / "phase3_2e_alignment_metadata.json"
    )


def alignment_report_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_alignment_report.json"
    )


def alignment_summary_txt_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_alignment_summary.txt"
    )


def expected_lagged_frame_paths(
    cfg: Phase32EConfig,
) -> Tuple[Path, ...]:
    """
    Return the exact set of frame files registered by the active config.
    """
    return tuple(
        lagged_frame_path(
            cfg,
            int(window_seconds),
            int(lag_windows),
        )
        for window_seconds in cfg.all_window_seconds
        for lag_windows in cfg.all_lags_windows
    )


def remove_stale_lagged_frames(
    cfg: Phase32EConfig,
) -> List[str]:
    """
    Remove cached Phase III.2E frames that no longer belong to the active
    window/lag grid.

    This prevents obsolete files from the previous 1/2/5/10/30-second and
    -5/-3 lag grid from being picked up by downstream tooling.
    """
    frames_dir = Path(cfg.lagged_frames_dir)

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected = {
        path.resolve()
        for path in expected_lagged_frame_paths(cfg)
    }

    removed: List[str] = []

    for candidate in frames_dir.glob(
        "phase3_2e_w*s_lag_*.csv"
    ):
        if candidate.resolve() in expected:
            continue

        candidate.unlink(
            missing_ok=True
        )

        removed.append(
            str(candidate)
        )

    return removed


# =========================================================
# Empty outputs
# =========================================================

def empty_alignment_frame() -> pd.DataFrame:
    columns = [
        "alignment_row_id",
        "source_feature_row_id",
        "target_feature_row_id",
        "phase3_2e_window_key",
        "target_phase3_2e_window_key",
        "subject_id",
        "session_id",
        "window_id",
        "target_window_id",
        "window_seconds",
        "lag_windows",
        "lag_seconds",
        "lag_direction",
        "alignment_scope",
        "primary_alignment",
        "exploratory_alignment",
        "valid_alignment_row",
        "valid_primary_alignment_row",
        "valid_exploratory_alignment_row",
        "valid_conditional_row",
        "valid_mc_conditional_row",
        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "eeg_next_state",
        "qrng_next_state",
        "mc_eeg_next_state",
        "target_eeg_state",
        "target_qrng_state",
        "target_mc_eeg_state",
        "alignment_status",
    ]

    return pd.DataFrame(
        columns=columns
    )


def empty_alignment_inventory(
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for window_seconds in cfg.all_window_seconds:
        for lag_windows in cfg.all_lags_windows:
            rows.append(
                {
                    "window_seconds": int(
                        window_seconds
                    ),
                    "lag_windows": int(
                        lag_windows
                    ),
                    "lag_seconds": (
                        int(window_seconds)
                        * int(lag_windows)
                    ),
                    "lag_direction": _lag_direction(
                        int(lag_windows)
                    ),
                    "alignment_scope": _alignment_scope(
                        cfg,
                        window_seconds,
                        lag_windows,
                    ),
                    "primary_alignment": _is_primary_alignment(
                        cfg,
                        window_seconds,
                        lag_windows,
                    ),
                    "exploratory_alignment": bool(
                        _is_exploratory_alignment(
                            cfg,
                            window_seconds,
                            lag_windows,
                        )
                    ),
                    "frame_file": str(
                        lagged_frame_path(
                            cfg,
                            window_seconds,
                            lag_windows,
                        )
                    ),
                    "n_rows": 0,
                    "n_valid_alignment_rows": 0,
                    "n_valid_primary_alignment_rows": 0,
                    "n_valid_exploratory_alignment_rows": 0,
                    "n_valid_conditional_rows": 0,
                    "n_valid_mc_conditional_rows": 0,
                    "n_subjects": 0,
                    "n_sessions": 0,
                    "status": "pending_synchronized_data",
                }
            )

    return pd.DataFrame(rows)


# =========================================================
# Scope helpers
# =========================================================

def _is_primary_alignment(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> bool:
    primary_windows = {
        int(value)
        for value in cfg.primary_test_window_seconds
    }

    primary_lags = {
        int(value)
        for value in cfg.primary_test_lags_windows
    }

    return (
        int(window_seconds) in primary_windows
        and int(lag_windows) in primary_lags
    )


def _is_exploratory_alignment(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> bool:
    """
    Return True only for registered window/lag combinations that are not
    part of the primary evidence-guided replication scope.

    Primary scope always has precedence when a window duration appears in both
    primary and secondary configuration lists.
    """
    registered_windows = {
        int(value)
        for value in cfg.all_window_seconds
    }

    registered_lags = {
        int(value)
        for value in cfg.all_lags_windows
    }

    return (
        int(window_seconds) in registered_windows
        and int(lag_windows) in registered_lags
        and not _is_primary_alignment(
            cfg,
            window_seconds,
            lag_windows,
        )
    )


def _alignment_scope(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> str:
    if _is_primary_alignment(
        cfg,
        window_seconds,
        lag_windows,
    ):
        return "primary"

    if _is_exploratory_alignment(
        cfg,
        window_seconds,
        lag_windows,
    ):
        return "exploratory"

    return "unregistered"


def _valid_feature_mask(
    df: pd.DataFrame,
) -> pd.Series:
    if "valid_feature_row" in df.columns:
        return (
            pd.to_numeric(
                df["valid_feature_row"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    if (
        "valid_for_exploratory_analysis"
        in df.columns
    ):
        return (
            pd.to_numeric(
                df[
                    "valid_for_exploratory_analysis"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    return pd.Series(
        np.zeros(
            len(df),
            dtype=bool,
        ),
        index=df.index,
    )


def _valid_primary_mask(
    df: pd.DataFrame,
) -> pd.Series:
    if (
        "valid_for_primary_analysis"
        in df.columns
    ):
        return (
            pd.to_numeric(
                df[
                    "valid_for_primary_analysis"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    return pd.Series(
        np.zeros(
            len(df),
            dtype=bool,
        ),
        index=df.index,
    )


def _sort_alignment_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    sort_cols = [
        column
        for column in [
            "window_start_utc",
            "window_id",
            "window_row_id",
            "feature_row_id",
        ]
        if column in df.columns
    ]

    if sort_cols:
        return (
            df
            .sort_values(
                sort_cols,
                kind="stable",
            )
            .copy()
        )

    return df.copy()


# =========================================================
# Core alignment
# =========================================================

def _target_columns() -> List[str]:
    return [
        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",
        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",
    ]


def build_lagged_alignment_frame(
    features_df: pd.DataFrame,
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> pd.DataFrame:
    if features_df.empty:
        return empty_alignment_frame()

    if (
        "window_seconds"
        not in features_df.columns
    ):
        raise KeyError(
            "window_seconds column is required "
            "for Phase III.2E alignment."
        )

    window_seconds = int(
        window_seconds
    )

    lag_windows = int(
        lag_windows
    )

    subset = features_df[
        pd.to_numeric(
            features_df[
                "window_seconds"
            ],
            errors="coerce",
        ).eq(
            window_seconds
        )
    ].copy()

    if subset.empty:
        return empty_alignment_frame()

    subset = _sort_alignment_group(
        subset
    )

    if "feature_row_id" in subset.columns:
        subset[
            "source_feature_row_id"
        ] = (
            pd.to_numeric(
                subset["feature_row_id"],
                errors="coerce",
            )
            .fillna(
                pd.Series(
                    subset.index,
                    index=subset.index,
                )
            )
            .astype(int)
        )
    else:
        subset[
            "source_feature_row_id"
        ] = subset.index.astype(int)

    aligned_parts: List[
        pd.DataFrame
    ] = []

    group_cols = [
        "subject_id",
        "session_id",
        "window_seconds",
    ]

    for _, group in subset.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        g = _sort_alignment_group(
            group.copy()
        )

        if g.empty:
            continue

        source = g.copy()

        source[
            "target_feature_row_id"
        ] = (
            g[
                "source_feature_row_id"
            ]
            .shift(
                -lag_windows
            )
        )

        source[
            "target_phase3_2e_window_key"
        ] = (
            g[
                "phase3_2e_window_key"
            ]
            .shift(
                -lag_windows
            )
            if (
                "phase3_2e_window_key"
                in g.columns
            )
            else pd.NA
        )

        source[
            "target_window_id"
        ] = (
            g["window_id"]
            .shift(
                -lag_windows
            )
            if "window_id" in g.columns
            else pd.NA
        )

        for column in _target_columns():
            if column not in g.columns:
                continue

            source[
                f"target_{column}"
            ] = (
                g[column]
                .shift(
                    -lag_windows
                )
            )

        aligned_parts.append(
            source
        )

    if not aligned_parts:
        return empty_alignment_frame()

    out = pd.concat(
        aligned_parts,
        axis=0,
        ignore_index=True,
    )

    out[
        "alignment_row_id"
    ] = np.arange(
        len(out),
        dtype=int,
    )

    out["lag_windows"] = int(
        lag_windows
    )

    out["lag_seconds"] = (
        int(window_seconds)
        * int(lag_windows)
    )

    out["lag_direction"] = (
        _lag_direction(
            lag_windows
        )
    )

    out["alignment_scope"] = (
        _alignment_scope(
            cfg,
            window_seconds,
            lag_windows,
        )
    )

    out["primary_alignment"] = int(
        _is_primary_alignment(
            cfg,
            window_seconds,
            lag_windows,
        )
    )

    out[
        "exploratory_alignment"
    ] = int(
        _is_exploratory_alignment(
            cfg,
            window_seconds,
            lag_windows,
        )
    )

    # -----------------------------------------------------
    # Lagged target aliases expected by conditional models
    # -----------------------------------------------------

    if "target_eeg_state" in out.columns:
        out["eeg_next_state"] = (
            pd.to_numeric(
                out[
                    "target_eeg_state"
                ],
                errors="coerce",
            )
            .astype("Int64")
        )
    else:
        out["eeg_next_state"] = pd.Series(
            [pd.NA] * len(out),
            dtype="Int64",
        )

    if "target_qrng_state" in out.columns:
        out["qrng_next_state"] = (
            pd.to_numeric(
                out[
                    "target_qrng_state"
                ],
                errors="coerce",
            )
            .astype("Int64")
        )
    else:
        out["qrng_next_state"] = pd.Series(
            [pd.NA] * len(out),
            dtype="Int64",
        )

    if (
        "target_mc_eeg_state"
        in out.columns
    ):
        out[
            "mc_eeg_next_state"
        ] = (
            pd.to_numeric(
                out[
                    "target_mc_eeg_state"
                ],
                errors="coerce",
            )
            .astype("Int64")
        )
    else:
        out[
            "mc_eeg_next_state"
        ] = pd.Series(
            [pd.NA] * len(out),
            dtype="Int64",
        )

    current_valid = _valid_feature_mask(
        out
    )

    current_primary_valid = (
        _valid_primary_mask(
            out
        )
    )

    target_single_valid = (
        out["eeg_next_state"].notna()
        & out["qrng_next_state"].notna()
        & out[
            "target_feature_row_id"
        ].notna()
    )

    target_mc_valid = (
        out[
            "mc_eeg_next_state"
        ].notna()
        & out[
            "qrng_next_state"
        ].notna()
        & out[
            "target_feature_row_id"
        ].notna()
    )

    out[
        "valid_alignment_row"
    ] = (
        current_valid
        & target_single_valid
    ).astype(int)

    out[
        "valid_primary_alignment_row"
    ] = (
        out[
            "valid_alignment_row"
        ].astype(int).eq(1)
        & current_primary_valid
        & bool(
            _is_primary_alignment(
                cfg,
                window_seconds,
                lag_windows,
            )
        )
    ).astype(int)

    out[
        "valid_exploratory_alignment_row"
    ] = (
        out[
            "valid_alignment_row"
        ].astype(int).eq(1)
        & bool(
            _is_exploratory_alignment(
                cfg,
                window_seconds,
                lag_windows,
            )
        )
    ).astype(int)

    out[
        "valid_conditional_row"
    ] = (
        out[
            "valid_alignment_row"
        ]
        .astype(int)
    )

    out[
        "valid_mc_conditional_row"
    ] = (
        current_valid
        & target_mc_valid
    ).astype(int)

    out[
        "alignment_status"
    ] = np.where(
        out[
            "valid_alignment_row"
        ].astype(int).eq(1),
        "valid_alignment_row",
        "invalid_lag_boundary_or_window_row",
    )

    # Cast target IDs safely.
    for column in [
        "target_feature_row_id",
        "target_window_id",
        "target_eeg_state",
        "target_qrng_state",
        "target_mc_eeg_state",
        "target_eeg_info_bin",
        "target_qrng_info_bin",
        "target_mc_eeg_info_bin",
        "target_observed_joint_state",
        "target_mc_observed_joint_state",
        "target_informational_joint_state",
        "target_mc_informational_joint_state",
    ]:
        if column not in out.columns:
            continue

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    return enforce_alignment_column_order(
        out,
        cfg,
    )


def enforce_alignment_column_order(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = [
        "alignment_row_id",
        "source_feature_row_id",
        "target_feature_row_id",
        "phase3_2e_window_key",
        "target_phase3_2e_window_key",

        "subject_id",
        "session_id",
        "window_id",
        "target_window_id",

        "window_seconds",
        "lag_windows",
        "lag_seconds",
        "lag_direction",
        "alignment_scope",
        "primary_alignment",
        "exploratory_alignment",

        "valid_alignment_row",
        "valid_primary_alignment_row",
        "valid_exploratory_alignment_row",
        "valid_conditional_row",
        "valid_mc_conditional_row",

        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "eeg_next_state",
        "qrng_next_state",
        "mc_eeg_next_state",

        "target_eeg_state",
        "target_qrng_state",
        "target_mc_eeg_state",

        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",
        "target_eeg_info_bin",
        "target_qrng_info_bin",
        "target_mc_eeg_info_bin",

        "observed_joint_state",
        "observed_joint_state_next",
        "target_observed_joint_state",

        "mc_observed_joint_state",
        "mc_observed_joint_state_next",
        "target_mc_observed_joint_state",

        "informational_joint_state",
        "informational_joint_state_next",
        "target_informational_joint_state",

        "mc_informational_joint_state",
        "mc_informational_joint_state_next",
        "target_mc_informational_joint_state",

        "window_start_utc",
        "window_end_utc",
        "window_midpoint_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_qrng_gap_ms",
        "sync_quality",
        "clock_drift_ms",
        "clock_drift_abs_ms",

        "alignment_status",
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
# Inventory / summaries
# =========================================================

def summarize_alignment_frame(
    frame: pd.DataFrame,
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
    frame_file: Path,
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "window_seconds": int(
                window_seconds
            ),
            "lag_windows": int(
                lag_windows
            ),
            "lag_seconds": (
                int(window_seconds)
                * int(lag_windows)
            ),
            "lag_direction": _lag_direction(
                lag_windows
            ),
            "alignment_scope": _alignment_scope(
                cfg,
                window_seconds,
                lag_windows,
            ),
            "primary_alignment": bool(
                _is_primary_alignment(
                    cfg,
                    window_seconds,
                    lag_windows,
                )
            ),
            "exploratory_alignment": bool(
                _is_exploratory_alignment(
                    cfg,
                    window_seconds,
                    lag_windows,
                )
            ),
            "frame_file": str(
                frame_file
            ),
            "n_rows": 0,
            "n_valid_alignment_rows": 0,
            "n_valid_primary_alignment_rows": 0,
            "n_valid_exploratory_alignment_rows": 0,
            "n_valid_conditional_rows": 0,
            "n_valid_mc_conditional_rows": 0,
            "n_subjects": 0,
            "n_sessions": 0,
            "status": "empty_frame",
        }

    if {
        "subject_id",
        "session_id",
    }.issubset(frame.columns):
        n_sessions = int(
            frame[
                [
                    "subject_id",
                    "session_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )
    else:
        n_sessions = 0

    n_valid = int(
        pd.to_numeric(
            frame.get(
                "valid_alignment_row",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    n_valid_primary = int(
        pd.to_numeric(
            frame.get(
                "valid_primary_alignment_row",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    n_valid_exploratory = int(
        pd.to_numeric(
            frame.get(
                "valid_exploratory_alignment_row",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    n_valid_conditional = int(
        pd.to_numeric(
            frame.get(
                "valid_conditional_row",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    n_valid_mc = int(
        pd.to_numeric(
            frame.get(
                "valid_mc_conditional_row",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if n_valid > 0:
        status = "alignment_ready"
    else:
        status = "alignment_no_valid_rows"

    return {
        "window_seconds": int(
            window_seconds
        ),
        "lag_windows": int(
            lag_windows
        ),
        "lag_seconds": (
            int(window_seconds)
            * int(lag_windows)
        ),
        "lag_direction": _lag_direction(
            lag_windows
        ),
        "alignment_scope": _alignment_scope(
            cfg,
            window_seconds,
            lag_windows,
        ),
        "primary_alignment": bool(
            _is_primary_alignment(
                cfg,
                window_seconds,
                lag_windows,
            )
        ),
        "exploratory_alignment": bool(
            _is_exploratory_alignment(
                cfg,
                window_seconds,
                lag_windows,
            )
        ),
        "frame_file": str(
            frame_file
        ),
        "n_rows": int(
            len(frame)
        ),
        "n_valid_alignment_rows": (
            n_valid
        ),
        "n_valid_primary_alignment_rows": (
            n_valid_primary
        ),
        "n_valid_exploratory_alignment_rows": (
            n_valid_exploratory
        ),
        "n_valid_conditional_rows": (
            n_valid_conditional
        ),
        "n_valid_mc_conditional_rows": (
            n_valid_mc
        ),
        "n_subjects": (
            int(
                frame[
                    "subject_id"
                ].nunique()
            )
            if "subject_id" in frame.columns
            else 0
        ),
        "n_sessions": n_sessions,
        "alignment_status_counts": (
            _value_counts_dict(
                frame,
                "alignment_status",
            )
        ),
        "status": status,
    }


def summarize_alignment_inventory(
    inventory_df: pd.DataFrame,
) -> Dict[str, Any]:
    if inventory_df.empty:
        return {
            "n_frames": 0,
            "n_window_sizes": 0,
            "n_lags": 0,
            "n_total_aligned_rows": 0,
            "n_valid_alignment_rows": 0,
            "n_valid_primary_alignment_rows": 0,
            "n_valid_exploratory_alignment_rows": 0,
            "n_valid_conditional_rows": 0,
            "n_valid_mc_conditional_rows": 0,
            "ready_frames": [],
            "primary_ready_frames": [],
        }

    ready = inventory_df[
        pd.to_numeric(
            inventory_df[
                "n_valid_alignment_rows"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .gt(0)
    ].copy()

    primary_ready = inventory_df[
        pd.to_numeric(
            inventory_df[
                "n_valid_primary_alignment_rows"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .gt(0)
    ].copy()

    return {
        "n_frames": int(
            len(inventory_df)
        ),
        "n_window_sizes": (
            int(
                inventory_df[
                    "window_seconds"
                ].nunique()
            )
            if (
                "window_seconds"
                in inventory_df.columns
            )
            else 0
        ),
        "n_lags": (
            int(
                inventory_df[
                    "lag_windows"
                ].nunique()
            )
            if (
                "lag_windows"
                in inventory_df.columns
            )
            else 0
        ),
        "n_total_aligned_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "n_valid_alignment_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_valid_alignment_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "n_valid_primary_alignment_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_valid_primary_alignment_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "n_valid_exploratory_alignment_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_valid_exploratory_alignment_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "n_valid_conditional_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_valid_conditional_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "n_valid_mc_conditional_rows": int(
            pd.to_numeric(
                inventory_df[
                    "n_valid_mc_conditional_rows"
                ],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        ),
        "scope_counts": _value_counts_dict(
            inventory_df,
            "alignment_scope",
        ),
        "lag_direction_counts": (
            _value_counts_dict(
                inventory_df,
                "lag_direction",
            )
        ),
        "status_counts": _value_counts_dict(
            inventory_df,
            "status",
        ),
        "ready_frames": (
            ready["frame_file"]
            .astype(str)
            .tolist()
            if "frame_file" in ready.columns
            else []
        ),
        "primary_ready_frames": (
            primary_ready[
                "frame_file"
            ]
            .astype(str)
            .tolist()
            if (
                "frame_file"
                in primary_ready.columns
            )
            else []
        ),
    }


def build_alignment_result(
    status: str,
    protocol_only: bool,
    summary: Dict[str, Any],
    interpretation: str,
) -> AlignmentBuildResult:
    return AlignmentBuildResult(
        status=status,
        protocol_only=bool(
            protocol_only
        ),
        input_rows=int(
            summary.get(
                "n_total_aligned_rows",
                0,
            )
        ),
        n_frames=int(
            summary.get(
                "n_frames",
                0,
            )
        ),
        n_window_sizes=int(
            summary.get(
                "n_window_sizes",
                0,
            )
        ),
        n_lags=int(
            summary.get(
                "n_lags",
                0,
            )
        ),
        n_total_aligned_rows=int(
            summary.get(
                "n_total_aligned_rows",
                0,
            )
        ),
        n_valid_alignment_rows=int(
            summary.get(
                "n_valid_alignment_rows",
                0,
            )
        ),
        n_valid_primary_alignment_rows=int(
            summary.get(
                "n_valid_primary_alignment_rows",
                0,
            )
        ),
        n_valid_exploratory_alignment_rows=int(
            summary.get(
                "n_valid_exploratory_alignment_rows",
                0,
            )
        ),
        n_valid_conditional_rows=int(
            summary.get(
                "n_valid_conditional_rows",
                0,
            )
        ),
        n_valid_mc_conditional_rows=int(
            summary.get(
                "n_valid_mc_conditional_rows",
                0,
            )
        ),
        can_proceed_to_conditional_cdr=(
            int(
                summary.get(
                    "n_valid_conditional_rows",
                    0,
                )
            )
            > 0
        ),
        can_proceed_to_empirical_cdr=(
            int(
                summary.get(
                    "n_valid_primary_alignment_rows",
                    0,
                )
            )
            > 0
        ),
        interpretation=interpretation,
    )


def write_alignment_summary_txt(
    path: Path,
    result: AlignmentBuildResult,
    summary: Dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append(
        "Phase III.2E — Lagged Alignment Summary"
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
        f"n_frames: {result.n_frames}"
    )
    lines.append(
        f"n_window_sizes: {result.n_window_sizes}"
    )
    lines.append(
        f"n_lags: {result.n_lags}"
    )
    lines.append("")

    lines.append(
        f"n_total_aligned_rows: "
        f"{result.n_total_aligned_rows}"
    )
    lines.append(
        f"n_valid_alignment_rows: "
        f"{result.n_valid_alignment_rows}"
    )
    lines.append(
        f"n_valid_primary_alignment_rows: "
        f"{result.n_valid_primary_alignment_rows}"
    )
    lines.append(
        f"n_valid_exploratory_alignment_rows: "
        f"{result.n_valid_exploratory_alignment_rows}"
    )
    lines.append(
        f"n_valid_conditional_rows: "
        f"{result.n_valid_conditional_rows}"
    )
    lines.append(
        f"n_valid_mc_conditional_rows: "
        f"{result.n_valid_mc_conditional_rows}"
    )
    lines.append("")

    lines.append(
        "can_proceed_to_conditional_cdr: "
        f"{result.can_proceed_to_conditional_cdr}"
    )
    lines.append(
        "can_proceed_to_empirical_cdr: "
        f"{result.can_proceed_to_empirical_cdr}"
    )
    lines.append("")

    lines.append(
        f"scope_counts: "
        f"{summary.get('scope_counts')}"
    )
    lines.append(
        f"lag_direction_counts: "
        f"{summary.get('lag_direction_counts')}"
    )
    lines.append(
        f"status_counts: "
        f"{summary.get('status_counts')}"
    )
    lines.append("")

    lines.append(
        f"ready_frames: "
        f"{summary.get('ready_frames')}"
    )
    lines.append(
        f"primary_ready_frames: "
        f"{summary.get('primary_ready_frames')}"
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


# =========================================================
# Save outputs
# =========================================================

def save_alignment_outputs(
    cfg: Phase32EConfig,
    inventory_df: pd.DataFrame,
    report: Dict[str, Any],
    result: AlignmentBuildResult,
) -> None:
    Path(
        cfg.lagged_frames_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

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

    inventory_df.to_csv(
        alignment_inventory_csv_path(
            cfg
        ),
        index=False,
    )

    metadata = {
        "phase": cfg.phase_name,
        "project_name": (
            cfg.project_name
        ),
        "module": ALIGNMENT_MODULE,
        "alignment_version": (
            ALIGNMENT_VERSION
        ),
        "created_at": _now_str(),
        "result": asdict(result),
        "alignment_inventory_csv": str(
            alignment_inventory_csv_path(
                cfg
            )
        ),
        "alignment_report_json": str(
            alignment_report_json_path(
                cfg
            )
        ),
        "alignment_summary_txt": str(
            alignment_summary_txt_path(
                cfg
            )
        ),
        "lagged_frames_dir": str(
            cfg.lagged_frames_dir
        ),
        "active_conditional_pairs": list(
            active_conditional_pairs(cfg)
        ),
        "config": describe_config(cfg),
    }

    save_json(
        alignment_metadata_json_path(
            cfg
        ),
        metadata,
    )

    save_json(
        alignment_report_json_path(
            cfg
        ),
        report,
    )

    write_alignment_summary_txt(
        path=alignment_summary_txt_path(
            cfg
        ),
        result=result,
        summary=report.get(
            "summary",
            {},
        ),
    )

    print(
        "[Phase3.2E Alignment] "
        f"Saved inventory: "
        f"{alignment_inventory_csv_path(cfg)}"
    )

    print(
        "[Phase3.2E Alignment] "
        f"Saved metadata: "
        f"{alignment_metadata_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Alignment] "
        f"Saved report: "
        f"{alignment_report_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Alignment] "
        f"Saved summary: "
        f"{alignment_summary_txt_path(cfg)}"
    )


# =========================================================
# Main builder
# =========================================================

def build_phase3_2e_alignment(
    cfg: Optional[
        Phase32EConfig
    ] = None,
    force_rebuild_features: bool = False,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
    save_frames: bool = True,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print(
        "[Phase3.2E Alignment] "
        "Loading Phase III.2E features"
    )

    feature_frame = (
        load_or_build_phase3_2e_features(
            cfg=cfg,
            force_rebuild=bool(
                force_rebuild_features
            ),
            force_rebuild_windows=bool(
                force_rebuild_windows
            ),
            force_rebuild_loader=bool(
                force_rebuild_loader
            ),
        )
    )

    features_df = feature_frame.df

    feature_status = str(
        feature_frame.metadata.get(
            "status",
            "",
        )
    )

    # -----------------------------------------------------
    # Protocol-only / blocked path
    # -----------------------------------------------------

    if (
        features_df.empty
        or feature_status
        == cfg.status_pending_synchronized_data
    ):
        inventory_df = (
            empty_alignment_inventory(
                cfg
            )
        )

        summary = (
            summarize_alignment_inventory(
                inventory_df
            )
        )

        status = (
            cfg.status_pending_synchronized_data
        )

        interpretation = (
            "No synchronized EEG–QRNG dataset is available. "
            "Lagged alignment completed in protocol-only mode "
            "with empty canonical inventory."
        )

        result = build_alignment_result(
            status=status,
            protocol_only=True,
            summary=summary,
            interpretation=interpretation,
        )

        report = {
            "phase": cfg.phase_name,
            "project_name": (
                cfg.project_name
            ),
            "module": ALIGNMENT_MODULE,
            "alignment_version": (
                ALIGNMENT_VERSION
            ),
            "status": result.status,
            "created_at": _now_str(),
            "feature_status": (
                feature_status
            ),
            "feature_metadata": (
                feature_frame.metadata
            ),
            "result": asdict(result),
            "summary": summary,
            "inventory": (
                inventory_df.to_dict(
                    orient="records"
                )
            ),
            "active_conditional_pairs": list(
                active_conditional_pairs(cfg)
            ),
        }

        save_alignment_outputs(
            cfg,
            inventory_df,
            report,
            result,
        )

        return inventory_df, report

    if (
        "valid_conditional_row"
        not in features_df.columns
    ):
        inventory_df = (
            empty_alignment_inventory(
                cfg
            )
        )

        summary = (
            summarize_alignment_inventory(
                inventory_df
            )
        )

        status = (
            "alignment_blocked_by_features"
        )

        interpretation = (
            "Feature dataframe does not contain "
            "valid_conditional_row. Lagged alignment "
            "cannot proceed."
        )

        result = build_alignment_result(
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
            "module": ALIGNMENT_MODULE,
            "alignment_version": (
                ALIGNMENT_VERSION
            ),
            "status": result.status,
            "created_at": _now_str(),
            "feature_status": (
                feature_status
            ),
            "feature_metadata": (
                feature_frame.metadata
            ),
            "result": asdict(result),
            "summary": summary,
            "inventory": (
                inventory_df.to_dict(
                    orient="records"
                )
            ),
            "active_conditional_pairs": list(
                active_conditional_pairs(cfg)
            ),
        }

        save_alignment_outputs(
            cfg,
            inventory_df,
            report,
            result,
        )

        return inventory_df, report

    # -----------------------------------------------------
    # Full empirical path
    # -----------------------------------------------------

    stale_frames_removed = (
        remove_stale_lagged_frames(
            cfg
        )
    )

    inventory_rows: List[
        Dict[str, Any]
    ] = []

    for window_seconds in cfg.all_window_seconds:
        for lag_windows in cfg.all_lags_windows:
            frame_file = lagged_frame_path(
                cfg,
                window_seconds,
                lag_windows,
            )

            frame = build_lagged_alignment_frame(
                features_df=features_df,
                cfg=cfg,
                window_seconds=int(
                    window_seconds
                ),
                lag_windows=int(
                    lag_windows
                ),
            )

            if save_frames:
                frame_file.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                frame.to_csv(
                    frame_file,
                    index=False,
                )

            row = summarize_alignment_frame(
                frame=frame,
                cfg=cfg,
                window_seconds=int(
                    window_seconds
                ),
                lag_windows=int(
                    lag_windows
                ),
                frame_file=frame_file,
            )

            inventory_rows.append(
                row
            )

            print(
                "[Phase3.2E Alignment] "
                f"window={window_seconds}s "
                f"lag={lag_windows} "
                f"valid={row.get('n_valid_alignment_rows')} "
                f"file={frame_file.name}"
            )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    summary = (
        summarize_alignment_inventory(
            inventory_df
        )
    )

    if (
        int(
            summary.get(
                "n_valid_primary_alignment_rows",
                0,
            )
        )
        > 0
    ):
        status = "alignment_ready"

        interpretation = (
            "Lagged synchronized alignment frames were built "
            "successfully. At least one primary registered "
            "alignment is ready for conditional CDR."
        )

    elif (
        int(
            summary.get(
                "n_valid_conditional_rows",
                0,
            )
        )
        > 0
    ):
        status = (
            "alignment_ready_exploratory_only"
        )

        interpretation = (
            "Lagged synchronized alignment frames were built "
            "and exploratory conditional rows are available, "
            "but no primary registered alignment is ready "
            "for strong inference."
        )

    else:
        status = (
            "alignment_no_valid_rows"
        )

        interpretation = (
            "Lagged alignment frames were generated, "
            "but no valid conditional alignment rows "
            "are available."
        )

    result = build_alignment_result(
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
        "module": ALIGNMENT_MODULE,
        "alignment_version": (
            ALIGNMENT_VERSION
        ),
        "status": result.status,
        "created_at": _now_str(),
        "feature_status": (
            feature_status
        ),
        "feature_metadata": (
            feature_frame.metadata
        ),
        "result": asdict(result),
        "summary": summary,
        "inventory": (
            inventory_df.to_dict(
                orient="records"
            )
        ),
        "active_conditional_pairs": list(
            active_conditional_pairs(cfg)
        ),
        "registered_window_seconds": [
            int(value)
            for value in (
                cfg.all_window_seconds
            )
        ],
        "primary_window_seconds": [
            int(value)
            for value in (
                cfg.primary_test_window_seconds
            )
        ],
        "registered_lags_windows": [
            int(value)
            for value in (
                cfg.all_lags_windows
            )
        ],
        "primary_lags_windows": [
            int(value)
            for value in (
                cfg.primary_test_lags_windows
            )
        ],
        "stale_frames_removed": (
            stale_frames_removed
        ),
        "lag_convention": {
            "lag_0": (
                "target is current synchronized window"
            ),
            "lag_positive": (
                "current row at t predicts target row "
                "at t + lag"
            ),
            "lag_negative": (
                "current row at t predicts target row "
                "at t - |lag|"
            ),
        },
    }

    save_alignment_outputs(
        cfg,
        inventory_df,
        report,
        result,
    )

    return inventory_df, report


def load_or_build_phase3_2e_alignment(
    cfg: Optional[
        Phase32EConfig
    ] = None,
    force_rebuild: bool = False,
    force_rebuild_features: bool = False,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
) -> Tuple[
    pd.DataFrame,
    Dict[str, Any],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    inventory_path = (
        alignment_inventory_csv_path(
            cfg
        )
    )

    report_path = (
        alignment_report_json_path(
            cfg
        )
    )

    if (
        inventory_path.exists()
        and report_path.exists()
        and not force_rebuild
    ):
        print(
            "[Phase3.2E Alignment] "
            f"Loading cached inventory: "
            f"{inventory_path}"
        )

        try:
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
                    "alignment_version"
                )
                == ALIGNMENT_VERSION
            ):
                print(
                    "[Phase3.2E Alignment] "
                    "Cached alignment compatible."
                )

                return inventory_df, report

            print(
                "[Phase3.2E Alignment] "
                "Cached version changed. Rebuilding..."
            )

        except Exception as exc:
            print(
                "[Phase3.2E Alignment] "
                f"Failed to load cache: {exc}. "
                "Rebuilding..."
            )

    return build_phase3_2e_alignment(
        cfg=cfg,
        force_rebuild_features=bool(
            force_rebuild_features
        ),
        force_rebuild_windows=bool(
            force_rebuild_windows
        ),
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
            "Build Phase III.2E synchronized "
            "lagged alignment frames."
        )
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help=(
            "Force rebuilding alignment outputs."
        ),
    )

    parser.add_argument(
        "--force-rebuild-features",
        action="store_true",
        help=(
            "Force rebuilding features before alignment."
        ),
    )

    parser.add_argument(
        "--force-rebuild-windows",
        action="store_true",
        help=(
            "Force rebuilding windows before "
            "features/alignment."
        ),
    )

    parser.add_argument(
        "--force-rebuild-loader",
        action="store_true",
        help=(
            "Force rebuilding loader before "
            "windows/features/alignment."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    inventory_df, report = (
        load_or_build_phase3_2e_alignment(
            cfg=cfg,
            force_rebuild=bool(
                args.force_rebuild
            ),
            force_rebuild_features=bool(
                args.force_rebuild_features
            ),
            force_rebuild_windows=bool(
                args.force_rebuild_windows
            ),
            force_rebuild_loader=bool(
                args.force_rebuild_loader
            ),
        )
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
        "Phase III.2E lagged alignment completed"
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
        f"n_frames: "
        f"{result.get('n_frames')}"
    )

    print(
        f"n_window_sizes: "
        f"{result.get('n_window_sizes')}"
    )

    print(
        f"n_lags: "
        f"{result.get('n_lags')}"
    )

    print(
        f"n_total_aligned_rows: "
        f"{result.get('n_total_aligned_rows')}"
    )

    print(
        f"n_valid_alignment_rows: "
        f"{result.get('n_valid_alignment_rows')}"
    )

    print(
        f"n_valid_primary_alignment_rows: "
        f"{result.get('n_valid_primary_alignment_rows')}"
    )

    print(
        f"n_valid_conditional_rows: "
        f"{result.get('n_valid_conditional_rows')}"
    )

    print(
        f"n_valid_mc_conditional_rows: "
        f"{result.get('n_valid_mc_conditional_rows')}"
    )

    print(
        "can_proceed_to_conditional_cdr: "
        f"{result.get('can_proceed_to_conditional_cdr')}"
    )

    print(
        "can_proceed_to_empirical_cdr: "
        f"{result.get('can_proceed_to_empirical_cdr')}"
    )

    print(
        f"scope_counts: "
        f"{summary.get('scope_counts')}"
    )

    print(
        f"lag_direction_counts: "
        f"{summary.get('lag_direction_counts')}"
    )

    print(
        f"registered_windows: "
        f"{list(cfg.all_window_seconds)}"
    )

    print(
        f"primary_windows: "
        f"{list(cfg.primary_test_window_seconds)}"
    )

    print(
        f"registered_lags: "
        f"{list(cfg.all_lags_windows)}"
    )

    print(
        f"primary_lags: "
        f"{list(cfg.primary_test_lags_windows)}"
    )

    print(
        f"inventory_csv: "
        f"{alignment_inventory_csv_path(cfg)}"
    )

    print(
        f"lagged_frames_dir: "
        f"{cfg.lagged_frames_dir}"
    )

    print(
        f"interpretation: "
        f"{result.get('interpretation')}"
    )

    print(
        "=" * 78
        + "\n"
    )


if __name__ == "__main__":
    main()