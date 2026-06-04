from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    active_conditional_pairs,
    describe_config,
    load_phase3_2e_config,
)

from src.phase3_2e_alignment import (
    alignment_inventory_csv_path,
    load_or_build_phase3_2e_alignment,
)

from src.phase3_2e_conditional import (
    evaluate_conditional_models_for_frame,
)


# =========================================================
# Phase III.2E — Negative Controls
# Synchronized EEG–QRNG protocol version
# =========================================================
#
# Purpose:
#
#   Verify that synchronized conditional EEG–QRNG effects collapse when the
#   temporal, subject, QRNG, EEG, target, window or synchronization structure is
#   deliberately broken.
#
# Phase III.2E controls:
#
#   - within_subject_qrng_shuffle
#   - within_subject_eeg_shuffle
#   - circular_qrng_shift
#   - subject_mismatch
#   - timestamp_jitter_control
#   - phase_randomized_eeg_control
#   - window_permutation_control
#   - conditional_target_shuffle
#   - sync_break_control
#
# Critical interpretation:
#
#   Controls do not fail merely because eps_test is high.
#   Controls fail when conditional lift, subject-fraction persistence, BIC/LL
#   improvement or strong candidate persistence survive negative controls.
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   empty canonical control outputs are saved
#


# =========================================================
# Constants
# =========================================================

CONTROLS_MODULE = "phase3_2e_controls"
CONTROLS_VERSION = "phase3_2e_controls_v1_sync_break_timestamp_jitter"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class ControlRunResult:
    control_type: str
    replicate: int

    window_seconds: int
    lag_windows: int
    lag_seconds: int
    lag_label: str
    lag_direction: str
    alignment_scope: str

    valid: bool
    primary_alignment: bool

    n_rows: int
    n_subjects: int
    n_sessions: int

    max_baseline_eps_test: float
    max_augmented_eps_test: float
    max_conditional_lift: float
    max_fraction_positive_lift: float
    max_bic_improvement: float
    max_ll_improvement: float

    n_positive_lift_rows: int
    n_strong_candidate_rows: int

    n_multichannel_pair_rows: int
    n_multichannel_positive_lift_rows: int
    n_multichannel_strong_candidate_rows: int

    eps_saturation_value: float
    n_eps_saturated_rows: int
    eps_saturation_fraction: float

    collapsed_by_lift: bool
    collapsed_by_bic: bool
    collapsed_by_subject_fraction: bool
    collapsed_by_strong_candidates: bool
    collapsed_overall: bool

    reason: str = "ok"


@dataclass
class ControlSummary:
    control_type: str
    n_runs: int
    n_valid_runs: int

    median_max_baseline_eps_test: float
    median_max_augmented_eps_test: float
    mean_max_augmented_eps_test: float
    max_augmented_eps_test: float

    median_max_conditional_lift: float
    mean_max_conditional_lift: float
    max_conditional_lift: float

    median_max_bic_improvement: float
    max_bic_improvement: float

    median_max_ll_improvement: float
    max_ll_improvement: float

    median_eps_saturation_fraction: float
    max_eps_saturation_fraction: float

    total_multichannel_pair_rows: int
    total_multichannel_positive_lift_rows: int
    total_multichannel_strong_candidate_rows: int

    fraction_runs_below_control_tol: float
    fraction_runs_collapsed_overall: float
    control_tol: float

    passed: bool
    reason: str


@dataclass
class ControlsBuildResult:
    status: str
    protocol_only: bool

    n_control_runs: int
    n_valid_control_runs: int
    n_control_pair_rows: int
    n_control_types: int

    overall_passed: bool
    failed_control_types: List[str]

    can_proceed_to_metrics: bool
    can_proceed_to_empirical_cdr: bool

    interpretation: str


# =========================================================
# Generic helpers
# =========================================================

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default

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


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def _as_records(items: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for item in items:
        if hasattr(item, "__dict__"):
            rows.append(asdict(item))
        elif isinstance(item, dict):
            rows.append(item)
        else:
            rows.append({"value": item})

    return rows


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def _existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def _lag_label(lag_windows: int) -> str:
    lag_windows = int(lag_windows)

    if lag_windows < 0:
        return f"m{abs(lag_windows)}"

    if lag_windows > 0:
        return f"p{lag_windows}"

    return "0"


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        c for c in [
            "subject_id",
            "session_id",
            "window_seconds",
            "window_start_utc",
            "window_id",
            "alignment_row_id",
            "feature_row_id",
        ]
        if c in df.columns
    ]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


def _group_columns(df: pd.DataFrame) -> List[str]:
    candidates = ["subject_id", "session_id", "window_seconds"]
    cols = [c for c in candidates if c in df.columns]

    if cols:
        return cols

    if "subject_id" in df.columns:
        return ["subject_id"]

    return []


# =========================================================
# Output paths
# =========================================================

def control_runs_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.control_runs_csv)


def control_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.control_pair_scores_csv)


def controls_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.controls_json)


def controls_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.controls_summary_txt)


def controls_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_controls_report.json"


# =========================================================
# Empty outputs
# =========================================================

def empty_control_runs_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "control_type",
            "replicate",
            "window_seconds",
            "lag_windows",
            "lag_seconds",
            "lag_label",
            "lag_direction",
            "alignment_scope",
            "valid",
            "primary_alignment",
            "n_rows",
            "n_subjects",
            "n_sessions",
            "max_baseline_eps_test",
            "max_augmented_eps_test",
            "max_conditional_lift",
            "max_fraction_positive_lift",
            "max_bic_improvement",
            "max_ll_improvement",
            "n_positive_lift_rows",
            "n_strong_candidate_rows",
            "n_multichannel_pair_rows",
            "n_multichannel_positive_lift_rows",
            "n_multichannel_strong_candidate_rows",
            "eps_saturation_value",
            "n_eps_saturated_rows",
            "eps_saturation_fraction",
            "collapsed_by_lift",
            "collapsed_by_bic",
            "collapsed_by_subject_fraction",
            "collapsed_by_strong_candidates",
            "collapsed_overall",
            "reason",
        ]
    )


def empty_control_pair_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "control_type",
            "replicate",
            "control_seed",
            "window_seconds",
            "lag_windows",
            "family",
            "baseline_model",
            "augmented_model",
            "valid",
            "primary",
            "multichannel",
            "conditional_lift",
            "baseline_eps_test",
            "augmented_eps_test",
            "fraction_positive_lift",
            "bic_improvement",
            "ll_improvement",
            "aic_improvement",
            "control_collapsed_overall",
        ]
    )


# =========================================================
# Column groups — Phase III.2E alignment frames
# =========================================================

def qrng_current_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "qrng_state",
        "qrng_info_bin",
        "qrng_info_score",
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
        "qrng_run_length_mean",
        "qrng_run_length_std",
        "qrng_run_length_max",
        "qrng_compressibility_proxy",
        "qrng_surprise_index",
        "qrng_transition_asymmetry",
        "qrng_quality_score",
        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",
    ]

    return _existing_columns(df, candidates)


def eeg_current_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "eeg_state",
        "eeg_info_bin",
        "eeg_info_score",
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_entropy",
        "eeg_gamma_power",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
        "eeg_line_length",
        "eeg_signal_std",
        "eeg_artifact_score",
        "eeg_quality_score",
        "mc_eeg_state",
        "mc_eeg_info_bin",
        "mc_eeg_info_score",
        "mc_activation_score",
        "mc_cross_channel_score",
        "eeg_primary_delta_power",
        "eeg_primary_alpha_power",
        "eeg_secondary_delta_power",
        "eeg_secondary_alpha_power",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
        "cross_channel_corr",
        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",
    ]

    return _existing_columns(df, candidates)


def target_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "eeg_next_state",
        "qrng_next_state",
        "mc_eeg_next_state",
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
        "observed_joint_state_next",
        "mc_observed_joint_state_next",
        "informational_joint_state_next",
        "mc_informational_joint_state_next",
    ]

    return _existing_columns(df, candidates)


def qrng_target_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "qrng_next_state",
        "target_qrng_state",
        "target_qrng_info_bin",
    ]

    return _existing_columns(df, candidates)


def timestamp_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "window_start_utc",
        "window_end_utc",
        "window_midpoint_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_qrng_gap_ms",
        "sync_quality",
        "clock_drift_ms",
        "clock_drift_abs_ms",
        "sync_row_valid",
    ]

    return _existing_columns(df, candidates)


def _is_multichannel_pair_rows(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(np.zeros(len(df), dtype=bool), index=df.index)

    for col in ["baseline_model", "augmented_model", "family"]:
        if col not in df.columns:
            continue

        text = df[col].astype(str)

        mask |= text.str.contains("MCEEG", case=False, na=False)
        mask |= text.str.contains("MC_EEG", case=False, na=False)
        mask |= text.str.contains("multichannel", case=False, na=False)
        mask |= text.str.startswith("E4")
        mask |= text.str.startswith("E5")
        mask |= text.str.startswith("E6")

    return mask


# =========================================================
# Shuffle / shift utilities
# =========================================================

def _shuffle_values(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    arr = values.to_numpy(copy=True)
    rng.shuffle(arr)
    return pd.Series(arr, index=values.index)


def _shuffle_columns_within_groups(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_cols: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    if not columns:
        return out

    if not group_cols:
        for col in columns:
            out[col] = _shuffle_values(out[col], rng).to_numpy()
        return out

    _require_columns(out, group_cols, context="Phase III.2E grouped shuffle")

    for _, idx in out.groupby(list(group_cols), sort=False).groups.items():
        idx_list = list(idx)

        if len(idx_list) <= 1:
            continue

        for col in columns:
            if col not in out.columns:
                continue

            out.loc[idx_list, col] = _shuffle_values(out.loc[idx_list, col], rng).to_numpy()

    return out


def _circular_shift_columns_within_groups(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_cols: Sequence[str],
    rng: np.random.Generator,
    fixed_shift: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()

    if not columns:
        return out

    if not group_cols:
        n = len(out)
        if n <= 2:
            return out

        shift = int(fixed_shift) if fixed_shift is not None else int(rng.integers(1, n))

        for col in columns:
            arr = out[col].to_numpy(copy=True)
            out[col] = np.roll(arr, shift)

        return out

    _require_columns(out, group_cols, context="Phase III.2E circular shift")

    for _, idx in out.groupby(list(group_cols), sort=False).groups.items():
        idx_list = list(idx)
        n = len(idx_list)

        if n <= 2:
            continue

        if fixed_shift is None:
            shift = int(rng.integers(1, n))
        else:
            shift = int(fixed_shift) % n
            if shift == 0:
                shift = 1

        for col in columns:
            if col not in out.columns:
                continue

            arr = out.loc[idx_list, col].to_numpy(copy=True)
            arr = np.roll(arr, shift)
            out.loc[idx_list, col] = arr

    return out


def _subject_mismatch_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = _sort_frame(df.copy())

    if not columns:
        return out

    if "subject_id" not in out.columns:
        return out

    subjects = sorted(out["subject_id"].astype(str).unique().tolist())

    if len(subjects) < 2:
        return out

    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    if all(a == b for a, b in zip(subjects, shuffled_subjects)):
        shuffled_subjects = shuffled_subjects[1:] + shuffled_subjects[:1]

    mapping = dict(zip(subjects, shuffled_subjects))

    source_by_subject: Dict[str, pd.DataFrame] = {
        str(subject): out[out["subject_id"].astype(str) == str(subject)].copy().reset_index(drop=True)
        for subject in subjects
    }

    for target_subject in subjects:
        source_subject = mapping[target_subject]

        target_idx = out[out["subject_id"].astype(str) == str(target_subject)].index.to_numpy()
        source_df = source_by_subject[source_subject]

        if len(target_idx) == 0 or source_df.empty:
            continue

        for col in columns:
            if col not in out.columns or col not in source_df.columns:
                continue

            source_values = source_df[col].to_numpy(copy=True)

            if len(source_values) == 0:
                continue

            resized = np.resize(source_values, len(target_idx))
            out.loc[target_idx, col] = resized

    return out


# =========================================================
# Rebuild aliases after controls
# =========================================================

def synchronize_target_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    alias_pairs = [
        ("target_eeg_state", "eeg_next_state"),
        ("target_qrng_state", "qrng_next_state"),
        ("target_mc_eeg_state", "mc_eeg_next_state"),
    ]

    for target_col, alias_col in alias_pairs:
        if target_col in out.columns:
            out[alias_col] = pd.to_numeric(out[target_col], errors="coerce").astype("Int64")

    return out


def rebuild_control_joint_states(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    out = df.copy()

    if "eeg_state" in out.columns and "qrng_state" in out.columns:
        valid = out["eeg_state"].notna() & out["qrng_state"].notna()

        out.loc[valid, "observed_joint_state"] = (
            out.loc[valid, "eeg_state"].astype(int) * int(cfg.qrng_n_bins)
            + out.loc[valid, "qrng_state"].astype(int)
        )

    if "mc_eeg_state" in out.columns and "qrng_state" in out.columns:
        valid = out["mc_eeg_state"].notna() & out["qrng_state"].notna()

        out.loc[valid, "mc_observed_joint_state"] = (
            out.loc[valid, "mc_eeg_state"].astype(int) * int(cfg.qrng_n_bins)
            + out.loc[valid, "qrng_state"].astype(int)
        )

    if "eeg_info_bin" in out.columns and "qrng_info_bin" in out.columns:
        valid = out["eeg_info_bin"].notna() & out["qrng_info_bin"].notna()

        out.loc[valid, "informational_joint_state"] = (
            out.loc[valid, "eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
            + out.loc[valid, "qrng_info_bin"].astype(int)
        )

    if "mc_eeg_info_bin" in out.columns and "qrng_info_bin" in out.columns:
        valid = out["mc_eeg_info_bin"].notna() & out["qrng_info_bin"].notna()

        out.loc[valid, "mc_informational_joint_state"] = (
            out.loc[valid, "mc_eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
            + out.loc[valid, "qrng_info_bin"].astype(int)
        )

    if "target_eeg_state" in out.columns and "target_qrng_state" in out.columns:
        valid = out["target_eeg_state"].notna() & out["target_qrng_state"].notna()

        out.loc[valid, "target_observed_joint_state"] = (
            out.loc[valid, "target_eeg_state"].astype(int) * int(cfg.qrng_n_bins)
            + out.loc[valid, "target_qrng_state"].astype(int)
        )

    if "target_mc_eeg_state" in out.columns and "target_qrng_state" in out.columns:
        valid = out["target_mc_eeg_state"].notna() & out["target_qrng_state"].notna()

        out.loc[valid, "target_mc_observed_joint_state"] = (
            out.loc[valid, "target_mc_eeg_state"].astype(int) * int(cfg.qrng_n_bins)
            + out.loc[valid, "target_qrng_state"].astype(int)
        )

    out = synchronize_target_aliases(out)

    int_cols = [
        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "eeg_next_state",
        "qrng_next_state",
        "mc_eeg_next_state",
        "target_eeg_state",
        "target_qrng_state",
        "target_mc_eeg_state",
    ]

    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


# =========================================================
# Control application
# =========================================================

def apply_timestamp_jitter_control(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    # The actual conditional models operate on states, not timestamps.
    # Therefore timestamp jitter must also break the synchronized QRNG alignment
    # while retaining row count and marginal distributions.
    cols = qrng_current_columns(out)
    group_cols = _group_columns(out)

    out = _circular_shift_columns_within_groups(
        df=out,
        columns=cols,
        group_cols=group_cols,
        rng=rng,
        fixed_shift=1,
    )

    jitter_ms = float(cfg.timestamp_jitter_ms)

    if "eeg_qrng_gap_ms" in out.columns:
        out["eeg_qrng_gap_ms"] = float(cfg.max_timestamp_gap_ms) + abs(jitter_ms)

    if "sync_row_valid" in out.columns:
        out["sync_row_valid"] = 0

    if "sync_quality" in out.columns:
        out["sync_quality"] = np.minimum(
            pd.to_numeric(out["sync_quality"], errors="coerce").fillna(1.0),
            float(cfg.min_sync_quality) - 0.01,
        )

    out["timestamp_jitter_applied_ms"] = jitter_ms
    out["timestamp_jitter_control_active"] = 1

    return out


def apply_sync_break_control(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    cols = qrng_current_columns(out) + qrng_target_columns(out)
    cols = list(dict.fromkeys(cols))

    group_cols = _group_columns(out)
    shift_windows = int(getattr(cfg, "sync_break_shift_windows", 10))

    out = _circular_shift_columns_within_groups(
        df=out,
        columns=cols,
        group_cols=group_cols,
        rng=rng,
        fixed_shift=max(1, shift_windows),
    )

    if "sync_row_valid" in out.columns:
        out["sync_row_valid"] = 0

    if "eeg_qrng_gap_ms" in out.columns:
        out["eeg_qrng_gap_ms"] = float(cfg.max_timestamp_gap_ms) + abs(float(shift_windows)) * 1000.0

    if "sync_quality" in out.columns:
        out["sync_quality"] = np.minimum(
            pd.to_numeric(out["sync_quality"], errors="coerce").fillna(1.0),
            float(cfg.min_sync_quality) - 0.01,
        )

    out["sync_break_shift_windows"] = shift_windows
    out["sync_break_control_active"] = 1

    return out


def apply_phase_randomized_eeg_control(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    # We do not have raw phase spectra in the synchronized table yet.
    # This surrogate preserves EEG marginals but breaks EEG temporal phase/order.
    cols = eeg_current_columns(out)
    group_cols = _group_columns(out)

    out = _circular_shift_columns_within_groups(
        df=out,
        columns=cols,
        group_cols=group_cols,
        rng=rng,
        fixed_shift=None,
    )

    out["phase_randomized_eeg_control_active"] = 1

    return out


def apply_window_permutation_control(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()

    cols = target_columns(out)
    group_cols = _group_columns(out)

    out = _shuffle_columns_within_groups(
        df=out,
        columns=cols,
        group_cols=group_cols,
        rng=rng,
    )

    out["window_permutation_control_active"] = 1

    return out


def apply_control(
    df: pd.DataFrame,
    control_type: str,
    cfg: Optional[Phase32EConfig] = None,
    seed: int = 0,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_phase3_2e_config()

    out = _sort_frame(df.copy())
    rng = np.random.default_rng(int(seed))

    group_cols = _group_columns(out)

    if control_type == "within_subject_qrng_shuffle":
        cols = qrng_current_columns(out)

        out = _shuffle_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    elif control_type == "within_subject_eeg_shuffle":
        cols = eeg_current_columns(out)

        out = _shuffle_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    elif control_type == "circular_qrng_shift":
        cols = qrng_current_columns(out)

        out = _circular_shift_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    elif control_type == "subject_mismatch":
        cols = qrng_current_columns(out)

        out = _subject_mismatch_columns(
            df=out,
            columns=cols,
            rng=rng,
        )

    elif control_type == "timestamp_jitter_control":
        out = apply_timestamp_jitter_control(
            df=out,
            cfg=cfg,
            rng=rng,
        )

    elif control_type == "phase_randomized_eeg_control":
        out = apply_phase_randomized_eeg_control(
            df=out,
            cfg=cfg,
            rng=rng,
        )

    elif control_type == "window_permutation_control":
        out = apply_window_permutation_control(
            df=out,
            cfg=cfg,
            rng=rng,
        )

    elif control_type == "conditional_target_shuffle":
        cols = target_columns(out)

        out = _shuffle_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    elif control_type == "sync_break_control":
        out = apply_sync_break_control(
            df=out,
            cfg=cfg,
            rng=rng,
        )

    else:
        raise ValueError(f"Unknown Phase III.2E control_type: {control_type}")

    out = rebuild_control_joint_states(out, cfg)

    out["control_type"] = str(control_type)
    out["control_seed"] = int(seed)

    return _sort_frame(out)


# =========================================================
# Control evaluation helpers
# =========================================================

def _valid_pair_scores(pair_scores_df: pd.DataFrame) -> pd.DataFrame:
    if pair_scores_df.empty or "valid" not in pair_scores_df.columns:
        return pd.DataFrame()

    return pair_scores_df[_as_bool_series(pair_scores_df["valid"])].copy()


def _count_eps_saturation(
    valid_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[int, float]:
    if valid_df.empty:
        return 0, 0.0

    eps_saturation_value = float(cfg.eps_saturation_value)

    cols = [c for c in ["baseline_eps_test", "augmented_eps_test"] if c in valid_df.columns]

    if not cols:
        return 0, 0.0

    sat_mask = np.zeros(len(valid_df), dtype=bool)

    for col in cols:
        eps = pd.to_numeric(valid_df[col], errors="coerce").fillna(0.0)
        sat_mask |= (eps >= eps_saturation_value).to_numpy(dtype=bool)

    n_sat = int(sat_mask.sum())
    frac = float(np.mean(sat_mask)) if len(sat_mask) else 0.0

    return n_sat, frac


def _strong_control_candidates(
    valid_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    if valid_df.empty:
        return pd.DataFrame()

    required = [
        "conditional_lift",
        "augmented_eps_test",
        "fraction_positive_lift",
        "bic_improvement",
    ]

    if any(c not in valid_df.columns for c in required):
        return pd.DataFrame()

    return valid_df[
        (pd.to_numeric(valid_df["conditional_lift"], errors="coerce").fillna(0.0) >= float(cfg.conditional_lift_min))
        & (pd.to_numeric(valid_df["augmented_eps_test"], errors="coerce").fillna(0.0) >= float(cfg.strong_eps_min))
        & (pd.to_numeric(valid_df["fraction_positive_lift"], errors="coerce").fillna(0.0) >= float(cfg.subject_effect_fraction))
        & (pd.to_numeric(valid_df["bic_improvement"], errors="coerce").fillna(0.0) >= 0.0)
    ].copy()


def _control_run_from_pair_scores(
    pair_scores_df: pd.DataFrame,
    control_type: str,
    replicate: int,
    window_seconds: int,
    lag_windows: int,
    lag_direction: str,
    alignment_scope: str,
    primary_alignment: bool,
    n_rows: int,
    n_subjects: int,
    n_sessions: int,
    cfg: Phase32EConfig,
) -> ControlRunResult:
    eps_saturation_value = float(cfg.eps_saturation_value)
    lag_seconds = int(window_seconds) * int(lag_windows)

    if pair_scores_df.empty:
        return ControlRunResult(
            control_type=control_type,
            replicate=int(replicate),
            window_seconds=int(window_seconds),
            lag_windows=int(lag_windows),
            lag_seconds=int(lag_seconds),
            lag_label=_lag_label(int(lag_windows)),
            lag_direction=str(lag_direction),
            alignment_scope=str(alignment_scope),
            valid=False,
            primary_alignment=bool(primary_alignment),
            n_rows=int(n_rows),
            n_subjects=int(n_subjects),
            n_sessions=int(n_sessions),
            max_baseline_eps_test=0.0,
            max_augmented_eps_test=0.0,
            max_conditional_lift=0.0,
            max_fraction_positive_lift=0.0,
            max_bic_improvement=0.0,
            max_ll_improvement=0.0,
            n_positive_lift_rows=0,
            n_strong_candidate_rows=0,
            n_multichannel_pair_rows=0,
            n_multichannel_positive_lift_rows=0,
            n_multichannel_strong_candidate_rows=0,
            eps_saturation_value=eps_saturation_value,
            n_eps_saturated_rows=0,
            eps_saturation_fraction=0.0,
            collapsed_by_lift=False,
            collapsed_by_bic=False,
            collapsed_by_subject_fraction=False,
            collapsed_by_strong_candidates=False,
            collapsed_overall=False,
            reason="empty_pair_scores",
        )

    valid_df = _valid_pair_scores(pair_scores_df)

    if valid_df.empty:
        return ControlRunResult(
            control_type=control_type,
            replicate=int(replicate),
            window_seconds=int(window_seconds),
            lag_windows=int(lag_windows),
            lag_seconds=int(lag_seconds),
            lag_label=_lag_label(int(lag_windows)),
            lag_direction=str(lag_direction),
            alignment_scope=str(alignment_scope),
            valid=False,
            primary_alignment=bool(primary_alignment),
            n_rows=int(n_rows),
            n_subjects=int(n_subjects),
            n_sessions=int(n_sessions),
            max_baseline_eps_test=0.0,
            max_augmented_eps_test=0.0,
            max_conditional_lift=0.0,
            max_fraction_positive_lift=0.0,
            max_bic_improvement=0.0,
            max_ll_improvement=0.0,
            n_positive_lift_rows=0,
            n_strong_candidate_rows=0,
            n_multichannel_pair_rows=0,
            n_multichannel_positive_lift_rows=0,
            n_multichannel_strong_candidate_rows=0,
            eps_saturation_value=eps_saturation_value,
            n_eps_saturated_rows=0,
            eps_saturation_fraction=0.0,
            collapsed_by_lift=False,
            collapsed_by_bic=False,
            collapsed_by_subject_fraction=False,
            collapsed_by_strong_candidates=False,
            collapsed_overall=False,
            reason="no_valid_pair_scores",
        )

    baseline_eps = pd.to_numeric(valid_df.get("baseline_eps_test", 0.0), errors="coerce").fillna(0.0)
    augmented_eps = pd.to_numeric(valid_df.get("augmented_eps_test", 0.0), errors="coerce").fillna(0.0)
    lift = pd.to_numeric(valid_df.get("conditional_lift", 0.0), errors="coerce").fillna(0.0)
    frac = pd.to_numeric(valid_df.get("fraction_positive_lift", 0.0), errors="coerce").fillna(0.0)
    bic = pd.to_numeric(valid_df.get("bic_improvement", 0.0), errors="coerce").fillna(0.0)
    ll = pd.to_numeric(valid_df.get("ll_improvement", 0.0), errors="coerce").fillna(0.0)

    positive_lift = valid_df[lift > 0.0]
    strong_candidates = _strong_control_candidates(valid_df, cfg)

    multichannel_mask = _is_multichannel_pair_rows(valid_df)
    multichannel_df = valid_df[multichannel_mask].copy()

    if not multichannel_df.empty:
        mc_lift = pd.to_numeric(multichannel_df.get("conditional_lift", 0.0), errors="coerce").fillna(0.0)
        mc_positive = multichannel_df[mc_lift > 0.0]
        mc_strong = _strong_control_candidates(multichannel_df, cfg)
    else:
        mc_positive = pd.DataFrame()
        mc_strong = pd.DataFrame()

    n_sat, sat_frac = _count_eps_saturation(valid_df, cfg)

    max_lift = float(lift.max()) if len(lift) else 0.0
    max_frac = float(frac.max()) if len(frac) else 0.0
    max_bic = float(bic.max()) if len(bic) else 0.0
    max_ll = float(ll.max()) if len(ll) else 0.0

    collapsed_by_lift = max_lift <= float(cfg.control_tol)
    collapsed_by_bic = max_bic <= 0.0
    collapsed_by_subject_fraction = max_frac < float(cfg.subject_effect_fraction)
    collapsed_by_strong_candidates = len(strong_candidates) == 0

    collapsed_overall = (
        collapsed_by_lift
        and collapsed_by_bic
        and collapsed_by_subject_fraction
        and collapsed_by_strong_candidates
    )

    reason = "ok" if collapsed_overall else "control_effect_survived"

    return ControlRunResult(
        control_type=control_type,
        replicate=int(replicate),
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_seconds=int(lag_seconds),
        lag_label=_lag_label(int(lag_windows)),
        lag_direction=str(lag_direction),
        alignment_scope=str(alignment_scope),
        valid=True,
        primary_alignment=bool(primary_alignment),
        n_rows=int(n_rows),
        n_subjects=int(n_subjects),
        n_sessions=int(n_sessions),
        max_baseline_eps_test=float(baseline_eps.max()) if len(baseline_eps) else 0.0,
        max_augmented_eps_test=float(augmented_eps.max()) if len(augmented_eps) else 0.0,
        max_conditional_lift=max_lift,
        max_fraction_positive_lift=max_frac,
        max_bic_improvement=max_bic,
        max_ll_improvement=max_ll,
        n_positive_lift_rows=int(len(positive_lift)),
        n_strong_candidate_rows=int(len(strong_candidates)),
        n_multichannel_pair_rows=int(len(multichannel_df)),
        n_multichannel_positive_lift_rows=int(len(mc_positive)),
        n_multichannel_strong_candidate_rows=int(len(mc_strong)),
        eps_saturation_value=eps_saturation_value,
        n_eps_saturated_rows=int(n_sat),
        eps_saturation_fraction=float(sat_frac),
        collapsed_by_lift=bool(collapsed_by_lift),
        collapsed_by_bic=bool(collapsed_by_bic),
        collapsed_by_subject_fraction=bool(collapsed_by_subject_fraction),
        collapsed_by_strong_candidates=bool(collapsed_by_strong_candidates),
        collapsed_overall=bool(collapsed_overall),
        reason=reason,
    )


def evaluate_control_on_frame(
    df: pd.DataFrame,
    control_type: str,
    replicate: int,
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[ControlRunResult, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    window_seconds = _safe_int(df["window_seconds"].iloc[0]) if "window_seconds" in df.columns and not df.empty else 0
    lag_windows = _safe_int(df["lag_windows"].iloc[0]) if "lag_windows" in df.columns and not df.empty else 0
    lag_direction = str(df["lag_direction"].iloc[0]) if "lag_direction" in df.columns and not df.empty else "unknown"
    alignment_scope = str(df["alignment_scope"].iloc[0]) if "alignment_scope" in df.columns and not df.empty else "unknown"

    primary_alignment = False

    if "primary_alignment" in df.columns and not df.empty:
        primary_alignment = bool(
            pd.to_numeric(df["primary_alignment"], errors="coerce").fillna(0).astype(int).max() == 1
        )

    seed = int(
        cfg.control_seed
        + int(replicate) * 1009
        + abs(int(lag_windows)) * 17
        + int(window_seconds) * 31
    )

    controlled = apply_control(
        df=df,
        control_type=control_type,
        cfg=cfg,
        seed=seed,
    )

    _, pair_scores = evaluate_conditional_models_for_frame(
        df=controlled,
        cfg=cfg,
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
    )

    pair_scores_df = pd.DataFrame(_as_records(pair_scores))

    n_sessions = 0

    if {"subject_id", "session_id"}.issubset(controlled.columns):
        n_sessions = int(controlled[["subject_id", "session_id"]].drop_duplicates().shape[0])

    result = _control_run_from_pair_scores(
        pair_scores_df=pair_scores_df,
        control_type=str(control_type),
        replicate=int(replicate),
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_direction=str(lag_direction),
        alignment_scope=str(alignment_scope),
        primary_alignment=bool(primary_alignment),
        n_rows=int(len(controlled)),
        n_subjects=int(controlled["subject_id"].nunique()) if "subject_id" in controlled.columns else 0,
        n_sessions=int(n_sessions),
        cfg=cfg,
    )

    if not pair_scores_df.empty:
        pair_scores_df["control_type"] = str(control_type)
        pair_scores_df["replicate"] = int(replicate)
        pair_scores_df["control_seed"] = int(seed)
        pair_scores_df["control_collapsed_overall"] = bool(result.collapsed_overall)

    return result, pair_scores_df


# =========================================================
# Control frame selection
# =========================================================

def available_control_frames(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[List[Tuple[int, int, Path, Dict[str, Any]]], Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    inventory_df, alignment_report = load_or_build_phase3_2e_alignment(
        cfg=cfg,
        force_rebuild=bool(force_rebuild_alignment),
    )

    frames: List[Tuple[int, int, Path, Dict[str, Any]]] = []

    alignment_result = alignment_report.get("result", {})
    can_continue = bool(alignment_result.get("can_proceed_to_conditional_cdr", False))

    if inventory_df.empty or not can_continue:
        return frames, alignment_report

    work = inventory_df.copy()

    for col in [
        "n_valid_conditional_rows",
        "n_valid_mc_conditional_rows",
        "n_valid_primary_alignment_rows",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    primary_ready = work[
        work.get("n_valid_primary_alignment_rows", pd.Series([0] * len(work))).astype(int) > 0
    ].copy()

    if not primary_ready.empty:
        selected = primary_ready
    else:
        selected = work[
            (
                work.get("n_valid_conditional_rows", pd.Series([0] * len(work))).astype(int) > 0
            )
            | (
                work.get("n_valid_mc_conditional_rows", pd.Series([0] * len(work))).astype(int) > 0
            )
        ].copy()

    for _, row in selected.iterrows():
        frame_file = Path(str(row.get("frame_file", "")))

        if not frame_file.exists():
            continue

        window_seconds = _safe_int(row.get("window_seconds"))
        lag_windows = _safe_int(row.get("lag_windows"))

        frames.append(
            (
                window_seconds,
                lag_windows,
                frame_file,
                row.to_dict(),
            )
        )

    return frames, alignment_report


# =========================================================
# Batch controls
# =========================================================

def evaluate_all_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    frames, alignment_report = available_control_frames(
        cfg=cfg,
        force_rebuild_alignment=bool(force_rebuild_alignment),
    )

    alignment_status = str(alignment_report.get("status", ""))
    alignment_result = alignment_report.get("result", {})

    if not frames:
        context = {
            "alignment_status": alignment_status,
            "alignment_result": alignment_result,
            "protocol_only": bool(alignment_status == cfg.status_pending_synchronized_data),
            "reason": "no_alignment_frames_available_for_controls",
            "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        }

        return empty_control_runs_frame(), empty_control_pair_scores_frame(), context

    run_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for window_seconds, lag_windows, path, frame_meta in frames:
        df = pd.read_csv(path)
        df = _sort_frame(df)

        for control_type in cfg.control_types:
            for replicate in range(int(cfg.n_controls)):
                print(
                    "[Phase3.2E Controls] "
                    f"control={control_type} replicate={replicate} "
                    f"window={window_seconds}s lag={lag_windows}"
                )

                try:
                    run_result, pair_scores_df = evaluate_control_on_frame(
                        df=df,
                        control_type=str(control_type),
                        replicate=int(replicate),
                        cfg=cfg,
                    )

                    run_rows.append(asdict(run_result))

                    if not pair_scores_df.empty:
                        pair_rows.extend(pair_scores_df.to_dict(orient="records"))

                except Exception as exc:
                    lag_direction = str(frame_meta.get("lag_direction", "unknown"))
                    alignment_scope = str(frame_meta.get("alignment_scope", "unknown"))
                    primary_alignment = bool(frame_meta.get("primary_alignment", False))
                    lag_seconds = int(window_seconds) * int(lag_windows)

                    n_sessions = 0
                    if {"subject_id", "session_id"}.issubset(df.columns):
                        n_sessions = int(df[["subject_id", "session_id"]].drop_duplicates().shape[0])

                    run_rows.append(
                        asdict(
                            ControlRunResult(
                                control_type=str(control_type),
                                replicate=int(replicate),
                                window_seconds=int(window_seconds),
                                lag_windows=int(lag_windows),
                                lag_seconds=int(lag_seconds),
                                lag_label=_lag_label(int(lag_windows)),
                                lag_direction=lag_direction,
                                alignment_scope=alignment_scope,
                                valid=False,
                                primary_alignment=primary_alignment,
                                n_rows=int(len(df)),
                                n_subjects=int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
                                n_sessions=int(n_sessions),
                                max_baseline_eps_test=0.0,
                                max_augmented_eps_test=0.0,
                                max_conditional_lift=0.0,
                                max_fraction_positive_lift=0.0,
                                max_bic_improvement=0.0,
                                max_ll_improvement=0.0,
                                n_positive_lift_rows=0,
                                n_strong_candidate_rows=0,
                                n_multichannel_pair_rows=0,
                                n_multichannel_positive_lift_rows=0,
                                n_multichannel_strong_candidate_rows=0,
                                eps_saturation_value=float(cfg.eps_saturation_value),
                                n_eps_saturated_rows=0,
                                eps_saturation_fraction=0.0,
                                collapsed_by_lift=False,
                                collapsed_by_bic=False,
                                collapsed_by_subject_fraction=False,
                                collapsed_by_strong_candidates=False,
                                collapsed_overall=False,
                                reason=str(exc),
                            )
                        )
                    )

    runs_df = pd.DataFrame(run_rows) if run_rows else empty_control_runs_frame()
    pairs_df = pd.DataFrame(pair_rows) if pair_rows else empty_control_pair_scores_frame()

    if not runs_df.empty:
        runs_df = runs_df.sort_values(
            ["control_type", "window_seconds", "lag_windows", "replicate"]
        ).reset_index(drop=True)

    if not pairs_df.empty and {"control_type", "replicate", "window_seconds", "lag_windows"}.issubset(pairs_df.columns):
        sort_cols = [
            c for c in [
                "control_type",
                "replicate",
                "window_seconds",
                "lag_windows",
                "family",
                "augmented_model",
            ]
            if c in pairs_df.columns
        ]

        pairs_df = pairs_df.sort_values(sort_cols).reset_index(drop=True)

    context = {
        "alignment_status": alignment_status,
        "alignment_result": alignment_result,
        "protocol_only": False,
        "reason": "controls_completed",
        "n_selected_frames": int(len(frames)),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
    }

    return runs_df, pairs_df, context


# =========================================================
# Summaries
# =========================================================

def summarize_control_runs(
    runs_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    if runs_df.empty:
        return {
            "available": False,
            "reason": "empty_control_runs",
            "passed": False,
            "n_control_types": 0,
            "control_summaries": [],
            "failed_control_types": list(cfg.required_control_types),
            "interpretation": (
                "No control runs are available. This is expected only in protocol-only "
                "mode or when no synchronized alignment frames can proceed to controls."
            ),
        }

    summaries: List[ControlSummary] = []

    for control_type, group in runs_df.groupby("control_type", sort=True):
        valid_group = group[_as_bool_series(group["valid"])].copy()

        if valid_group.empty:
            summaries.append(
                ControlSummary(
                    control_type=str(control_type),
                    n_runs=int(len(group)),
                    n_valid_runs=0,
                    median_max_baseline_eps_test=0.0,
                    median_max_augmented_eps_test=0.0,
                    mean_max_augmented_eps_test=0.0,
                    max_augmented_eps_test=0.0,
                    median_max_conditional_lift=0.0,
                    mean_max_conditional_lift=0.0,
                    max_conditional_lift=0.0,
                    median_max_bic_improvement=0.0,
                    max_bic_improvement=0.0,
                    median_max_ll_improvement=0.0,
                    max_ll_improvement=0.0,
                    median_eps_saturation_fraction=0.0,
                    max_eps_saturation_fraction=0.0,
                    total_multichannel_pair_rows=0,
                    total_multichannel_positive_lift_rows=0,
                    total_multichannel_strong_candidate_rows=0,
                    fraction_runs_below_control_tol=0.0,
                    fraction_runs_collapsed_overall=0.0,
                    control_tol=float(cfg.control_tol),
                    passed=False,
                    reason="no_valid_runs",
                )
            )
            continue

        baseline_eps = pd.to_numeric(valid_group["max_baseline_eps_test"], errors="coerce").fillna(0.0)
        augmented_eps = pd.to_numeric(valid_group["max_augmented_eps_test"], errors="coerce").fillna(0.0)
        max_lift = pd.to_numeric(valid_group["max_conditional_lift"], errors="coerce").fillna(0.0)
        max_bic = pd.to_numeric(valid_group["max_bic_improvement"], errors="coerce").fillna(0.0)
        max_ll = pd.to_numeric(valid_group["max_ll_improvement"], errors="coerce").fillna(0.0)
        max_frac = pd.to_numeric(valid_group["max_fraction_positive_lift"], errors="coerce").fillna(0.0)
        sat_frac = pd.to_numeric(valid_group["eps_saturation_fraction"], errors="coerce").fillna(0.0)

        collapsed_overall = _as_bool_series(valid_group["collapsed_overall"])

        below_tol = (
            (max_lift <= float(cfg.control_tol))
            & (max_bic <= 0.0)
            & (max_frac < float(cfg.subject_effect_fraction))
            & (pd.to_numeric(valid_group["n_strong_candidate_rows"], errors="coerce").fillna(0).astype(int) == 0)
        )

        fraction_below = float(np.mean(below_tol.to_numpy(dtype=bool)))
        fraction_collapsed = float(np.mean(collapsed_overall.to_numpy(dtype=bool)))

        passed = fraction_collapsed >= float(cfg.required_control_fraction)

        reason = "ok" if passed else (
            f"fraction_collapsed_overall<{cfg.required_control_fraction}"
        )

        summaries.append(
            ControlSummary(
                control_type=str(control_type),
                n_runs=int(len(group)),
                n_valid_runs=int(len(valid_group)),
                median_max_baseline_eps_test=float(baseline_eps.median()),
                median_max_augmented_eps_test=float(augmented_eps.median()),
                mean_max_augmented_eps_test=float(augmented_eps.mean()),
                max_augmented_eps_test=float(augmented_eps.max()),
                median_max_conditional_lift=float(max_lift.median()),
                mean_max_conditional_lift=float(max_lift.mean()),
                max_conditional_lift=float(max_lift.max()),
                median_max_bic_improvement=float(max_bic.median()),
                max_bic_improvement=float(max_bic.max()),
                median_max_ll_improvement=float(max_ll.median()),
                max_ll_improvement=float(max_ll.max()),
                median_eps_saturation_fraction=float(sat_frac.median()),
                max_eps_saturation_fraction=float(sat_frac.max()),
                total_multichannel_pair_rows=int(
                    pd.to_numeric(valid_group.get("n_multichannel_pair_rows", 0), errors="coerce").fillna(0).sum()
                ),
                total_multichannel_positive_lift_rows=int(
                    pd.to_numeric(valid_group.get("n_multichannel_positive_lift_rows", 0), errors="coerce").fillna(0).sum()
                ),
                total_multichannel_strong_candidate_rows=int(
                    pd.to_numeric(valid_group.get("n_multichannel_strong_candidate_rows", 0), errors="coerce").fillna(0).sum()
                ),
                fraction_runs_below_control_tol=fraction_below,
                fraction_runs_collapsed_overall=fraction_collapsed,
                control_tol=float(cfg.control_tol),
                passed=bool(passed),
                reason=reason,
            )
        )

    summary_by_type = {item.control_type: item for item in summaries}

    failed_control_types = [
        item.control_type
        for item in summaries
        if not item.passed
    ]

    for required_type in cfg.required_control_types:
        if required_type not in summary_by_type:
            failed_control_types.append(str(required_type))

    failed_control_types = sorted(set(failed_control_types))

    all_required_present = all(str(t) in summary_by_type for t in cfg.required_control_types)

    all_required_passed = (
        all(summary_by_type[str(t)].passed for t in cfg.required_control_types if str(t) in summary_by_type)
        and all_required_present
    )

    return {
        "available": True,
        "passed": bool(all_required_passed),
        "n_control_types": int(len(summaries)),
        "n_required_control_types": int(len(cfg.required_control_types)),
        "required_control_types": list(cfg.required_control_types),
        "all_required_present": bool(all_required_present),
        "control_summaries": [asdict(item) for item in summaries],
        "failed_control_types": failed_control_types,
        "interpretation": (
            "Controls pass only when required synchronized negative controls collapse "
            "conditional lift, BIC improvement, subject-level persistence and strong "
            "candidate survival. Epsilon saturation is tracked as a diagnostic, not "
            "as a direct failure criterion."
        ),
    }


def build_controls_result(
    status: str,
    protocol_only: bool,
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    summary: Mapping[str, Any],
    interpretation: str,
) -> ControlsBuildResult:
    valid_runs = 0

    if not runs_df.empty and "valid" in runs_df.columns:
        valid_runs = int(_as_bool_series(runs_df["valid"]).sum())

    return ControlsBuildResult(
        status=status,
        protocol_only=bool(protocol_only),
        n_control_runs=int(len(runs_df)),
        n_valid_control_runs=int(valid_runs),
        n_control_pair_rows=int(len(pairs_df)),
        n_control_types=int(summary.get("n_control_types", 0)),
        overall_passed=bool(summary.get("passed", False)),
        failed_control_types=list(summary.get("failed_control_types", [])),
        can_proceed_to_metrics=True,
        can_proceed_to_empirical_cdr=bool(summary.get("passed", False)),
        interpretation=interpretation,
    )


def build_controls_summary_text(
    summary: Mapping[str, Any],
    result: ControlsBuildResult,
) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Negative Controls Summary")
    lines.append("Synchronized EEG–QRNG protocol version")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append(f"overall_passed: {result.overall_passed}")
    lines.append(f"n_control_runs: {result.n_control_runs}")
    lines.append(f"n_valid_control_runs: {result.n_valid_control_runs}")
    lines.append(f"n_control_pair_rows: {result.n_control_pair_rows}")
    lines.append(f"n_control_types: {result.n_control_types}")
    lines.append(f"failed_control_types: {result.failed_control_types}")
    lines.append("")
    lines.append(f"can_proceed_to_metrics: {result.can_proceed_to_metrics}")
    lines.append(f"can_proceed_to_empirical_cdr: {result.can_proceed_to_empirical_cdr}")
    lines.append("")
    lines.append("interpretation:")
    lines.append(result.interpretation)
    lines.append("")

    if not summary.get("available"):
        lines.append(f"Unavailable: {summary.get('reason')}")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(f"required_control_types: {summary.get('required_control_types')}")
    lines.append(f"all_required_present: {summary.get('all_required_present')}")
    lines.append("")

    for item in summary.get("control_summaries", []):
        lines.append(f"Control: {item.get('control_type')}")
        lines.append("-" * 78)
        lines.append(f"Runs: {item.get('n_valid_runs')}/{item.get('n_runs')}")
        lines.append(f"Median max baseline eps: {item.get('median_max_baseline_eps_test')}")
        lines.append(f"Median max augmented eps: {item.get('median_max_augmented_eps_test')}")
        lines.append(f"Mean max augmented eps: {item.get('mean_max_augmented_eps_test')}")
        lines.append(f"Max augmented eps: {item.get('max_augmented_eps_test')}")
        lines.append(f"Median max lift: {item.get('median_max_conditional_lift')}")
        lines.append(f"Mean max lift: {item.get('mean_max_conditional_lift')}")
        lines.append(f"Max lift: {item.get('max_conditional_lift')}")
        lines.append(f"Median max BIC improvement: {item.get('median_max_bic_improvement')}")
        lines.append(f"Max BIC improvement: {item.get('max_bic_improvement')}")
        lines.append(f"Median max LL improvement: {item.get('median_max_ll_improvement')}")
        lines.append(f"Max LL improvement: {item.get('max_ll_improvement')}")
        lines.append(f"Median eps saturation fraction: {item.get('median_eps_saturation_fraction')}")
        lines.append(f"Max eps saturation fraction: {item.get('max_eps_saturation_fraction')}")
        lines.append(f"Total multichannel pair rows: {item.get('total_multichannel_pair_rows')}")
        lines.append(f"Total multichannel positive lift rows: {item.get('total_multichannel_positive_lift_rows')}")
        lines.append(f"Total multichannel strong candidate rows: {item.get('total_multichannel_strong_candidate_rows')}")
        lines.append(f"Fraction below control tol: {item.get('fraction_runs_below_control_tol')}")
        lines.append(f"Fraction collapsed overall: {item.get('fraction_runs_collapsed_overall')}")
        lines.append(f"Passed: {item.get('passed')}")
        lines.append(f"Reason: {item.get('reason')}")
        lines.append("")

    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Save outputs
# =========================================================

def save_control_outputs(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    context: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(control_runs_csv_path(cfg), index=False)
    pairs_df.to_csv(control_pair_scores_csv_path(cfg), index=False)

    summary = summarize_control_runs(runs_df, cfg)

    if context.get("protocol_only"):
        status = cfg.status_pending_synchronized_data
        protocol_only = True
        interpretation = (
            "No synchronized EEG–QRNG dataset is available. Controls completed "
            "in protocol-only mode with empty canonical outputs."
        )

    elif not summary.get("available"):
        status = "controls_unavailable"
        protocol_only = False
        interpretation = (
            "Controls could not be evaluated because no valid synchronized control "
            "runs were available."
        )

    elif summary.get("passed"):
        status = "controls_passed"
        protocol_only = False
        interpretation = (
            "Required synchronized negative controls passed. This permits metrics "
            "and gate evaluation, but does not by itself establish a positive claim."
        )

    else:
        status = "controls_failed"
        protocol_only = False
        interpretation = (
            "One or more required synchronized negative controls failed to collapse. "
            "Any positive conditional CDR interpretation is blocked."
        )

    result = build_controls_result(
        status=status,
        protocol_only=protocol_only,
        runs_df=runs_df,
        pairs_df=pairs_df,
        summary=summary,
        interpretation=interpretation,
    )

    payload = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": CONTROLS_MODULE,
        "controls_version": CONTROLS_VERSION,
        "created_at": _now_str(),
        "status": result.status,
        "result": asdict(result),
        "summary": summary,
        "context": dict(context),
        "runs_file": str(control_runs_csv_path(cfg)),
        "pair_scores_file": str(control_pair_scores_csv_path(cfg)),
        "controls_json": str(controls_json_path(cfg)),
        "controls_summary_txt": str(controls_summary_txt_path(cfg)),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "config": describe_config(cfg),
    }

    save_json(controls_json_path(cfg), payload)
    save_json(controls_report_json_path(cfg), payload)

    with open(controls_summary_txt_path(cfg), "w", encoding="utf-8") as f:
        f.write(build_controls_summary_text(summary, result))

    print(f"[Phase3.2E Controls] Saved runs: {control_runs_csv_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved pair scores: {control_pair_scores_csv_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved controls JSON: {controls_json_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved controls TXT: {controls_summary_txt_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved report JSON: {controls_report_json_path(cfg)}")

    return payload


# =========================================================
# Main API
# =========================================================

def build_phase3_2e_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    runs_df, pairs_df, context = evaluate_all_controls(
        cfg=cfg,
        force_rebuild_alignment=bool(force_rebuild_alignment),
    )

    report = save_control_outputs(
        runs_df=runs_df,
        pairs_df=pairs_df,
        context=context,
        cfg=cfg,
    )

    return runs_df, pairs_df, report


def load_or_build_phase3_2e_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    runs_path = control_runs_csv_path(cfg)
    pairs_path = control_pair_scores_csv_path(cfg)
    report_path = controls_report_json_path(cfg)

    if runs_path.exists() and pairs_path.exists() and report_path.exists() and not force_rebuild:
        print(f"[Phase3.2E Controls] Loading cached controls: {runs_path}")

        try:
            runs_df = pd.read_csv(runs_path)
            pairs_df = pd.read_csv(pairs_path)

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            if report.get("controls_version") == CONTROLS_VERSION:
                print("[Phase3.2E Controls] Cached controls compatible.")
                return runs_df, pairs_df, report

            print("[Phase3.2E Controls] Cached controls version changed. Rebuilding...")

        except Exception as exc:
            print(f"[Phase3.2E Controls] Failed to load cached controls: {exc}. Rebuilding...")

    return build_phase3_2e_controls(
        cfg=cfg,
        force_rebuild_alignment=bool(force_rebuild_alignment),
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III.2E synchronized negative controls."
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding control outputs.",
    )

    parser.add_argument(
        "--force-rebuild-alignment",
        action="store_true",
        help="Force rebuilding alignment before controls.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    runs_df, pairs_df, report = load_or_build_phase3_2e_controls(
        cfg=cfg,
        force_rebuild=bool(args.force_rebuild),
        force_rebuild_alignment=bool(args.force_rebuild_alignment),
    )

    result = report.get("result", {})
    summary = report.get("summary", {})

    print("\n" + "=" * 78)
    print("Phase III.2E controls completed")
    print("Synchronized EEG–QRNG negative controls")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"overall_passed: {result.get('overall_passed')}")
    print(f"n_control_runs: {result.get('n_control_runs')}")
    print(f"n_valid_control_runs: {result.get('n_valid_control_runs')}")
    print(f"n_control_pair_rows: {result.get('n_control_pair_rows')}")
    print(f"n_control_types: {result.get('n_control_types')}")
    print(f"failed_control_types: {result.get('failed_control_types')}")
    print(f"required_control_types: {summary.get('required_control_types')}")
    print(f"can_proceed_to_metrics: {result.get('can_proceed_to_metrics')}")
    print(f"can_proceed_to_empirical_cdr: {result.get('can_proceed_to_empirical_cdr')}")
    print(f"runs_csv: {control_runs_csv_path(cfg)}")
    print(f"pair_scores_csv: {control_pair_scores_csv_path(cfg)}")
    print(f"controls_json: {controls_json_path(cfg)}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()