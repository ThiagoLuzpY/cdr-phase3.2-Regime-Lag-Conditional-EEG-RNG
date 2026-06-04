from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    ConditionalModelSpec,
    Phase32EConfig,
    active_conditional_pairs,
    conditional_model_map,
    describe_config,
    load_phase3_2e_config,
)

from src.phase3_2e_alignment import (
    alignment_inventory_csv_path,
    lagged_frame_path,
    load_or_build_phase3_2e_alignment,
)


# =========================================================
# Phase III.2E — Conditional CDR
# Leakage-safe synchronized EEG–QRNG estimator
# =========================================================
#
# Purpose:
#
#   Evaluate whether synchronized QRNG_t adds conditional predictive structure
#   about EEG_{t+τ}, and whether synchronized EEG_t adds conditional predictive
#   structure about QRNG_{t+τ}.
#
# Phase III.2E models:
#
#   E0: P(EEG_{t+τ} | EEG_t)
#   E1: P(EEG_{t+τ} | EEG_t, QRNG_t)
#
#   E2: P(QRNG_{t+τ} | QRNG_t)
#   E3: P(QRNG_{t+τ} | QRNG_t, EEG_t)
#
#   E4: P(MC_EEG_{t+τ} | MC_EEG_t)
#   E5: P(MC_EEG_{t+τ} | MC_EEG_t, QRNG_t)
#
#   E6: P(QRNG_{t+τ} | QRNG_t, MC_EEG_t)
#
# Estimator:
#
#   reference_train → fits P0
#   calibration     → estimates Δχ and selects ε
#   test_holdout    → evaluates selected ε
#
# Critical rule:
#
#   eps_test is confirmed only if selected ε improves held-out test likelihood
#   over ε = 0. Otherwise eps_test is forced to 0.0.
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   empty canonical outputs are saved
#


# =========================================================
# Constants
# =========================================================

CONDITIONAL_MODULE = "phase3_2e_conditional"
CONDITIONAL_VERSION = "phase3_2e_conditional_v1_leakage_safe_sync_ref_calib_test"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class ConditionalScore:
    model: str
    label: str
    family: str

    window_seconds: int
    lag_windows: int
    lag_seconds: int
    lag_label: str
    lag_direction: str
    alignment_scope: str

    valid: bool
    primary: bool
    multichannel: bool

    n_rows_total: int
    n_rows_model: int
    n_ref: int
    n_calib: int
    n_test: int

    n_current_states: int
    n_target_states: int
    n_params: int

    eps_train: float
    eps_calib: float
    eps_candidate: float
    eps_test: float
    eps_saturated: bool

    ll_train: float

    ll_calib: float
    ll_calib_eps0: float
    ll_calib_gain_from_eps: float

    ll_test: float
    ll_test_eps0: float
    ll_test_gain_from_eps: float

    bic_test: float
    aic_test: float

    transitions_per_current_state: float
    reason: str = "ok"


@dataclass
class ConditionalPairScore:
    window_seconds: int
    lag_windows: int
    lag_seconds: int
    lag_label: str
    lag_direction: str
    alignment_scope: str

    family: str

    baseline_model: str
    augmented_model: str

    valid: bool
    primary: bool
    multichannel: bool

    baseline_eps_test: float
    augmented_eps_test: float
    conditional_lift: float

    baseline_ll_test: float
    augmented_ll_test: float
    ll_improvement: float

    baseline_bic_test: float
    augmented_bic_test: float
    bic_improvement: float

    baseline_aic_test: float
    augmented_aic_test: float
    aic_improvement: float

    n_subjects_evaluated: int
    n_subjects_positive_lift: int
    fraction_positive_lift: float

    reason: str = "ok"


@dataclass
class ConditionalBuildResult:
    status: str
    protocol_only: bool

    n_model_rows: int
    n_pair_rows: int
    n_valid_pair_rows: int
    n_primary_pair_rows: int
    n_positive_lift_rows: int
    n_primary_positive_lift_rows: int
    n_strong_candidate_rows: int
    n_multichannel_rows: int
    n_multichannel_positive_lift_rows: int

    max_baseline_eps_test: float
    max_augmented_eps_test: float
    max_conditional_lift: float

    can_proceed_to_controls: bool
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


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


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


# =========================================================
# Output paths
# =========================================================

def conditional_model_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_model_scores_csv)


def conditional_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_pair_scores_csv)


def conditional_summary_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_summary_json)


def conditional_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_summary_txt)


def conditional_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_conditional_report.json"


# =========================================================
# Model helpers
# =========================================================

def model_uses_multichannel(model: ConditionalModelSpec) -> bool:
    components = list(model.current_components) + [model.target_component]

    return bool(model.multichannel) or any(
        str(component).lower().startswith("mc_eeg")
        or str(component).lower().startswith("mceeg")
        or "multichannel" in str(component).lower()
        for component in components
    ) or model.family == "MCEEG_next"


def resolve_component_column(component: str, df: pd.DataFrame) -> str:
    direct = str(component)

    aliases: Dict[str, List[str]] = {
        "eeg_state": [
            "eeg_state",
        ],
        "eeg_next_state": [
            "eeg_next_state",
            "target_eeg_state",
        ],
        "qrng_state": [
            "qrng_state",
        ],
        "qrng_next_state": [
            "qrng_next_state",
            "target_qrng_state",
        ],
        "mc_eeg_state": [
            "mc_eeg_state",
        ],
        "mc_eeg_next_state": [
            "mc_eeg_next_state",
            "target_mc_eeg_state",
        ],
        "eeg_info_bin": [
            "eeg_info_bin",
        ],
        "qrng_info_bin": [
            "qrng_info_bin",
        ],
        "mc_eeg_info_bin": [
            "mc_eeg_info_bin",
        ],
        "observed_joint_state": [
            "observed_joint_state",
        ],
        "mc_observed_joint_state": [
            "mc_observed_joint_state",
        ],
        "informational_joint_state": [
            "informational_joint_state",
        ],
        "mc_informational_joint_state": [
            "mc_informational_joint_state",
        ],
    }

    candidates = aliases.get(direct, [direct])

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not resolve component column '{component}'. "
        f"Tried: {candidates}. Available columns include: {list(df.columns)[:80]}..."
    )


def model_validity_column(model: ConditionalModelSpec, df: pd.DataFrame) -> str:
    if model_uses_multichannel(model):
        for col in [
            "valid_mc_conditional_row",
            "valid_mc_next_state",
            "valid_mc_alignment_row",
        ]:
            if col in df.columns:
                return col

    for col in [
        "valid_conditional_row",
        "valid_alignment_row",
        "valid_next_state",
    ]:
        if col in df.columns:
            return col

    raise KeyError(f"No validity column found for model={model.name}, family={model.family}")


def _model_primary_for_frame(
    model: ConditionalModelSpec,
    cfg: Phase32EConfig,
    frame: pd.DataFrame,
) -> bool:
    if not bool(model.primary):
        return False

    if frame.empty:
        return False

    if "primary_alignment" in frame.columns:
        return bool(pd.to_numeric(frame["primary_alignment"], errors="coerce").fillna(0).astype(int).max() == 1)

    window_seconds = _safe_int(frame["window_seconds"].iloc[0]) if "window_seconds" in frame.columns else -1
    lag_windows = _safe_int(frame["lag_windows"].iloc[0]) if "lag_windows" in frame.columns else -999

    return (
        window_seconds in set(int(x) for x in cfg.primary_test_window_seconds)
        and lag_windows in set(int(x) for x in cfg.primary_test_lags_windows)
    )


# =========================================================
# Splits
# =========================================================

def make_subject_session_chronological_three_way_split(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    min_ref: int = 20,
    min_calib: int = 10,
    min_test: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df.empty:
        raise RuntimeError("Cannot split empty dataframe.")

    _require_columns(df, ["subject_id", "session_id"], context="Phase III.2E chronological split")

    out = _sort_frame(df.copy())

    ref_ratio = float(cfg.conditional_split_ref_ratio)
    calib_ratio = float(cfg.conditional_split_calib_ratio)

    if ref_ratio <= 0 or calib_ratio <= 0 or ref_ratio + calib_ratio >= 1:
        ref_ratio = 0.50
        calib_ratio = 0.20

    ref_indices: List[int] = []
    calib_indices: List[int] = []
    test_indices: List[int] = []

    group_cols = ["subject_id", "session_id"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        n = len(idx_arr)

        if n < min_ref + min_calib + min_test:
            continue

        n_ref = int(math.floor(n * ref_ratio))
        n_calib = int(math.floor(n * calib_ratio))

        n_ref = max(min_ref, n_ref)
        n_calib = max(min_calib, n_calib)

        if n_ref + n_calib > n - min_test:
            overflow = (n_ref + n_calib) - (n - min_test)

            reduce_calib = min(overflow, max(n_calib - min_calib, 0))
            n_calib -= reduce_calib
            overflow -= reduce_calib

            reduce_ref = min(overflow, max(n_ref - min_ref, 0))
            n_ref -= reduce_ref

        n_test = n - n_ref - n_calib

        if n_ref < min_ref or n_calib < min_calib or n_test < min_test:
            continue

        ref_indices.extend(idx_arr[:n_ref].tolist())
        calib_indices.extend(idx_arr[n_ref:n_ref + n_calib].tolist())
        test_indices.extend(idx_arr[n_ref + n_calib:].tolist())

    if len(ref_indices) < min_ref or len(calib_indices) < min_calib or len(test_indices) < min_test:
        raise RuntimeError(
            "Three-way split too small: "
            f"ref={len(ref_indices)}, calib={len(calib_indices)}, test={len(test_indices)}"
        )

    return (
        np.asarray(ref_indices, dtype=int),
        np.asarray(calib_indices, dtype=int),
        np.asarray(test_indices, dtype=int),
    )


# =========================================================
# Conditional state construction
# =========================================================

def _row_key_from_components(row: pd.Series, columns: Sequence[str]) -> Tuple[int, ...]:
    return tuple(int(row[col]) for col in columns)


def build_model_dataframe(
    df: pd.DataFrame,
    model: ConditionalModelSpec,
) -> Tuple[pd.DataFrame, List[str], str]:
    validity_col = model_validity_column(model, df)

    current_cols = [
        resolve_component_column(component, df)
        for component in model.current_components
    ]

    target_col = resolve_component_column(model.target_component, df)

    required = ["subject_id", "session_id", validity_col, target_col] + current_cols

    for optional_col in [
        "window_seconds",
        "lag_windows",
        "lag_seconds",
        "alignment_row_id",
        "window_start_utc",
        "window_id",
    ]:
        if optional_col in df.columns:
            required.append(optional_col)

    required = list(dict.fromkeys(required))

    _require_columns(df, required, context=f"conditional model {model.name}")

    work = df[required].copy()
    work = work[pd.to_numeric(work[validity_col], errors="coerce").fillna(0).astype(int) == 1].copy()

    for col in current_cols + [target_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=current_cols + [target_col]).copy()

    if work.empty:
        raise RuntimeError(f"No valid rows for model {model.name}")

    for col in current_cols:
        work[col] = work[col].astype(int)

    work[target_col] = work[target_col].astype(int)

    work["current_key"] = [
        _row_key_from_components(row, current_cols)
        for _, row in work.iterrows()
    ]

    work["target_value"] = work[target_col].astype(int)

    work = _sort_frame(work)

    return work, current_cols, target_col


# =========================================================
# Conditional CDR estimator
# =========================================================

def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return mat / row_sums


def _fit_transition_counts(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    alpha: float = 1e-6,
) -> Tuple[Dict[Tuple[int, ...], int], Dict[int, int], np.ndarray, np.ndarray]:
    unique_currents = sorted(set(current_keys))
    unique_targets = sorted(set(int(x) for x in targets))

    if not unique_currents:
        raise RuntimeError("No current states available.")

    if not unique_targets:
        raise RuntimeError("No target states available.")

    current_map = {key: i for i, key in enumerate(unique_currents)}
    target_map = {target: j for j, target in enumerate(unique_targets)}

    counts = np.zeros((len(current_map), len(target_map)), dtype=float)

    for key, target in zip(current_keys, targets):
        target = int(target)

        if key in current_map and target in target_map:
            counts[current_map[key], target_map[target]] += 1.0

    p0 = counts + float(alpha)
    p0 = _normalize_rows(p0)

    return current_map, target_map, counts, p0


def _global_target_distribution(
    targets: Sequence[int],
    target_map: Mapping[int, int],
    alpha: float = 1e-6,
) -> np.ndarray:
    counts = np.zeros(len(target_map), dtype=float)

    for target in targets:
        target = int(target)

        if target in target_map:
            counts[target_map[target]] += 1.0

    p = counts + float(alpha)
    total = float(p.sum())

    if total <= 0:
        return np.ones(len(target_map), dtype=float) / max(len(target_map), 1)

    return p / total


def _log_likelihood(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
    prob_matrix: np.ndarray,
    fallback_prob: np.ndarray,
) -> float:
    ll = 0.0
    tiny = 1e-12

    for key, target in zip(current_keys, targets):
        target = int(target)

        if target not in target_map:
            p = tiny
        else:
            target_idx = target_map[target]

            if key in current_map:
                p = float(prob_matrix[current_map[key], target_idx])
            else:
                p = float(fallback_prob[target_idx])

        ll += math.log(max(p, tiny))

    return float(ll)


def _counts_for_existing_maps(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
) -> np.ndarray:
    counts = np.zeros((len(current_map), len(target_map)), dtype=float)

    for key, target in zip(current_keys, targets):
        target = int(target)

        if key in current_map and target in target_map:
            counts[current_map[key], target_map[target]] += 1.0

    return counts


def _build_delta_from_calibration(
    p0: np.ndarray,
    calib_counts: np.ndarray,
    alpha: float = 1e-6,
    clip_abs: float = 5.0,
) -> np.ndarray:
    p_calib = calib_counts + float(alpha)
    p_calib = _normalize_rows(p_calib)

    observed_state_mask = calib_counts.sum(axis=1) > 0

    delta = np.zeros_like(p0, dtype=float)

    delta[observed_state_mask] = np.log(
        np.maximum(p_calib[observed_state_mask], alpha)
        / np.maximum(p0[observed_state_mask], alpha)
    )

    delta = np.clip(delta, -float(clip_abs), float(clip_abs))

    if not np.all(np.isfinite(delta)):
        delta = np.nan_to_num(delta, nan=0.0, posinf=float(clip_abs), neginf=-float(clip_abs))

    return delta


def _weighted_kernel(p0: np.ndarray, delta: np.ndarray, eps: float) -> np.ndarray:
    weighted = np.asarray(p0, dtype=float) * np.exp(float(eps) * np.asarray(delta, dtype=float))
    weighted = _normalize_rows(weighted)
    return weighted


def _select_eps_on_calibration(
    calib_keys: Sequence[Tuple[int, ...]],
    calib_targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
    p0: np.ndarray,
    fallback: np.ndarray,
    delta: np.ndarray,
    eps_grid: Sequence[float],
    min_gain: float = 1e-6,
) -> Tuple[float, float, float, float]:
    ll_eps0 = _log_likelihood(
        current_keys=calib_keys,
        targets=calib_targets,
        current_map=current_map,
        target_map=target_map,
        prob_matrix=p0,
        fallback_prob=fallback,
    )

    best_eps = 0.0
    best_ll = ll_eps0

    sorted_grid = sorted({float(eps) for eps in eps_grid})

    for eps in sorted_grid:
        weighted = _weighted_kernel(p0, delta, eps)

        ll = _log_likelihood(
            current_keys=calib_keys,
            targets=calib_targets,
            current_map=current_map,
            target_map=target_map,
            prob_matrix=weighted,
            fallback_prob=fallback,
        )

        if ll > best_ll + float(min_gain):
            best_ll = ll
            best_eps = eps

    gain = float(best_ll - ll_eps0)

    if gain <= float(min_gain):
        return 0.0, float(ll_eps0), float(ll_eps0), 0.0

    return float(best_eps), float(best_ll), float(ll_eps0), gain


def _evaluate_selected_eps_on_test(
    test_keys: Sequence[Tuple[int, ...]],
    test_targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
    p0: np.ndarray,
    fallback: np.ndarray,
    delta: np.ndarray,
    eps_candidate: float,
    min_test_gain: float = 1e-6,
) -> Tuple[float, float, float, float]:
    ll_eps0 = _log_likelihood(
        current_keys=test_keys,
        targets=test_targets,
        current_map=current_map,
        target_map=target_map,
        prob_matrix=p0,
        fallback_prob=fallback,
    )

    if float(eps_candidate) <= 0.0:
        return 0.0, float(ll_eps0), float(ll_eps0), 0.0

    weighted = _weighted_kernel(p0, delta, float(eps_candidate))

    ll_candidate = _log_likelihood(
        current_keys=test_keys,
        targets=test_targets,
        current_map=current_map,
        target_map=target_map,
        prob_matrix=weighted,
        fallback_prob=fallback,
    )

    gain = float(ll_candidate - ll_eps0)

    if gain <= float(min_test_gain):
        return 0.0, float(ll_eps0), float(ll_eps0), 0.0

    return float(eps_candidate), float(ll_candidate), float(ll_eps0), gain


def _estimate_eps_leakage_safe(
    ref_keys: Sequence[Tuple[int, ...]],
    ref_targets: Sequence[int],
    calib_keys: Sequence[Tuple[int, ...]],
    calib_targets: Sequence[int],
    test_keys: Sequence[Tuple[int, ...]],
    test_targets: Sequence[int],
    eps_grid: Sequence[float],
    eps_saturation_value: float,
    alpha: float = 1e-6,
    min_calib_gain: float = 1e-6,
    min_test_gain: float = 1e-6,
) -> Dict[str, Any]:
    current_map, target_map, ref_counts, p0 = _fit_transition_counts(
        current_keys=ref_keys,
        targets=ref_targets,
        alpha=alpha,
    )

    fallback = _global_target_distribution(
        targets=ref_targets,
        target_map=target_map,
        alpha=alpha,
    )

    calib_counts = _counts_for_existing_maps(
        current_keys=calib_keys,
        targets=calib_targets,
        current_map=current_map,
        target_map=target_map,
    )

    delta = _build_delta_from_calibration(
        p0=p0,
        calib_counts=calib_counts,
        alpha=alpha,
        clip_abs=5.0,
    )

    eps_candidate, ll_calib, ll_calib_eps0, ll_calib_gain = _select_eps_on_calibration(
        calib_keys=calib_keys,
        calib_targets=calib_targets,
        current_map=current_map,
        target_map=target_map,
        p0=p0,
        fallback=fallback,
        delta=delta,
        eps_grid=eps_grid,
        min_gain=min_calib_gain,
    )

    eps_test, ll_test, ll_test_eps0, ll_test_gain = _evaluate_selected_eps_on_test(
        test_keys=test_keys,
        test_targets=test_targets,
        current_map=current_map,
        target_map=target_map,
        p0=p0,
        fallback=fallback,
        delta=delta,
        eps_candidate=eps_candidate,
        min_test_gain=min_test_gain,
    )

    ll_train = _log_likelihood(
        current_keys=ref_keys,
        targets=ref_targets,
        current_map=current_map,
        target_map=target_map,
        prob_matrix=p0,
        fallback_prob=fallback,
    )

    eps_grid_max = max(float(x) for x in eps_grid) if eps_grid else 0.0
    saturation_threshold = min(float(eps_saturation_value), eps_grid_max)

    eps_saturated = bool(
        eps_candidate >= saturation_threshold
        and saturation_threshold > 0
    )

    return {
        "current_map": current_map,
        "target_map": target_map,
        "p0": p0,
        "fallback": fallback,
        "delta": delta,
        "ll_train": float(ll_train),
        "eps_calib": float(eps_candidate),
        "eps_candidate": float(eps_candidate),
        "eps_test": float(eps_test),
        "eps_saturated": bool(eps_saturated),
        "ll_calib": float(ll_calib),
        "ll_calib_eps0": float(ll_calib_eps0),
        "ll_calib_gain_from_eps": float(ll_calib_gain),
        "ll_test": float(ll_test),
        "ll_test_eps0": float(ll_test_eps0),
        "ll_test_gain_from_eps": float(ll_test_gain),
    }


# =========================================================
# Model evaluation
# =========================================================

def evaluate_conditional_model(
    df: pd.DataFrame,
    model: ConditionalModelSpec,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
) -> ConditionalScore:
    if cfg is None:
        cfg = load_phase3_2e_config()

    window_seconds = int(window_seconds)
    lag_windows = int(lag_windows)
    lag_seconds = int(window_seconds) * int(lag_windows)

    lag_direction = "concurrent"
    alignment_scope = "unknown"

    if not df.empty:
        if "lag_direction" in df.columns:
            lag_direction = str(df["lag_direction"].iloc[0])
        if "alignment_scope" in df.columns:
            alignment_scope = str(df["alignment_scope"].iloc[0])

    primary = _model_primary_for_frame(model, cfg, df)
    multichannel = model_uses_multichannel(model)

    try:
        model_df, current_cols, target_col = build_model_dataframe(df, model)
        model_df = model_df.reset_index(drop=True)

        min_ref = int(cfg.conditional_min_ref_rows)
        min_calib = int(cfg.conditional_min_calib_rows)
        min_test = int(cfg.conditional_min_test_rows)

        idx_ref_arr, idx_calib_arr, idx_test_arr = make_subject_session_chronological_three_way_split(
            model_df,
            cfg=cfg,
            min_ref=min_ref,
            min_calib=min_calib,
            min_test=min_test,
        )

        ref_df = model_df.iloc[idx_ref_arr].copy()
        calib_df = model_df.iloc[idx_calib_arr].copy()
        test_df = model_df.iloc[idx_test_arr].copy()

        if ref_df.empty or calib_df.empty or test_df.empty:
            raise RuntimeError(
                f"Empty ref/calib/test split for model={model.name}: "
                f"ref={len(ref_df)}, calib={len(calib_df)}, test={len(test_df)}"
            )

        estimate = _estimate_eps_leakage_safe(
            ref_keys=ref_df["current_key"].tolist(),
            ref_targets=ref_df["target_value"].astype(int).tolist(),
            calib_keys=calib_df["current_key"].tolist(),
            calib_targets=calib_df["target_value"].astype(int).tolist(),
            test_keys=test_df["current_key"].tolist(),
            test_targets=test_df["target_value"].astype(int).tolist(),
            eps_grid=cfg.eps_grid,
            eps_saturation_value=float(cfg.eps_saturation_value),
            alpha=1e-6,
            min_calib_gain=float(cfg.conditional_min_calib_ll_gain),
            min_test_gain=float(cfg.conditional_min_test_ll_gain),
        )

        current_map = estimate["current_map"]
        target_map = estimate["target_map"]

        n_current_states = int(len(current_map))
        n_target_states = int(len(target_map))
        n_params = int(n_current_states * max(n_target_states - 1, 1))

        n_test = int(len(test_df))
        ll_test = float(estimate["ll_test"])

        bic_test = float(-2.0 * ll_test + n_params * math.log(max(n_test, 2)))
        aic_test = float(-2.0 * ll_test + 2.0 * n_params)

        transitions_per_state = float(len(ref_df) / max(n_current_states, 1))

        return ConditionalScore(
            model=model.name,
            label=model.label,
            family=model.family,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
            lag_seconds=lag_seconds,
            lag_label=_lag_label(lag_windows),
            lag_direction=lag_direction,
            alignment_scope=alignment_scope,
            valid=True,
            primary=bool(primary),
            multichannel=bool(multichannel),
            n_rows_total=int(len(df)),
            n_rows_model=int(len(model_df)),
            n_ref=int(len(ref_df)),
            n_calib=int(len(calib_df)),
            n_test=n_test,
            n_current_states=n_current_states,
            n_target_states=n_target_states,
            n_params=n_params,
            eps_train=0.0,
            eps_calib=float(estimate["eps_calib"]),
            eps_candidate=float(estimate["eps_candidate"]),
            eps_test=float(estimate["eps_test"]),
            eps_saturated=bool(estimate["eps_saturated"]),
            ll_train=float(estimate["ll_train"]),
            ll_calib=float(estimate["ll_calib"]),
            ll_calib_eps0=float(estimate["ll_calib_eps0"]),
            ll_calib_gain_from_eps=float(estimate["ll_calib_gain_from_eps"]),
            ll_test=ll_test,
            ll_test_eps0=float(estimate["ll_test_eps0"]),
            ll_test_gain_from_eps=float(estimate["ll_test_gain_from_eps"]),
            bic_test=bic_test,
            aic_test=aic_test,
            transitions_per_current_state=transitions_per_state,
            reason="ok",
        )

    except Exception as exc:
        return ConditionalScore(
            model=model.name,
            label=model.label,
            family=model.family,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
            lag_seconds=lag_seconds,
            lag_label=_lag_label(lag_windows),
            lag_direction=lag_direction,
            alignment_scope=alignment_scope,
            valid=False,
            primary=bool(primary),
            multichannel=bool(multichannel),
            n_rows_total=int(len(df)),
            n_rows_model=0,
            n_ref=0,
            n_calib=0,
            n_test=0,
            n_current_states=0,
            n_target_states=0,
            n_params=0,
            eps_train=0.0,
            eps_calib=0.0,
            eps_candidate=0.0,
            eps_test=0.0,
            eps_saturated=False,
            ll_train=0.0,
            ll_calib=0.0,
            ll_calib_eps0=0.0,
            ll_calib_gain_from_eps=0.0,
            ll_test=0.0,
            ll_test_eps0=0.0,
            ll_test_gain_from_eps=0.0,
            bic_test=float("inf"),
            aic_test=float("inf"),
            transitions_per_current_state=0.0,
            reason=str(exc),
        )


# =========================================================
# Pair evaluation
# =========================================================

def evaluate_subject_fraction_positive_lift(
    df: pd.DataFrame,
    baseline_model: ConditionalModelSpec,
    augmented_model: ConditionalModelSpec,
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> Tuple[int, int, float]:
    if "subject_id" not in df.columns:
        return 0, 0, 0.0

    n_eval = 0
    n_positive = 0

    for _, group in df.groupby("subject_id", sort=True):
        g = _sort_frame(group.copy())

        if len(g) < (
            int(cfg.conditional_min_ref_rows)
            + int(cfg.conditional_min_calib_rows)
            + int(cfg.conditional_min_test_rows)
        ):
            continue

        baseline_score = evaluate_conditional_model(
            df=g,
            model=baseline_model,
            cfg=cfg,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
        )

        augmented_score = evaluate_conditional_model(
            df=g,
            model=augmented_model,
            cfg=cfg,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
        )

        if not baseline_score.valid or not augmented_score.valid:
            continue

        n_eval += 1

        lift = float(augmented_score.eps_test - baseline_score.eps_test)

        if lift > 0.0:
            n_positive += 1

    fraction = float(n_positive / n_eval) if n_eval > 0 else 0.0

    return n_eval, n_positive, fraction


def evaluate_conditional_pair(
    df: pd.DataFrame,
    baseline_model: ConditionalModelSpec,
    augmented_model: ConditionalModelSpec,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
) -> Tuple[ConditionalScore, ConditionalScore, ConditionalPairScore]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    window_seconds = int(window_seconds)
    lag_windows = int(lag_windows)
    lag_seconds = int(window_seconds) * int(lag_windows)

    lag_direction = str(df["lag_direction"].iloc[0]) if not df.empty and "lag_direction" in df.columns else "unknown"
    alignment_scope = str(df["alignment_scope"].iloc[0]) if not df.empty and "alignment_scope" in df.columns else "unknown"

    baseline_score = evaluate_conditional_model(
        df=df,
        model=baseline_model,
        cfg=cfg,
        window_seconds=window_seconds,
        lag_windows=lag_windows,
    )

    augmented_score = evaluate_conditional_model(
        df=df,
        model=augmented_model,
        cfg=cfg,
        window_seconds=window_seconds,
        lag_windows=lag_windows,
    )

    valid = baseline_score.valid and augmented_score.valid

    n_eval, n_pos, frac_pos = evaluate_subject_fraction_positive_lift(
        df=df,
        baseline_model=baseline_model,
        augmented_model=augmented_model,
        cfg=cfg,
        window_seconds=window_seconds,
        lag_windows=lag_windows,
    )

    conditional_lift = float(augmented_score.eps_test - baseline_score.eps_test)
    ll_improvement = float(augmented_score.ll_test - baseline_score.ll_test)

    bic_improvement = float(baseline_score.bic_test - augmented_score.bic_test)
    aic_improvement = float(baseline_score.aic_test - augmented_score.aic_test)

    pair_tuple = (baseline_model.name, augmented_model.name)

    primary = (
        bool(baseline_model.primary)
        and bool(augmented_model.primary)
        and pair_tuple in set(tuple(x) for x in cfg.primary_conditional_pairs)
        and window_seconds in set(int(x) for x in cfg.primary_test_window_seconds)
        and lag_windows in set(int(x) for x in cfg.primary_test_lags_windows)
    )

    multichannel = model_uses_multichannel(baseline_model) or model_uses_multichannel(augmented_model)

    reason = "ok"

    if not valid:
        reason = f"baseline_valid={baseline_score.valid}; augmented_valid={augmented_score.valid}"

    pair = ConditionalPairScore(
        window_seconds=window_seconds,
        lag_windows=lag_windows,
        lag_seconds=lag_seconds,
        lag_label=_lag_label(lag_windows),
        lag_direction=lag_direction,
        alignment_scope=alignment_scope,
        family=augmented_model.family,
        baseline_model=baseline_model.name,
        augmented_model=augmented_model.name,
        valid=valid,
        primary=bool(primary),
        multichannel=bool(multichannel),
        baseline_eps_test=float(baseline_score.eps_test),
        augmented_eps_test=float(augmented_score.eps_test),
        conditional_lift=conditional_lift,
        baseline_ll_test=float(baseline_score.ll_test),
        augmented_ll_test=float(augmented_score.ll_test),
        ll_improvement=ll_improvement,
        baseline_bic_test=float(baseline_score.bic_test),
        augmented_bic_test=float(augmented_score.bic_test),
        bic_improvement=bic_improvement,
        baseline_aic_test=float(baseline_score.aic_test),
        augmented_aic_test=float(augmented_score.aic_test),
        aic_improvement=aic_improvement,
        n_subjects_evaluated=int(n_eval),
        n_subjects_positive_lift=int(n_pos),
        fraction_positive_lift=float(frac_pos),
        reason=reason,
    )

    return baseline_score, augmented_score, pair


def evaluate_conditional_models_for_frame(
    df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
) -> Tuple[List[ConditionalScore], List[ConditionalPairScore]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    model_map = conditional_model_map(cfg)

    scores_by_name: Dict[str, ConditionalScore] = {}
    pair_scores: List[ConditionalPairScore] = []

    for baseline_name, augmented_name in active_conditional_pairs(cfg):
        baseline_model = model_map[baseline_name]
        augmented_model = model_map[augmented_name]

        baseline_score, augmented_score, pair_score = evaluate_conditional_pair(
            df=df,
            baseline_model=baseline_model,
            augmented_model=augmented_model,
            cfg=cfg,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
        )

        scores_by_name[baseline_score.model] = baseline_score
        scores_by_name[augmented_score.model] = augmented_score
        pair_scores.append(pair_score)

    model_scores = list(scores_by_name.values())

    return model_scores, pair_scores


# =========================================================
# Batch evaluation
# =========================================================

def evaluate_all_conditional_from_alignment_frames(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    inventory_df, alignment_report = load_or_build_phase3_2e_alignment(
        cfg=cfg,
        force_rebuild=bool(force_rebuild_alignment),
    )

    alignment_status = str(alignment_report.get("status", ""))
    alignment_result = alignment_report.get("result", {})

    model_score_rows: List[Dict[str, Any]] = []
    pair_score_rows: List[Dict[str, Any]] = []

    if (
        inventory_df.empty
        or alignment_status == cfg.status_pending_synchronized_data
        or not bool(alignment_result.get("can_proceed_to_conditional_cdr", False))
    ):
        context = {
            "alignment_status": alignment_status,
            "alignment_result": alignment_result,
            "protocol_only": bool(alignment_status == cfg.status_pending_synchronized_data),
            "reason": "alignment_not_ready_for_conditional_cdr",
        }

        return pd.DataFrame(), pd.DataFrame(), context

    for _, inv_row in inventory_df.iterrows():
        window_seconds = _safe_int(inv_row.get("window_seconds"))
        lag_windows = _safe_int(inv_row.get("lag_windows"))

        frame_file = Path(str(inv_row.get("frame_file", "")))

        if not frame_file.exists():
            continue

        n_valid_conditional = _safe_int(inv_row.get("n_valid_conditional_rows"), 0)
        n_valid_mc = _safe_int(inv_row.get("n_valid_mc_conditional_rows"), 0)

        if n_valid_conditional <= 0 and n_valid_mc <= 0:
            continue

        print(
            "[Phase3.2E Conditional] "
            f"Evaluating window={window_seconds}s lag={lag_windows} file={frame_file.name}"
        )

        frame = pd.read_csv(frame_file)
        frame = _sort_frame(frame)

        model_scores, pair_scores = evaluate_conditional_models_for_frame(
            df=frame,
            cfg=cfg,
            window_seconds=window_seconds,
            lag_windows=lag_windows,
        )

        model_score_rows.extend(_as_records(model_scores))
        pair_score_rows.extend(_as_records(pair_scores))

    model_scores_df = pd.DataFrame(model_score_rows)
    pair_scores_df = pd.DataFrame(pair_score_rows)

    if not model_scores_df.empty:
        model_scores_df = model_scores_df.sort_values(
            ["window_seconds", "lag_windows", "family", "model"]
        ).reset_index(drop=True)

    if not pair_scores_df.empty:
        pair_scores_df = pair_scores_df.sort_values(
            ["window_seconds", "lag_windows", "family", "augmented_model"]
        ).reset_index(drop=True)

    context = {
        "alignment_status": alignment_status,
        "alignment_result": alignment_result,
        "protocol_only": False,
        "reason": "conditional_evaluation_completed",
    }

    return model_scores_df, pair_scores_df, context


# =========================================================
# Summaries
# =========================================================

def summarize_conditional_pairs(
    pair_scores_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    if pair_scores_df.empty:
        return {
            "available": False,
            "reason": "empty_pair_scores",
            "n_rows": 0,
            "n_valid_rows": 0,
            "n_primary_rows": 0,
            "n_positive_lift_rows": 0,
            "n_primary_positive_lift_rows": 0,
            "n_strong_candidate_rows": 0,
            "n_multichannel_rows": 0,
            "n_multichannel_positive_lift_rows": 0,
            "max_baseline_eps_test": 0.0,
            "max_augmented_eps_test": 0.0,
            "max_conditional_lift": 0.0,
        }

    if "valid" not in pair_scores_df.columns:
        return {
            "available": False,
            "reason": "missing_valid_column",
            "n_rows": int(len(pair_scores_df)),
        }

    valid = pair_scores_df[_as_bool_series(pair_scores_df["valid"])].copy()

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_pair_scores",
            "n_rows": int(len(pair_scores_df)),
            "n_valid_rows": 0,
        }

    primary = (
        valid[_as_bool_series(valid["primary"])].copy()
        if "primary" in valid.columns
        else pd.DataFrame()
    )

    positive_lift = valid[
        pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0) > 0.0
    ].copy()

    primary_positive_lift = (
        primary[pd.to_numeric(primary["conditional_lift"], errors="coerce").fillna(0.0) > 0.0].copy()
        if not primary.empty
        else pd.DataFrame()
    )

    multichannel = (
        valid[_as_bool_series(valid["multichannel"])].copy()
        if "multichannel" in valid.columns
        else pd.DataFrame()
    )

    multichannel_positive_lift = (
        multichannel[
            pd.to_numeric(multichannel["conditional_lift"], errors="coerce").fillna(0.0) > 0.0
        ].copy()
        if not multichannel.empty
        else pd.DataFrame()
    )

    strong_candidates = valid[
        (pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0) >= float(cfg.conditional_lift_min))
        & (pd.to_numeric(valid["augmented_eps_test"], errors="coerce").fillna(0.0) >= float(cfg.strong_eps_min))
        & (pd.to_numeric(valid["fraction_positive_lift"], errors="coerce").fillna(0.0) >= float(cfg.subject_effect_fraction))
        & (pd.to_numeric(valid["bic_improvement"], errors="coerce").fillna(0.0) >= 0.0)
    ].copy()

    best_lift = valid.sort_values("conditional_lift", ascending=False).iloc[0]
    best_bic = valid.sort_values("bic_improvement", ascending=False).iloc[0]
    best_ll = valid.sort_values("ll_improvement", ascending=False).iloc[0]

    if not multichannel.empty:
        best_multichannel_lift = multichannel.sort_values("conditional_lift", ascending=False).iloc[0]
        best_multichannel_bic = multichannel.sort_values("bic_improvement", ascending=False).iloc[0]
    else:
        best_multichannel_lift = None
        best_multichannel_bic = None

    def _row_payload(row: Optional[pd.Series]) -> Dict[str, Any]:
        if row is None:
            return {}

        return {
            "window_seconds": _safe_int(row.get("window_seconds")),
            "lag_windows": _safe_int(row.get("lag_windows")),
            "lag_seconds": _safe_int(row.get("lag_seconds")),
            "lag_direction": str(row.get("lag_direction")),
            "alignment_scope": str(row.get("alignment_scope")),
            "family": str(row.get("family")),
            "baseline_model": str(row.get("baseline_model")),
            "augmented_model": str(row.get("augmented_model")),
            "conditional_lift": _safe_float(row.get("conditional_lift")),
            "baseline_eps_test": _safe_float(row.get("baseline_eps_test")),
            "augmented_eps_test": _safe_float(row.get("augmented_eps_test")),
            "fraction_positive_lift": _safe_float(row.get("fraction_positive_lift")),
            "ll_improvement": _safe_float(row.get("ll_improvement")),
            "bic_improvement": _safe_float(row.get("bic_improvement")),
            "aic_improvement": _safe_float(row.get("aic_improvement")),
            "primary": bool(row.get("primary")),
            "multichannel": bool(row.get("multichannel")),
        }

    max_baseline_eps = float(pd.to_numeric(valid["baseline_eps_test"], errors="coerce").fillna(0.0).max())
    max_augmented_eps = float(pd.to_numeric(valid["augmented_eps_test"], errors="coerce").fillna(0.0).max())
    max_conditional_lift = float(pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0).max())

    summary: Dict[str, Any] = {
        "available": True,
        "module": CONDITIONAL_MODULE,
        "conditional_version": CONDITIONAL_VERSION,
        "n_rows": int(len(pair_scores_df)),
        "n_valid_rows": int(len(valid)),
        "n_primary_rows": int(len(primary)),
        "n_positive_lift_rows": int(len(positive_lift)),
        "n_primary_positive_lift_rows": int(len(primary_positive_lift)),
        "n_strong_candidate_rows": int(len(strong_candidates)),
        "n_multichannel_rows": int(len(multichannel)),
        "n_multichannel_positive_lift_rows": int(len(multichannel_positive_lift)),
        "max_baseline_eps_test": max_baseline_eps,
        "max_augmented_eps_test": max_augmented_eps,
        "max_conditional_lift": max_conditional_lift,
        "best_by_conditional_lift": _row_payload(best_lift),
        "best_by_bic_improvement": _row_payload(best_bic),
        "best_by_ll_improvement": _row_payload(best_ll),
        "best_multichannel_by_conditional_lift": _row_payload(best_multichannel_lift),
        "best_multichannel_by_bic_improvement": _row_payload(best_multichannel_bic),
        "positive_lift_rows": positive_lift.to_dict(orient="records"),
        "multichannel_positive_lift_rows": multichannel_positive_lift.to_dict(orient="records"),
        "strong_candidate_rows": strong_candidates.to_dict(orient="records"),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "primary_conditional_pairs": list(cfg.primary_conditional_pairs),
        "multichannel_conditional_pairs": list(cfg.multichannel_conditional_pairs),
    }

    return summary


def build_conditional_result(
    status: str,
    protocol_only: bool,
    summary: Mapping[str, Any],
    interpretation: str,
) -> ConditionalBuildResult:
    return ConditionalBuildResult(
        status=status,
        protocol_only=bool(protocol_only),
        n_model_rows=int(summary.get("n_model_rows", 0)),
        n_pair_rows=int(summary.get("n_rows", 0)),
        n_valid_pair_rows=int(summary.get("n_valid_rows", 0)),
        n_primary_pair_rows=int(summary.get("n_primary_rows", 0)),
        n_positive_lift_rows=int(summary.get("n_positive_lift_rows", 0)),
        n_primary_positive_lift_rows=int(summary.get("n_primary_positive_lift_rows", 0)),
        n_strong_candidate_rows=int(summary.get("n_strong_candidate_rows", 0)),
        n_multichannel_rows=int(summary.get("n_multichannel_rows", 0)),
        n_multichannel_positive_lift_rows=int(summary.get("n_multichannel_positive_lift_rows", 0)),
        max_baseline_eps_test=float(summary.get("max_baseline_eps_test", 0.0)),
        max_augmented_eps_test=float(summary.get("max_augmented_eps_test", 0.0)),
        max_conditional_lift=float(summary.get("max_conditional_lift", 0.0)),
        can_proceed_to_controls=int(summary.get("n_valid_rows", 0)) > 0,
        can_proceed_to_metrics=True,
        can_proceed_to_empirical_cdr=int(summary.get("n_primary_rows", 0)) > 0,
        interpretation=interpretation,
    )


def build_conditional_summary_text(
    summary: Mapping[str, Any],
    result: ConditionalBuildResult,
) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Conditional CDR summary")
    lines.append("Leakage-safe synchronized ref/calib/test estimator")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append("")

    if not summary.get("available"):
        lines.append(f"Summary unavailable: {summary.get('reason')}")
        lines.append("")
        lines.append("interpretation:")
        lines.append(result.interpretation)
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(f"Model rows: {summary.get('n_model_rows')}")
    lines.append(f"Valid pair rows: {summary.get('n_valid_rows')}/{summary.get('n_rows')}")
    lines.append(f"Primary rows: {summary.get('n_primary_rows')}")
    lines.append(f"Positive lift rows: {summary.get('n_positive_lift_rows')}")
    lines.append(f"Primary positive lift rows: {summary.get('n_primary_positive_lift_rows')}")
    lines.append(f"Strong candidate rows: {summary.get('n_strong_candidate_rows')}")
    lines.append(f"Multichannel rows: {summary.get('n_multichannel_rows')}")
    lines.append(f"Multichannel positive lift rows: {summary.get('n_multichannel_positive_lift_rows')}")
    lines.append(f"Max baseline eps_test: {summary.get('max_baseline_eps_test')}")
    lines.append(f"Max augmented eps_test: {summary.get('max_augmented_eps_test')}")
    lines.append(f"Max conditional lift: {summary.get('max_conditional_lift')}")
    lines.append("")

    lines.append("Best by conditional lift:")
    lines.append(str(summary.get("best_by_conditional_lift")))
    lines.append("")

    lines.append("Best by BIC improvement:")
    lines.append(str(summary.get("best_by_bic_improvement")))
    lines.append("")

    lines.append("Best by LL improvement:")
    lines.append(str(summary.get("best_by_ll_improvement")))
    lines.append("")

    lines.append("Best multichannel by conditional lift:")
    lines.append(str(summary.get("best_multichannel_by_conditional_lift")))
    lines.append("")

    lines.append("Best multichannel by BIC improvement:")
    lines.append(str(summary.get("best_multichannel_by_bic_improvement")))
    lines.append("")

    lines.append("interpretation:")
    lines.append(result.interpretation)
    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Save outputs
# =========================================================

def save_conditional_outputs(
    model_scores_df: pd.DataFrame,
    pair_scores_df: pd.DataFrame,
    context: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    model_scores_df.to_csv(conditional_model_scores_csv_path(cfg), index=False)
    pair_scores_df.to_csv(conditional_pair_scores_csv_path(cfg), index=False)

    summary = summarize_conditional_pairs(pair_scores_df, cfg)
    summary["n_model_rows"] = int(len(model_scores_df))

    if context.get("protocol_only"):
        status = cfg.status_pending_synchronized_data
        protocol_only = True
        interpretation = (
            "No synchronized EEG–QRNG dataset is available. Conditional CDR "
            "completed in protocol-only mode with empty canonical outputs."
        )

    elif not summary.get("available"):
        status = "conditional_no_valid_pairs"
        protocol_only = False
        interpretation = (
            "Conditional CDR ran but no valid synchronized conditional pair scores "
            "were available. Empirical interpretation remains blocked."
        )

    elif int(summary.get("n_strong_candidate_rows", 0)) > 0:
        status = "conditional_candidate_rows_detected"
        protocol_only = False
        interpretation = (
            "Conditional CDR produced strong candidate rows before controls and "
            "final gates. This is not a claim until controls, metrics and gate "
            "sensitivity diagnostics pass."
        )

    elif int(summary.get("n_positive_lift_rows", 0)) > 0:
        status = "conditional_exploratory_leads"
        protocol_only = False
        interpretation = (
            "Conditional CDR produced exploratory positive-lift rows. These require "
            "controls, F7/F8/F12 diagnostics and final gates before interpretation."
        )

    else:
        status = "conditional_synchronized_null_at_current_stage"
        protocol_only = False
        interpretation = (
            "Conditional CDR found no positive synchronized conditional lift at the "
            "current stage."
        )

    result = build_conditional_result(
        status=status,
        protocol_only=protocol_only,
        summary=summary,
        interpretation=interpretation,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": CONDITIONAL_MODULE,
        "conditional_version": CONDITIONAL_VERSION,
        "created_at": _now_str(),
        "status": result.status,
        "result": asdict(result),
        "summary": summary,
        "context": dict(context),
        "model_scores_file": str(conditional_model_scores_csv_path(cfg)),
        "pair_scores_file": str(conditional_pair_scores_csv_path(cfg)),
        "summary_json_file": str(conditional_summary_json_path(cfg)),
        "summary_txt_file": str(conditional_summary_txt_path(cfg)),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "config": describe_config(cfg),
    }

    save_json(
        conditional_summary_json_path(cfg),
        {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": CONDITIONAL_MODULE,
            "conditional_version": CONDITIONAL_VERSION,
            "summary": summary,
            "result": asdict(result),
            "model_scores_file": str(conditional_model_scores_csv_path(cfg)),
            "pair_scores_file": str(conditional_pair_scores_csv_path(cfg)),
        },
    )

    save_json(conditional_report_json_path(cfg), report)

    with open(conditional_summary_txt_path(cfg), "w", encoding="utf-8") as f:
        f.write(build_conditional_summary_text(summary, result))

    print(f"[Phase3.2E Conditional] Saved model scores: {conditional_model_scores_csv_path(cfg)}")
    print(f"[Phase3.2E Conditional] Saved pair scores: {conditional_pair_scores_csv_path(cfg)}")
    print(f"[Phase3.2E Conditional] Saved summary JSON: {conditional_summary_json_path(cfg)}")
    print(f"[Phase3.2E Conditional] Saved summary TXT: {conditional_summary_txt_path(cfg)}")
    print(f"[Phase3.2E Conditional] Saved report JSON: {conditional_report_json_path(cfg)}")

    return report


# =========================================================
# Main API
# =========================================================

def build_phase3_2e_conditional(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    model_scores_df, pair_scores_df, context = evaluate_all_conditional_from_alignment_frames(
        cfg=cfg,
        force_rebuild_alignment=bool(force_rebuild_alignment),
    )

    report = save_conditional_outputs(
        model_scores_df=model_scores_df,
        pair_scores_df=pair_scores_df,
        context=context,
        cfg=cfg,
    )

    return model_scores_df, pair_scores_df, report


def load_or_build_phase3_2e_conditional(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    model_path = conditional_model_scores_csv_path(cfg)
    pair_path = conditional_pair_scores_csv_path(cfg)
    report_path = conditional_report_json_path(cfg)

    if model_path.exists() and pair_path.exists() and report_path.exists() and not force_rebuild:
        print(f"[Phase3.2E Conditional] Loading cached outputs: {pair_path}")

        try:
            model_scores_df = pd.read_csv(model_path)
            pair_scores_df = pd.read_csv(pair_path)

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)

            if report.get("conditional_version") == CONDITIONAL_VERSION:
                print("[Phase3.2E Conditional] Cached conditional outputs compatible.")
                return model_scores_df, pair_scores_df, report

            print("[Phase3.2E Conditional] Cached version changed. Rebuilding...")

        except Exception as exc:
            print(f"[Phase3.2E Conditional] Failed to load cache: {exc}. Rebuilding...")

    return build_phase3_2e_conditional(
        cfg=cfg,
        force_rebuild_alignment=bool(force_rebuild_alignment),
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III.2E leakage-safe synchronized conditional CDR."
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding conditional outputs.",
    )

    parser.add_argument(
        "--force-rebuild-alignment",
        action="store_true",
        help="Force rebuilding alignment before conditional CDR.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    model_scores_df, pair_scores_df, report = load_or_build_phase3_2e_conditional(
        cfg=cfg,
        force_rebuild=bool(args.force_rebuild),
        force_rebuild_alignment=bool(args.force_rebuild_alignment),
    )

    result = report.get("result", {})
    summary = report.get("summary", {})

    print("\n" + "=" * 78)
    print("Phase III.2E conditional CDR completed")
    print("Leakage-safe synchronized ref/calib/test estimator")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"model_rows: {len(model_scores_df)}")
    print(f"pair_rows: {len(pair_scores_df)}")
    print(f"valid_pair_rows: {summary.get('n_valid_rows')}")
    print(f"primary_pair_rows: {summary.get('n_primary_rows')}")
    print(f"positive_lift_rows: {summary.get('n_positive_lift_rows')}")
    print(f"primary_positive_lift_rows: {summary.get('n_primary_positive_lift_rows')}")
    print(f"strong_candidate_rows: {summary.get('n_strong_candidate_rows')}")
    print(f"multichannel_rows: {summary.get('n_multichannel_rows')}")
    print(f"multichannel_positive_lift_rows: {summary.get('n_multichannel_positive_lift_rows')}")
    print(f"max_baseline_eps_test: {summary.get('max_baseline_eps_test')}")
    print(f"max_augmented_eps_test: {summary.get('max_augmented_eps_test')}")
    print(f"max_conditional_lift: {summary.get('max_conditional_lift')}")
    print(f"can_proceed_to_controls: {result.get('can_proceed_to_controls')}")
    print(f"can_proceed_to_metrics: {result.get('can_proceed_to_metrics')}")
    print(f"can_proceed_to_empirical_cdr: {result.get('can_proceed_to_empirical_cdr')}")
    print(f"model_scores_csv: {conditional_model_scores_csv_path(cfg)}")
    print(f"pair_scores_csv: {conditional_pair_scores_csv_path(cfg)}")
    print(f"summary_json: {conditional_summary_json_path(cfg)}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()