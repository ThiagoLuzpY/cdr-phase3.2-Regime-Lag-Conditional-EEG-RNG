from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import (
    ConditionalModelSpec,
    Phase32Config,
    active_conditional_pairs,
    conditional_model_map,
    load_phase3_2_config,
)

from src.phase3_2_lagging import lag_label, lagged_csv_path
from src.phase3_2_regimes import save_json


# =========================================================
# Phase III.2C/D — Conditional CDR
# Leakage-safe ref/calib/test estimator
# =========================================================
#
# This module tests whether one domain improves conditional transition
# modeling of the other domain:
#
#   III.2C — single-channel EEG/RNG:
#
#       C0: P(EEG_{t+1} | EEG_t)
#       C1: P(EEG_{t+1} | EEG_t, RNG_t)
#
#       C2: P(RNG_{t+1} | RNG_t)
#       C3: P(RNG_{t+1} | RNG_t, EEG_t)
#
#   III.2D — multichannel EEG enrichment:
#
#       D0: P(MC-EEG_{t+1} | MC-EEG_t)
#       D1: P(MC-EEG_{t+1} | MC-EEG_t, RNG_t)
#       D2: P(RNG_{t+1} | RNG_t, MC-EEG_t)
#
# Estimator:
#
#   reference_train → fits P0
#   calibration     → estimates Δχ and selects ε
#   test_holdout    → evaluates selected ε
#
# The final reported eps_test is confirmed only if the selected ε improves
# held-out test likelihood over ε = 0. Otherwise eps_test is set to 0.0.
#
# Interpretation:
#
#   III.2D is exploratory. It enriches EEG representation but does not convert
#   multichannel rows into primary claims. Primary status remains controlled by
#   model.primary, cfg.primary_regimes and cfg.primary_lags_epochs.


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class ConditionalScore:
    model: str
    label: str
    family: str
    regime: str
    lag_epochs: int
    lag_label: str
    valid: bool

    n_rows_total: int
    n_rows_model: int
    n_train: int
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
    regime: str
    lag_epochs: int
    lag_label: str
    family: str

    baseline_model: str
    augmented_model: str

    valid: bool
    primary: bool

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


# =========================================================
# Generic helpers
# =========================================================

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

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


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


def _get_cfg_float(cfg: Phase32Config, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _get_cfg_int(cfg: Phase32Config, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def model_uses_multichannel(model: ConditionalModelSpec) -> bool:
    components = list(model.current_components) + [model.target_component]

    return any(
        str(component).startswith("eeg_multichannel")
        or str(component).startswith("mc_eeg")
        or str(component).startswith("mceeg")
        for component in components
    ) or model.family == "MCEEG_next"


# =========================================================
# Column resolution
# =========================================================

def resolve_component_column(component: str, df: pd.DataFrame) -> str:
    direct = str(component)

    aliases: Dict[str, List[str]] = {
        "eeg_state": [
            "eeg_state",
        ],
        "rng_state": [
            "rng_state_model",
            "rng_state_aligned",
            "rng_state",
        ],
        "eeg_next_state": [
            "eeg_next_state",
        ],
        "rng_next_state": [
            "rng_next_state_model",
            "rng_next_state",
            "rng_next_bit_raw",
        ],
        "eeg_info_bin": [
            "eeg_info_bin",
        ],
        "rng_info_bin": [
            "rng_info_bin_model",
            "rng_info_bin_aligned",
            "rng_info_bin",
        ],
        "q_rng_bin": [
            "q_rng_bin_model",
            "q_rng_bin_aligned",
            "q_rng_bin",
        ],
        "observed_joint_state": [
            "observed_joint_state_lagged",
            "observed_joint_state",
        ],
        "informational_joint_state": [
            "informational_joint_state_lagged",
            "informational_joint_state",
        ],
        "latent_q_state": [
            "latent_q_state_lagged",
            "latent_q_state",
        ],

        # -------------------------------------------------
        # Phase III.2D — multichannel EEG aliases
        # -------------------------------------------------
        "eeg_multichannel_state": [
            "eeg_multichannel_state_model",
            "eeg_multichannel_state_lagged",
            "eeg_multichannel_state_aligned",
            "eeg_multichannel_state",
        ],
        "eeg_multichannel_next_state": [
            "eeg_multichannel_next_state_model",
            "eeg_multichannel_next_state",
            "eeg_multichannel_state_next",
        ],
        "eeg_multichannel_info_bin": [
            "eeg_multichannel_info_bin_model",
            "eeg_multichannel_info_bin_lagged",
            "eeg_multichannel_info_bin_aligned",
            "eeg_multichannel_info_bin",
        ],
        "eeg_multichannel_info_bin_next": [
            "eeg_multichannel_info_bin_next",
        ],
    }

    candidates = aliases.get(direct, [direct])

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not resolve component column '{component}'. "
        f"Tried: {candidates}. Available columns include: {list(df.columns)[:60]}..."
    )


def model_validity_column(model: ConditionalModelSpec, df: pd.DataFrame) -> str:
    """
    Resolves the validity mask for the model.

    Multichannel models must use valid_multichannel_conditional_row when
    available, because D0/D1/D2 require the multichannel next-state layer.
    """
    if model_uses_multichannel(model):
        for col in [
            "valid_multichannel_conditional_row",
            "valid_multichannel_next_state",
        ]:
            if col in df.columns:
                return col

    if model.family == "EEG_next":
        if "valid_eeg_conditional_row" in df.columns:
            return "valid_eeg_conditional_row"

    if model.family == "MCEEG_next":
        for col in [
            "valid_multichannel_conditional_row",
            "valid_multichannel_next_state",
        ]:
            if col in df.columns:
                return col

    if model.family == "RNG_next":
        # For D2, RNG_next uses multichannel EEG as an additional current
        # component, so the multichannel validity column is preferred when the
        # model uses multichannel state.
        if model_uses_multichannel(model):
            for col in [
                "valid_multichannel_conditional_row",
                "valid_multichannel_next_state",
            ]:
                if col in df.columns:
                    return col

        if "valid_rng_conditional_row" in df.columns:
            return "valid_rng_conditional_row"

    if "valid_conditional_row" in df.columns:
        return "valid_conditional_row"

    if "valid_next_state" in df.columns:
        return "valid_next_state"

    raise KeyError(f"No validity column found for model family={model.family}")


# =========================================================
# Splits
# =========================================================

def make_subject_chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    min_train: int = 20,
    min_test: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible two-way split.

    Kept because other modules may still call it. The main conditional estimator
    uses make_subject_chronological_three_way_split().
    """
    if df.empty:
        raise RuntimeError("Cannot split empty dataframe.")

    _require_columns(df, ["subject_id", "recording_id"], context="chronological split")

    out = _sort_frame(df.copy())

    train_indices: List[int] = []
    test_indices: List[int] = []

    group_cols = ["subject_id", "recording_id"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)
        n = len(idx_arr)

        if n < min_train + min_test:
            continue

        n_train = int(math.floor(n * float(train_ratio)))

        n_train = max(min_train, n_train)
        n_train = min(n_train, n - min_test)

        if n_train <= 0 or n_train >= n:
            continue

        train_indices.extend(idx_arr[:n_train].tolist())
        test_indices.extend(idx_arr[n_train:].tolist())

    if len(train_indices) < min_train or len(test_indices) < min_test:
        raise RuntimeError(
            f"Split too small: train={len(train_indices)}, test={len(test_indices)}"
        )

    return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)


def make_subject_chronological_three_way_split(
    df: pd.DataFrame,
    cfg: Phase32Config,
    min_ref: int = 20,
    min_calib: int = 10,
    min_test: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds leakage-safe chronological splits inside each subject/recording.

    Partitions:
        reference_train: fits P0
        calibration: estimates Δχ and selects ε
        test_holdout: evaluates selected ε
    """
    if df.empty:
        raise RuntimeError("Cannot split empty dataframe.")

    _require_columns(df, ["subject_id", "recording_id"], context="three-way chronological split")

    out = _sort_frame(df.copy())

    ref_ratio = _get_cfg_float(cfg, "conditional_split_ref_ratio", 0.50)
    calib_ratio = _get_cfg_float(cfg, "conditional_split_calib_ratio", 0.20)

    if ref_ratio <= 0 or calib_ratio <= 0 or ref_ratio + calib_ratio >= 1:
        ref_ratio = 0.50
        calib_ratio = 0.20

    ref_indices: List[int] = []
    calib_indices: List[int] = []
    test_indices: List[int] = []

    group_cols = ["subject_id", "recording_id"]

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


def split_existing_train_indices_into_ref_calib(
    train_indices: Sequence[int],
    min_ref: int = 10,
    min_calib: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compatibility helper when external code provides idx_train/idx_test.

    The provided train fold is split chronologically into reference/calibration.
    """
    arr = np.asarray(train_indices, dtype=int)

    if arr.size < min_ref + min_calib:
        raise RuntimeError(
            f"Provided idx_train too small for ref/calib split: {arr.size}"
        )

    cut = int(math.floor(arr.size * 0.70))
    cut = max(min_ref, cut)
    cut = min(cut, arr.size - min_calib)

    return arr[:cut], arr[cut:]


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

    required = ["subject_id", "recording_id", "epoch_idx", validity_col, target_col] + current_cols
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
        if key not in current_map:
            continue

        target = int(target)

        if target not in target_map:
            continue

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
    """
    Builds Δχ from calibration only.

    Rows absent from calibration receive Δχ = 0, preventing artificial
    reweighting in unsupported current states.
    """
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
    """
    Evaluates selected ε on held-out test.

    If the selected ε does not improve test likelihood over ε=0, the confirmed
    eps_test is forced to 0.0 and ll_test is reported as the ε=0 likelihood.
    """
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
    alpha: float = 1e-6,
    min_calib_gain: float = 1e-6,
    min_test_gain: float = 1e-6,
) -> Dict[str, Any]:
    """
    Leakage-safe conditional ε estimator.

    Reference:
        Fits P0.

    Calibration:
        Builds Δχ and selects candidate ε.

    Test:
        Evaluates the candidate ε on a disjoint holdout.
        If the candidate does not improve held-out likelihood, confirmed eps_test = 0.
    """
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
    eps_saturated = bool(eps_candidate >= eps_grid_max and eps_grid_max > 0)

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
        "eps_saturated": eps_saturated,
        "ll_calib": float(ll_calib),
        "ll_calib_eps0": float(ll_calib_eps0),
        "ll_calib_gain_from_eps": float(ll_calib_gain),
        "ll_test": float(ll_test),
        "ll_test_eps0": float(ll_test_eps0),
        "ll_test_gain_from_eps": float(ll_test_gain),
    }


def evaluate_conditional_model(
    df: pd.DataFrame,
    model: ConditionalModelSpec,
    cfg: Optional[Phase32Config] = None,
    regime_name: str = "",
    lag_epochs: int = 0,
    idx_train: Optional[Sequence[int]] = None,
    idx_test: Optional[Sequence[int]] = None,
) -> ConditionalScore:
    """
    Evaluates one conditional model on one regime + lag dataframe.

    Leakage-safe behavior:
        - P0 is fitted on reference_train.
        - Δχ and ε candidate are estimated on calibration.
        - eps_test is confirmed only on held-out test.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    lag_epochs = int(lag_epochs)

    try:
        model_df, current_cols, target_col = build_model_dataframe(df, model)
        model_df = model_df.reset_index(drop=True)

        min_ref = _get_cfg_int(cfg, "conditional_min_ref_rows", 20)
        min_calib = _get_cfg_int(cfg, "conditional_min_calib_rows", 10)
        min_test = _get_cfg_int(cfg, "conditional_min_test_rows", 10)

        if idx_train is None or idx_test is None:
            idx_ref_arr, idx_calib_arr, idx_test_arr = make_subject_chronological_three_way_split(
                model_df,
                cfg=cfg,
                min_ref=min_ref,
                min_calib=min_calib,
                min_test=min_test,
            )
        else:
            idx_ref_arr, idx_calib_arr = split_existing_train_indices_into_ref_calib(
                idx_train,
                min_ref=min_ref,
                min_calib=min_calib,
            )
            idx_test_arr = np.asarray(idx_test, dtype=int)

        ref_df = model_df.iloc[idx_ref_arr].copy()
        calib_df = model_df.iloc[idx_calib_arr].copy()
        test_df = model_df.iloc[idx_test_arr].copy()

        if ref_df.empty or calib_df.empty or test_df.empty:
            raise RuntimeError(
                f"Empty ref/calib/test split for model={model.name}: "
                f"ref={len(ref_df)}, calib={len(calib_df)}, test={len(test_df)}"
            )

        ref_keys = ref_df["current_key"].tolist()
        ref_targets = ref_df["target_value"].astype(int).tolist()

        calib_keys = calib_df["current_key"].tolist()
        calib_targets = calib_df["target_value"].astype(int).tolist()

        test_keys = test_df["current_key"].tolist()
        test_targets = test_df["target_value"].astype(int).tolist()

        min_calib_gain = _get_cfg_float(cfg, "conditional_min_calib_ll_gain", 1e-6)
        min_test_gain = _get_cfg_float(cfg, "conditional_min_test_ll_gain", 1e-6)

        estimate = _estimate_eps_leakage_safe(
            ref_keys=ref_keys,
            ref_targets=ref_targets,
            calib_keys=calib_keys,
            calib_targets=calib_targets,
            test_keys=test_keys,
            test_targets=test_targets,
            eps_grid=cfg.eps_grid,
            alpha=1e-6,
            min_calib_gain=min_calib_gain,
            min_test_gain=min_test_gain,
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
            regime=regime_name,
            lag_epochs=lag_epochs,
            lag_label=lag_label(lag_epochs),
            valid=True,
            n_rows_total=int(len(df)),
            n_rows_model=int(len(model_df)),
            n_train=int(len(ref_df)),
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
            regime=regime_name,
            lag_epochs=lag_epochs,
            lag_label=lag_label(lag_epochs),
            valid=False,
            n_rows_total=int(len(df)),
            n_rows_model=0,
            n_train=0,
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
    cfg: Phase32Config,
    regime_name: str,
    lag_epochs: int,
) -> Tuple[int, int, float]:
    """
    Evaluates within-subject positive confirmed eps lift.

    Uses the same leakage-safe estimator. A subject counts as positive only if
    augmented confirmed eps_test > baseline confirmed eps_test.
    """
    if "subject_id" not in df.columns:
        return 0, 0, 0.0

    n_eval = 0
    n_positive = 0

    for subject_id, group in df.groupby("subject_id", sort=True):
        g = _sort_frame(group.copy())

        if len(g) < 60:
            continue

        baseline_score = evaluate_conditional_model(
            df=g,
            model=baseline_model,
            cfg=cfg,
            regime_name=regime_name,
            lag_epochs=lag_epochs,
        )

        augmented_score = evaluate_conditional_model(
            df=g,
            model=augmented_model,
            cfg=cfg,
            regime_name=regime_name,
            lag_epochs=lag_epochs,
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
    cfg: Optional[Phase32Config] = None,
    regime_name: str = "",
    lag_epochs: int = 0,
) -> Tuple[ConditionalScore, ConditionalScore, ConditionalPairScore]:
    if cfg is None:
        cfg = load_phase3_2_config()

    baseline_score = evaluate_conditional_model(
        df=df,
        model=baseline_model,
        cfg=cfg,
        regime_name=regime_name,
        lag_epochs=lag_epochs,
    )

    augmented_score = evaluate_conditional_model(
        df=df,
        model=augmented_model,
        cfg=cfg,
        regime_name=regime_name,
        lag_epochs=lag_epochs,
    )

    valid = baseline_score.valid and augmented_score.valid

    n_eval, n_pos, frac_pos = evaluate_subject_fraction_positive_lift(
        df=df,
        baseline_model=baseline_model,
        augmented_model=augmented_model,
        cfg=cfg,
        regime_name=regime_name,
        lag_epochs=lag_epochs,
    )

    conditional_lift = float(augmented_score.eps_test - baseline_score.eps_test)
    ll_improvement = float(augmented_score.ll_test - baseline_score.ll_test)

    bic_improvement = float(baseline_score.bic_test - augmented_score.bic_test)
    aic_improvement = float(baseline_score.aic_test - augmented_score.aic_test)

    primary = (
        baseline_model.primary
        and augmented_model.primary
        and regime_name in cfg.primary_regimes
        and int(lag_epochs) in cfg.primary_lags_epochs
    )

    reason = "ok"

    if not valid:
        reason = f"baseline_valid={baseline_score.valid}; augmented_valid={augmented_score.valid}"

    pair = ConditionalPairScore(
        regime=regime_name,
        lag_epochs=int(lag_epochs),
        lag_label=lag_label(int(lag_epochs)),
        family=augmented_model.family,
        baseline_model=baseline_model.name,
        augmented_model=augmented_model.name,
        valid=valid,
        primary=bool(primary),
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


# =========================================================
# Batch evaluation
# =========================================================

def evaluate_conditional_models_for_frame(
    df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
    regime_name: str = "",
    lag_epochs: int = 0,
) -> Tuple[List[ConditionalScore], List[ConditionalPairScore]]:
    if cfg is None:
        cfg = load_phase3_2_config()

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
            regime_name=regime_name,
            lag_epochs=int(lag_epochs),
        )

        scores_by_name[baseline_score.model] = baseline_score
        scores_by_name[augmented_score.model] = augmented_score
        pair_scores.append(pair_score)

    model_scores = list(scores_by_name.values())

    return model_scores, pair_scores


def evaluate_all_conditional_from_lagged_csvs(
    cfg: Optional[Phase32Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2_config()

    model_score_rows: List[Dict[str, Any]] = []
    pair_score_rows: List[Dict[str, Any]] = []

    regimes_order: List[str] = []

    for name in list(cfg.primary_regimes) + list(cfg.secondary_regimes) + ["full"]:
        if name not in regimes_order:
            regimes_order.append(name)

    for regime_name in regimes_order:
        for lag_value in cfg.lags_epochs:
            path = lagged_csv_path(
                regime_name=regime_name,
                lag_epochs=int(lag_value),
                cfg=cfg,
            )

            if not path.exists():
                continue

            print(
                f"[Phase3.2 Conditional] Evaluating regime={regime_name} "
                f"lag={lag_value} file={path.name}"
            )

            df = pd.read_csv(path)
            df = _sort_frame(df)

            model_scores, pair_scores = evaluate_conditional_models_for_frame(
                df=df,
                cfg=cfg,
                regime_name=regime_name,
                lag_epochs=int(lag_value),
            )

            model_score_rows.extend(_as_records(model_scores))
            pair_score_rows.extend(_as_records(pair_scores))

    model_scores_df = pd.DataFrame(model_score_rows)
    pair_scores_df = pd.DataFrame(pair_score_rows)

    if not model_scores_df.empty:
        model_scores_df = model_scores_df.sort_values(
            ["regime", "lag_epochs", "family", "model"]
        ).reset_index(drop=True)

    if not pair_scores_df.empty:
        pair_scores_df = pair_scores_df.sort_values(
            ["regime", "lag_epochs", "family", "augmented_model"]
        ).reset_index(drop=True)

    return model_scores_df, pair_scores_df


# =========================================================
# Summaries and output
# =========================================================

def summarize_conditional_pairs(
    pair_scores_df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2_config()

    if pair_scores_df.empty:
        return {
            "available": False,
            "reason": "empty_pair_scores",
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
        }

    primary = valid[_as_bool_series(valid["primary"])].copy() if "primary" in valid.columns else pd.DataFrame()

    positive_lift = valid[
        pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0) > 0.0
    ].copy()

    primary_positive_lift = (
        primary[pd.to_numeric(primary["conditional_lift"], errors="coerce").fillna(0.0) > 0.0].copy()
        if not primary.empty
        else pd.DataFrame()
    )

    multichannel = valid[
        valid["augmented_model"].astype(str).str.startswith("D")
        | valid["baseline_model"].astype(str).str.startswith("D")
        | valid["family"].astype(str).str.contains("MCEEG", case=False, na=False)
    ].copy()

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
            "regime": str(row.get("regime")),
            "lag_epochs": _safe_int(row.get("lag_epochs")),
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
        }

    summary: Dict[str, Any] = {
        "available": True,
        "module": "III.2C_D_conditional_CDR_leakage_safe",
        "n_rows": int(len(pair_scores_df)),
        "n_valid_rows": int(len(valid)),
        "n_primary_rows": int(len(primary)),
        "n_positive_lift_rows": int(len(positive_lift)),
        "n_primary_positive_lift_rows": int(len(primary_positive_lift)),
        "n_strong_candidate_rows": int(len(strong_candidates)),
        "n_multichannel_rows": int(len(multichannel)),
        "n_multichannel_positive_lift_rows": int(len(multichannel_positive_lift)),
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

    if "baseline_eps_test" in valid.columns and "augmented_eps_test" in valid.columns:
        summary["max_baseline_eps_test"] = float(
            pd.to_numeric(valid["baseline_eps_test"], errors="coerce").fillna(0.0).max()
        )
        summary["max_augmented_eps_test"] = float(
            pd.to_numeric(valid["augmented_eps_test"], errors="coerce").fillna(0.0).max()
        )

    return summary


def build_conditional_summary_text(
    pair_scores_df: pd.DataFrame,
    summary: Mapping[str, Any],
) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2C/D — Conditional CDR summary")
    lines.append("Leakage-safe ref/calib/test estimator")
    lines.append("=" * 78)
    lines.append("")

    if not summary.get("available"):
        lines.append(f"Summary unavailable: {summary.get('reason')}")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(f"Valid rows: {summary.get('n_valid_rows')}/{summary.get('n_rows')}")
    lines.append(f"Primary rows: {summary.get('n_primary_rows')}")
    lines.append(f"Positive lift rows: {summary.get('n_positive_lift_rows')}")
    lines.append(f"Primary positive lift rows: {summary.get('n_primary_positive_lift_rows')}")
    lines.append(f"Strong candidate rows: {summary.get('n_strong_candidate_rows')}")
    lines.append(f"Multichannel rows: {summary.get('n_multichannel_rows')}")
    lines.append(f"Multichannel positive lift rows: {summary.get('n_multichannel_positive_lift_rows')}")
    lines.append(f"Max baseline eps_test: {summary.get('max_baseline_eps_test')}")
    lines.append(f"Max augmented eps_test: {summary.get('max_augmented_eps_test')}")
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

    lines.append("Positive lift rows:")
    for row in summary.get("positive_lift_rows", []):
        lines.append(f"  {row}")

    lines.append("")
    lines.append("Multichannel positive lift rows:")
    for row in summary.get("multichannel_positive_lift_rows", []):
        lines.append(f"  {row}")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


def save_conditional_outputs(
    model_scores_df: pd.DataFrame,
    pair_scores_df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    model_scores_path = Path(cfg.results_dir) / "phase3_2c_conditional_model_scores.csv"
    pair_scores_path = Path(cfg.conditional_results_csv)
    summary_json_path = Path(cfg.conditional_summary_json)
    summary_txt_path = Path(cfg.results_dir) / "phase3_2c_conditional_summary.txt"

    multichannel_results_path = Path(cfg.multichannel_results_csv)

    model_scores_df.to_csv(model_scores_path, index=False)
    pair_scores_df.to_csv(pair_scores_path, index=False)

    if not pair_scores_df.empty:
        multichannel_mask = (
            pair_scores_df["augmented_model"].astype(str).str.startswith("D")
            | pair_scores_df["baseline_model"].astype(str).str.startswith("D")
            | pair_scores_df["family"].astype(str).str.contains("MCEEG", case=False, na=False)
        )

        multichannel_df = pair_scores_df[multichannel_mask].copy()
    else:
        multichannel_df = pd.DataFrame()

    multichannel_results_path.parent.mkdir(parents=True, exist_ok=True)
    multichannel_df.to_csv(multichannel_results_path, index=False)

    summary = summarize_conditional_pairs(pair_scores_df, cfg)

    save_json(
        summary_json_path,
        {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": "III.2C_D_conditional_CDR_leakage_safe",
            "summary": summary,
            "model_scores_file": str(model_scores_path),
            "pair_scores_file": str(pair_scores_path),
            "multichannel_results_file": str(multichannel_results_path),
        },
    )

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(build_conditional_summary_text(pair_scores_df, summary))

    print(f"[Phase3.2 Conditional] Saved model scores: {model_scores_path}")
    print(f"[Phase3.2 Conditional] Saved pair scores: {pair_scores_path}")
    print(f"[Phase3.2 Conditional] Saved multichannel results: {multichannel_results_path}")
    print(f"[Phase3.2 Conditional] Saved summary JSON: {summary_json_path}")
    print(f"[Phase3.2 Conditional] Saved summary TXT: {summary_txt_path}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    model_scores_df, pair_scores_df = evaluate_all_conditional_from_lagged_csvs(cfg)

    save_conditional_outputs(
        model_scores_df=model_scores_df,
        pair_scores_df=pair_scores_df,
        cfg=cfg,
    )

    summary = summarize_conditional_pairs(pair_scores_df, cfg)

    print("\n============================================================")
    print("Phase III.2C/D conditional CDR completed")
    print("Leakage-safe ref/calib/test estimator")
    print("============================================================")

    if not summary.get("available"):
        print(f"Summary unavailable: {summary.get('reason')}")
    else:
        print(f"Valid pair rows: {summary.get('n_valid_rows')}")
        print(f"Positive lift rows: {summary.get('n_positive_lift_rows')}")
        print(f"Primary positive lift rows: {summary.get('n_primary_positive_lift_rows')}")
        print(f"Strong candidate rows: {summary.get('n_strong_candidate_rows')}")
        print(f"Multichannel rows: {summary.get('n_multichannel_rows')}")
        print(f"Multichannel positive lift rows: {summary.get('n_multichannel_positive_lift_rows')}")
        print(f"Max baseline eps_test: {summary.get('max_baseline_eps_test')}")
        print(f"Max augmented eps_test: {summary.get('max_augmented_eps_test')}")
        print(f"Best by lift: {summary.get('best_by_conditional_lift')}")
        print(f"Best by BIC: {summary.get('best_by_bic_improvement')}")
        print(f"Best multichannel by lift: {summary.get('best_multichannel_by_conditional_lift')}")
        print(f"Best multichannel by BIC: {summary.get('best_multichannel_by_bic_improvement')}")

    print("============================================================\n")


if __name__ == "__main__":
    main()