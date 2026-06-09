from __future__ import annotations

import argparse
import hashlib
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
    load_phase3_2e_config,
)
from src.phase3_2e_alignment import (
    lagged_frame_path,
    load_or_build_phase3_2e_alignment,
)

CONDITIONAL_MODULE = "phase3_2e_conditional"
CONDITIONAL_VERSION = "phase3_2e_conditional_v5_primary_lead_cache_safe"


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
    empirical_claim_allowed: bool
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
    eps_grid_initial_max: float
    eps_grid_final_max: float
    eps_grid_expansions: int
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
    empirical_claim_allowed: bool
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
    required_positive_subjects: int
    required_positive_fraction: float
    subject_consistency_pass: bool
    baseline_eps_saturated: bool
    augmented_eps_saturated: bool
    reason: str = "ok"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
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


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)

    except Exception:
        return default


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "sim", "t"})
    )


def _lag_label(lag_windows: int) -> str:
    lag_windows = int(lag_windows)

    if lag_windows < 0:
        return f"m{abs(lag_windows)}"

    if lag_windows > 0:
        return f"p{lag_windows}"

    return "0"


def _lag_direction(lag_windows: int) -> str:
    lag_windows = int(lag_windows)

    if lag_windows > 0:
        return "future_target"

    if lag_windows < 0:
        return "past_target"

    return "concurrent"


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "subject_id",
            "session_id",
            "window_start_utc",
            "window_id",
            "alignment_row_id",
        )
        if column in df.columns
    ]

    if not columns:
        return df.reset_index(drop=True)

    return df.sort_values(
        columns,
        kind="stable",
    ).reset_index(drop=True)


def _require_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns for {context}: {missing}"
        )


def _get_cfg_float(
    cfg: Phase32EConfig,
    name: str,
    default: float,
) -> float:
    return float(getattr(cfg, name, default))


def _get_cfg_int(
    cfg: Phase32EConfig,
    name: str,
    default: int,
) -> int:
    return int(getattr(cfg, name, default))


def _registered_windows(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    return tuple(
        int(value)
        for value in cfg.all_window_seconds
    )


def _registered_lags(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    return tuple(
        int(value)
        for value in cfg.all_lags_windows
    )


def _primary_windows(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    values = getattr(
        cfg,
        "primary_test_window_seconds",
        getattr(
            cfg,
            "primary_window_seconds",
            tuple(),
        ),
    )

    return tuple(
        int(value)
        for value in values
    )


def _primary_lags(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    values = getattr(
        cfg,
        "primary_test_lags_windows",
        getattr(
            cfg,
            "primary_lags_windows",
            tuple(),
        ),
    )

    return tuple(
        int(value)
        for value in values
    )


def _primary_pairs(
    cfg: Phase32EConfig,
) -> Tuple[Tuple[str, str], ...]:
    values = getattr(
        cfg,
        "primary_test_pairs",
        getattr(
            cfg,
            "primary_conditional_pairs",
            tuple(),
        ),
    )

    return tuple(
        (
            str(baseline),
            str(augmented),
        )
        for baseline, augmented in values
    )


def _config_signature_payload(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    return {
        "registered_windows": list(_registered_windows(cfg)),
        "primary_windows": list(_primary_windows(cfg)),
        "registered_lags": list(_registered_lags(cfg)),
        "primary_lags": list(_primary_lags(cfg)),
        "active_conditional_pairs": [
            list(pair)
            for pair in active_conditional_pairs(cfg)
        ],
        "primary_test_pairs": [
            list(pair)
            for pair in _primary_pairs(cfg)
        ],
        "eps_grid": [
            float(value)
            for value in cfg.eps_grid
        ],
        "eps_auto_expand_enabled": bool(
            getattr(
                cfg,
                "eps_auto_expand_enabled",
                False,
            )
        ),
        "eps_auto_expand_factor": _get_cfg_float(
            cfg,
            "eps_auto_expand_factor",
            1.5,
        ),
        "eps_auto_expand_max": _get_cfg_float(
            cfg,
            "eps_auto_expand_max",
            max(cfg.eps_grid) if cfg.eps_grid else 0.0,
        ),
        "conditional_lift_min": _get_cfg_float(
            cfg,
            "conditional_lift_min",
            0.03,
        ),
        "strong_eps_min": _get_cfg_float(
            cfg,
            "strong_eps_min",
            0.07,
        ),
        "subject_effect_fraction": _get_cfg_float(
            cfg,
            "subject_effect_fraction",
            0.10,
        ),
        "min_positive_subjects": _get_cfg_int(
            cfg,
            "min_positive_subjects",
            1,
        ),
        "large_cohort_subject_fraction_min": _get_cfg_float(
            cfg,
            "large_cohort_subject_fraction_min",
            0.01,
        ),
        "large_cohort_subject_fraction_max": _get_cfg_float(
            cfg,
            "large_cohort_subject_fraction_max",
            0.03,
        ),
        "conditional_split_ref_ratio": _get_cfg_float(
            cfg,
            "conditional_split_ref_ratio",
            0.50,
        ),
        "conditional_split_calib_ratio": _get_cfg_float(
            cfg,
            "conditional_split_calib_ratio",
            0.20,
        ),
        "conditional_min_ref_rows": _get_cfg_int(
            cfg,
            "conditional_min_ref_rows",
            20,
        ),
        "conditional_min_calib_rows": _get_cfg_int(
            cfg,
            "conditional_min_calib_rows",
            10,
        ),
        "conditional_min_test_rows": _get_cfg_int(
            cfg,
            "conditional_min_test_rows",
            10,
        ),
    }


def _config_fingerprint(
    cfg: Phase32EConfig,
) -> str:
    encoded = json.dumps(
        _config_signature_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _alignment_signature(
    alignment_report: Mapping[str, Any],
) -> Dict[str, Any]:
    result = (
        alignment_report.get("result", {})
        if isinstance(alignment_report, Mapping)
        else {}
    )

    return {
        "status": (
            alignment_report.get("status")
            if isinstance(alignment_report, Mapping)
            else None
        ),
        "n_frames": result.get("n_frames"),
        "n_window_sizes": result.get("n_window_sizes"),
        "n_lags": result.get("n_lags"),
        "n_total_aligned_rows": result.get(
            "n_total_aligned_rows"
        ),
        "n_valid_alignment_rows": result.get(
            "n_valid_alignment_rows"
        ),
        "n_valid_primary_alignment_rows": result.get(
            "n_valid_primary_alignment_rows"
        ),
        "registered_windows": result.get(
            "registered_windows"
        ),
        "primary_windows": result.get(
            "primary_windows"
        ),
        "registered_lags": result.get(
            "registered_lags"
        ),
        "primary_lags": result.get(
            "primary_lags"
        ),
    }


def _strong_failure_reasons(
    row: pd.Series,
    cfg: Phase32EConfig,
) -> List[str]:
    reasons: List[str] = []

    if _safe_float(
        row.get("conditional_lift")
    ) < _get_cfg_float(
        cfg,
        "conditional_lift_min",
        0.03,
    ):
        reasons.append(
            "conditional_lift_below_threshold"
        )

    if _safe_float(
        row.get("augmented_eps_test")
    ) < _get_cfg_float(
        cfg,
        "strong_eps_min",
        0.07,
    ):
        reasons.append(
            "augmented_epsilon_below_threshold"
        )

    if not bool(
        row.get(
            "subject_consistency_pass",
            False,
        )
    ):
        reasons.append(
            "subject_consistency_not_met"
        )

    if _safe_float(
        row.get("bic_improvement"),
        float("-inf"),
    ) < 0.0:
        reasons.append(
            "bic_does_not_favor_augmented_model"
        )

    if bool(
        row.get(
            "augmented_eps_saturated",
            False,
        )
    ):
        reasons.append(
            "augmented_epsilon_saturated_diagnostic"
        )

    return reasons


def _records_with_strong_diagnostics(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        record = row.to_dict()
        record["strong_failure_reasons"] = (
            _strong_failure_reasons(
                row,
                cfg,
            )
        )
        record["meets_strong_candidate_criteria"] = (
            len(record["strong_failure_reasons"]) == 0
        )
        records.append(record)

    return records


def _alignment_scope(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> str:
    if (
        int(window_seconds) in set(_primary_windows(cfg))
        and int(lag_windows) in set(_primary_lags(cfg))
    ):
        return "primary"

    if (
        int(window_seconds) in set(_registered_windows(cfg))
        and int(lag_windows) in set(_registered_lags(cfg))
    ):
        return "exploratory"

    return "unregistered"


def _frame_claim_allowed(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> bool:
    if "empirical_claim_allowed" in df.columns:
        values = _as_bool_series(
            df["empirical_claim_allowed"]
        )

        return bool(
            not values.empty
            and values.all()
        )

    return not bool(
        cfg.require_real_synchronized_data_for_claim
    )


def model_uses_multichannel(
    model: ConditionalModelSpec,
) -> bool:
    components = (
        list(model.current_components)
        + [model.target_component]
    )

    return bool(
        getattr(
            model,
            "multichannel",
            False,
        )
        or model.family == "MCEEG_next"
        or any(
            str(component).startswith(
                (
                    "mc_eeg",
                    "mceeg",
                    "eeg_multichannel",
                )
            )
            for component in components
        )
    )


def pair_is_primary(
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
    baseline_model: str,
    augmented_model: str,
) -> bool:
    return bool(
        int(window_seconds) in set(_primary_windows(cfg))
        and int(lag_windows) in set(_primary_lags(cfg))
        and (
            (
                str(baseline_model),
                str(augmented_model),
            )
            in set(_primary_pairs(cfg))
        )
    )


def resolve_component_column(
    component: str,
    df: pd.DataFrame,
) -> str:
    aliases: Dict[str, Tuple[str, ...]] = {
        "eeg_state": (
            "eeg_state",
        ),
        "eeg_next_state": (
            "eeg_next_state",
            "target_eeg_state",
        ),
        "qrng_state": (
            "qrng_state",
            "qrng_state_model",
            "rng_state",
        ),
        "qrng_next_state": (
            "qrng_next_state",
            "target_qrng_state",
            "rng_next_state",
        ),
        "mc_eeg_state": (
            "mc_eeg_state",
            "eeg_multichannel_state",
        ),
        "mc_eeg_next_state": (
            "mc_eeg_next_state",
            "target_mc_eeg_state",
            "eeg_multichannel_next_state",
        ),
        "eeg_info_bin": (
            "eeg_info_bin",
        ),
        "qrng_info_bin": (
            "qrng_info_bin",
            "rng_info_bin",
        ),
        "mc_eeg_info_bin": (
            "mc_eeg_info_bin",
            "eeg_multichannel_info_bin",
        ),
        "observed_joint_state": (
            "observed_joint_state",
        ),
        "informational_joint_state": (
            "informational_joint_state",
        ),
        "mc_observed_joint_state": (
            "mc_observed_joint_state",
        ),
        "mc_informational_joint_state": (
            "mc_informational_joint_state",
        ),
    }

    candidates = aliases.get(
        str(component),
        (str(component),),
    )

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        f"Could not resolve component "
        f"'{component}'. Tried {list(candidates)}."
    )


def model_validity_column(
    model: ConditionalModelSpec,
    df: pd.DataFrame,
) -> str:
    if model_uses_multichannel(model):
        for column in (
            "valid_mc_conditional_row",
            "valid_multichannel_conditional_row",
            "valid_mc_next_state",
        ):
            if column in df.columns:
                return column

    for column in (
        "valid_conditional_row",
        "valid_alignment_row",
    ):
        if column in df.columns:
            return column

    raise KeyError(
        f"No validity column found for "
        f"model={model.name}."
    )


def _model_columns(
    model: ConditionalModelSpec,
    df: pd.DataFrame,
) -> Tuple[List[str], str, str]:
    current_columns = [
        resolve_component_column(
            component,
            df,
        )
        for component in model.current_components
    ]

    target_column = resolve_component_column(
        model.target_component,
        df,
    )

    validity_column = model_validity_column(
        model,
        df,
    )

    return (
        current_columns,
        target_column,
        validity_column,
    )


def build_model_dataframe(
    df: pd.DataFrame,
    model: ConditionalModelSpec,
) -> Tuple[pd.DataFrame, List[str], str]:
    (
        current_columns,
        target_column,
        validity_column,
    ) = _model_columns(
        model,
        df,
    )

    identity_columns = [
        column
        for column in (
            "subject_id",
            "session_id",
            "window_start_utc",
            "window_id",
            "alignment_row_id",
            "empirical_claim_allowed",
        )
        if column in df.columns
    ]

    required = [
        "subject_id",
        "session_id",
        validity_column,
        target_column,
        *current_columns,
    ]

    _require_columns(
        df,
        required,
        f"conditional model {model.name}",
    )

    selected = list(
        dict.fromkeys(
            identity_columns
            + [validity_column]
            + current_columns
            + [target_column]
        )
    )

    work = df[selected].copy()

    valid_mask = (
        pd.to_numeric(
            work[validity_column],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    work = work.loc[valid_mask].copy()

    for column in current_columns + [target_column]:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.dropna(
        subset=current_columns + [target_column]
    ).copy()

    if work.empty:
        raise RuntimeError(
            f"No valid rows for model {model.name}."
        )

    for column in current_columns + [target_column]:
        work[column] = work[column].astype(int)

    work["current_key"] = list(
        zip(
            *(
                work[column].to_numpy()
                for column in current_columns
            )
        )
    )

    work["target_value"] = (
        work[target_column].astype(int)
    )

    return (
        _sort_frame(work),
        current_columns,
        target_column,
    )


def build_pair_common_dataframe(
    df: pd.DataFrame,
    baseline_model: ConditionalModelSpec,
    augmented_model: ConditionalModelSpec,
) -> pd.DataFrame:
    (
        baseline_current,
        baseline_target,
        baseline_validity,
    ) = _model_columns(
        baseline_model,
        df,
    )

    (
        augmented_current,
        augmented_target,
        augmented_validity,
    ) = _model_columns(
        augmented_model,
        df,
    )

    numeric_columns = list(
        dict.fromkeys(
            baseline_current
            + augmented_current
            + [
                baseline_target,
                augmented_target,
            ]
        )
    )

    identity_columns = [
        column
        for column in (
            "subject_id",
            "session_id",
            "window_start_utc",
            "window_id",
            "alignment_row_id",
            "empirical_claim_allowed",
        )
        if column in df.columns
    ]

    required = [
        "subject_id",
        "session_id",
        baseline_validity,
        augmented_validity,
        *numeric_columns,
    ]

    _require_columns(
        df,
        required,
        (
            f"pair {baseline_model.name} -> "
            f"{augmented_model.name}"
        ),
    )

    selected = list(
        dict.fromkeys(
            identity_columns
            + [
                baseline_validity,
                augmented_validity,
            ]
            + numeric_columns
        )
    )

    work = df[selected].copy()

    baseline_mask = (
        pd.to_numeric(
            work[baseline_validity],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    augmented_mask = (
        pd.to_numeric(
            work[augmented_validity],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    work = work.loc[
        baseline_mask & augmented_mask
    ].copy()

    for column in numeric_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.dropna(
        subset=numeric_columns
    ).copy()

    if work.empty:
        raise RuntimeError(
            f"No common rows for "
            f"{baseline_model.name} -> "
            f"{augmented_model.name}."
        )

    return _sort_frame(work)


def make_subject_chronological_three_way_split(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    lag_windows: int,
    min_ref: int = 20,
    min_calib: int = 10,
    min_test: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if df.empty:
        raise RuntimeError(
            "Cannot split an empty dataframe."
        )

    _require_columns(
        df,
        (
            "subject_id",
            "session_id",
        ),
        "three-way chronological split",
    )

    work = _sort_frame(df.copy())

    ref_ratio = _get_cfg_float(
        cfg,
        "conditional_split_ref_ratio",
        0.50,
    )

    calib_ratio = _get_cfg_float(
        cfg,
        "conditional_split_calib_ratio",
        0.20,
    )

    if (
        ref_ratio <= 0
        or calib_ratio <= 0
        or ref_ratio + calib_ratio >= 1
    ):
        ref_ratio = 0.50
        calib_ratio = 0.20

    purge = abs(int(lag_windows))

    ref_indices: List[int] = []
    calib_indices: List[int] = []
    test_indices: List[int] = []

    grouped = work.groupby(
        [
            "subject_id",
            "session_id",
        ],
        sort=False,
        dropna=False,
    )

    for _, indices in grouped.groups.items():
        arr = np.asarray(
            list(indices),
            dtype=int,
        )

        n_rows = len(arr)

        if (
            n_rows
            < min_ref
            + min_calib
            + min_test
            + 4 * purge
        ):
            continue

        ref_boundary = max(
            min_ref + purge,
            int(
                math.floor(
                    n_rows * ref_ratio
                )
            ),
        )

        calib_size = max(
            min_calib + 2 * purge,
            int(
                math.floor(
                    n_rows * calib_ratio
                )
            ),
        )

        calib_boundary = min(
            ref_boundary + calib_size,
            n_rows - min_test - purge,
        )

        ref_part = arr[
            : max(ref_boundary - purge, 0)
        ]

        calib_part = arr[
            min(ref_boundary + purge, n_rows)
            : max(calib_boundary - purge, 0)
        ]

        test_part = arr[
            min(calib_boundary + purge, n_rows) :
        ]

        if (
            len(ref_part) < min_ref
            or len(calib_part) < min_calib
            or len(test_part) < min_test
        ):
            continue

        ref_indices.extend(ref_part.tolist())
        calib_indices.extend(calib_part.tolist())
        test_indices.extend(test_part.tolist())

    if (
        len(ref_indices) < min_ref
        or len(calib_indices) < min_calib
        or len(test_indices) < min_test
    ):
        raise RuntimeError(
            "Three-way split too small after lag purge: "
            f"ref={len(ref_indices)}, "
            f"calib={len(calib_indices)}, "
            f"test={len(test_indices)}, "
            f"purge={purge}."
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
    arr = np.asarray(
        train_indices,
        dtype=int,
    )

    if arr.size < min_ref + min_calib:
        raise RuntimeError(
            "idx_train too small for ref/calib split: "
            f"{arr.size}."
        )

    cut = int(
        math.floor(
            arr.size * 0.70
        )
    )

    cut = max(
        min_ref,
        min(
            cut,
            arr.size - min_calib,
        ),
    )

    return (
        arr[:cut],
        arr[cut:],
    )


def _normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    sums[sums <= 0] = 1.0

    return matrix / sums


def _fit_reference(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    alpha: float,
) -> Tuple[
    Dict[Tuple[int, ...], int],
    Dict[int, int],
    np.ndarray,
    np.ndarray,
]:
    unique_current = sorted(
        set(current_keys)
    )

    unique_target = sorted(
        set(
            int(value)
            for value in targets
        )
    )

    if not unique_current or not unique_target:
        raise RuntimeError(
            "Reference split has no usable states."
        )

    current_map = {
        key: index
        for index, key in enumerate(unique_current)
    }

    target_map = {
        value: index
        for index, value in enumerate(unique_target)
    }

    counts = np.zeros(
        (
            len(current_map),
            len(target_map),
        ),
        dtype=float,
    )

    for key, target in zip(
        current_keys,
        targets,
    ):
        target = int(target)

        if (
            key in current_map
            and target in target_map
        ):
            counts[
                current_map[key],
                target_map[target],
            ] += 1.0

    p0 = _normalize_rows(
        counts + alpha
    )

    return (
        current_map,
        target_map,
        counts,
        p0,
    )


def _fallback_distribution(
    targets: Sequence[int],
    target_map: Mapping[int, int],
    alpha: float,
) -> np.ndarray:
    counts = np.zeros(
        len(target_map),
        dtype=float,
    )

    for target in targets:
        target = int(target)

        if target in target_map:
            counts[
                target_map[target]
            ] += 1.0

    probabilities = counts + alpha

    return probabilities / probabilities.sum()


def _encode(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    current = np.fromiter(
        (
            current_map.get(key, -1)
            for key in current_keys
        ),
        dtype=np.int64,
        count=len(current_keys),
    )

    target = np.fromiter(
        (
            target_map.get(int(value), -1)
            for value in targets
        ),
        dtype=np.int64,
        count=len(targets),
    )

    return current, target


def _log_likelihood(
    current_index: np.ndarray,
    target_index: np.ndarray,
    probabilities: np.ndarray,
    fallback: np.ndarray,
) -> float:
    tiny = 1e-12

    observed = np.full(
        len(target_index),
        tiny,
        dtype=float,
    )

    known_target = target_index >= 0
    known_current = current_index >= 0

    direct = known_target & known_current
    unknown_current = known_target & ~known_current

    if direct.any():
        observed[direct] = probabilities[
            current_index[direct],
            target_index[direct],
        ]

    if unknown_current.any():
        observed[unknown_current] = fallback[
            target_index[unknown_current]
        ]

    return float(
        np.log(
            np.maximum(
                observed,
                tiny,
            )
        ).sum()
    )


def _counts_for_maps(
    current_keys: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
) -> np.ndarray:
    counts = np.zeros(
        (
            len(current_map),
            len(target_map),
        ),
        dtype=float,
    )

    for key, target in zip(
        current_keys,
        targets,
    ):
        target = int(target)

        if (
            key in current_map
            and target in target_map
        ):
            counts[
                current_map[key],
                target_map[target],
            ] += 1.0

    return counts


def _build_delta(
    p0: np.ndarray,
    calibration_counts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    p_calibration = _normalize_rows(
        calibration_counts + alpha
    )

    observed = (
        calibration_counts.sum(axis=1) > 0
    )

    delta = np.zeros_like(
        p0,
        dtype=float,
    )

    delta[observed] = np.log(
        np.maximum(
            p_calibration[observed],
            alpha,
        )
        / np.maximum(
            p0[observed],
            alpha,
        )
    )

    return np.nan_to_num(
        np.clip(
            delta,
            -5.0,
            5.0,
        ),
        nan=0.0,
        posinf=5.0,
        neginf=-5.0,
    )


def _weighted_kernel(
    p0: np.ndarray,
    delta: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    exponent = np.clip(
        float(epsilon) * delta,
        -700.0,
        700.0,
    )

    return _normalize_rows(
        p0 * np.exp(exponent)
    )


def _grid_step(
    grid: Sequence[float],
) -> float:
    values = sorted(
        set(
            float(value)
            for value in grid
        )
    )

    differences = [
        values[index + 1] - values[index]
        for index in range(len(values) - 1)
        if values[index + 1] > values[index]
    ]

    if differences:
        return min(differences)

    return 0.01


def _expanded_grid(
    grid: Sequence[float],
    maximum: float,
) -> List[float]:
    step = _grid_step(grid)

    count = int(
        math.floor(
            float(maximum) / step
        )
    )

    values = [
        round(
            index * step,
            10,
        )
        for index in range(count + 1)
    ]

    if values[-1] < maximum:
        values.append(float(maximum))

    return values


def _select_epsilon(
    calibration_keys: Sequence[Tuple[int, ...]],
    calibration_targets: Sequence[int],
    current_map: Mapping[Tuple[int, ...], int],
    target_map: Mapping[int, int],
    p0: np.ndarray,
    fallback: np.ndarray,
    delta: np.ndarray,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    grid = sorted(
        set(
            float(value)
            for value in cfg.eps_grid
        )
    )

    if not grid:
        grid = [0.0]

    initial_max = max(grid)

    min_gain = _get_cfg_float(
        cfg,
        "conditional_min_calib_ll_gain",
        1e-6,
    )

    auto_expand = bool(
        getattr(
            cfg,
            "eps_auto_expand_enabled",
            False,
        )
    )

    expand_factor = max(
        _get_cfg_float(
            cfg,
            "eps_auto_expand_factor",
            1.5,
        ),
        1.01,
    )

    expand_max = max(
        _get_cfg_float(
            cfg,
            "eps_auto_expand_max",
            initial_max,
        ),
        initial_max,
    )

    (
        current_index,
        target_index,
    ) = _encode(
        calibration_keys,
        calibration_targets,
        current_map,
        target_map,
    )

    ll_zero = _log_likelihood(
        current_index,
        target_index,
        p0,
        fallback,
    )

    best_epsilon = 0.0
    best_ll = ll_zero
    expansions = 0

    while True:
        for epsilon in grid:
            ll_value = _log_likelihood(
                current_index,
                target_index,
                _weighted_kernel(
                    p0,
                    delta,
                    epsilon,
                ),
                fallback,
            )

            if ll_value > best_ll + min_gain:
                best_epsilon = float(epsilon)
                best_ll = float(ll_value)

        current_max = max(grid)
        step = _grid_step(grid)

        at_boundary = bool(
            best_ll > ll_zero + min_gain
            and best_epsilon
            >= current_max - step / 2.0
        )

        if (
            not auto_expand
            or not at_boundary
            or current_max >= expand_max
        ):
            break

        next_max = min(
            expand_max,
            max(
                current_max * expand_factor,
                current_max + step,
            ),
        )

        if next_max <= current_max:
            break

        grid = _expanded_grid(
            grid,
            next_max,
        )

        expansions += 1

    gain = float(best_ll - ll_zero)

    if gain <= min_gain:
        best_epsilon = 0.0
        best_ll = ll_zero
        gain = 0.0

    final_max = max(grid)

    saturated = bool(
        gain > min_gain
        and best_epsilon
        >= final_max - _grid_step(grid) / 2.0
    )

    return {
        "epsilon": float(best_epsilon),
        "ll": float(best_ll),
        "ll_zero": float(ll_zero),
        "gain": float(gain),
        "initial_max": float(initial_max),
        "final_max": float(final_max),
        "expansions": int(expansions),
        "saturated": saturated,
    }


def _estimate_model(
    work: pd.DataFrame,
    current_columns: Sequence[str],
    target_column: str,
    ref_indices: np.ndarray,
    calib_indices: np.ndarray,
    test_indices: np.ndarray,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    current_keys = list(
        zip(
            *(
                work[column]
                .astype(int)
                .to_numpy()
                for column in current_columns
            )
        )
    )

    targets = (
        work[target_column]
        .astype(int)
        .to_numpy()
    )

    ref_keys = [
        current_keys[index]
        for index in ref_indices
    ]

    calib_keys = [
        current_keys[index]
        for index in calib_indices
    ]

    test_keys = [
        current_keys[index]
        for index in test_indices
    ]

    ref_targets = targets[ref_indices].tolist()
    calib_targets = targets[calib_indices].tolist()
    test_targets = targets[test_indices].tolist()

    alpha = 1e-6

    (
        current_map,
        target_map,
        _,
        p0,
    ) = _fit_reference(
        ref_keys,
        ref_targets,
        alpha,
    )

    fallback = _fallback_distribution(
        ref_targets,
        target_map,
        alpha,
    )

    calibration_counts = _counts_for_maps(
        calib_keys,
        calib_targets,
        current_map,
        target_map,
    )

    delta = _build_delta(
        p0,
        calibration_counts,
        alpha,
    )

    selection = _select_epsilon(
        calib_keys,
        calib_targets,
        current_map,
        target_map,
        p0,
        fallback,
        delta,
        cfg,
    )

    (
        ref_current,
        ref_target,
    ) = _encode(
        ref_keys,
        ref_targets,
        current_map,
        target_map,
    )

    (
        test_current,
        test_target,
    ) = _encode(
        test_keys,
        test_targets,
        current_map,
        target_map,
    )

    ll_train = _log_likelihood(
        ref_current,
        ref_target,
        p0,
        fallback,
    )

    ll_test_zero = _log_likelihood(
        test_current,
        test_target,
        p0,
        fallback,
    )

    epsilon = float(selection["epsilon"])

    if epsilon > 0:
        ll_test_candidate = _log_likelihood(
            test_current,
            test_target,
            _weighted_kernel(
                p0,
                delta,
                epsilon,
            ),
            fallback,
        )
    else:
        ll_test_candidate = ll_test_zero

    min_test_gain = _get_cfg_float(
        cfg,
        "conditional_min_test_ll_gain",
        1e-6,
    )

    test_gain = float(
        ll_test_candidate - ll_test_zero
    )

    if (
        epsilon <= 0
        or test_gain <= min_test_gain
    ):
        eps_test = 0.0
        ll_test = ll_test_zero
        test_gain = 0.0
    else:
        eps_test = epsilon
        ll_test = ll_test_candidate

    return {
        "current_map": current_map,
        "target_map": target_map,
        "ll_train": float(ll_train),
        "eps_calib": epsilon,
        "eps_candidate": epsilon,
        "eps_test": float(eps_test),
        "eps_grid_initial_max": selection[
            "initial_max"
        ],
        "eps_grid_final_max": selection[
            "final_max"
        ],
        "eps_grid_expansions": selection[
            "expansions"
        ],
        "eps_saturated": selection[
            "saturated"
        ],
        "ll_calib": selection["ll"],
        "ll_calib_eps0": selection[
            "ll_zero"
        ],
        "ll_calib_gain_from_eps": selection[
            "gain"
        ],
        "ll_test": float(ll_test),
        "ll_test_eps0": float(ll_test_zero),
        "ll_test_gain_from_eps": float(test_gain),
    }


def _invalid_score(
    model: ConditionalModelSpec,
    cfg: Phase32EConfig,
    df: pd.DataFrame,
    window_seconds: int,
    lag_windows: int,
    reason: str,
) -> ConditionalScore:
    grid_max = (
        max(cfg.eps_grid)
        if cfg.eps_grid
        else 0.0
    )

    return ConditionalScore(
        model=model.name,
        label=model.label,
        family=model.family,
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_seconds=(
            int(window_seconds)
            * int(lag_windows)
        ),
        lag_label=_lag_label(lag_windows),
        lag_direction=_lag_direction(lag_windows),
        alignment_scope=_alignment_scope(
            cfg,
            window_seconds,
            lag_windows,
        ),
        valid=False,
        empirical_claim_allowed=_frame_claim_allowed(
            df,
            cfg,
        ),
        n_rows_total=len(df),
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
        eps_grid_initial_max=float(grid_max),
        eps_grid_final_max=float(grid_max),
        eps_grid_expansions=0,
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
        reason=str(reason),
    )


def _score_from_work(
    source_df: pd.DataFrame,
    work: pd.DataFrame,
    model: ConditionalModelSpec,
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
    ref_indices: np.ndarray,
    calib_indices: np.ndarray,
    test_indices: np.ndarray,
) -> ConditionalScore:
    current_columns = [
        resolve_component_column(
            component,
            work,
        )
        for component in model.current_components
    ]

    target_column = resolve_component_column(
        model.target_component,
        work,
    )

    estimate = _estimate_model(
        work,
        current_columns,
        target_column,
        ref_indices,
        calib_indices,
        test_indices,
        cfg,
    )

    n_current = len(estimate["current_map"])
    n_target = len(estimate["target_map"])

    n_params = (
        n_current
        * max(
            n_target - 1,
            1,
        )
    )

    n_test = len(test_indices)
    ll_test = float(estimate["ll_test"])

    return ConditionalScore(
        model=model.name,
        label=model.label,
        family=model.family,
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_seconds=(
            int(window_seconds)
            * int(lag_windows)
        ),
        lag_label=_lag_label(lag_windows),
        lag_direction=_lag_direction(lag_windows),
        alignment_scope=_alignment_scope(
            cfg,
            window_seconds,
            lag_windows,
        ),
        valid=True,
        empirical_claim_allowed=_frame_claim_allowed(
            source_df,
            cfg,
        ),
        n_rows_total=len(source_df),
        n_rows_model=len(work),
        n_train=len(ref_indices),
        n_calib=len(calib_indices),
        n_test=n_test,
        n_current_states=n_current,
        n_target_states=n_target,
        n_params=n_params,
        eps_train=0.0,
        eps_calib=float(
            estimate["eps_calib"]
        ),
        eps_candidate=float(
            estimate["eps_candidate"]
        ),
        eps_test=float(
            estimate["eps_test"]
        ),
        eps_grid_initial_max=float(
            estimate["eps_grid_initial_max"]
        ),
        eps_grid_final_max=float(
            estimate["eps_grid_final_max"]
        ),
        eps_grid_expansions=int(
            estimate["eps_grid_expansions"]
        ),
        eps_saturated=bool(
            estimate["eps_saturated"]
        ),
        ll_train=float(
            estimate["ll_train"]
        ),
        ll_calib=float(
            estimate["ll_calib"]
        ),
        ll_calib_eps0=float(
            estimate["ll_calib_eps0"]
        ),
        ll_calib_gain_from_eps=float(
            estimate["ll_calib_gain_from_eps"]
        ),
        ll_test=ll_test,
        ll_test_eps0=float(
            estimate["ll_test_eps0"]
        ),
        ll_test_gain_from_eps=float(
            estimate["ll_test_gain_from_eps"]
        ),
        bic_test=float(
            -2.0 * ll_test
            + n_params
            * math.log(max(n_test, 2))
        ),
        aic_test=float(
            -2.0 * ll_test
            + 2.0 * n_params
        ),
        transitions_per_current_state=float(
            len(ref_indices)
            / max(n_current, 1)
        ),
        reason="ok",
    )


def evaluate_conditional_model(
    df: pd.DataFrame,
    model: ConditionalModelSpec,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
    idx_train: Optional[Sequence[int]] = None,
    idx_test: Optional[Sequence[int]] = None,
) -> ConditionalScore:
    if cfg is None:
        cfg = load_phase3_2e_config()

    try:
        work, _, _ = build_model_dataframe(
            df,
            model,
        )

        work = work.reset_index(drop=True)

        min_ref = _get_cfg_int(
            cfg,
            "conditional_min_ref_rows",
            20,
        )

        min_calib = _get_cfg_int(
            cfg,
            "conditional_min_calib_rows",
            10,
        )

        min_test = _get_cfg_int(
            cfg,
            "conditional_min_test_rows",
            10,
        )

        if idx_train is None or idx_test is None:
            (
                ref_indices,
                calib_indices,
                test_indices,
            ) = make_subject_chronological_three_way_split(
                work,
                cfg,
                lag_windows,
                min_ref,
                min_calib,
                min_test,
            )
        else:
            (
                ref_indices,
                calib_indices,
            ) = split_existing_train_indices_into_ref_calib(
                idx_train,
                min_ref,
                min_calib,
            )

            test_indices = np.asarray(
                idx_test,
                dtype=int,
            )

        return _score_from_work(
            df,
            work,
            model,
            cfg,
            window_seconds,
            lag_windows,
            ref_indices,
            calib_indices,
            test_indices,
        )

    except Exception as exc:
        return _invalid_score(
            model,
            cfg,
            df,
            window_seconds,
            lag_windows,
            str(exc),
        )


def required_subject_consistency(
    n_subjects_evaluated: int,
    cfg: Phase32EConfig,
) -> Tuple[int, float]:
    n_subjects = max(
        int(n_subjects_evaluated),
        0,
    )

    minimum_count = max(
        _get_cfg_int(
            cfg,
            "min_positive_subjects",
            1,
        ),
        1,
    )

    if n_subjects <= 10:
        fraction = _get_cfg_float(
            cfg,
            "subject_effect_fraction",
            0.10,
        )

    elif n_subjects < 1000:
        fraction = _get_cfg_float(
            cfg,
            "large_cohort_subject_fraction_max",
            0.03,
        )

    else:
        fraction = _get_cfg_float(
            cfg,
            "large_cohort_subject_fraction_min",
            0.01,
        )

    required = max(
        minimum_count,
        int(
            math.ceil(
                n_subjects * fraction - 1e-12
            )
        ),
    )

    return required, fraction


def _evaluate_pair_core(
    df: pd.DataFrame,
    baseline_model: ConditionalModelSpec,
    augmented_model: ConditionalModelSpec,
    cfg: Phase32EConfig,
    window_seconds: int,
    lag_windows: int,
) -> Tuple[ConditionalScore, ConditionalScore]:
    common = build_pair_common_dataframe(
        df,
        baseline_model,
        augmented_model,
    ).reset_index(drop=True)

    (
        ref_indices,
        calib_indices,
        test_indices,
    ) = make_subject_chronological_three_way_split(
        common,
        cfg,
        lag_windows,
        _get_cfg_int(
            cfg,
            "conditional_min_ref_rows",
            20,
        ),
        _get_cfg_int(
            cfg,
            "conditional_min_calib_rows",
            10,
        ),
        _get_cfg_int(
            cfg,
            "conditional_min_test_rows",
            10,
        ),
    )

    baseline = _score_from_work(
        df,
        common,
        baseline_model,
        cfg,
        window_seconds,
        lag_windows,
        ref_indices,
        calib_indices,
        test_indices,
    )

    augmented = _score_from_work(
        df,
        common,
        augmented_model,
        cfg,
        window_seconds,
        lag_windows,
        ref_indices,
        calib_indices,
        test_indices,
    )

    return baseline, augmented


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

    n_evaluated = 0
    n_positive = 0

    grouped = df.groupby(
        "subject_id",
        sort=True,
        dropna=False,
    )

    for _, subject_df in grouped:
        try:
            (
                baseline,
                augmented,
            ) = _evaluate_pair_core(
                _sort_frame(
                    subject_df.copy()
                ),
                baseline_model,
                augmented_model,
                cfg,
                window_seconds,
                lag_windows,
            )

        except Exception:
            continue

        if (
            not baseline.valid
            or not augmented.valid
        ):
            continue

        n_evaluated += 1

        if (
            augmented.eps_test
            - baseline.eps_test
            > 0.0
        ):
            n_positive += 1

    fraction = (
        float(n_positive / n_evaluated)
        if n_evaluated > 0
        else 0.0
    )

    return (
        n_evaluated,
        n_positive,
        fraction,
    )


def evaluate_conditional_pair(
    df: pd.DataFrame,
    baseline_model: ConditionalModelSpec,
    augmented_model: ConditionalModelSpec,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
) -> Tuple[
    ConditionalScore,
    ConditionalScore,
    ConditionalPairScore,
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    try:
        (
            baseline,
            augmented,
        ) = _evaluate_pair_core(
            df,
            baseline_model,
            augmented_model,
            cfg,
            window_seconds,
            lag_windows,
        )

    except Exception as exc:
        baseline = _invalid_score(
            baseline_model,
            cfg,
            df,
            window_seconds,
            lag_windows,
            str(exc),
        )

        augmented = _invalid_score(
            augmented_model,
            cfg,
            df,
            window_seconds,
            lag_windows,
            str(exc),
        )

    valid = bool(
        baseline.valid
        and augmented.valid
    )

    if valid:
        (
            n_evaluated,
            n_positive,
            fraction_positive,
        ) = evaluate_subject_fraction_positive_lift(
            df,
            baseline_model,
            augmented_model,
            cfg,
            window_seconds,
            lag_windows,
        )
    else:
        n_evaluated = 0
        n_positive = 0
        fraction_positive = 0.0

    (
        required_count,
        required_fraction,
    ) = required_subject_consistency(
        n_evaluated,
        cfg,
    )

    pair = ConditionalPairScore(
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_seconds=(
            int(window_seconds)
            * int(lag_windows)
        ),
        lag_label=_lag_label(lag_windows),
        lag_direction=_lag_direction(lag_windows),
        alignment_scope=_alignment_scope(
            cfg,
            window_seconds,
            lag_windows,
        ),
        family=augmented_model.family,
        baseline_model=baseline_model.name,
        augmented_model=augmented_model.name,
        valid=valid,
        primary=pair_is_primary(
            cfg,
            window_seconds,
            lag_windows,
            baseline_model.name,
            augmented_model.name,
        ),
        multichannel=bool(
            model_uses_multichannel(
                baseline_model
            )
            or model_uses_multichannel(
                augmented_model
            )
        ),
        empirical_claim_allowed=bool(
            baseline.empirical_claim_allowed
            and augmented.empirical_claim_allowed
        ),
        baseline_eps_test=float(
            baseline.eps_test
        ),
        augmented_eps_test=float(
            augmented.eps_test
        ),
        conditional_lift=float(
            augmented.eps_test
            - baseline.eps_test
        ),
        baseline_ll_test=float(
            baseline.ll_test
        ),
        augmented_ll_test=float(
            augmented.ll_test
        ),
        ll_improvement=float(
            augmented.ll_test
            - baseline.ll_test
        ),
        baseline_bic_test=float(
            baseline.bic_test
        ),
        augmented_bic_test=float(
            augmented.bic_test
        ),
        bic_improvement=float(
            baseline.bic_test
            - augmented.bic_test
        ),
        baseline_aic_test=float(
            baseline.aic_test
        ),
        augmented_aic_test=float(
            augmented.aic_test
        ),
        aic_improvement=float(
            baseline.aic_test
            - augmented.aic_test
        ),
        n_subjects_evaluated=int(
            n_evaluated
        ),
        n_subjects_positive_lift=int(
            n_positive
        ),
        fraction_positive_lift=float(
            fraction_positive
        ),
        required_positive_subjects=int(
            required_count
        ),
        required_positive_fraction=float(
            required_fraction
        ),
        subject_consistency_pass=bool(
            n_evaluated > 0
            and n_positive >= required_count
        ),
        baseline_eps_saturated=bool(
            baseline.eps_saturated
        ),
        augmented_eps_saturated=bool(
            augmented.eps_saturated
        ),
        reason=(
            "ok"
            if valid
            else (
                f"baseline={baseline.reason}; "
                f"augmented={augmented.reason}"
            )
        ),
    )

    return (
        baseline,
        augmented,
        pair,
    )


def evaluate_conditional_models_for_frame(
    df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
    window_seconds: int = 0,
    lag_windows: int = 0,
) -> Tuple[
    List[ConditionalScore],
    List[ConditionalPairScore],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    models = conditional_model_map(cfg)

    scores: Dict[
        str,
        ConditionalScore,
    ] = {}

    pairs: List[
        ConditionalPairScore
    ] = []

    for (
        baseline_name,
        augmented_name,
    ) in active_conditional_pairs(cfg):
        (
            baseline,
            augmented,
            pair,
        ) = evaluate_conditional_pair(
            df,
            models[baseline_name],
            models[augmented_name],
            cfg,
            window_seconds,
            lag_windows,
        )

        scores[baseline.model] = baseline
        scores[augmented.model] = augmented
        pairs.append(pair)

    return (
        list(scores.values()),
        pairs,
    )


def evaluate_all_conditional_from_lagged_csvs(
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print(
        "[Phase3.2E Conditional] "
        f"version={CONDITIONAL_VERSION}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"windows={list(_registered_windows(cfg))} "
        f"lags={list(_registered_lags(cfg))}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"primary_windows={list(_primary_windows(cfg))} "
        f"primary_lags={list(_primary_lags(cfg))} "
        f"primary_pairs={list(_primary_pairs(cfg))}"
    )

    model_rows: List[
        Dict[str, Any]
    ] = []

    pair_rows: List[
        Dict[str, Any]
    ] = []

    for window_seconds in _registered_windows(cfg):
        for lag_windows in _registered_lags(cfg):
            path = lagged_frame_path(
                cfg,
                window_seconds,
                lag_windows,
            )

            if not path.exists():
                print(
                    "[Phase3.2E Conditional] "
                    f"missing frame "
                    f"window={window_seconds}s "
                    f"lag={lag_windows}: "
                    f"{path.name}"
                )
                continue

            frame = pd.read_csv(path)

            if frame.empty:
                print(
                    "[Phase3.2E Conditional] "
                    f"empty frame "
                    f"window={window_seconds}s "
                    f"lag={lag_windows}"
                )
                continue

            print(
                "[Phase3.2E Conditional] "
                f"evaluating "
                f"window={window_seconds}s "
                f"lag={lag_windows} "
                f"rows={len(frame)}"
            )

            (
                model_scores,
                pair_scores,
            ) = evaluate_conditional_models_for_frame(
                _sort_frame(frame),
                cfg,
                window_seconds,
                lag_windows,
            )

            model_rows.extend(
                asdict(score)
                for score in model_scores
            )

            pair_rows.extend(
                asdict(score)
                for score in pair_scores
            )

    model_df = pd.DataFrame(model_rows)
    pair_df = pd.DataFrame(pair_rows)

    if not model_df.empty:
        model_df = (
            model_df.sort_values(
                [
                    "window_seconds",
                    "lag_windows",
                    "family",
                    "model",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    if not pair_df.empty:
        pair_df = (
            pair_df.sort_values(
                [
                    "window_seconds",
                    "lag_windows",
                    "family",
                    "augmented_model",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return model_df, pair_df


def _row_payload(
    row: Optional[pd.Series],
) -> Dict[str, Any]:
    if row is None:
        return {}

    keys = (
        "window_seconds",
        "lag_windows",
        "lag_seconds",
        "lag_direction",
        "alignment_scope",
        "family",
        "baseline_model",
        "augmented_model",
        "conditional_lift",
        "baseline_eps_test",
        "augmented_eps_test",
        "ll_improvement",
        "bic_improvement",
        "aic_improvement",
        "n_subjects_evaluated",
        "n_subjects_positive_lift",
        "fraction_positive_lift",
        "required_positive_subjects",
        "subject_consistency_pass",
        "primary",
        "multichannel",
        "empirical_claim_allowed",
        "augmented_eps_saturated",
    )

    return {
        key: _json_default(row.get(key))
        for key in keys
    }


def summarize_conditional_pairs(
    pair_scores_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    if pair_scores_df.empty:
        return {
            "available": False,
            "status": "conditional_no_results",
            "protocol_only": False,
            "reason": "empty_pair_scores",
            "can_proceed_to_controls": False,
            "can_proceed_to_metrics": True,
            "can_proceed_to_empirical_cdr": False,
            "can_claim_empirical_cdr": False,
        }

    valid = pair_scores_df[
        _as_bool_series(
            pair_scores_df["valid"]
        )
    ].copy()

    if valid.empty:
        return {
            "available": False,
            "status": "conditional_no_valid_pairs",
            "protocol_only": False,
            "reason": "no_valid_pair_scores",
            "n_rows": int(len(pair_scores_df)),
            "can_proceed_to_controls": False,
            "can_proceed_to_metrics": True,
            "can_proceed_to_empirical_cdr": False,
            "can_claim_empirical_cdr": False,
        }

    primary = valid[
        _as_bool_series(
            valid["primary"]
        )
    ].copy()

    positive = valid[
        pd.to_numeric(
            valid["conditional_lift"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
    ].copy()

    primary_positive = primary[
        pd.to_numeric(
            primary["conditional_lift"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
    ].copy()

    exploratory_positive = positive[
        ~_as_bool_series(
            positive["primary"]
        )
    ].copy()

    multichannel = valid[
        _as_bool_series(
            valid["multichannel"]
        )
    ].copy()

    multichannel_positive = multichannel[
        pd.to_numeric(
            multichannel["conditional_lift"],
            errors="coerce",
        )
        .fillna(0.0)
        .gt(0.0)
    ].copy()

    strong_mask = (
        pd.to_numeric(
            valid["conditional_lift"],
            errors="coerce",
        )
        .fillna(0.0)
        .ge(float(cfg.conditional_lift_min))
        & pd.to_numeric(
            valid["augmented_eps_test"],
            errors="coerce",
        )
        .fillna(0.0)
        .ge(float(cfg.strong_eps_min))
        & _as_bool_series(
            valid["subject_consistency_pass"]
        )
        & pd.to_numeric(
            valid["bic_improvement"],
            errors="coerce",
        )
        .fillna(float("-inf"))
        .ge(0.0)
    )

    strong = valid[strong_mask].copy()

    primary_strong = strong[
        _as_bool_series(
            strong["primary"]
        )
    ].copy()

    saturated = valid[
        _as_bool_series(
            valid["baseline_eps_saturated"]
        )
        | _as_bool_series(
            valid["augmented_eps_saturated"]
        )
    ].copy()

    best_lift = (
        valid.sort_values(
            "conditional_lift",
            ascending=False,
        ).iloc[0]
    )

    best_primary = (
        primary.sort_values(
            "conditional_lift",
            ascending=False,
        ).iloc[0]
        if not primary.empty
        else None
    )

    best_bic = (
        valid.sort_values(
            "bic_improvement",
            ascending=False,
        ).iloc[0]
    )

    best_ll = (
        valid.sort_values(
            "ll_improvement",
            ascending=False,
        ).iloc[0]
    )

    best_multichannel = (
        multichannel.sort_values(
            "conditional_lift",
            ascending=False,
        ).iloc[0]
        if not multichannel.empty
        else None
    )

    if not primary_positive.empty:
        status = "conditional_primary_leads"
        interpretation = (
            f"Conditional CDR identified "
            f"{len(primary_positive)} primary "
            f"positive-lift row(s) within "
            f"{len(positive)} positive row(s) overall. "
            f"These are primary leads, not yet strong claims, "
            f"and must proceed through negative controls, "
            f"F7/F8/F12 diagnostics and final gates."
        )

    elif not positive.empty:
        status = "conditional_exploratory_leads"
        interpretation = (
            f"Conditional CDR identified "
            f"{len(exploratory_positive)} exploratory "
            f"positive-lift row(s) and no primary positive row. "
            f"These results may proceed to controls and "
            f"diagnostics but do not support confirmatory "
            f"interpretation."
        )

    else:
        status = "conditional_no_positive_lift"
        interpretation = (
            "Conditional CDR completed on the registered "
            "synchronized window/lag grid without a positive "
            "conditional-lift row. Controls and metrics may "
            "still be generated for a complete null analysis."
        )

    source_claim_allowed = bool(
        _as_bool_series(
            valid["empirical_claim_allowed"]
        ).all()
    )

    return {
        "available": True,
        "status": status,
        "protocol_only": False,
        "module": CONDITIONAL_MODULE,
        "conditional_version": CONDITIONAL_VERSION,
        "config_fingerprint": _config_fingerprint(cfg),
        "cache_policy": (
            "always_rebuild_conditional_outputs"
        ),
        "n_rows": int(len(pair_scores_df)),
        "n_valid_rows": int(len(valid)),
        "n_primary_rows": int(len(primary)),
        "n_exploratory_rows": int(
            len(valid) - len(primary)
        ),
        "n_positive_lift_rows": int(
            len(positive)
        ),
        "n_primary_positive_lift_rows": int(
            len(primary_positive)
        ),
        "n_exploratory_positive_lift_rows": int(
            len(exploratory_positive)
        ),
        "n_strong_candidate_rows": int(
            len(strong)
        ),
        "n_primary_strong_candidate_rows": int(
            len(primary_strong)
        ),
        "n_multichannel_rows": int(
            len(multichannel)
        ),
        "n_multichannel_positive_lift_rows": int(
            len(multichannel_positive)
        ),
        "n_subject_consistency_pass_rows": int(
            _as_bool_series(
                valid["subject_consistency_pass"]
            ).sum()
        ),
        "n_epsilon_saturation_rows": int(
            len(saturated)
        ),
        "primary_pair_rows": int(
            len(primary)
        ),
        "positive_lift_rows": int(
            len(positive)
        ),
        "primary_positive_lift_rows": int(
            len(primary_positive)
        ),
        "strong_candidate_rows": int(
            len(strong)
        ),
        "primary_strong_candidate_rows": int(
            len(primary_strong)
        ),
        "multichannel_rows": int(
            len(multichannel)
        ),
        "multichannel_positive_lift_rows": int(
            len(multichannel_positive)
        ),
        "best_by_conditional_lift": _row_payload(
            best_lift
        ),
        "best_primary_by_conditional_lift": _row_payload(
            best_primary
        ),
        "best_by_bic_improvement": _row_payload(
            best_bic
        ),
        "best_by_ll_improvement": _row_payload(
            best_ll
        ),
        "best_multichannel_by_conditional_lift": _row_payload(
            best_multichannel
        ),
        "positive_lift_records": (
            _records_with_strong_diagnostics(
                positive,
                cfg,
            )
        ),
        "primary_positive_lift_records": (
            _records_with_strong_diagnostics(
                primary_positive,
                cfg,
            )
        ),
        "exploratory_positive_lift_records": (
            _records_with_strong_diagnostics(
                exploratory_positive,
                cfg,
            )
        ),
        "strong_candidate_records": (
            strong.to_dict(orient="records")
        ),
        "primary_strong_candidate_records": (
            primary_strong.to_dict(
                orient="records"
            )
        ),
        "epsilon_saturation_records": (
            saturated.to_dict(
                orient="records"
            )
        ),
        "registered_windows": list(
            _registered_windows(cfg)
        ),
        "primary_windows": list(
            _primary_windows(cfg)
        ),
        "registered_lags": list(
            _registered_lags(cfg)
        ),
        "primary_lags": list(
            _primary_lags(cfg)
        ),
        "active_conditional_pairs": [
            list(pair)
            for pair in active_conditional_pairs(cfg)
        ],
        "primary_test_pairs": [
            list(pair)
            for pair in _primary_pairs(cfg)
        ],
        "max_baseline_eps_test": float(
            pd.to_numeric(
                valid["baseline_eps_test"],
                errors="coerce",
            )
            .fillna(0.0)
            .max()
        ),
        "max_augmented_eps_test": float(
            pd.to_numeric(
                valid["augmented_eps_test"],
                errors="coerce",
            )
            .fillna(0.0)
            .max()
        ),
        "max_conditional_lift": float(
            pd.to_numeric(
                valid["conditional_lift"],
                errors="coerce",
            )
            .fillna(0.0)
            .max()
        ),
        "source_claim_allowed": source_claim_allowed,
        "can_proceed_to_controls": True,
        "can_proceed_to_metrics": True,
        "can_proceed_to_empirical_cdr": bool(
            len(primary) > 0
        ),
        "can_claim_empirical_cdr": False,
        "interpretation": interpretation,
    }


def build_conditional_summary_text(
    summary: Mapping[str, Any],
) -> str:
    lines = [
        "=" * 78,
        "Phase III.2E — Conditional CDR Summary",
        (
            "Leakage-safe synchronized "
            "ref/calib/test estimator"
        ),
        "=" * 78,
        "",
        f"status: {summary.get('status')}",
        f"protocol_only: {summary.get('protocol_only')}",
        f"available: {summary.get('available')}",
        (
            f"conditional_version: "
            f"{summary.get('conditional_version')}"
        ),
        (
            f"config_fingerprint: "
            f"{summary.get('config_fingerprint')}"
        ),
        (
            f"cache_policy: "
            f"{summary.get('cache_policy')}"
        ),
    ]

    if not summary.get("available"):
        lines.extend(
            [
                f"reason: {summary.get('reason')}",
                "",
                "=" * 78,
            ]
        )

        return "\n".join(lines)

    lines.extend(
        [
            f"pair_rows: {summary.get('n_rows')}",
            (
                f"valid_pair_rows: "
                f"{summary.get('n_valid_rows')}"
            ),
            (
                f"primary_pair_rows: "
                f"{summary.get('n_primary_rows')}"
            ),
            (
                f"exploratory_pair_rows: "
                f"{summary.get('n_exploratory_rows')}"
            ),
            (
                f"positive_lift_rows: "
                f"{summary.get('n_positive_lift_rows')}"
            ),
            (
                f"primary_positive_lift_rows: "
                f"{summary.get('n_primary_positive_lift_rows')}"
            ),
            (
                f"exploratory_positive_lift_rows: "
                f"{summary.get('n_exploratory_positive_lift_rows')}"
            ),
            (
                f"strong_candidate_rows: "
                f"{summary.get('n_strong_candidate_rows')}"
            ),
            (
                f"multichannel_rows: "
                f"{summary.get('n_multichannel_rows')}"
            ),
            (
                f"multichannel_positive_lift_rows: "
                f"{summary.get('n_multichannel_positive_lift_rows')}"
            ),
            (
                f"subject_consistency_pass_rows: "
                f"{summary.get('n_subject_consistency_pass_rows')}"
            ),
            (
                f"epsilon_saturation_rows: "
                f"{summary.get('n_epsilon_saturation_rows')}"
            ),
            (
                f"max_baseline_eps_test: "
                f"{summary.get('max_baseline_eps_test')}"
            ),
            (
                f"max_augmented_eps_test: "
                f"{summary.get('max_augmented_eps_test')}"
            ),
            (
                f"max_conditional_lift: "
                f"{summary.get('max_conditional_lift')}"
            ),
            (
                f"can_proceed_to_controls: "
                f"{summary.get('can_proceed_to_controls')}"
            ),
            (
                f"can_proceed_to_metrics: "
                f"{summary.get('can_proceed_to_metrics')}"
            ),
            (
                f"can_proceed_to_empirical_cdr: "
                f"{summary.get('can_proceed_to_empirical_cdr')}"
            ),
            (
                f"can_claim_empirical_cdr: "
                f"{summary.get('can_claim_empirical_cdr')}"
            ),
            "",
            (
                f"interpretation: "
                f"{summary.get('interpretation')}"
            ),
            "=" * 78,
        ]
    )

    return "\n".join(lines)


def save_conditional_outputs(
    model_scores_df: pd.DataFrame,
    pair_scores_df: pd.DataFrame,
    summary: Mapping[str, Any],
    alignment_report: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    model_path = Path(
        cfg.conditional_model_scores_csv
    )

    pair_path = Path(
        cfg.conditional_pair_scores_csv
    )

    summary_json_path = Path(
        cfg.conditional_summary_json
    )

    summary_txt_path = Path(
        cfg.conditional_summary_txt
    )

    report_json_path = Path(
        getattr(
            cfg,
            "conditional_report_json",
            (
                Path(cfg.results_dir)
                / "phase3_2e_conditional_report.json"
            ),
        )
    )

    for path in (
        model_path,
        pair_path,
        summary_json_path,
        summary_txt_path,
        report_json_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    model_scores_df.to_csv(
        model_path,
        index=False,
    )

    pair_scores_df.to_csv(
        pair_path,
        index=False,
    )

    summary_payload = dict(summary)

    summary_payload["config_fingerprint"] = (
        _config_fingerprint(cfg)
    )

    summary_payload["cache_policy"] = (
        "always_rebuild_conditional_outputs"
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": CONDITIONAL_MODULE,
        "conditional_version": CONDITIONAL_VERSION,
        "created_at": _now_str(),
        "status": summary_payload.get("status"),
        "cache_policy": (
            "always_rebuild_conditional_outputs"
        ),
        "config_fingerprint": _config_fingerprint(cfg),
        "config_signature": _config_signature_payload(
            cfg
        ),
        "alignment_signature": _alignment_signature(
            alignment_report
        ),
        "summary": summary_payload,
        "alignment_status": alignment_report.get(
            "status"
        ),
        "alignment_result": alignment_report.get(
            "result",
            {},
        ),
        "model_scores_file": str(model_path),
        "pair_scores_file": str(pair_path),
        "summary_json_file": str(
            summary_json_path
        ),
        "summary_text_file": str(
            summary_txt_path
        ),
    }

    save_json(
        summary_json_path,
        summary_payload,
    )

    save_json(
        report_json_path,
        report,
    )

    summary_txt_path.write_text(
        build_conditional_summary_text(
            summary_payload
        ),
        encoding="utf-8",
    )

    print(
        "[Phase3.2E Conditional] "
        f"Saved model scores: {model_path}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"Saved pair scores: {pair_path}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"Saved summary JSON: {summary_json_path}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"Saved report JSON: {report_json_path}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"Saved summary TXT: {summary_txt_path}"
    )

    return report


def run_phase3_2e_conditional(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
    force_rebuild_features: bool = False,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print(
        "[Phase3.2E Conditional] "
        "cache_policy=always_rebuild_conditional_outputs"
    )

    print(
        "[Phase3.2E Conditional] "
        f"config_fingerprint={_config_fingerprint(cfg)}"
    )

    (
        _,
        alignment_report,
    ) = load_or_build_phase3_2e_alignment(
        cfg=cfg,
        force_rebuild=force_rebuild_alignment,
        force_rebuild_features=force_rebuild_features,
        force_rebuild_windows=force_rebuild_windows,
        force_rebuild_loader=force_rebuild_loader,
    )

    alignment_result = alignment_report.get(
        "result",
        {},
    )

    can_proceed = bool(
        alignment_result.get(
            "can_proceed_to_conditional_cdr",
            False,
        )
    )

    if can_proceed:
        (
            model_df,
            pair_df,
        ) = evaluate_all_conditional_from_lagged_csvs(
            cfg
        )

        summary = summarize_conditional_pairs(
            pair_df,
            cfg,
        )

    else:
        model_df = pd.DataFrame()
        pair_df = pd.DataFrame()

        protocol_only = bool(
            alignment_result.get(
                "protocol_only",
                False,
            )
        )

        summary = {
            "available": False,
            "status": (
                cfg.status_pending_synchronized_data
                if protocol_only
                else "conditional_blocked_by_alignment"
            ),
            "protocol_only": protocol_only,
            "reason": (
                "alignment_not_ready_for_conditional_cdr"
            ),
            "conditional_version": CONDITIONAL_VERSION,
            "config_fingerprint": _config_fingerprint(cfg),
            "cache_policy": (
                "always_rebuild_conditional_outputs"
            ),
            "can_proceed_to_controls": False,
            "can_proceed_to_metrics": True,
            "can_proceed_to_empirical_cdr": False,
            "can_claim_empirical_cdr": False,
            "interpretation": (
                "Conditional CDR was not executed because "
                "alignment is not ready."
            ),
        }

    report = save_conditional_outputs(
        model_df,
        pair_df,
        summary,
        alignment_report,
        cfg,
    )

    return (
        model_df,
        pair_df,
        report,
    )


def load_or_build_phase3_2e_conditional(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    force_rebuild_alignment: bool = False,
    force_rebuild_features: bool = False,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    return run_phase3_2e_conditional(
        cfg=cfg,
        force_rebuild_alignment=bool(
            force_rebuild
            or force_rebuild_alignment
        ),
        force_rebuild_features=bool(
            force_rebuild
            or force_rebuild_features
        ),
        force_rebuild_windows=bool(
            force_rebuild
            or force_rebuild_windows
        ),
        force_rebuild_loader=bool(
            force_rebuild
            or force_rebuild_loader
        ),
    )


build_phase3_2e_conditional = (
    run_phase3_2e_conditional
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase III.2E synchronized "
            "conditional CDR."
        )
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-alignment",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-features",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-windows",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-loader",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase3_2e_config()

    print(
        "[Phase3.2E Conditional] "
        f"executing={Path(__file__).resolve()}"
    )

    print(
        "[Phase3.2E Conditional] "
        f"version={CONDITIONAL_VERSION}"
    )

    (
        model_df,
        pair_df,
        report,
    ) = run_phase3_2e_conditional(
        cfg=cfg,
        force_rebuild_alignment=bool(
            args.force_rebuild
            or args.force_rebuild_alignment
        ),
        force_rebuild_features=bool(
            args.force_rebuild
            or args.force_rebuild_features
        ),
        force_rebuild_windows=bool(
            args.force_rebuild
            or args.force_rebuild_windows
        ),
        force_rebuild_loader=bool(
            args.force_rebuild
            or args.force_rebuild_loader
        ),
    )

    summary = report.get(
        "summary",
        {},
    )

    print("\n" + "=" * 78)
    print(
        "Phase III.2E conditional CDR completed"
    )
    print(
        "Leakage-safe synchronized "
        "ref/calib/test estimator"
    )
    print("=" * 78)

    print(
        f"status: {summary.get('status')}"
    )

    print(
        f"protocol_only: "
        f"{summary.get('protocol_only')}"
    )

    print(
        f"model_rows: {len(model_df)}"
    )

    print(
        f"pair_rows: {len(pair_df)}"
    )

    print(
        f"valid_pair_rows: "
        f"{summary.get('n_valid_rows')}"
    )

    print(
        f"primary_pair_rows: "
        f"{summary.get('n_primary_rows')}"
    )

    print(
        f"exploratory_pair_rows: "
        f"{summary.get('n_exploratory_rows')}"
    )

    print(
        f"positive_lift_rows: "
        f"{summary.get('n_positive_lift_rows')}"
    )

    print(
        f"primary_positive_lift_rows: "
        f"{summary.get('n_primary_positive_lift_rows')}"
    )

    print(
        f"exploratory_positive_lift_rows: "
        f"{summary.get('n_exploratory_positive_lift_rows')}"
    )

    print(
        f"strong_candidate_rows: "
        f"{summary.get('n_strong_candidate_rows')}"
    )

    print(
        f"multichannel_rows: "
        f"{summary.get('n_multichannel_rows')}"
    )

    print(
        f"multichannel_positive_lift_rows: "
        f"{summary.get('n_multichannel_positive_lift_rows')}"
    )

    print(
        f"max_baseline_eps_test: "
        f"{summary.get('max_baseline_eps_test')}"
    )

    print(
        f"max_augmented_eps_test: "
        f"{summary.get('max_augmented_eps_test')}"
    )

    print(
        f"max_conditional_lift: "
        f"{summary.get('max_conditional_lift')}"
    )

    print(
        f"can_proceed_to_controls: "
        f"{summary.get('can_proceed_to_controls')}"
    )

    print(
        f"can_proceed_to_metrics: "
        f"{summary.get('can_proceed_to_metrics')}"
    )

    print(
        f"can_proceed_to_empirical_cdr: "
        f"{summary.get('can_proceed_to_empirical_cdr')}"
    )

    print(
        f"can_claim_empirical_cdr: "
        f"{summary.get('can_claim_empirical_cdr')}"
    )

    print(
        f"interpretation: "
        f"{summary.get('interpretation')}"
    )

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()