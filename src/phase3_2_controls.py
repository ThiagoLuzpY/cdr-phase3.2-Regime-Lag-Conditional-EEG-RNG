from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import Phase32Config, load_phase3_2_config
from src.phase3_2_conditional import (
    evaluate_conditional_models_for_frame,
)
from src.phase3_2_lagging import lag_label, lagged_csv_path
from src.phase3_2_regimes import save_json


# =========================================================
# Phase III.2 — Negative Controls
# Leakage-safe conditional estimator version
# =========================================================
#
# Purpose:
#
#   Verify that conditional EEG-RNG effects collapse when temporal, subject,
#   RNG, EEG or target structure is broken.
#
# Important patch:
#
#   The leakage-safe conditional estimator can still produce high eps values
#   in baseline or augmented models. Therefore controls must not fail merely
#   because eps is high.
#
#   Controls now focus on:
#
#       - conditional_lift
#       - subject-level positive lift fraction
#       - BIC improvement
#       - LL improvement
#       - strong candidate persistence
#       - epsilon saturation as diagnostic, not direct failure criterion
#
# A valid positive Phase III.2 result should not survive these controls.


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class ControlRunResult:
    control_type: str
    replicate: int
    regime: str
    lag_epochs: int
    lag_label: str
    valid: bool

    n_rows: int
    n_subjects: int

    max_baseline_eps_test: float
    max_augmented_eps_test: float
    max_conditional_lift: float
    max_fraction_positive_lift: float
    max_bic_improvement: float
    max_ll_improvement: float

    n_positive_lift_rows: int
    n_strong_candidate_rows: int

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

    fraction_runs_below_control_tol: float
    fraction_runs_collapsed_overall: float
    control_tol: float

    passed: bool
    reason: str


# =========================================================
# Generic helpers
# =========================================================

def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


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


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _cfg_float(cfg: Phase32Config, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


# =========================================================
# Column groups
# =========================================================

def rng_current_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "rng_state_model",
        "rng_bit_model",
        "rng_state_aligned",
        "rng_bit_aligned",
        "rng_info_bin_model",
        "rng_info_bin_aligned",
        "q_rng_bin_model",
        "q_rng_bin_aligned",
        "rng_window_id_aligned",
        "rng_window_start_aligned",
        "rng_window_end_aligned",
        "bit_balance_local_aligned",
        "transition_rate_aligned",
        "rng_entropy_local_aligned",
        "entropy_rate_proxy_aligned",
        "run_length_mean_aligned",
        "run_length_max_aligned",
        "run_length_std_aligned",
        "compressibility_proxy_aligned",
        "surprise_index_aligned",
        "transition_asymmetry_aligned",
        "micro_cluster_deviation_aligned",
        "q_rng_score_aligned",
        "observed_joint_state_lagged",
        "informational_joint_state_lagged",
        "latent_q_state_lagged",
    ]

    return [c for c in candidates if c in df.columns]


def eeg_current_columns(df: pd.DataFrame) -> List[str]:
    candidates = [
        "eeg_state",
        "delta_power_bin",
        "alpha_power_bin",
        "eeg_info_bin",
        "latent_state",
        "sleep_stage_code",
        "is_wake",
        "is_sleep",
        "is_n1",
        "is_n2",
        "is_n3",
        "is_rem",
        "is_stable_sleep",
        "is_deep_sleep",
    ]

    return [c for c in candidates if c in df.columns]


# =========================================================
# Shuffle utilities
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

    _require_columns(out, group_cols, context="grouped shuffle")

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
) -> pd.DataFrame:
    out = df.copy()

    if not columns:
        return out

    _require_columns(out, group_cols, context="circular shift")

    for _, idx in out.groupby(list(group_cols), sort=False).groups.items():
        idx_list = list(idx)
        n = len(idx_list)

        if n <= 2:
            continue

        shift = int(rng.integers(1, max(n, 2)))

        for col in columns:
            if col not in out.columns:
                continue

            arr = out.loc[idx_list, col].to_numpy(copy=True)
            arr = np.roll(arr, shift)
            out.loc[idx_list, col] = arr

    return out


def _subject_mismatch_rng(
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
# State rebuilding after controls
# =========================================================

def rebuild_control_joint_states(
    df: pd.DataFrame,
    cfg: Phase32Config,
) -> pd.DataFrame:
    out = df.copy()

    if "eeg_state" in out.columns and "rng_state_model" in out.columns:
        valid = out["eeg_state"].notna() & out["rng_state_model"].notna()

        out.loc[valid, "observed_joint_state_lagged"] = (
            out.loc[valid, "eeg_state"].astype(int) * int(cfg.rng_n_bins)
            + out.loc[valid, "rng_state_model"].astype(int)
        )

    if "eeg_info_bin" in out.columns and "rng_info_bin_model" in out.columns:
        valid = out["eeg_info_bin"].notna() & out["rng_info_bin_model"].notna()

        out.loc[valid, "informational_joint_state_lagged"] = (
            out.loc[valid, "eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
            + out.loc[valid, "rng_info_bin_model"].astype(int)
        )

    if "latent_state" in out.columns and "q_rng_bin_model" in out.columns:
        valid = out["latent_state"].notna() & out["q_rng_bin_model"].notna()

        out.loc[valid, "latent_q_state_lagged"] = (
            out.loc[valid, "latent_state"].astype(int) * int(cfg.q_n_bins)
            + out.loc[valid, "q_rng_bin_model"].astype(int)
        )

    return out


# =========================================================
# Control application
# =========================================================

def apply_control(
    df: pd.DataFrame,
    control_type: str,
    cfg: Optional[Phase32Config] = None,
    seed: int = 0,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_phase3_2_config()

    out = _sort_frame(df.copy())
    rng = np.random.default_rng(int(seed))

    group_cols = ["subject_id", "recording_id"]

    if control_type == "within_subject_rng_shuffle":
        cols = rng_current_columns(out)
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

    elif control_type == "circular_rng_shift":
        cols = rng_current_columns(out)
        out = _circular_shift_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    elif control_type == "subject_mismatch":
        cols = rng_current_columns(out)
        out = _subject_mismatch_rng(
            df=out,
            columns=cols,
            rng=rng,
        )

    elif control_type == "stage_preserving_rng_shuffle":
        cols = rng_current_columns(out)

        if "sleep_stage" in out.columns:
            out = _shuffle_columns_within_groups(
                df=out,
                columns=cols,
                group_cols=["subject_id", "recording_id", "sleep_stage"],
                rng=rng,
            )
        else:
            out = _shuffle_columns_within_groups(
                df=out,
                columns=cols,
                group_cols=group_cols,
                rng=rng,
            )

    elif control_type == "conditional_target_shuffle":
        cols = [
            c for c in ["eeg_next_state", "rng_next_state_model"]
            if c in out.columns
        ]

        out = _shuffle_columns_within_groups(
            df=out,
            columns=cols,
            group_cols=group_cols,
            rng=rng,
        )

    else:
        raise ValueError(f"Unknown control_type: {control_type}")

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
    cfg: Phase32Config,
) -> Tuple[int, float]:
    if valid_df.empty:
        return 0, 0.0

    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)

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


def _strong_control_candidates(valid_df: pd.DataFrame, cfg: Phase32Config) -> pd.DataFrame:
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
    regime: str,
    lag_epochs: int,
    n_rows: int,
    n_subjects: int,
    cfg: Phase32Config,
) -> ControlRunResult:
    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)

    if pair_scores_df.empty:
        return ControlRunResult(
            control_type=control_type,
            replicate=int(replicate),
            regime=regime,
            lag_epochs=int(lag_epochs),
            lag_label=lag_label(int(lag_epochs)),
            valid=False,
            n_rows=int(n_rows),
            n_subjects=int(n_subjects),
            max_baseline_eps_test=0.0,
            max_augmented_eps_test=0.0,
            max_conditional_lift=0.0,
            max_fraction_positive_lift=0.0,
            max_bic_improvement=0.0,
            max_ll_improvement=0.0,
            n_positive_lift_rows=0,
            n_strong_candidate_rows=0,
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
            regime=regime,
            lag_epochs=int(lag_epochs),
            lag_label=lag_label(int(lag_epochs)),
            valid=False,
            n_rows=int(n_rows),
            n_subjects=int(n_subjects),
            max_baseline_eps_test=0.0,
            max_augmented_eps_test=0.0,
            max_conditional_lift=0.0,
            max_fraction_positive_lift=0.0,
            max_bic_improvement=0.0,
            max_ll_improvement=0.0,
            n_positive_lift_rows=0,
            n_strong_candidate_rows=0,
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
        regime=regime,
        lag_epochs=int(lag_epochs),
        lag_label=lag_label(int(lag_epochs)),
        valid=True,
        n_rows=int(n_rows),
        n_subjects=int(n_subjects),
        max_baseline_eps_test=float(baseline_eps.max()) if len(baseline_eps) else 0.0,
        max_augmented_eps_test=float(augmented_eps.max()) if len(augmented_eps) else 0.0,
        max_conditional_lift=max_lift,
        max_fraction_positive_lift=max_frac,
        max_bic_improvement=max_bic,
        max_ll_improvement=max_ll,
        n_positive_lift_rows=int(len(positive_lift)),
        n_strong_candidate_rows=int(len(strong_candidates)),
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
    regime: str,
    lag_epochs: int,
    replicate: int,
    cfg: Optional[Phase32Config] = None,
) -> Tuple[ControlRunResult, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2_config()

    seed = int(cfg.control_seed + replicate * 1009 + abs(int(lag_epochs)) * 17)

    controlled = apply_control(
        df=df,
        control_type=control_type,
        cfg=cfg,
        seed=seed,
    )

    _, pair_scores = evaluate_conditional_models_for_frame(
        df=controlled,
        cfg=cfg,
        regime_name=regime,
        lag_epochs=int(lag_epochs),
    )

    pair_scores_df = pd.DataFrame(_as_records(pair_scores))

    result = _control_run_from_pair_scores(
        pair_scores_df=pair_scores_df,
        control_type=control_type,
        replicate=int(replicate),
        regime=regime,
        lag_epochs=int(lag_epochs),
        n_rows=int(len(controlled)),
        n_subjects=int(controlled["subject_id"].nunique()) if "subject_id" in controlled.columns else 0,
        cfg=cfg,
    )

    if not pair_scores_df.empty:
        pair_scores_df["control_type"] = str(control_type)
        pair_scores_df["replicate"] = int(replicate)
        pair_scores_df["control_seed"] = int(seed)
        pair_scores_df["control_collapsed_overall"] = bool(result.collapsed_overall)

    return result, pair_scores_df


# =========================================================
# Control experiment selection
# =========================================================

def control_regime_order(cfg: Phase32Config) -> List[str]:
    order: List[str] = []

    for name in list(cfg.primary_regimes) + ["full"]:
        if name not in order:
            order.append(name)

    return order


def control_lag_order(cfg: Phase32Config) -> List[int]:
    order: List[int] = []

    for lag_value in list(cfg.primary_lags_epochs):
        if int(lag_value) not in order:
            order.append(int(lag_value))

    return order


def available_control_frames(
    cfg: Optional[Phase32Config] = None,
) -> List[Tuple[str, int, Path]]:
    if cfg is None:
        cfg = load_phase3_2_config()

    frames: List[Tuple[str, int, Path]] = []

    for regime in control_regime_order(cfg):
        for lag_value in control_lag_order(cfg):
            path = lagged_csv_path(
                regime_name=regime,
                lag_epochs=int(lag_value),
                cfg=cfg,
            )

            if path.exists():
                frames.append((regime, int(lag_value), path))

    return frames


# =========================================================
# Batch control evaluation
# =========================================================

def evaluate_all_controls(
    cfg: Optional[Phase32Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2_config()

    frames = available_control_frames(cfg)

    if not frames:
        raise FileNotFoundError(
            "No lagged frames found for controls. Run:\n"
            "python -m src.phase3_2_lagging"
        )

    run_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    for regime, lag_value, path in frames:
        df = pd.read_csv(path)
        df = _sort_frame(df)

        for control_type in cfg.control_types:
            for replicate in range(int(cfg.n_controls)):
                print(
                    f"[Phase3.2 Controls] control={control_type} "
                    f"replicate={replicate} regime={regime} lag={lag_value}"
                )

                try:
                    run_result, pair_scores_df = evaluate_control_on_frame(
                        df=df,
                        control_type=str(control_type),
                        regime=str(regime),
                        lag_epochs=int(lag_value),
                        replicate=int(replicate),
                        cfg=cfg,
                    )

                    run_rows.append(asdict(run_result))

                    if not pair_scores_df.empty:
                        pair_rows.extend(pair_scores_df.to_dict(orient="records"))

                except Exception as exc:
                    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)

                    run_rows.append(
                        asdict(
                            ControlRunResult(
                                control_type=str(control_type),
                                replicate=int(replicate),
                                regime=str(regime),
                                lag_epochs=int(lag_value),
                                lag_label=lag_label(int(lag_value)),
                                valid=False,
                                n_rows=int(len(df)),
                                n_subjects=int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
                                max_baseline_eps_test=0.0,
                                max_augmented_eps_test=0.0,
                                max_conditional_lift=0.0,
                                max_fraction_positive_lift=0.0,
                                max_bic_improvement=0.0,
                                max_ll_improvement=0.0,
                                n_positive_lift_rows=0,
                                n_strong_candidate_rows=0,
                                eps_saturation_value=eps_saturation_value,
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

    runs_df = pd.DataFrame(run_rows)
    pairs_df = pd.DataFrame(pair_rows)

    if not runs_df.empty:
        runs_df = runs_df.sort_values(
            ["control_type", "regime", "lag_epochs", "replicate"]
        ).reset_index(drop=True)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            ["control_type", "replicate", "regime", "lag_epochs", "family"]
        ).reset_index(drop=True)

    return runs_df, pairs_df


# =========================================================
# Control summaries
# =========================================================

def summarize_control_runs(
    runs_df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2_config()

    if runs_df.empty:
        return {
            "available": False,
            "reason": "empty_control_runs",
            "control_summaries": [],
            "passed": False,
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
        sat_frac = pd.to_numeric(valid_group["eps_saturation_fraction"], errors="coerce").fillna(0.0)

        collapsed_overall = _as_bool_series(valid_group["collapsed_overall"])
        below_tol = (
            (max_lift <= float(cfg.control_tol))
            & (max_bic <= 0.0)
            & (pd.to_numeric(valid_group["max_fraction_positive_lift"], errors="coerce").fillna(0.0) < float(cfg.subject_effect_fraction))
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
                fraction_runs_below_control_tol=fraction_below,
                fraction_runs_collapsed_overall=fraction_collapsed,
                control_tol=float(cfg.control_tol),
                passed=bool(passed),
                reason=reason,
            )
        )

    all_passed = all(item.passed for item in summaries) if summaries else False

    return {
        "available": True,
        "passed": bool(all_passed),
        "n_control_types": int(len(summaries)),
        "control_summaries": [asdict(item) for item in summaries],
        "failed_control_types": [
            item.control_type for item in summaries if not item.passed
        ],
        "interpretation": (
            "Controls pass when conditional lift, BIC improvement, subject-level "
            "positive lift fraction and strong-candidate persistence collapse. "
            "High epsilon alone is tracked as saturation diagnostic, not as direct failure."
        ),
    }


def build_controls_summary_text(summary: Mapping[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2 — Negative Controls Summary")
    lines.append("Leakage-safe conditional estimator version")
    lines.append("=" * 78)
    lines.append("")

    if not summary.get("available"):
        lines.append(f"Unavailable: {summary.get('reason')}")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(f"Overall passed: {summary.get('passed')}")
    lines.append(f"Control types: {summary.get('n_control_types')}")
    lines.append(f"Failed control types: {summary.get('failed_control_types')}")
    lines.append(f"Interpretation: {summary.get('interpretation')}")
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
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    runs_csv = Path(cfg.results_dir) / "phase3_2_control_runs.csv"
    pairs_csv = Path(cfg.results_dir) / "phase3_2_control_pair_scores.csv"
    summary_json = Path(cfg.controls_json)
    summary_txt = Path(cfg.results_dir) / "phase3_2_controls_summary.txt"

    runs_df.to_csv(runs_csv, index=False)
    pairs_df.to_csv(pairs_csv, index=False)

    summary = summarize_control_runs(runs_df, cfg)

    save_json(
        summary_json,
        {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "module": "Phase_III_2_negative_controls_leakage_safe",
            "summary": summary,
            "runs_file": str(runs_csv),
            "pair_scores_file": str(pairs_csv),
        },
    )

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(build_controls_summary_text(summary))

    print(f"[Phase3.2 Controls] Saved runs: {runs_csv}")
    print(f"[Phase3.2 Controls] Saved pair scores: {pairs_csv}")
    print(f"[Phase3.2 Controls] Saved controls JSON: {summary_json}")
    print(f"[Phase3.2 Controls] Saved controls TXT: {summary_txt}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    runs_df, pairs_df = evaluate_all_controls(cfg)

    save_control_outputs(
        runs_df=runs_df,
        pairs_df=pairs_df,
        cfg=cfg,
    )

    summary = summarize_control_runs(runs_df, cfg)

    print("\n============================================================")
    print("Phase III.2 controls completed")
    print("Leakage-safe conditional estimator version")
    print("============================================================")
    print(f"Overall passed: {summary.get('passed')}")
    print(f"Failed control types: {summary.get('failed_control_types')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()