from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_1_config import StateModelSpec, load_phase3_1_config

from src.phase3_1_features import prepare_phase3_1_features_for_modeling
from src.phase3_1_metrics import save_json
from src.phase3_1_runner import (
    evaluate_state_model,
    eeg_baseline_model,
    rng_baseline_model,
    get_model,
    make_chronological_split,
)


# =========================================================
# Diagnostics purpose
# =========================================================
#
# This file is intentionally separated from phase3_1_runner.py.
#
# The runner executes the registered Phase III.1 pipeline.
# This diagnostics module performs post-run analysis:
#
#   - model ranking
#   - subject-by-model LOSO matrix
#   - all-subject internal diagnostics
#   - RNG alignment audit
#   - lead inspection before Phase III.2
#
# It should be run AFTER phase3_1_runner.py has successfully generated:
#
#   results/phase3_1/phase3_1_results.json
#   data/interim/phase3_1/phase3_1_combined_features.csv
#
# Command:
#
#   python -m src.phase3_1_diagnostics


# =========================================================
# Generic helpers
# =========================================================

def _load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if not np.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        x = int(value)
        return x
    except Exception:
        return default


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


def _make_loso_indices(
    df: pd.DataFrame,
    heldout_subject: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if "subject_id" not in df.columns:
        raise KeyError("subject_id column is required for LOSO diagnostics")

    subject_values = df["subject_id"].astype(str)

    train_idx = np.flatnonzero((subject_values != str(heldout_subject)).to_numpy())
    test_idx = np.flatnonzero((subject_values == str(heldout_subject)).to_numpy())

    if len(train_idx) < 10 or len(test_idx) < 10:
        raise RuntimeError(
            f"Invalid LOSO split for subject={heldout_subject}: "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

    return train_idx.astype(int), test_idx.astype(int)


def _diagnostic_models(cfg: Any) -> Tuple[StateModelSpec, ...]:
    """
    Models evaluated by the diagnostics layer.

    Includes:
        - B0 EEG-only baseline
        - B1 RNG-only baseline
        - M0-M5 registered models
    """
    models: List[StateModelSpec] = [
        eeg_baseline_model(),
        rng_baseline_model(),
    ]

    for name in [
        "M0_observed_compact",
        "M1_eeg_informational",
        "M2_rng_quantum_proxy",
        "M3_informational_joint",
        "M4_latent_joint",
        "M5_augmented_final",
    ]:
        models.append(get_model(cfg, name))

    return tuple(models)


def _subjects(raw_df: pd.DataFrame) -> List[str]:
    if "subject_id" not in raw_df.columns:
        return []

    return sorted([str(x) for x in raw_df["subject_id"].dropna().unique()])


# =========================================================
# Ranking from existing result files
# =========================================================

def build_existing_model_ranking(results: Mapping[str, Any]) -> pd.DataFrame:
    """
    Builds model ranking using the already saved phase3_1_results.json.

    This does not recompute anything.
    """
    model_scores = results.get("model_scores", {})
    baseline_scores = results.get("baseline_scores", {})

    rows: List[Dict[str, Any]] = []

    for name, payload in model_scores.items():
        row = dict(payload)
        row["score_group"] = "registered_model"
        rows.append(row)

    for name, payload in baseline_scores.items():
        row = dict(payload)
        row["score_group"] = "baseline"
        row["score_key"] = name
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    numeric_cols = [
        "eps_train",
        "eps_test",
        "ll_train",
        "ll_test",
        "bic_train",
        "bic_test",
        "aic_train",
        "aic_test",
        "n_states",
        "n_params",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bic_test" in df.columns:
        df["rank_bic_test"] = df["bic_test"].rank(method="min", ascending=True)

    if "ll_test" in df.columns:
        df["rank_ll_test"] = df["ll_test"].rank(method="min", ascending=False)

    if "eps_test" in df.columns:
        df["rank_eps_test"] = df["eps_test"].rank(method="min", ascending=False)

    return df.sort_values(
        by=[c for c in ["rank_bic_test", "rank_ll_test"] if c in df.columns],
        ascending=True,
    ).reset_index(drop=True)


def summarize_existing_ranking(ranking_df: pd.DataFrame) -> Dict[str, Any]:
    if ranking_df.empty:
        return {
            "available": False,
            "reason": "empty_model_ranking",
        }

    summary: Dict[str, Any] = {
        "available": True,
    }

    if "bic_test" in ranking_df.columns:
        best_bic = ranking_df.sort_values("bic_test", ascending=True).iloc[0]
        summary["best_by_bic_test"] = {
            "model": str(best_bic.get("model")),
            "label": str(best_bic.get("label")),
            "bic_test": _safe_float(best_bic.get("bic_test")),
            "eps_test": _safe_float(best_bic.get("eps_test")),
            "ll_test": _safe_float(best_bic.get("ll_test")),
            "n_states": int(_safe_float(best_bic.get("n_states"))),
        }

    if "ll_test" in ranking_df.columns:
        best_ll = ranking_df.sort_values("ll_test", ascending=False).iloc[0]
        summary["best_by_ll_test"] = {
            "model": str(best_ll.get("model")),
            "label": str(best_ll.get("label")),
            "ll_test": _safe_float(best_ll.get("ll_test")),
            "eps_test": _safe_float(best_ll.get("eps_test")),
            "bic_test": _safe_float(best_ll.get("bic_test")),
            "n_states": int(_safe_float(best_ll.get("n_states"))),
        }

    if "eps_test" in ranking_df.columns:
        best_eps = ranking_df.sort_values("eps_test", ascending=False).iloc[0]
        summary["best_by_eps_test"] = {
            "model": str(best_eps.get("model")),
            "label": str(best_eps.get("label")),
            "eps_test": _safe_float(best_eps.get("eps_test")),
            "bic_test": _safe_float(best_eps.get("bic_test")),
            "ll_test": _safe_float(best_eps.get("ll_test")),
            "n_states": int(_safe_float(best_eps.get("n_states"))),
        }

    return summary


# =========================================================
# Subject-by-model LOSO diagnostics
# =========================================================

def evaluate_subject_by_model_matrix(
    raw_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """
    Evaluates B0/B1/M0-M5 for each held-out subject.

    Important:
        For each held-out subject, Phase III.1 feature layers are refit using
        train rows only. This avoids train/test leakage during diagnostics.

    This is diagnostic, not a replacement for the registered runner.
    """
    if "subject_id" not in raw_df.columns:
        raise KeyError("subject_id column is required")

    raw_df = _sort_feature_frame(raw_df)

    subjects = _subjects(raw_df)
    models = _diagnostic_models(cfg)

    rows: List[Dict[str, Any]] = []

    for heldout_subject in subjects:
        print(f"[Diagnostics] LOSO subject-by-model: heldout={heldout_subject}")

        try:
            idx_train, idx_test = _make_loso_indices(
                df=raw_df,
                heldout_subject=heldout_subject,
            )

            layered = prepare_phase3_1_features_for_modeling(
                df=raw_df,
                cfg=cfg,
                train_index=idx_train,
            ).df

            for model in models:
                try:
                    ev = evaluate_state_model(
                        df=layered,
                        model=model,
                        cfg=cfg,
                        idx_train_rows=idx_train,
                        idx_test_rows=idx_test,
                        label=f"diagnostic_loso_{heldout_subject}_{model.name}",
                        verbose=0,
                    )

                    rows.append(
                        {
                            "subject_id": heldout_subject,
                            "model": model.name,
                            "label": model.label,
                            "diagnostic_mode": "loso",
                            "valid": True,
                            "train_rows": int(len(idx_train)),
                            "test_rows": int(len(idx_test)),
                            "n_states": int(model.n_states),
                            "n_components": int(model.n_components),
                            "eps_train": float(ev.eps_train),
                            "eps_test": float(ev.eps_test),
                            "ll_train": float(ev.ll_train),
                            "ll_test": float(ev.ll_test),
                            "bic_train": float(ev.score.bic_train),
                            "bic_test": float(ev.score.bic_test),
                            "aic_train": float(ev.score.aic_train),
                            "aic_test": float(ev.score.aic_test),
                            "transitions_per_state": float(ev.score.transitions_per_state),
                            "transitions_per_observed_state": float(
                                ev.score.transitions_per_observed_state
                            ),
                        }
                    )

                except Exception as exc:
                    rows.append(
                        {
                            "subject_id": heldout_subject,
                            "model": model.name,
                            "label": model.label,
                            "diagnostic_mode": "loso",
                            "valid": False,
                            "error": str(exc),
                            "train_rows": int(len(idx_train)),
                            "test_rows": int(len(idx_test)),
                        }
                    )

        except Exception as exc:
            rows.append(
                {
                    "subject_id": heldout_subject,
                    "model": "__split__",
                    "label": "split_failure",
                    "diagnostic_mode": "loso",
                    "valid": False,
                    "error": str(exc),
                }
            )

    matrix = pd.DataFrame(rows)

    if matrix.empty:
        return matrix

    return add_subject_lift_columns(matrix)


def add_subject_lift_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EEG/RNG baseline references and lift values to each subject/model row.
    """
    out = matrix.copy()

    out["eps_eeg_baseline"] = np.nan
    out["eps_rng_baseline"] = np.nan
    out["baseline_max"] = np.nan
    out["lift_over_eeg"] = np.nan
    out["lift_over_rng"] = np.nan
    out["lift_over_baseline"] = np.nan

    if "subject_id" not in out.columns or "model" not in out.columns:
        return out

    for subject_id, group in out.groupby("subject_id"):
        eeg_rows = group[group["model"] == "B0_eeg_only"]
        rng_rows = group[group["model"] == "B1_rng_only"]

        eps_eeg = np.nan
        eps_rng = np.nan

        if not eeg_rows.empty:
            eps_eeg = _safe_float(eeg_rows.iloc[0].get("eps_test"), default=np.nan)

        if not rng_rows.empty:
            eps_rng = _safe_float(rng_rows.iloc[0].get("eps_test"), default=np.nan)

        baseline_max = np.nanmax([eps_eeg, eps_rng])

        idx = out["subject_id"].astype(str) == str(subject_id)

        out.loc[idx, "eps_eeg_baseline"] = eps_eeg
        out.loc[idx, "eps_rng_baseline"] = eps_rng
        out.loc[idx, "baseline_max"] = baseline_max

        if "eps_test" in out.columns:
            out.loc[idx, "lift_over_eeg"] = out.loc[idx, "eps_test"].astype(float) - eps_eeg
            out.loc[idx, "lift_over_rng"] = out.loc[idx, "eps_test"].astype(float) - eps_rng
            out.loc[idx, "lift_over_baseline"] = (
                out.loc[idx, "eps_test"].astype(float) - baseline_max
            )

    return out


def summarize_subject_model_matrix(matrix: pd.DataFrame) -> Dict[str, Any]:
    if matrix.empty:
        return {
            "available": False,
            "reason": "empty_subject_model_matrix",
        }

    valid = matrix[matrix.get("valid", False) == True].copy()

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_subject_model_rows",
        }

    summary: Dict[str, Any] = {
        "available": True,
        "n_rows": int(len(matrix)),
        "n_valid_rows": int(len(valid)),
        "subjects": sorted(valid["subject_id"].astype(str).unique().tolist()),
        "models": sorted(valid["model"].astype(str).unique().tolist()),
    }

    positive_eps = valid[valid["eps_test"].astype(float) > 0.0]
    positive_lift = valid[valid["lift_over_baseline"].astype(float) > 0.0]

    display_cols = [
        "subject_id",
        "model",
        "eps_test",
        "eps_eeg_baseline",
        "eps_rng_baseline",
        "lift_over_baseline",
        "bic_test",
        "ll_test",
    ]

    summary["positive_eps_rows"] = positive_eps[display_cols].to_dict(orient="records")
    summary["positive_lift_rows"] = positive_lift[display_cols].to_dict(orient="records")

    by_model: Dict[str, Any] = {}

    for model, group in valid.groupby("model"):
        by_model[str(model)] = {
            "n_subjects": int(group["subject_id"].nunique()),
            "median_eps_test": float(group["eps_test"].median()),
            "max_eps_test": float(group["eps_test"].max()),
            "median_lift_over_baseline": float(group["lift_over_baseline"].median()),
            "max_lift_over_baseline": float(group["lift_over_baseline"].max()),
            "fraction_positive_lift": float(
                np.mean(group["lift_over_baseline"].astype(float) > 0.0)
            ),
            "median_bic_test": float(group["bic_test"].median()),
            "median_ll_test": float(group["ll_test"].median()),
        }

    summary["by_model"] = by_model

    by_subject: Dict[str, Any] = {}

    for subject_id, group in valid.groupby("subject_id"):
        best_eps = group.sort_values("eps_test", ascending=False).iloc[0]
        best_bic = group.sort_values("bic_test", ascending=True).iloc[0]
        best_ll = group.sort_values("ll_test", ascending=False).iloc[0]
        best_lift = group.sort_values("lift_over_baseline", ascending=False).iloc[0]

        by_subject[str(subject_id)] = {
            "best_by_eps": {
                "model": str(best_eps["model"]),
                "eps_test": _safe_float(best_eps["eps_test"]),
                "lift_over_baseline": _safe_float(best_eps["lift_over_baseline"]),
            },
            "best_by_bic": {
                "model": str(best_bic["model"]),
                "bic_test": _safe_float(best_bic["bic_test"]),
                "eps_test": _safe_float(best_bic["eps_test"]),
            },
            "best_by_ll": {
                "model": str(best_ll["model"]),
                "ll_test": _safe_float(best_ll["ll_test"]),
                "eps_test": _safe_float(best_ll["eps_test"]),
            },
            "best_by_lift": {
                "model": str(best_lift["model"]),
                "lift_over_baseline": _safe_float(best_lift["lift_over_baseline"]),
                "eps_test": _safe_float(best_lift["eps_test"]),
            },
        }

    summary["by_subject"] = by_subject

    return summary


# =========================================================
# Internal subject diagnostics
# =========================================================

def evaluate_single_subject_internal(
    raw_df: pd.DataFrame,
    cfg: Any,
    subject_id: str,
) -> pd.DataFrame:
    """
    Evaluates B0/B1/M0-M5 inside one subject using chronological split.

    This answers:
        Does the subject show structure internally, or only when used as
        a held-out subject against a multi-subject training set?
    """
    if "subject_id" not in raw_df.columns:
        raise KeyError("subject_id column is required")

    subject_df = raw_df[raw_df["subject_id"].astype(str) == str(subject_id)].copy()
    subject_df = _sort_feature_frame(subject_df)

    if len(subject_df) < 100:
        raise RuntimeError(
            f"Not enough rows for subject-only diagnostic: {subject_id}, rows={len(subject_df)}"
        )

    idx_train, idx_test = make_chronological_split(
        n=len(subject_df),
        train_ratio=float(cfg.train_ratio),
    )

    layered = prepare_phase3_1_features_for_modeling(
        df=subject_df,
        cfg=cfg,
        train_index=idx_train,
    ).df

    rows: List[Dict[str, Any]] = []

    for model in _diagnostic_models(cfg):
        try:
            ev = evaluate_state_model(
                df=layered,
                model=model,
                cfg=cfg,
                idx_train_rows=idx_train,
                idx_test_rows=idx_test,
                label=f"diagnostic_internal_{subject_id}_{model.name}",
                verbose=0,
            )

            rows.append(
                {
                    "subject_id": subject_id,
                    "model": model.name,
                    "label": model.label,
                    "diagnostic_mode": "chronological_within_subject",
                    "valid": True,
                    "train_rows": int(len(idx_train)),
                    "test_rows": int(len(idx_test)),
                    "n_states": int(model.n_states),
                    "n_components": int(model.n_components),
                    "eps_train": float(ev.eps_train),
                    "eps_test": float(ev.eps_test),
                    "ll_train": float(ev.ll_train),
                    "ll_test": float(ev.ll_test),
                    "bic_train": float(ev.score.bic_train),
                    "bic_test": float(ev.score.bic_test),
                    "aic_train": float(ev.score.aic_train),
                    "aic_test": float(ev.score.aic_test),
                    "transitions_per_state": float(ev.score.transitions_per_state),
                    "transitions_per_observed_state": float(
                        ev.score.transitions_per_observed_state
                    ),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "subject_id": subject_id,
                    "model": model.name,
                    "label": model.label,
                    "diagnostic_mode": "chronological_within_subject",
                    "valid": False,
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = add_subject_lift_columns(result)

    return result


def evaluate_all_subjects_internal(
    raw_df: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """
    Runs chronological within-subject diagnostics for every subject.

    This is the final audit before Phase III.2:
        - SC4002/M3 lead
        - SC4022/RNG anomaly
        - SC4032/M2 lead
        - all other subjects for comparison
    """
    rows: List[pd.DataFrame] = []

    for subject_id in _subjects(raw_df):
        print(f"[Diagnostics] Internal subject diagnostic: subject={subject_id}")

        try:
            result = evaluate_single_subject_internal(
                raw_df=raw_df,
                cfg=cfg,
                subject_id=subject_id,
            )
            rows.append(result)

        except Exception as exc:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "subject_id": subject_id,
                            "model": "__internal_failure__",
                            "label": "internal_subject_failure",
                            "diagnostic_mode": "chronological_within_subject",
                            "valid": False,
                            "error": str(exc),
                        }
                    ]
                )
            )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize_internal_subject_diagnostics(internal_df: pd.DataFrame) -> Dict[str, Any]:
    if internal_df.empty:
        return {
            "available": False,
            "reason": "empty_internal_subject_diagnostics",
        }

    valid = internal_df[internal_df.get("valid", False) == True].copy()

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_internal_subject_rows",
        }

    positive_eps = valid[valid["eps_test"].astype(float) > 0.0].copy()
    positive_lift = valid[valid["lift_over_baseline"].astype(float) > 0.0].copy()

    display_cols = [
        "subject_id",
        "model",
        "eps_train",
        "eps_test",
        "eps_eeg_baseline",
        "eps_rng_baseline",
        "lift_over_baseline",
        "bic_test",
        "ll_test",
        "n_states",
    ]

    by_subject: Dict[str, Any] = {}

    for subject_id, group in valid.groupby("subject_id"):
        best_eps = group.sort_values("eps_test", ascending=False).iloc[0]
        best_bic = group.sort_values("bic_test", ascending=True).iloc[0]
        best_ll = group.sort_values("ll_test", ascending=False).iloc[0]
        best_lift = group.sort_values("lift_over_baseline", ascending=False).iloc[0]

        by_subject[str(subject_id)] = {
            "best_by_eps": {
                "model": str(best_eps["model"]),
                "eps_test": _safe_float(best_eps["eps_test"]),
                "lift_over_baseline": _safe_float(best_eps["lift_over_baseline"]),
                "bic_test": _safe_float(best_eps["bic_test"]),
            },
            "best_by_bic": {
                "model": str(best_bic["model"]),
                "bic_test": _safe_float(best_bic["bic_test"]),
                "eps_test": _safe_float(best_bic["eps_test"]),
            },
            "best_by_ll": {
                "model": str(best_ll["model"]),
                "ll_test": _safe_float(best_ll["ll_test"]),
                "eps_test": _safe_float(best_ll["eps_test"]),
            },
            "best_by_lift": {
                "model": str(best_lift["model"]),
                "lift_over_baseline": _safe_float(best_lift["lift_over_baseline"]),
                "eps_test": _safe_float(best_lift["eps_test"]),
            },
        }

    by_model: Dict[str, Any] = {}

    for model, group in valid.groupby("model"):
        by_model[str(model)] = {
            "n_subjects": int(group["subject_id"].nunique()),
            "median_eps_test": float(group["eps_test"].median()),
            "max_eps_test": float(group["eps_test"].max()),
            "median_lift_over_baseline": float(group["lift_over_baseline"].median()),
            "max_lift_over_baseline": float(group["lift_over_baseline"].max()),
            "fraction_positive_lift": float(
                np.mean(group["lift_over_baseline"].astype(float) > 0.0)
            ),
            "median_bic_test": float(group["bic_test"].median()),
            "median_ll_test": float(group["ll_test"].median()),
        }

    return {
        "available": True,
        "n_rows": int(len(internal_df)),
        "n_valid_rows": int(len(valid)),
        "subjects": sorted(valid["subject_id"].astype(str).unique().tolist()),
        "models": sorted(valid["model"].astype(str).unique().tolist()),
        "positive_eps_rows": positive_eps[display_cols].to_dict(orient="records"),
        "positive_lift_rows": positive_lift[display_cols].to_dict(orient="records"),
        "by_subject": by_subject,
        "by_model": by_model,
    }


# =========================================================
# RNG alignment audit
# =========================================================

def rng_alignment_audit(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Audits RNG mapping/alignment by subject.

    This is important because the RNG source is a fixed ANU sample:
        1024 uint8 -> 8192 bits

    It is resampled/aligned to EEG epochs. A subject-specific RNG-only
    epsilon anomaly may indicate a mapping artifact rather than a physical
    or informational coupling effect.
    """
    if "subject_id" not in raw_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    expected_cols = [
        "rng_bit",
        "rng_window_id",
        "rng_window_start",
        "rng_window_end",
        "rng_info_bin",
        "q_rng_bin",
        "bit_balance_local",
        "transition_rate",
        "rng_entropy_local",
        "entropy_rate_proxy",
        "run_length_mean",
        "run_length_max",
        "run_length_std",
        "compressibility_proxy",
        "surprise_index",
        "transition_asymmetry",
        "micro_cluster_deviation",
        "q_rng_score",
    ]

    existing_cols = [c for c in expected_cols if c in raw_df.columns]

    for subject_id, group in raw_df.groupby("subject_id"):
        group = group.copy().reset_index(drop=True)

        row: Dict[str, Any] = {
            "subject_id": str(subject_id),
            "n_rows": int(len(group)),
            "existing_rng_columns": ",".join(existing_cols),
        }

        if "recording_id" in group.columns:
            row["recording_ids"] = ",".join(sorted(group["recording_id"].astype(str).unique()))

        if "sleep_stage" in group.columns:
            row["n_sleep_stages"] = int(group["sleep_stage"].nunique())
            row["sleep_stages"] = ",".join(sorted(group["sleep_stage"].astype(str).unique()))

        if "rng_bit" in group.columns:
            bits = pd.to_numeric(group["rng_bit"], errors="coerce").dropna()
            row["rng_bit_mean"] = float(bits.mean()) if len(bits) else np.nan
            row["rng_bit_std"] = float(bits.std(ddof=0)) if len(bits) else np.nan
            row["rng_bit_ones"] = int((bits == 1).sum()) if len(bits) else 0
            row["rng_bit_zeros"] = int((bits == 0).sum()) if len(bits) else 0

            if len(bits) > 1:
                row["rng_bit_transition_rate_subject"] = float(np.mean(bits.to_numpy()[1:] != bits.to_numpy()[:-1]))
            else:
                row["rng_bit_transition_rate_subject"] = np.nan

        if "rng_window_id" in group.columns:
            ids = pd.to_numeric(group["rng_window_id"], errors="coerce").dropna()
            row["rng_window_unique"] = int(ids.nunique()) if len(ids) else 0
            row["rng_window_min"] = int(ids.min()) if len(ids) else -1
            row["rng_window_max"] = int(ids.max()) if len(ids) else -1
            row["rng_window_reuse_ratio"] = (
                float(len(group) / max(ids.nunique(), 1)) if len(ids) else np.nan
            )
            row["rng_window_most_common_count"] = (
                int(ids.value_counts().iloc[0]) if len(ids) else 0
            )

        if "rng_window_start" in group.columns:
            starts = pd.to_numeric(group["rng_window_start"], errors="coerce").dropna()
            row["rng_window_start_min"] = int(starts.min()) if len(starts) else -1
            row["rng_window_start_max"] = int(starts.max()) if len(starts) else -1
            row["rng_window_start_unique"] = int(starts.nunique()) if len(starts) else 0

        if "rng_window_end" in group.columns:
            ends = pd.to_numeric(group["rng_window_end"], errors="coerce").dropna()
            row["rng_window_end_min"] = int(ends.min()) if len(ends) else -1
            row["rng_window_end_max"] = int(ends.max()) if len(ends) else -1
            row["rng_window_end_unique"] = int(ends.nunique()) if len(ends) else 0

        for col in [
            "bit_balance_local",
            "transition_rate",
            "rng_entropy_local",
            "entropy_rate_proxy",
            "run_length_mean",
            "run_length_max",
            "run_length_std",
            "compressibility_proxy",
            "surprise_index",
            "transition_asymmetry",
            "micro_cluster_deviation",
            "q_rng_score",
        ]:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{col}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
                row[f"{col}_min"] = float(values.min()) if len(values) else np.nan
                row[f"{col}_max"] = float(values.max()) if len(values) else np.nan
                row[f"{col}_unique"] = int(values.nunique()) if len(values) else 0

        if "rng_info_bin" in group.columns:
            row["rng_info_bin_counts"] = json.dumps(
                group["rng_info_bin"].value_counts(dropna=False).sort_index().to_dict()
            )

        if "q_rng_bin" in group.columns:
            row["q_rng_bin_counts"] = json.dumps(
                group["q_rng_bin"].value_counts(dropna=False).sort_index().to_dict()
            )

        rows.append(row)

    audit = pd.DataFrame(rows)

    if not audit.empty and "rng_window_reuse_ratio" in audit.columns:
        audit = audit.sort_values(
            by=["rng_window_reuse_ratio", "subject_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return audit


def summarize_rng_alignment_audit(audit_df: pd.DataFrame) -> Dict[str, Any]:
    if audit_df.empty:
        return {
            "available": False,
            "reason": "empty_rng_alignment_audit",
        }

    summary: Dict[str, Any] = {
        "available": True,
        "n_subjects": int(len(audit_df)),
    }

    if "rng_window_reuse_ratio" in audit_df.columns:
        max_row = audit_df.sort_values("rng_window_reuse_ratio", ascending=False).iloc[0]
        min_row = audit_df.sort_values("rng_window_reuse_ratio", ascending=True).iloc[0]

        summary["max_reuse_subject"] = {
            "subject_id": str(max_row.get("subject_id")),
            "rng_window_reuse_ratio": _safe_float(max_row.get("rng_window_reuse_ratio")),
            "rng_window_unique": _safe_int(max_row.get("rng_window_unique")),
            "n_rows": _safe_int(max_row.get("n_rows")),
        }

        summary["min_reuse_subject"] = {
            "subject_id": str(min_row.get("subject_id")),
            "rng_window_reuse_ratio": _safe_float(min_row.get("rng_window_reuse_ratio")),
            "rng_window_unique": _safe_int(min_row.get("rng_window_unique")),
            "n_rows": _safe_int(min_row.get("n_rows")),
        }

    if "rng_bit_transition_rate_subject" in audit_df.columns:
        trans = pd.to_numeric(audit_df["rng_bit_transition_rate_subject"], errors="coerce")
        summary["rng_bit_transition_rate_subject"] = {
            "min": float(trans.min()),
            "median": float(trans.median()),
            "max": float(trans.max()),
        }

    if "rng_bit_mean" in audit_df.columns:
        means = pd.to_numeric(audit_df["rng_bit_mean"], errors="coerce")
        summary["rng_bit_mean"] = {
            "min": float(means.min()),
            "median": float(means.median()),
            "max": float(means.max()),
        }

    return summary


# =========================================================
# Sleep-stage counts
# =========================================================

def sleep_stage_counts(raw_df: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in raw_df.columns or "sleep_stage" not in raw_df.columns:
        return pd.DataFrame()

    counts = (
        raw_df.groupby(["subject_id", "sleep_stage"])
        .size()
        .reset_index(name="n_epochs")
        .sort_values(["subject_id", "sleep_stage"])
        .reset_index(drop=True)
    )

    return counts


# =========================================================
# Report text
# =========================================================

def build_diagnostics_summary_text(
    results: Mapping[str, Any],
    ranking_summary: Mapping[str, Any],
    subject_summary: Mapping[str, Any],
    internal_summary: Mapping[str, Any],
    rng_audit_summary: Mapping[str, Any],
) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("CDR Phase III.1 Diagnostics")
    lines.append("=" * 78)
    lines.append("")

    final_status = results.get("final_status", {})
    primary = results.get("primary_results", {})
    gates = results.get("gates", {})

    lines.append("1. Registered run status")
    lines.append("-" * 78)
    lines.append(f"status: {final_status.get('status')}")
    lines.append(f"interpretation: {final_status.get('interpretation')}")
    lines.append(f"failed_gates: {gates.get('failed')}")
    lines.append(f"eps_joint_test: {primary.get('eps_joint_test')}")
    lines.append(f"eps_eeg_test: {primary.get('eps_eeg_test')}")
    lines.append(f"eps_rng_test: {primary.get('eps_rng_test')}")
    lines.append("")

    lines.append("2. Existing model ranking")
    lines.append("-" * 78)

    if ranking_summary.get("available"):
        for key in ["best_by_bic_test", "best_by_ll_test", "best_by_eps_test"]:
            item = ranking_summary.get(key)
            if item:
                lines.append(
                    f"{key}: model={item.get('model')} | "
                    f"eps_test={item.get('eps_test')} | "
                    f"BIC_test={item.get('bic_test')} | "
                    f"LL_test={item.get('ll_test')} | "
                    f"states={item.get('n_states')}"
                )
    else:
        lines.append(f"ranking unavailable: {ranking_summary.get('reason')}")

    lines.append("")

    lines.append("3. LOSO subject-by-model diagnostic")
    lines.append("-" * 78)

    if subject_summary.get("available"):
        lines.append(f"subjects: {subject_summary.get('subjects')}")
        lines.append(f"models: {subject_summary.get('models')}")
        lines.append("")
        lines.append("positive_eps_rows:")
        for row in subject_summary.get("positive_eps_rows", []):
            lines.append(f"  {row}")
        lines.append("")
        lines.append("positive_lift_rows:")
        for row in subject_summary.get("positive_lift_rows", []):
            lines.append(f"  {row}")
    else:
        lines.append(f"subject diagnostic unavailable: {subject_summary.get('reason')}")

    lines.append("")

    lines.append("4. Internal all-subject diagnostics")
    lines.append("-" * 78)

    if internal_summary.get("available"):
        lines.append(f"subjects: {internal_summary.get('subjects')}")
        lines.append(f"models: {internal_summary.get('models')}")
        lines.append("")
        lines.append("internal_positive_eps_rows:")
        for row in internal_summary.get("positive_eps_rows", []):
            lines.append(f"  {row}")
        lines.append("")
        lines.append("internal_positive_lift_rows:")
        for row in internal_summary.get("positive_lift_rows", []):
            lines.append(f"  {row}")
    else:
        lines.append(f"internal diagnostic unavailable: {internal_summary.get('reason')}")

    lines.append("")

    lines.append("5. RNG alignment audit")
    lines.append("-" * 78)

    if rng_audit_summary.get("available"):
        lines.append(f"n_subjects: {rng_audit_summary.get('n_subjects')}")
        lines.append(f"max_reuse_subject: {rng_audit_summary.get('max_reuse_subject')}")
        lines.append(f"min_reuse_subject: {rng_audit_summary.get('min_reuse_subject')}")
        lines.append(
            f"rng_bit_transition_rate_subject: "
            f"{rng_audit_summary.get('rng_bit_transition_rate_subject')}"
        )
        lines.append(f"rng_bit_mean: {rng_audit_summary.get('rng_bit_mean')}")
    else:
        lines.append(f"RNG alignment audit unavailable: {rng_audit_summary.get('reason')}")

    lines.append("")

    lines.append("6. Diagnostic interpretation")
    lines.append("-" * 78)
    lines.append(
        "This diagnostics module does not replace the registered runner. "
        "It is a post-run audit layer designed to identify whether the null_result "
        "is broad, model-specific, subject-specific, or possibly influenced by RNG "
        "alignment artifacts."
    )
    lines.append(
        "The main purpose of this patched version is to inspect all subjects internally, "
        "not only SC4011. This is required before Phase III.2 because the 10-subject "
        "diagnostic produced isolated LOSO leads such as SC4002/M3, SC4032/M2, and "
        "the SC4022 RNG-only anomaly."
    )
    lines.append(
        "If a LOSO lead disappears in within-subject chronological diagnostics, it should "
        "be treated as domain-shift or split-dependent structure, not as robust EEG-RNG "
        "coupling."
    )
    lines.append(
        "If SC4022's RNG-only epsilon remains high internally, the RNG alignment strategy "
        "must be audited before it is reused in Phase III.2."
    )

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Main diagnostics runner
# =========================================================

def run_phase3_1_diagnostics() -> None:
    cfg = load_phase3_1_config()

    diagnostics_dir = cfg.results_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    results_path = cfg.results_dir / "phase3_1_results.json"
    model_comparison_path = cfg.results_dir / "phase3_1_model_comparison.json"
    subject_results_path = cfg.results_dir / "phase3_1_subject_results.json"
    ablation_path = cfg.results_dir / "phase3_1_ablation.json"
    controls_path = cfg.results_dir / "phase3_1_controls.json"
    injection_path = cfg.results_dir / "phase3_1_injection_curve.json"

    combined_features_path = cfg.interim_dir / "phase3_1_combined_features.csv"

    print("\n============================================================")
    print("CDR Phase III.1 — Diagnostics")
    print("============================================================")

    print(f"[Diagnostics] Loading results: {results_path}")
    results = _load_json(results_path)

    print(f"[Diagnostics] Loading combined features: {combined_features_path}")
    if not combined_features_path.exists():
        raise FileNotFoundError(
            "Combined feature file not found. Run the main runner first:\n"
            "python -m src.phase3_1_runner"
        )

    raw_df = pd.read_csv(combined_features_path)
    raw_df = _sort_feature_frame(raw_df)

    print(f"[Diagnostics] Rows loaded: {len(raw_df)}")

    print("[Diagnostics] Building existing model ranking...")
    ranking_df = build_existing_model_ranking(results)
    ranking_summary = summarize_existing_ranking(ranking_df)

    ranking_csv = diagnostics_dir / "phase3_1_model_ranking_diagnostic.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    print("[Diagnostics] Running LOSO subject-by-model matrix...")
    subject_matrix = evaluate_subject_by_model_matrix(
        raw_df=raw_df,
        cfg=cfg,
    )

    subject_matrix_csv = diagnostics_dir / "phase3_1_subject_model_matrix.csv"
    subject_matrix.to_csv(subject_matrix_csv, index=False)

    subject_summary = summarize_subject_model_matrix(subject_matrix)

    print("[Diagnostics] Running internal all-subject diagnostics...")
    internal_df = evaluate_all_subjects_internal(
        raw_df=raw_df,
        cfg=cfg,
    )

    internal_csv = diagnostics_dir / "phase3_1_all_subjects_internal_diagnostic.csv"
    internal_df.to_csv(internal_csv, index=False)

    internal_summary = summarize_internal_subject_diagnostics(internal_df)

    print("[Diagnostics] Running RNG alignment audit...")
    rng_audit_df = rng_alignment_audit(raw_df)

    rng_audit_csv = diagnostics_dir / "phase3_1_rng_alignment_audit.csv"
    rng_audit_df.to_csv(rng_audit_csv, index=False)

    rng_audit_summary = summarize_rng_alignment_audit(rng_audit_df)

    print("[Diagnostics] Building sleep-stage counts...")
    stage_counts = sleep_stage_counts(raw_df)

    stage_counts_csv = diagnostics_dir / "phase3_1_sleep_stage_counts.csv"
    stage_counts.to_csv(stage_counts_csv, index=False)

    diagnostics_payload: Dict[str, Any] = {
        "diagnostics_version": "phase3_1_diagnostics_v2_all_subject_internal_audit",
        "source_files": {
            "results": str(results_path),
            "model_comparison": str(model_comparison_path),
            "subject_results": str(subject_results_path),
            "ablation": str(ablation_path),
            "controls": str(controls_path),
            "injection": str(injection_path),
            "combined_features": str(combined_features_path),
        },
        "registered_run_status": results.get("final_status", {}),
        "primary_results": results.get("primary_results", {}),
        "gates": results.get("gates", {}),
        "dataset_metadata": results.get("dataset_metadata", {}),
        "ranking_summary": ranking_summary,
        "subject_model_summary_loso": subject_summary,
        "internal_subject_summary": internal_summary,
        "rng_alignment_audit_summary": rng_audit_summary,
        "sleep_stage_counts": (
            stage_counts.to_dict(orient="records") if not stage_counts.empty else []
        ),
        "notes": {
            "diagnostic_status": "post_run_audit",
            "does_not_replace_registered_runner": True,
            "main_patch": (
                "Adds internal chronological diagnostics for all subjects and RNG "
                "alignment audit before Phase III.2."
            ),
            "critical_subjects_from_previous_run": [
                "SC4002_M3_positive_lift",
                "SC4022_rng_only_anomaly",
                "SC4032_M2_positive_lift",
            ],
            "recommended_next_step_if_internal_leads_disappear": (
                "Close Phase III.1 as null_result and proceed to Phase III.2."
            ),
            "recommended_next_step_if_internal_leads_survive": (
                "Inspect surviving subject/model lead before Phase III.2."
            ),
            "recommended_next_step_if_rng_alignment_anomaly_detected": (
                "Patch RNG alignment strategy or flag affected subject before Phase III.2."
            ),
        },
    }

    diagnostics_json = diagnostics_dir / "phase3_1_diagnostics_report.json"
    save_json(diagnostics_json, diagnostics_payload)

    summary_text = build_diagnostics_summary_text(
        results=results,
        ranking_summary=ranking_summary,
        subject_summary=subject_summary,
        internal_summary=internal_summary,
        rng_audit_summary=rng_audit_summary,
    )

    summary_txt = diagnostics_dir / "phase3_1_diagnostics_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n============================================================")
    print("Diagnostics complete")
    print("============================================================")
    print(f"Saved: {diagnostics_json}")
    print(f"Saved: {summary_txt}")
    print(f"Saved: {ranking_csv}")
    print(f"Saved: {subject_matrix_csv}")
    print(f"Saved: {internal_csv}")
    print(f"Saved: {rng_audit_csv}")
    print(f"Saved: {stage_counts_csv}")
    print("============================================================\n")


if __name__ == "__main__":
    run_phase3_1_diagnostics()