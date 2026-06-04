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
    describe_config,
    load_phase3_2e_config,
)


# =========================================================
# Phase III.2E — Gate Sensitivity Diagnostics
# F7 / F8 / F12 without changing official gates
# =========================================================
#
# Purpose:
#
#   Diagnose why the official gates F7, F8 and F12 pass/fail under the
#   synchronized EEG–QRNG protocol, without weakening or changing the official
#   gate rules.
#
# This module answers:
#
#   F7 — Subject consistency
#       - Is F7 failing because the subject threshold is too strict?
#       - How close is the best primary row to the official subject threshold?
#       - Would the result pass under exploratory thresholds?
#
#   F8 — Complexity penalty / BIC
#       - Is F8 failing because BIC penalizes the augmented model too strongly?
#       - Are AIC or held-out LL improvements positive even when BIC fails?
#       - How far is the best primary row from the official BIC >= 0 rule?
#
#   F12 — Epsilon saturation
#       - Are epsilon values saturating at the upper grid?
#       - Is saturation model-wide or pair-specific?
#       - How sensitive is the warning to saturation threshold and warning fraction?
#
# Critical rule:
#
#   This module never changes official gates.
#   It produces diagnostic / sensitivity reports only.
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   empty canonical sensitivity outputs are saved
#


# =========================================================
# Constants
# =========================================================

GATE_SENSITIVITY_MODULE = "phase3_2e_gate_sensitivity"
GATE_SENSITIVITY_VERSION = "phase3_2e_gate_sensitivity_v1_f7_f8_f12_diagnostics"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class GateSensitivityResult:
    status: str
    protocol_only: bool

    official_f7_threshold: float
    official_f8_threshold: float
    official_f12_eps_saturation_value: float
    official_f12_warning_fraction: float

    f7_available: bool
    f8_available: bool
    f12_available: bool

    f7_official_passed: bool
    f8_official_passed: bool
    f12_official_warning: bool

    f7_best_value: float
    f8_best_bic_value: float
    f12_max_saturation_fraction: float

    f7_distance_to_threshold: float
    f8_distance_to_threshold: float
    f12_distance_to_warning: float

    can_proceed_to_diagnostics: bool
    can_claim_empirical_cdr: bool

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


def _load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def _numeric_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=float)

    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)

    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _bool_cfg(cfg: Phase32EConfig, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


def _float_cfg(cfg: Phase32EConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _list_cfg(cfg: Phase32EConfig, name: str, default: Sequence[Any]) -> List[Any]:
    value = getattr(cfg, name, None)

    if value is None:
        return list(default)

    return list(value)


# =========================================================
# Paths
# =========================================================

def conditional_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "conditional_pair_scores_csv",
            Path(cfg.results_dir) / "phase3_2e_conditional_pair_scores.csv",
        )
    )


def conditional_model_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "conditional_model_scores_csv",
            Path(cfg.results_dir) / "phase3_2e_conditional_model_scores.csv",
        )
    )


def gates_json_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "gates_json",
            Path(cfg.results_dir) / "phase3_2e_gates.json",
        )
    )


def gate_sensitivity_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_gate_sensitivity_report.json"


def gate_sensitivity_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_gate_sensitivity_summary.txt"


def f7_sensitivity_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_f7_subject_sensitivity.csv"


def f8_sensitivity_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_f8_complexity_sensitivity.csv"


def f12_sensitivity_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_f12_epsilon_saturation_sensitivity.csv"


# =========================================================
# Row selection
# =========================================================

def valid_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty or "valid" not in pair_df.columns:
        return pd.DataFrame()

    return pair_df[_as_bool_series(pair_df["valid"])].copy()


def primary_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty or "primary" not in valid.columns:
        return pd.DataFrame()

    return valid[_as_bool_series(valid["primary"])].copy()


def positive_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty or "conditional_lift" not in valid.columns:
        return pd.DataFrame()

    return valid[_numeric_col(valid, "conditional_lift") > 0.0].copy()


def best_primary_row(pair_df: pd.DataFrame, column: str) -> Dict[str, Any]:
    primary = primary_pair_rows(pair_df)

    if primary.empty or column not in primary.columns:
        return {}

    row = primary.sort_values(column, ascending=False).iloc[0]

    return row.to_dict()


def best_valid_row(pair_df: pd.DataFrame, column: str) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty or column not in valid.columns:
        return {}

    row = valid.sort_values(column, ascending=False).iloc[0]

    return row.to_dict()


# =========================================================
# Official gate status helpers
# =========================================================

def load_official_gate_status(cfg: Phase32EConfig) -> Dict[str, Any]:
    payload = _load_json(gates_json_path(cfg))

    if not payload:
        return {
            "available": False,
            "reason": "gates_json_missing",
            "gates": {},
            "final_status": {},
        }

    gates = payload.get("gates", [])

    gate_map: Dict[str, Any] = {}

    for gate in gates:
        if not isinstance(gate, dict):
            continue

        gate_id = str(gate.get("gate", ""))

        if gate_id:
            gate_map[gate_id] = gate

    return {
        "available": True,
        "reason": "ok",
        "gates": gate_map,
        "final_status": payload.get("final_status", {}),
        "raw": payload,
    }


def official_gate_passed(gates_payload: Mapping[str, Any], gate_id: str) -> bool:
    gates = gates_payload.get("gates", {})

    if gate_id not in gates:
        return False

    return str(gates[gate_id].get("status", "")).upper() == "PASS"


def official_gate_warning(gates_payload: Mapping[str, Any], gate_id: str) -> bool:
    gates = gates_payload.get("gates", {})

    if gate_id not in gates:
        return False

    return str(gates[gate_id].get("status", "")).upper() == "WARN"


def official_gate_status(gates_payload: Mapping[str, Any], gate_id: str) -> str:
    gates = gates_payload.get("gates", {})

    if gate_id not in gates:
        return "MISSING"

    return str(gates[gate_id].get("status", "UNKNOWN"))


# =========================================================
# F7 — Subject consistency sensitivity
# =========================================================

def default_f7_threshold_grid(cfg: Phase32EConfig) -> List[float]:
    official = float(cfg.subject_effect_fraction)

    grid = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        official,
    ]

    grid = sorted(set(float(x) for x in grid if 0.0 <= float(x) <= 1.0))

    return grid


def build_f7_subject_sensitivity(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    primary = primary_pair_rows(pair_df)

    official_threshold = float(cfg.subject_effect_fraction)

    if primary.empty:
        empty = pd.DataFrame(
            columns=[
                "threshold",
                "official_threshold",
                "max_primary_fraction_positive_lift",
                "would_pass",
                "distance_to_threshold",
                "n_primary_rows",
                "n_rows_at_or_above_threshold",
            ]
        )

        summary = {
            "available": False,
            "reason": "no_primary_pair_rows",
            "official_threshold": official_threshold,
            "max_primary_fraction_positive_lift": 0.0,
            "distance_to_official_threshold": -official_threshold,
            "official_passed_by_data": False,
            "closest_threshold_passed": None,
            "strictest_threshold_passed": None,
            "interpretation": (
                "F7 cannot be diagnosed because there are no primary pair rows."
            ),
        }

        return empty, summary

    max_fraction = float(_numeric_col(primary, "fraction_positive_lift").max())

    thresholds = _list_cfg(
        cfg,
        "f7_subject_sensitivity_thresholds",
        default_f7_threshold_grid(cfg),
    )

    thresholds = sorted(set(float(x) for x in thresholds if 0.0 <= float(x) <= 1.0))

    rows: List[Dict[str, Any]] = []

    fractions = _numeric_col(primary, "fraction_positive_lift")

    for threshold in thresholds:
        threshold = float(threshold)
        rows.append(
            {
                "threshold": threshold,
                "official_threshold": official_threshold,
                "max_primary_fraction_positive_lift": max_fraction,
                "would_pass": bool(max_fraction >= threshold),
                "distance_to_threshold": float(max_fraction - threshold),
                "n_primary_rows": int(len(primary)),
                "n_rows_at_or_above_threshold": int((fractions >= threshold).sum()),
            }
        )

    sensitivity_df = pd.DataFrame(rows)

    passed_thresholds = sensitivity_df[sensitivity_df["would_pass"].astype(bool)]["threshold"].tolist()

    closest_passed = min(passed_thresholds) if passed_thresholds else None
    strictest_passed = max(passed_thresholds) if passed_thresholds else None

    official_passed = bool(max_fraction >= official_threshold)

    if official_passed:
        interpretation = (
            "F7 passes under the official subject-consistency threshold."
        )
    elif max_fraction > 0:
        interpretation = (
            "F7 does not pass officially, but nonzero subject-level persistence exists. "
            "This is diagnostic only and does not weaken the official gate."
        )
    else:
        interpretation = (
            "F7 does not pass and no subject-level positive-lift persistence was detected."
        )

    summary = {
        "available": True,
        "reason": "ok",
        "official_threshold": official_threshold,
        "max_primary_fraction_positive_lift": max_fraction,
        "distance_to_official_threshold": float(max_fraction - official_threshold),
        "official_passed_by_data": official_passed,
        "closest_threshold_passed": closest_passed,
        "strictest_threshold_passed": strictest_passed,
        "n_primary_rows": int(len(primary)),
        "best_primary_by_fraction": best_primary_row(pair_df, "fraction_positive_lift"),
        "interpretation": interpretation,
    }

    return sensitivity_df, summary


# =========================================================
# F8 — Complexity penalty / BIC sensitivity
# =========================================================

def default_f8_bic_threshold_grid() -> List[float]:
    return [
        -100.0,
        -50.0,
        -25.0,
        -10.0,
        -5.0,
        -1.0,
        0.0,
        1.0,
        5.0,
        10.0,
    ]


def build_f8_complexity_sensitivity(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    primary = primary_pair_rows(pair_df)

    official_threshold = 0.0

    if primary.empty:
        empty = pd.DataFrame(
            columns=[
                "bic_threshold",
                "official_threshold",
                "max_primary_bic_improvement",
                "max_primary_aic_improvement",
                "max_primary_ll_improvement",
                "would_pass_bic",
                "distance_to_bic_threshold",
                "n_primary_rows",
                "n_rows_at_or_above_bic_threshold",
            ]
        )

        summary = {
            "available": False,
            "reason": "no_primary_pair_rows",
            "official_threshold": official_threshold,
            "max_primary_bic_improvement": 0.0,
            "max_primary_aic_improvement": 0.0,
            "max_primary_ll_improvement": 0.0,
            "distance_to_official_threshold": 0.0,
            "official_passed_by_data": False,
            "bic_failed_but_aic_positive": False,
            "bic_failed_but_ll_positive": False,
            "interpretation": (
                "F8 cannot be diagnosed because there are no primary pair rows."
            ),
        }

        return empty, summary

    bic = _numeric_col(primary, "bic_improvement")
    aic = _numeric_col(primary, "aic_improvement")
    ll = _numeric_col(primary, "ll_improvement")

    max_bic = float(bic.max()) if len(bic) else 0.0
    max_aic = float(aic.max()) if len(aic) else 0.0
    max_ll = float(ll.max()) if len(ll) else 0.0

    thresholds = _list_cfg(
        cfg,
        "f8_bic_sensitivity_thresholds",
        default_f8_bic_threshold_grid(),
    )

    thresholds = sorted(set(float(x) for x in thresholds))

    rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        threshold = float(threshold)

        rows.append(
            {
                "bic_threshold": threshold,
                "official_threshold": official_threshold,
                "max_primary_bic_improvement": max_bic,
                "max_primary_aic_improvement": max_aic,
                "max_primary_ll_improvement": max_ll,
                "would_pass_bic": bool(max_bic >= threshold),
                "distance_to_bic_threshold": float(max_bic - threshold),
                "n_primary_rows": int(len(primary)),
                "n_rows_at_or_above_bic_threshold": int((bic >= threshold).sum()),
                "n_rows_with_positive_aic": int((aic > 0.0).sum()),
                "n_rows_with_positive_ll": int((ll > 0.0).sum()),
            }
        )

    sensitivity_df = pd.DataFrame(rows)

    official_passed = bool(max_bic >= official_threshold)
    bic_failed_but_aic_positive = bool((not official_passed) and max_aic > 0.0)
    bic_failed_but_ll_positive = bool((not official_passed) and max_ll > 0.0)

    if official_passed:
        interpretation = (
            "F8 passes under the official BIC >= 0 rule."
        )
    elif bic_failed_but_aic_positive or bic_failed_but_ll_positive:
        interpretation = (
            "F8 fails officially under BIC, but AIC and/or held-out log-likelihood "
            "show weaker exploratory support. This is diagnostic only and does not "
            "change the official BIC gate."
        )
    else:
        interpretation = (
            "F8 fails under BIC and no weaker AIC/LL support was detected."
        )

    summary = {
        "available": True,
        "reason": "ok",
        "official_threshold": official_threshold,
        "max_primary_bic_improvement": max_bic,
        "max_primary_aic_improvement": max_aic,
        "max_primary_ll_improvement": max_ll,
        "distance_to_official_threshold": float(max_bic - official_threshold),
        "official_passed_by_data": official_passed,
        "bic_failed_but_aic_positive": bic_failed_but_aic_positive,
        "bic_failed_but_ll_positive": bic_failed_but_ll_positive,
        "n_primary_rows": int(len(primary)),
        "best_primary_by_bic": best_primary_row(pair_df, "bic_improvement"),
        "best_primary_by_aic": best_primary_row(pair_df, "aic_improvement"),
        "best_primary_by_ll": best_primary_row(pair_df, "ll_improvement"),
        "interpretation": interpretation,
    }

    return sensitivity_df, summary


# =========================================================
# F12 — Epsilon saturation sensitivity
# =========================================================

def default_eps_saturation_values(cfg: Phase32EConfig) -> List[float]:
    official = _float_cfg(cfg, "eps_saturation_value", 0.80)

    values = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        official,
    ]

    values = sorted(set(float(x) for x in values if 0.0 <= float(x) <= 1.0))

    return values


def default_eps_warning_fractions(cfg: Phase32EConfig) -> List[float]:
    official = _float_cfg(cfg, "eps_saturation_warning_fraction", 0.50)

    values = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        official,
    ]

    values = sorted(set(float(x) for x in values if 0.0 <= float(x) <= 1.0))

    return values


def epsilon_values_from_model_scores(model_df: pd.DataFrame) -> pd.Series:
    if model_df.empty:
        return pd.Series([], dtype=float)

    eps_cols = [
        col for col in [
            "eps_candidate",
            "eps_calib",
            "eps_test",
        ]
        if col in model_df.columns
    ]

    if not eps_cols:
        return pd.Series([], dtype=float)

    values: List[float] = []

    for col in eps_cols:
        series = pd.to_numeric(model_df[col], errors="coerce").dropna()
        values.extend(series.astype(float).tolist())

    return pd.Series(values, dtype=float)


def epsilon_values_from_pair_scores(pair_df: pd.DataFrame) -> pd.Series:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.Series([], dtype=float)

    eps_cols = [
        col for col in [
            "baseline_eps_test",
            "augmented_eps_test",
        ]
        if col in valid.columns
    ]

    if not eps_cols:
        return pd.Series([], dtype=float)

    values: List[float] = []

    for col in eps_cols:
        series = pd.to_numeric(valid[col], errors="coerce").dropna()
        values.extend(series.astype(float).tolist())

    return pd.Series(values, dtype=float)


def build_f12_epsilon_sensitivity(
    model_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    official_eps_threshold = _float_cfg(cfg, "eps_saturation_value", 0.80)
    official_warning_fraction = _float_cfg(cfg, "eps_saturation_warning_fraction", 0.50)

    model_eps = epsilon_values_from_model_scores(model_df)
    pair_eps = epsilon_values_from_pair_scores(pair_df)

    all_eps = pd.concat([model_eps, pair_eps], ignore_index=True)

    if all_eps.empty:
        empty = pd.DataFrame(
            columns=[
                "eps_saturation_value",
                "warning_fraction",
                "saturation_fraction",
                "would_warn",
                "n_eps_values",
                "n_saturated_values",
            ]
        )

        summary = {
            "available": False,
            "reason": "no_epsilon_values_available",
            "official_eps_saturation_value": official_eps_threshold,
            "official_warning_fraction": official_warning_fraction,
            "max_saturation_fraction": 0.0,
            "official_warning_by_data": False,
            "distance_to_warning_fraction": -official_warning_fraction,
            "interpretation": (
                "F12 cannot be diagnosed because no epsilon values are available."
            ),
        }

        return empty, summary

    eps_thresholds = _list_cfg(
        cfg,
        "f12_eps_saturation_thresholds",
        default_eps_saturation_values(cfg),
    )

    warning_fractions = _list_cfg(
        cfg,
        "f12_warning_fraction_thresholds",
        default_eps_warning_fractions(cfg),
    )

    eps_thresholds = sorted(set(float(x) for x in eps_thresholds if 0.0 <= float(x) <= 1.0))
    warning_fractions = sorted(set(float(x) for x in warning_fractions if 0.0 <= float(x) <= 1.0))

    rows: List[Dict[str, Any]] = []

    for eps_threshold in eps_thresholds:
        saturated = all_eps >= float(eps_threshold)
        saturation_fraction = float(np.mean(saturated.to_numpy(dtype=bool))) if len(all_eps) else 0.0

        model_saturated = model_eps >= float(eps_threshold)
        pair_saturated = pair_eps >= float(eps_threshold)

        model_fraction = float(np.mean(model_saturated.to_numpy(dtype=bool))) if len(model_eps) else 0.0
        pair_fraction = float(np.mean(pair_saturated.to_numpy(dtype=bool))) if len(pair_eps) else 0.0

        for warning_fraction in warning_fractions:
            rows.append(
                {
                    "eps_saturation_value": float(eps_threshold),
                    "warning_fraction": float(warning_fraction),
                    "official_eps_saturation_value": official_eps_threshold,
                    "official_warning_fraction": official_warning_fraction,
                    "n_eps_values": int(len(all_eps)),
                    "n_model_eps_values": int(len(model_eps)),
                    "n_pair_eps_values": int(len(pair_eps)),
                    "n_saturated_values": int(saturated.sum()),
                    "saturation_fraction": saturation_fraction,
                    "model_saturation_fraction": model_fraction,
                    "pair_saturation_fraction": pair_fraction,
                    "would_warn": bool(saturation_fraction >= float(warning_fraction)),
                    "distance_to_warning_fraction": float(saturation_fraction - float(warning_fraction)),
                    "max_epsilon": float(all_eps.max()) if len(all_eps) else 0.0,
                }
            )

    sensitivity_df = pd.DataFrame(rows)

    official_row = sensitivity_df[
        (np.isclose(sensitivity_df["eps_saturation_value"], official_eps_threshold))
        & (np.isclose(sensitivity_df["warning_fraction"], official_warning_fraction))
    ].copy()

    if official_row.empty:
        official_saturation_fraction = float((all_eps >= official_eps_threshold).mean())
        official_warning_by_data = bool(official_saturation_fraction >= official_warning_fraction)
    else:
        official_saturation_fraction = float(official_row.iloc[0]["saturation_fraction"])
        official_warning_by_data = bool(official_row.iloc[0]["would_warn"])

    max_saturation_fraction = float(sensitivity_df["saturation_fraction"].max()) if not sensitivity_df.empty else 0.0

    if official_warning_by_data:
        interpretation = (
            "F12 raises an official epsilon saturation warning. This does not prove "
            "absence of signal, but blocks strong interpretation until additional "
            "diagnostics explain why epsilon saturates."
        )
    elif max_saturation_fraction > 0:
        interpretation = (
            "F12 does not warn at the official threshold, but saturation exists under "
            "some exploratory thresholds. This should be documented diagnostically."
        )
    else:
        interpretation = (
            "No epsilon saturation pattern was detected under the tested thresholds."
        )

    summary = {
        "available": True,
        "reason": "ok",
        "official_eps_saturation_value": official_eps_threshold,
        "official_warning_fraction": official_warning_fraction,
        "official_saturation_fraction": official_saturation_fraction,
        "official_warning_by_data": official_warning_by_data,
        "max_saturation_fraction": max_saturation_fraction,
        "distance_to_warning_fraction": float(official_saturation_fraction - official_warning_fraction),
        "n_model_eps_values": int(len(model_eps)),
        "n_pair_eps_values": int(len(pair_eps)),
        "n_total_eps_values": int(len(all_eps)),
        "max_epsilon": float(all_eps.max()) if len(all_eps) else 0.0,
        "mean_epsilon": float(all_eps.mean()) if len(all_eps) else 0.0,
        "median_epsilon": float(all_eps.median()) if len(all_eps) else 0.0,
        "interpretation": interpretation,
    }

    return sensitivity_df, summary


# =========================================================
# Combined report
# =========================================================

def build_gate_sensitivity_result(
    f7_summary: Mapping[str, Any],
    f8_summary: Mapping[str, Any],
    f12_summary: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> GateSensitivityResult:
    protocol_only = False

    final_status = gates_payload.get("final_status", {})

    if isinstance(final_status, dict):
        protocol_only = bool(final_status.get("protocol_only", False))

    if not gates_payload.get("available", False):
        protocol_only = False

    pending_status = getattr(cfg, "status_pending_synchronized_data", "pending_synchronized_data")

    if protocol_only:
        status = pending_status
        interpretation = (
            "No synchronized EEG–QRNG dataset is available. Gate sensitivity was "
            "generated in protocol-only mode. F7/F8/F12 cannot be empirically diagnosed "
            "until synchronized data exist."
        )

    else:
        f7_available = bool(f7_summary.get("available", False))
        f8_available = bool(f8_summary.get("available", False))
        f12_available = bool(f12_summary.get("available", False))

        if not any([f7_available, f8_available, f12_available]):
            status = "gate_sensitivity_unavailable"
            interpretation = (
                "Gate sensitivity could not be evaluated because no valid pair/model "
                "rows are available."
            )

        else:
            status = "gate_sensitivity_completed"
            interpretation = (
                "Gate sensitivity diagnostics completed. These diagnostics do not alter "
                "official F7/F8/F12 gates; they only explain how close the current data "
                "are to the official thresholds."
            )

    f7_best = _safe_float(f7_summary.get("max_primary_fraction_positive_lift"), 0.0)
    f8_best = _safe_float(f8_summary.get("max_primary_bic_improvement"), 0.0)
    f12_best = _safe_float(f12_summary.get("official_saturation_fraction"), 0.0)

    official_f7 = float(cfg.subject_effect_fraction)
    official_f8 = 0.0
    official_f12_eps = _float_cfg(cfg, "eps_saturation_value", 0.80)
    official_f12_warn = _float_cfg(cfg, "eps_saturation_warning_fraction", 0.50)

    return GateSensitivityResult(
        status=status,
        protocol_only=bool(protocol_only),

        official_f7_threshold=official_f7,
        official_f8_threshold=official_f8,
        official_f12_eps_saturation_value=official_f12_eps,
        official_f12_warning_fraction=official_f12_warn,

        f7_available=bool(f7_summary.get("available", False)),
        f8_available=bool(f8_summary.get("available", False)),
        f12_available=bool(f12_summary.get("available", False)),

        f7_official_passed=official_gate_passed(gates_payload, "F7"),
        f8_official_passed=official_gate_passed(gates_payload, "F8"),
        f12_official_warning=official_gate_warning(gates_payload, "F12"),

        f7_best_value=f7_best,
        f8_best_bic_value=f8_best,
        f12_max_saturation_fraction=f12_best,

        f7_distance_to_threshold=float(f7_best - official_f7),
        f8_distance_to_threshold=float(f8_best - official_f8),
        f12_distance_to_warning=float(f12_best - official_f12_warn),

        can_proceed_to_diagnostics=True,
        can_claim_empirical_cdr=False,

        interpretation=interpretation,
    )


def write_gate_sensitivity_summary_txt(
    path: Path,
    result: GateSensitivityResult,
    f7_summary: Mapping[str, Any],
    f8_summary: Mapping[str, Any],
    f12_summary: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Gate Sensitivity Summary")
    lines.append("F7 / F8 / F12 diagnostics without changing official gates")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append(f"can_proceed_to_diagnostics: {result.can_proceed_to_diagnostics}")
    lines.append(f"can_claim_empirical_cdr: {result.can_claim_empirical_cdr}")
    lines.append("")
    lines.append("Official gate statuses")
    lines.append("-" * 78)
    lines.append(f"F7 official status: {official_gate_status(gates_payload, 'F7')}")
    lines.append(f"F8 official status: {official_gate_status(gates_payload, 'F8')}")
    lines.append(f"F12 official status: {official_gate_status(gates_payload, 'F12')}")
    lines.append("")
    lines.append("F7 — Subject consistency")
    lines.append("-" * 78)
    lines.append(f"available: {f7_summary.get('available')}")
    lines.append(f"official_threshold: {f7_summary.get('official_threshold')}")
    lines.append(f"max_primary_fraction_positive_lift: {f7_summary.get('max_primary_fraction_positive_lift')}")
    lines.append(f"distance_to_official_threshold: {f7_summary.get('distance_to_official_threshold')}")
    lines.append(f"official_passed_by_data: {f7_summary.get('official_passed_by_data')}")
    lines.append(f"strictest_threshold_passed: {f7_summary.get('strictest_threshold_passed')}")
    lines.append(f"interpretation: {f7_summary.get('interpretation')}")
    lines.append("")
    lines.append("F8 — Complexity penalty / BIC")
    lines.append("-" * 78)
    lines.append(f"available: {f8_summary.get('available')}")
    lines.append(f"official_threshold: {f8_summary.get('official_threshold')}")
    lines.append(f"max_primary_bic_improvement: {f8_summary.get('max_primary_bic_improvement')}")
    lines.append(f"max_primary_aic_improvement: {f8_summary.get('max_primary_aic_improvement')}")
    lines.append(f"max_primary_ll_improvement: {f8_summary.get('max_primary_ll_improvement')}")
    lines.append(f"distance_to_official_threshold: {f8_summary.get('distance_to_official_threshold')}")
    lines.append(f"official_passed_by_data: {f8_summary.get('official_passed_by_data')}")
    lines.append(f"bic_failed_but_aic_positive: {f8_summary.get('bic_failed_but_aic_positive')}")
    lines.append(f"bic_failed_but_ll_positive: {f8_summary.get('bic_failed_but_ll_positive')}")
    lines.append(f"interpretation: {f8_summary.get('interpretation')}")
    lines.append("")
    lines.append("F12 — Epsilon saturation")
    lines.append("-" * 78)
    lines.append(f"available: {f12_summary.get('available')}")
    lines.append(f"official_eps_saturation_value: {f12_summary.get('official_eps_saturation_value')}")
    lines.append(f"official_warning_fraction: {f12_summary.get('official_warning_fraction')}")
    lines.append(f"official_saturation_fraction: {f12_summary.get('official_saturation_fraction')}")
    lines.append(f"distance_to_warning_fraction: {f12_summary.get('distance_to_warning_fraction')}")
    lines.append(f"official_warning_by_data: {f12_summary.get('official_warning_by_data')}")
    lines.append(f"max_epsilon: {f12_summary.get('max_epsilon')}")
    lines.append(f"interpretation: {f12_summary.get('interpretation')}")
    lines.append("")
    lines.append("Overall interpretation")
    lines.append("-" * 78)
    lines.append(result.interpretation)
    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Save outputs
# =========================================================

def save_gate_sensitivity_outputs(
    cfg: Phase32EConfig,
    result: GateSensitivityResult,
    f7_df: pd.DataFrame,
    f8_df: pd.DataFrame,
    f12_df: pd.DataFrame,
    f7_summary: Mapping[str, Any],
    f8_summary: Mapping[str, Any],
    f12_summary: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> Dict[str, Any]:
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    f7_df.to_csv(f7_sensitivity_csv_path(cfg), index=False)
    f8_df.to_csv(f8_sensitivity_csv_path(cfg), index=False)
    f12_df.to_csv(f12_sensitivity_csv_path(cfg), index=False)

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": GATE_SENSITIVITY_MODULE,
        "gate_sensitivity_version": GATE_SENSITIVITY_VERSION,
        "created_at": _now_str(),
        "status": result.status,
        "result": asdict(result),
        "official_gate_statuses": {
            "F7": official_gate_status(gates_payload, "F7"),
            "F8": official_gate_status(gates_payload, "F8"),
            "F12": official_gate_status(gates_payload, "F12"),
        },
        "f7_subject_consistency": dict(f7_summary),
        "f8_complexity_penalty": dict(f8_summary),
        "f12_epsilon_saturation": dict(f12_summary),
        "input_shapes": {
            "pair_rows": int(len(pair_df)),
            "model_rows": int(len(model_df)),
        },
        "files": {
            "f7_sensitivity_csv": str(f7_sensitivity_csv_path(cfg)),
            "f8_sensitivity_csv": str(f8_sensitivity_csv_path(cfg)),
            "f12_sensitivity_csv": str(f12_sensitivity_csv_path(cfg)),
            "gate_sensitivity_report_json": str(gate_sensitivity_report_json_path(cfg)),
            "gate_sensitivity_summary_txt": str(gate_sensitivity_summary_txt_path(cfg)),
        },
        "config": describe_config(cfg),
        "official_gate_rule_preservation": {
            "official_gates_changed": False,
            "note": (
                "This module evaluates sensitivity scenarios only. It does not alter "
                "official F7/F8/F12 thresholds or final empirical claim rules."
            ),
        },
    }

    save_json(gate_sensitivity_report_json_path(cfg), report)

    write_gate_sensitivity_summary_txt(
        path=gate_sensitivity_summary_txt_path(cfg),
        result=result,
        f7_summary=f7_summary,
        f8_summary=f8_summary,
        f12_summary=f12_summary,
        gates_payload=gates_payload,
    )

    print(f"[Phase3.2E GateSensitivity] Saved F7 CSV: {f7_sensitivity_csv_path(cfg)}")
    print(f"[Phase3.2E GateSensitivity] Saved F8 CSV: {f8_sensitivity_csv_path(cfg)}")
    print(f"[Phase3.2E GateSensitivity] Saved F12 CSV: {f12_sensitivity_csv_path(cfg)}")
    print(f"[Phase3.2E GateSensitivity] Saved report: {gate_sensitivity_report_json_path(cfg)}")
    print(f"[Phase3.2E GateSensitivity] Saved summary: {gate_sensitivity_summary_txt_path(cfg)}")

    return report


# =========================================================
# Main API
# =========================================================

def run_phase3_2e_gate_sensitivity(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    pair_df = _load_csv(conditional_pair_scores_csv_path(cfg))
    model_df = _load_csv(conditional_model_scores_csv_path(cfg))
    gates_payload = load_official_gate_status(cfg)

    f7_df, f7_summary = build_f7_subject_sensitivity(pair_df, cfg)
    f8_df, f8_summary = build_f8_complexity_sensitivity(pair_df, cfg)
    f12_df, f12_summary = build_f12_epsilon_sensitivity(model_df, pair_df, cfg)

    result = build_gate_sensitivity_result(
        f7_summary=f7_summary,
        f8_summary=f8_summary,
        f12_summary=f12_summary,
        gates_payload=gates_payload,
        cfg=cfg,
    )

    report = save_gate_sensitivity_outputs(
        cfg=cfg,
        result=result,
        f7_df=f7_df,
        f8_df=f8_df,
        f12_df=f12_df,
        f7_summary=f7_summary,
        f8_summary=f8_summary,
        f12_summary=f12_summary,
        gates_payload=gates_payload,
        pair_df=pair_df,
        model_df=model_df,
    )

    return report


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III.2E F7/F8/F12 gate sensitivity diagnostics."
    )

    return parser.parse_args()


def main() -> None:
    _ = parse_args()

    cfg = load_phase3_2e_config()

    report = run_phase3_2e_gate_sensitivity(cfg)

    result = report.get("result", {})
    f7 = report.get("f7_subject_consistency", {})
    f8 = report.get("f8_complexity_penalty", {})
    f12 = report.get("f12_epsilon_saturation", {})

    print("\n" + "=" * 78)
    print("Phase III.2E gate sensitivity completed")
    print("F7 / F8 / F12 diagnostics without changing official gates")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"can_proceed_to_diagnostics: {result.get('can_proceed_to_diagnostics')}")
    print(f"can_claim_empirical_cdr: {result.get('can_claim_empirical_cdr')}")
    print("")
    print(f"F7 available: {f7.get('available')}")
    print(f"F7 best value: {f7.get('max_primary_fraction_positive_lift')}")
    print(f"F7 distance to official: {f7.get('distance_to_official_threshold')}")
    print("")
    print(f"F8 available: {f8.get('available')}")
    print(f"F8 best BIC: {f8.get('max_primary_bic_improvement')}")
    print(f"F8 distance to official: {f8.get('distance_to_official_threshold')}")
    print("")
    print(f"F12 available: {f12.get('available')}")
    print(f"F12 official saturation fraction: {f12.get('official_saturation_fraction')}")
    print(f"F12 distance to warning: {f12.get('distance_to_warning_fraction')}")
    print("")
    print(f"report_json: {gate_sensitivity_report_json_path(cfg)}")
    print(f"summary_txt: {gate_sensitivity_summary_txt_path(cfg)}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()