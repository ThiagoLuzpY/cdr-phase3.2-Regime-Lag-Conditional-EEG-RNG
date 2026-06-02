from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config.phase3_2_config import Phase32Config, load_phase3_2_config


# =========================================================
# Phase III.2 — Metrics and Gates
# Leakage-safe conditional estimator aware version
# =========================================================
#
# This module consolidates:
#
#   - III.2C conditional model results
#   - negative controls
#   - conditional lift
#   - subject consistency
#   - BIC / complexity penalty
#   - regime specificity
#   - multiple-comparison guard
#   - ε saturation diagnostics
#
# Important:
#
#   The conditional estimator was patched to use:
#
#       reference_train -> calibration -> test_holdout
#
#   Therefore, old controls generated before that patch must not be reused
#   silently. This metrics module detects stale/missing controls and reports:
#
#       pending_controls
#
#   instead of incorrectly treating old controls as current evidence.


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class GateResult:
    gate: str
    name: str
    passed: bool
    status: str
    value: float
    threshold: float
    reason: str
    details: Dict[str, Any]


@dataclass
class Phase32FinalStatus:
    status: str
    interpretation: str
    passed_gates: List[str]
    failed_gates: List[str]
    warning_gates: List[str]
    not_evaluated_gates: List[str]


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

    return pd.read_csv(path)


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)

    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _gate_status(passed: bool, available: bool = True, warning: bool = False) -> str:
    if not available:
        return "NOT_EVALUATED"

    if warning:
        return "WARN"

    return "PASS" if passed else "FAIL"


def _gate(
    gate: str,
    name: str,
    passed: bool,
    value: float,
    threshold: float,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
    available: bool = True,
    warning: bool = False,
) -> GateResult:
    return GateResult(
        gate=gate,
        name=name,
        passed=bool(passed),
        status=_gate_status(passed, available=available, warning=warning),
        value=float(value),
        threshold=float(threshold),
        reason=str(reason),
        details=details or {},
    )


def _cfg_float(cfg: Phase32Config, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


# =========================================================
# Loading Phase III.2 outputs
# =========================================================

def load_conditional_pair_scores(cfg: Phase32Config) -> pd.DataFrame:
    return _load_csv(Path(cfg.conditional_results_csv))


def load_conditional_model_scores(cfg: Phase32Config) -> pd.DataFrame:
    path = Path(cfg.results_dir) / "phase3_2c_conditional_model_scores.csv"
    return _load_csv(path)


def load_controls_summary(cfg: Phase32Config) -> Dict[str, Any]:
    """
    Loads controls summary.

    Old controls generated before the leakage-safe conditional patch are treated
    as stale, because they were calculated from an older estimator.
    """
    payload = _load_json(Path(cfg.controls_json))

    if not payload:
        return {
            "available": False,
            "stale": False,
            "passed": False,
            "reason": "controls_json_missing",
        }

    module = str(payload.get("module", ""))
    summary = payload.get("summary", {})

    # The new controls.py patch will write this module name.
    # Until then, old controls are not used as current evidence.
    expected_module = "Phase_III_2_negative_controls_leakage_safe"

    if module != expected_module:
        return {
            "available": True,
            "stale": True,
            "passed": False,
            "reason": "controls_generated_before_leakage_safe_patch",
            "module_found": module,
            "expected_module": expected_module,
            "raw_summary": summary,
        }

    if not summary:
        return {
            "available": True,
            "stale": False,
            "passed": False,
            "reason": "controls_summary_empty",
        }

    summary = dict(summary)
    summary["available"] = True
    summary["stale"] = False
    summary["reason"] = summary.get("reason", "ok")

    return summary


def load_control_runs(cfg: Phase32Config) -> pd.DataFrame:
    path = Path(cfg.results_dir) / "phase3_2_control_runs.csv"
    return _load_csv(path)


# =========================================================
# Candidate selection
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


def positive_lift_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty or "conditional_lift" not in valid.columns:
        return pd.DataFrame()

    return valid[pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0) > 0.0].copy()


def primary_positive_lift_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    primary = primary_pair_rows(pair_df)

    if primary.empty or "conditional_lift" not in primary.columns:
        return pd.DataFrame()

    return primary[pd.to_numeric(primary["conditional_lift"], errors="coerce").fillna(0.0) > 0.0].copy()


def strong_candidate_rows(pair_df: pd.DataFrame, cfg: Phase32Config) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    required = [
        "conditional_lift",
        "augmented_eps_test",
        "fraction_positive_lift",
        "bic_improvement",
    ]

    for col in required:
        if col not in valid.columns:
            return pd.DataFrame()

    strong = valid[
        (pd.to_numeric(valid["conditional_lift"], errors="coerce").fillna(0.0) >= float(cfg.conditional_lift_min))
        & (pd.to_numeric(valid["augmented_eps_test"], errors="coerce").fillna(0.0) >= float(cfg.strong_eps_min))
        & (pd.to_numeric(valid["fraction_positive_lift"], errors="coerce").fillna(0.0) >= float(cfg.subject_effect_fraction))
        & (pd.to_numeric(valid["bic_improvement"], errors="coerce").fillna(0.0) >= 0.0)
    ].copy()

    return strong


def primary_strong_candidate_rows(pair_df: pd.DataFrame, cfg: Phase32Config) -> pd.DataFrame:
    strong = strong_candidate_rows(pair_df, cfg)

    if strong.empty or "primary" not in strong.columns:
        return pd.DataFrame()

    return strong[_as_bool_series(strong["primary"])].copy()


def exploratory_lead_rows(pair_df: pd.DataFrame, cfg: Phase32Config) -> pd.DataFrame:
    """
    Rows with positive lift but not strong registered candidates.
    """
    positive = positive_lift_rows(pair_df)

    if positive.empty:
        return pd.DataFrame()

    strong_idx = set(strong_candidate_rows(pair_df, cfg).index.tolist())
    exploratory = positive[~positive.index.isin(strong_idx)].copy()

    return exploratory


def best_row_by(pair_df: pd.DataFrame, column: str, descending: bool = True) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty or column not in valid.columns:
        return {}

    row = valid.sort_values(column, ascending=not descending).iloc[0]

    return row.to_dict()


# =========================================================
# Saturation diagnostics
# =========================================================

def model_saturation_summary(
    model_df: pd.DataFrame,
    cfg: Phase32Config,
) -> Dict[str, Any]:
    if model_df.empty:
        return {
            "available": False,
            "reason": "model_scores_missing",
            "saturation_fraction": 0.0,
            "warning": False,
        }

    eps_col = None

    for candidate in ["eps_candidate", "eps_calib", "eps_test"]:
        if candidate in model_df.columns:
            eps_col = candidate
            break

    if eps_col is None:
        return {
            "available": False,
            "reason": "no_epsilon_column_available",
            "saturation_fraction": 0.0,
            "warning": False,
        }

    eps_values = pd.to_numeric(model_df[eps_col], errors="coerce").fillna(0.0)

    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)
    warn_fraction = _cfg_float(cfg, "eps_saturation_warning_fraction", 0.50)

    saturated = eps_values >= eps_saturation_value
    saturation_fraction = float(np.mean(saturated.to_numpy(dtype=bool))) if len(eps_values) else 0.0

    warning = saturation_fraction >= warn_fraction

    summary = {
        "available": True,
        "epsilon_column_used": eps_col,
        "eps_saturation_value": eps_saturation_value,
        "eps_saturation_warning_fraction": warn_fraction,
        "n_model_rows": int(len(model_df)),
        "n_saturated_rows": int(saturated.sum()),
        "saturation_fraction": saturation_fraction,
        "max_epsilon": float(eps_values.max()) if len(eps_values) else 0.0,
        "warning": bool(warning),
        "reason": "eps_saturation_warning" if warning else "ok",
    }

    if "eps_saturated" in model_df.columns:
        explicit = _as_bool_series(model_df["eps_saturated"])
        summary["explicit_eps_saturated_rows"] = int(explicit.sum())
        summary["explicit_eps_saturated_fraction"] = float(np.mean(explicit.to_numpy(dtype=bool))) if len(explicit) else 0.0

    return summary


def pair_saturation_summary(
    pair_df: pd.DataFrame,
    cfg: Phase32Config,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_pair_rows",
            "warning": False,
        }

    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)
    warn_fraction = _cfg_float(cfg, "eps_saturation_warning_fraction", 0.50)

    baseline_eps = pd.to_numeric(valid.get("baseline_eps_test", 0.0), errors="coerce").fillna(0.0)
    augmented_eps = pd.to_numeric(valid.get("augmented_eps_test", 0.0), errors="coerce").fillna(0.0)

    baseline_sat = baseline_eps >= eps_saturation_value
    augmented_sat = augmented_eps >= eps_saturation_value
    either_sat = baseline_sat | augmented_sat
    both_sat = baseline_sat & augmented_sat

    either_fraction = float(np.mean(either_sat.to_numpy(dtype=bool))) if len(valid) else 0.0
    both_fraction = float(np.mean(both_sat.to_numpy(dtype=bool))) if len(valid) else 0.0

    warning = either_fraction >= warn_fraction

    return {
        "available": True,
        "eps_saturation_value": eps_saturation_value,
        "eps_saturation_warning_fraction": warn_fraction,
        "n_valid_pair_rows": int(len(valid)),
        "n_either_saturated_rows": int(either_sat.sum()),
        "n_both_saturated_rows": int(both_sat.sum()),
        "either_saturation_fraction": either_fraction,
        "both_saturation_fraction": both_fraction,
        "max_baseline_eps_test": float(baseline_eps.max()) if len(baseline_eps) else 0.0,
        "max_augmented_eps_test": float(augmented_eps.max()) if len(augmented_eps) else 0.0,
        "warning": bool(warning),
        "reason": "pair_level_eps_saturation_warning" if warning else "ok",
    }


# =========================================================
# Gate computations
# =========================================================

def gate_f1_injection_recovery(cfg: Phase32Config) -> GateResult:
    return _gate(
        gate="F1",
        name="Injection recovery",
        passed=False,
        value=0.0,
        threshold=float(cfg.gate_tol_abs),
        reason="not_evaluated_in_standalone_metrics; reserved for runner-level injection test",
        details={
            "inj_eps_true": float(cfg.inj_eps_true),
            "note": "This gate is marked NOT_EVALUATED here, not failed scientifically.",
        },
        available=False,
    )


def gate_f2_controls_collapse(
    controls_summary: Mapping[str, Any],
    cfg: Phase32Config,
) -> GateResult:
    if not controls_summary:
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(cfg.required_control_fraction),
            reason="controls_summary_missing",
            details={},
            available=False,
        )

    if controls_summary.get("stale", False):
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(cfg.required_control_fraction),
            reason=str(controls_summary.get("reason", "controls_stale")),
            details=dict(controls_summary),
            available=False,
        )

    if controls_summary.get("available") is False:
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(cfg.required_control_fraction),
            reason=str(controls_summary.get("reason", "controls_unavailable")),
            details=dict(controls_summary),
            available=False,
        )

    passed = bool(controls_summary.get("passed", False))
    failed_types = controls_summary.get("failed_control_types", [])

    control_summaries = controls_summary.get("control_summaries", [])

    fractions = [
        _safe_float(item.get("fraction_runs_below_control_tol"))
        for item in control_summaries
        if isinstance(item, dict)
    ]

    value = float(min(fractions)) if fractions else 0.0

    reason = "ok" if passed else f"failed_control_types={failed_types}"

    return _gate(
        gate="F2",
        name="Controls collapse",
        passed=passed,
        value=value,
        threshold=float(cfg.required_control_fraction),
        reason=reason,
        details={
            "failed_control_types": failed_types,
            "control_summaries": control_summaries,
        },
    )


def gate_f3_holdout_generalization(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F3",
            name="Holdout generalization",
            passed=False,
            value=0.0,
            threshold=1.0,
            reason="no_valid_conditional_rows",
            details={},
            available=False,
        )

    n_valid = int(len(valid))
    n_total = int(len(pair_df))

    value = float(n_valid / max(n_total, 1))
    passed = n_valid > 0

    return _gate(
        gate="F3",
        name="Holdout generalization",
        passed=passed,
        value=value,
        threshold=1.0,
        reason="valid ref/calib/test conditional evaluations exist" if passed else "no valid evaluations",
        details={
            "n_valid_pair_rows": n_valid,
            "n_total_pair_rows": n_total,
            "split": "reference_train -> calibration -> test_holdout",
        },
    )


def gate_f5_sensitivity_stability(cfg: Phase32Config) -> GateResult:
    return _gate(
        gate="F5",
        name="Sensitivity stability",
        passed=False,
        value=0.0,
        threshold=float(cfg.sensitivity_max_delta),
        reason="not_evaluated_in_standalone_metrics; reserved for runner-level sensitivity test",
        details={},
        available=False,
    )


def gate_f6_conditional_lift(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    primary = primary_pair_rows(pair_df)

    if primary.empty:
        return _gate(
            gate="F6",
            name="Conditional lift",
            passed=False,
            value=0.0,
            threshold=float(cfg.conditional_lift_min),
            reason="no_primary_pair_rows",
            details={},
            available=False,
        )

    max_lift = float(pd.to_numeric(primary["conditional_lift"], errors="coerce").fillna(0.0).max())
    max_eps = float(pd.to_numeric(primary["augmented_eps_test"], errors="coerce").fillna(0.0).max())

    passed = (
        max_lift >= float(cfg.conditional_lift_min)
        and max_eps >= float(cfg.strong_eps_min)
    )

    reason = "ok" if passed else (
        f"max_primary_lift={max_lift:.6f}, max_primary_augmented_eps={max_eps:.6f}"
    )

    return _gate(
        gate="F6",
        name="Conditional lift",
        passed=passed,
        value=max_lift,
        threshold=float(cfg.conditional_lift_min),
        reason=reason,
        details={
            "max_primary_conditional_lift": max_lift,
            "max_primary_augmented_eps_test": max_eps,
            "strong_eps_min": float(cfg.strong_eps_min),
            "primary_rows": primary.to_dict(orient="records"),
        },
    )


def gate_f7_subject_consistency(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    primary = primary_pair_rows(pair_df)

    if primary.empty:
        return _gate(
            gate="F7",
            name="Subject consistency",
            passed=False,
            value=0.0,
            threshold=float(cfg.subject_effect_fraction),
            reason="no_primary_pair_rows",
            details={},
            available=False,
        )

    max_fraction = float(pd.to_numeric(primary["fraction_positive_lift"], errors="coerce").fillna(0.0).max())
    passed = max_fraction >= float(cfg.subject_effect_fraction)

    reason = "ok" if passed else (
        f"max_primary_fraction_positive_lift={max_fraction:.6f}"
    )

    return _gate(
        gate="F7",
        name="Subject consistency",
        passed=passed,
        value=max_fraction,
        threshold=float(cfg.subject_effect_fraction),
        reason=reason,
        details={
            "max_primary_fraction_positive_lift": max_fraction,
            "primary_rows": primary.to_dict(orient="records"),
        },
    )


def gate_f8_complexity_penalty(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    primary = primary_pair_rows(pair_df)

    if primary.empty:
        return _gate(
            gate="F8",
            name="Complexity penalty / BIC",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="no_primary_pair_rows",
            details={},
            available=False,
        )

    max_bic_improvement = float(pd.to_numeric(primary["bic_improvement"], errors="coerce").fillna(0.0).max())
    passed = max_bic_improvement >= 0.0

    reason = "ok" if passed else (
        f"best_primary_bic_improvement={max_bic_improvement:.6f}"
    )

    return _gate(
        gate="F8",
        name="Complexity penalty / BIC",
        passed=passed,
        value=max_bic_improvement,
        threshold=0.0,
        reason=reason,
        details={
            "max_primary_bic_improvement": max_bic_improvement,
            "primary_rows": primary.to_dict(orient="records"),
        },
    )


def gate_f9_ablation_or_lag_stability(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F9",
            name="Lag / ablation stability",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="no_valid_pair_rows",
            details={},
            available=False,
        )

    primary_lag_rows = valid[valid["lag_epochs"].astype(int).isin(list(cfg.primary_lags_epochs))].copy()
    secondary_lag_rows = valid[~valid["lag_epochs"].astype(int).isin(list(cfg.primary_lags_epochs))].copy()

    primary_max_lift = (
        float(pd.to_numeric(primary_lag_rows["conditional_lift"], errors="coerce").fillna(0.0).max())
        if not primary_lag_rows.empty
        else 0.0
    )

    secondary_max_lift = (
        float(pd.to_numeric(secondary_lag_rows["conditional_lift"], errors="coerce").fillna(0.0).max())
        if not secondary_lag_rows.empty
        else 0.0
    )

    any_meaningful_secondary = secondary_max_lift >= float(cfg.conditional_lift_min)
    primary_competitive = primary_max_lift >= (secondary_max_lift - 1e-12)

    passed = (not any_meaningful_secondary) or primary_competitive

    reason = "ok" if passed else (
        "secondary_lag_effect_exceeds_primary_lag_effect"
    )

    return _gate(
        gate="F9",
        name="Lag / ablation stability",
        passed=passed,
        value=primary_max_lift - secondary_max_lift,
        threshold=0.0,
        reason=reason,
        details={
            "primary_max_lift": primary_max_lift,
            "secondary_max_lift": secondary_max_lift,
            "any_meaningful_secondary": bool(any_meaningful_secondary),
        },
    )


def gate_f10_regime_specificity(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F10",
            name="Regime specificity",
            passed=False,
            value=0.0,
            threshold=float(cfg.conditional_lift_min),
            reason="no_valid_pair_rows",
            details={},
            available=False,
        )

    primary = valid[valid["regime"].astype(str).isin(list(cfg.primary_regimes))].copy()
    secondary = valid[valid["regime"].astype(str).isin(list(cfg.secondary_regimes))].copy()

    primary_max = (
        float(pd.to_numeric(primary["conditional_lift"], errors="coerce").fillna(0.0).max())
        if not primary.empty
        else 0.0
    )

    secondary_max = (
        float(pd.to_numeric(secondary["conditional_lift"], errors="coerce").fillna(0.0).max())
        if not secondary.empty
        else 0.0
    )

    passed = primary_max >= float(cfg.conditional_lift_min) or secondary_max < float(cfg.conditional_lift_min)

    reason = "ok" if passed else (
        "only_secondary_or_exploratory_regime_has_meaningful_lift"
    )

    return _gate(
        gate="F10",
        name="Regime specificity",
        passed=passed,
        value=primary_max,
        threshold=float(cfg.conditional_lift_min),
        reason=reason,
        details={
            "primary_max_lift": primary_max,
            "secondary_max_lift": secondary_max,
            "primary_regimes": list(cfg.primary_regimes),
            "secondary_regimes": list(cfg.secondary_regimes),
        },
    )


def gate_f11_multiple_comparison_guard(pair_df: pd.DataFrame, cfg: Phase32Config) -> GateResult:
    strong = strong_candidate_rows(pair_df, cfg)
    primary_strong = primary_strong_candidate_rows(pair_df, cfg)

    n_strong = int(len(strong))
    n_primary_strong = int(len(primary_strong))

    if n_strong == 0:
        return _gate(
            gate="F11",
            name="Multiple-comparison guard",
            passed=True,
            value=0.0,
            threshold=0.0,
            reason="no_strong_candidates_detected",
            details={
                "n_strong_candidates": 0,
                "n_primary_strong_candidates": 0,
            },
        )

    passed = n_primary_strong > 0

    reason = "ok" if passed else (
        "strong_candidates_exist_only_in_secondary_or_exploratory_tests"
    )

    return _gate(
        gate="F11",
        name="Multiple-comparison guard",
        passed=passed,
        value=float(n_primary_strong),
        threshold=1.0,
        reason=reason,
        details={
            "n_strong_candidates": n_strong,
            "n_primary_strong_candidates": n_primary_strong,
            "strong_candidates": strong.to_dict(orient="records"),
            "primary_strong_candidates": primary_strong.to_dict(orient="records"),
        },
    )


def gate_f12_epsilon_saturation(
    model_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cfg: Phase32Config,
) -> GateResult:
    model_summary = model_saturation_summary(model_df, cfg)
    pair_summary = pair_saturation_summary(pair_df, cfg)

    warning = bool(model_summary.get("warning", False) or pair_summary.get("warning", False))

    value = max(
        _safe_float(model_summary.get("saturation_fraction")),
        _safe_float(pair_summary.get("either_saturation_fraction")),
    )

    threshold = _cfg_float(cfg, "eps_saturation_warning_fraction", 0.50)

    if warning:
        return _gate(
            gate="F12",
            name="Epsilon saturation diagnostic",
            passed=False,
            value=value,
            threshold=threshold,
            reason="epsilon_saturation_warning",
            details={
                "model_saturation": model_summary,
                "pair_saturation": pair_summary,
            },
            warning=True,
        )

    return _gate(
        gate="F12",
        name="Epsilon saturation diagnostic",
        passed=True,
        value=value,
        threshold=threshold,
        reason="ok",
        details={
            "model_saturation": model_summary,
            "pair_saturation": pair_summary,
        },
    )


# =========================================================
# Full gate evaluation
# =========================================================

def evaluate_phase3_2_gates(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Optional[Phase32Config] = None,
) -> List[GateResult]:
    if cfg is None:
        cfg = load_phase3_2_config()

    gates = [
        gate_f1_injection_recovery(cfg),
        gate_f2_controls_collapse(controls_summary, cfg),
        gate_f3_holdout_generalization(pair_df, cfg),
        gate_f5_sensitivity_stability(cfg),
        gate_f6_conditional_lift(pair_df, cfg),
        gate_f7_subject_consistency(pair_df, cfg),
        gate_f8_complexity_penalty(pair_df, cfg),
        gate_f9_ablation_or_lag_stability(pair_df, cfg),
        gate_f10_regime_specificity(pair_df, cfg),
        gate_f11_multiple_comparison_guard(pair_df, cfg),
        gate_f12_epsilon_saturation(model_df, pair_df, cfg),
    ]

    return gates


def summarize_gate_status(gates: Sequence[GateResult]) -> Dict[str, List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    warning: List[str] = []
    not_evaluated: List[str] = []

    for gate in gates:
        if gate.status == "PASS":
            passed.append(gate.gate)
        elif gate.status == "FAIL":
            failed.append(gate.gate)
        elif gate.status == "WARN":
            warning.append(gate.gate)
        elif gate.status == "NOT_EVALUATED":
            not_evaluated.append(gate.gate)

    return {
        "passed": passed,
        "failed": failed,
        "warning": warning,
        "not_evaluated": not_evaluated,
    }


def determine_final_status(
    gates: Sequence[GateResult],
    pair_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Phase32Config,
) -> Phase32FinalStatus:
    status_map = {gate.gate: gate for gate in gates}
    summary = summarize_gate_status(gates)

    f2_status = status_map.get("F2").status if "F2" in status_map else "NOT_EVALUATED"
    f2_pass = f2_status == "PASS"
    f6_pass = status_map.get("F6").status == "PASS" if "F6" in status_map else False
    f7_pass = status_map.get("F7").status == "PASS" if "F7" in status_map else False
    f8_pass = status_map.get("F8").status == "PASS" if "F8" in status_map else False
    f11_pass = status_map.get("F11").status == "PASS" if "F11" in status_map else False

    f12_warn = status_map.get("F12").status == "WARN" if "F12" in status_map else False

    strong_primary = primary_strong_candidate_rows(pair_df, cfg)
    strong_any = strong_candidate_rows(pair_df, cfg)
    positive_any = positive_lift_rows(pair_df)
    exploratory = exploratory_lead_rows(pair_df, cfg)

    controls_stale_or_missing = (
        f2_status == "NOT_EVALUATED"
        or controls_summary.get("stale", False)
        or controls_summary.get("available") is False
    )

    if not strong_primary.empty and f2_pass and f6_pass and f7_pass and f8_pass and f11_pass and not f12_warn:
        status = "candidate_signal"
        interpretation = (
            "A primary conditional EEG-RNG candidate signal passed lift, subject "
            "consistency, complexity, controls, saturation diagnostics and multiple-comparison safeguards. "
            "This is a candidate signal requiring diagnostic review and independent replication."
        )

    elif controls_stale_or_missing:
        if not positive_any.empty:
            status = "pending_controls_exploratory_lead"
            interpretation = (
                "The leakage-safe conditional estimator produced positive exploratory lift rows, "
                "but updated negative controls have not yet been run. No positive claim is allowed "
                "until controls are regenerated with the patched estimator."
            )
        else:
            status = "pending_controls_conditional_null"
            interpretation = (
                "No positive conditional lift rows were detected, but updated negative controls "
                "are still pending. A clean null interpretation requires controls generated with "
                "the patched leakage-safe estimator."
            )

    elif not f2_pass:
        status = "blocked_by_controls"
        interpretation = (
            "The conditional analysis cannot support a positive claim because the updated "
            "negative controls did not collapse within the required tolerance. This blocks "
            "strong interpretation even if exploratory lift appears."
        )

    elif f12_warn and not positive_any.empty:
        status = "exploratory_lead_with_saturation_warning"
        interpretation = (
            "Exploratory conditional lift appeared, but epsilon saturation diagnostics raised "
            "a warning. The result cannot be interpreted as evidence without additional diagnostics."
        )

    elif strong_any.empty and positive_any.empty:
        status = "conditional_null_result"
        interpretation = (
            "No positive conditional EEG-RNG lift was detected under the leakage-safe estimator. "
            "With controls acceptable, this supports a conditional null result under the current "
            "public-data design."
        )

    elif strong_any.empty and not exploratory.empty:
        status = "exploratory_lead_only"
        interpretation = (
            "Some positive conditional lift rows appeared, but they did not satisfy the full "
            "registered gate structure. They are exploratory leads only, not evidence."
        )

    else:
        status = "exploratory_lead_only"
        interpretation = (
            "Conditional leads exist, but they do not satisfy the full primary gate structure. "
            "They must be treated as exploratory."
        )

    return Phase32FinalStatus(
        status=status,
        interpretation=interpretation,
        passed_gates=summary["passed"],
        failed_gates=summary["failed"],
        warning_gates=summary["warning"],
        not_evaluated_gates=summary["not_evaluated"],
    )


# =========================================================
# Multiple-comparison guard report
# =========================================================

def build_multiple_comparison_guard_report(
    pair_df: pd.DataFrame,
    cfg: Phase32Config,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)
    positive = positive_lift_rows(pair_df)
    primary_positive = primary_positive_lift_rows(pair_df)
    exploratory = exploratory_lead_rows(pair_df, cfg)
    strong = strong_candidate_rows(pair_df, cfg)
    primary_strong = primary_strong_candidate_rows(pair_df, cfg)

    return {
        "enabled": bool(cfg.enable_multiple_comparison_guard),
        "allow_strong_claim_from_secondary_tests": bool(cfg.allow_strong_claim_from_secondary_tests),
        "n_valid_tests": int(len(valid)),
        "n_positive_lift_tests": int(len(positive)),
        "n_primary_positive_lift_tests": int(len(primary_positive)),
        "n_exploratory_lift_tests": int(len(exploratory)),
        "n_strong_candidate_tests": int(len(strong)),
        "n_primary_strong_candidate_tests": int(len(primary_strong)),
        "primary_regimes": list(cfg.primary_regimes),
        "secondary_regimes": list(cfg.secondary_regimes),
        "primary_lags_epochs": list(cfg.primary_lags_epochs),
        "secondary_lags_epochs": list(cfg.secondary_lags_epochs),
        "positive_lift_rows": positive.to_dict(orient="records") if not positive.empty else [],
        "exploratory_lift_rows": exploratory.to_dict(orient="records") if not exploratory.empty else [],
        "strong_candidates": strong.to_dict(orient="records") if not strong.empty else [],
        "primary_strong_candidates": (
            primary_strong.to_dict(orient="records")
            if not primary_strong.empty
            else []
        ),
        "interpretation": (
            "Strong claims are allowed only from primary tests. Secondary regimes or "
            "non-primary lags can generate leads but not claims without replication."
        ),
    }


# =========================================================
# Output formatting
# =========================================================

def gates_to_dataframe(gates: Sequence[GateResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(gate) for gate in gates])


def build_metrics_summary_text(
    gates: Sequence[GateResult],
    final_status: Phase32FinalStatus,
    mc_guard: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Phase32Config,
) -> str:
    lines: List[str] = []

    model_sat = model_saturation_summary(model_df, cfg)
    pair_sat = pair_saturation_summary(pair_df, cfg)

    lines.append("=" * 78)
    lines.append("Phase III.2 — Metrics and Gates Summary")
    lines.append("Leakage-safe conditional estimator aware version")
    lines.append("=" * 78)
    lines.append("")

    lines.append("Final status")
    lines.append("-" * 78)
    lines.append(f"status: {final_status.status}")
    lines.append(f"interpretation: {final_status.interpretation}")
    lines.append(f"passed_gates: {final_status.passed_gates}")
    lines.append(f"failed_gates: {final_status.failed_gates}")
    lines.append(f"warning_gates: {final_status.warning_gates}")
    lines.append(f"not_evaluated_gates: {final_status.not_evaluated_gates}")
    lines.append("")

    lines.append("Gates")
    lines.append("-" * 78)

    for gate in gates:
        lines.append(
            f"{gate.gate} — {gate.name}: {gate.status} | "
            f"value={gate.value} | threshold={gate.threshold} | reason={gate.reason}"
        )

    lines.append("")

    lines.append("Multiple-comparison guard")
    lines.append("-" * 78)
    lines.append(f"enabled: {mc_guard.get('enabled')}")
    lines.append(f"n_valid_tests: {mc_guard.get('n_valid_tests')}")
    lines.append(f"n_positive_lift_tests: {mc_guard.get('n_positive_lift_tests')}")
    lines.append(f"n_primary_positive_lift_tests: {mc_guard.get('n_primary_positive_lift_tests')}")
    lines.append(f"n_exploratory_lift_tests: {mc_guard.get('n_exploratory_lift_tests')}")
    lines.append(f"n_strong_candidate_tests: {mc_guard.get('n_strong_candidate_tests')}")
    lines.append(f"n_primary_strong_candidate_tests: {mc_guard.get('n_primary_strong_candidate_tests')}")
    lines.append("")

    lines.append("Epsilon saturation")
    lines.append("-" * 78)
    lines.append(f"model_saturation: {model_sat}")
    lines.append(f"pair_saturation: {pair_sat}")
    lines.append("")

    lines.append("Controls")
    lines.append("-" * 78)
    lines.append(f"controls_available: {controls_summary.get('available')}")
    lines.append(f"controls_stale: {controls_summary.get('stale')}")
    lines.append(f"controls_passed: {controls_summary.get('passed')}")
    lines.append(f"failed_control_types: {controls_summary.get('failed_control_types')}")
    lines.append(f"controls_reason: {controls_summary.get('reason')}")
    lines.append("")

    lines.append("Best conditional rows")
    lines.append("-" * 78)

    for column in ["conditional_lift", "bic_improvement", "ll_improvement", "fraction_positive_lift"]:
        row = best_row_by(pair_df, column, descending=True)

        if row:
            lines.append(f"best_by_{column}: {row}")

    lines.append("")
    lines.append("=" * 78)

    return "\n".join(lines)


# =========================================================
# Save outputs
# =========================================================

def save_metrics_outputs(
    gates: Sequence[GateResult],
    final_status: Phase32FinalStatus,
    mc_guard: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    gates_df = gates_to_dataframe(gates)

    metrics_table_csv = Path(cfg.results_dir) / "phase3_2_metrics_table.csv"
    metrics_summary_txt = Path(cfg.results_dir) / "phase3_2_metrics_summary.txt"

    gates_df.to_csv(metrics_table_csv, index=False)

    gates_payload = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": "Phase_III_2_metrics_leakage_safe",
        "final_status": asdict(final_status),
        "gates": [asdict(gate) for gate in gates],
        "epsilon_saturation": {
            "model_saturation": model_saturation_summary(model_df, cfg),
            "pair_saturation": pair_saturation_summary(pair_df, cfg),
        },
    }

    save_json(Path(cfg.gates_json), gates_payload)
    save_json(Path(cfg.multiple_comparison_guard_json), dict(mc_guard))

    with open(metrics_summary_txt, "w", encoding="utf-8") as f:
        f.write(
            build_metrics_summary_text(
                gates=gates,
                final_status=final_status,
                mc_guard=mc_guard,
                pair_df=pair_df,
                model_df=model_df,
                controls_summary=controls_summary,
                cfg=cfg,
            )
        )

    print(f"[Phase3.2 Metrics] Saved gates JSON: {cfg.gates_json}")
    print(f"[Phase3.2 Metrics] Saved multiple-comparison guard: {cfg.multiple_comparison_guard_json}")
    print(f"[Phase3.2 Metrics] Saved metrics table: {metrics_table_csv}")
    print(f"[Phase3.2 Metrics] Saved metrics summary: {metrics_summary_txt}")


# =========================================================
# Main
# =========================================================

def run_phase3_2_metrics(cfg: Optional[Phase32Config] = None) -> Phase32FinalStatus:
    if cfg is None:
        cfg = load_phase3_2_config()

    pair_df = load_conditional_pair_scores(cfg)
    model_df = load_conditional_model_scores(cfg)
    controls_summary = load_controls_summary(cfg)

    gates = evaluate_phase3_2_gates(
        pair_df=pair_df,
        model_df=model_df,
        controls_summary=controls_summary,
        cfg=cfg,
    )

    final_status = determine_final_status(
        gates=gates,
        pair_df=pair_df,
        controls_summary=controls_summary,
        cfg=cfg,
    )

    mc_guard = build_multiple_comparison_guard_report(
        pair_df=pair_df,
        cfg=cfg,
    )

    save_metrics_outputs(
        gates=gates,
        final_status=final_status,
        mc_guard=mc_guard,
        pair_df=pair_df,
        model_df=model_df,
        controls_summary=controls_summary,
        cfg=cfg,
    )

    return final_status


def main() -> None:
    cfg = load_phase3_2_config()

    final_status = run_phase3_2_metrics(cfg)

    print("\n============================================================")
    print("Phase III.2 metrics completed")
    print("Leakage-safe conditional estimator aware version")
    print("============================================================")
    print(f"status: {final_status.status}")
    print(f"interpretation: {final_status.interpretation}")
    print(f"passed_gates: {final_status.passed_gates}")
    print(f"failed_gates: {final_status.failed_gates}")
    print(f"warning_gates: {final_status.warning_gates}")
    print(f"not_evaluated_gates: {final_status.not_evaluated_gates}")
    print("============================================================\n")


if __name__ == "__main__":
    main()