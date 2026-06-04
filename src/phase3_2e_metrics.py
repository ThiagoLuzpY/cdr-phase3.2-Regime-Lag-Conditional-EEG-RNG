from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    active_conditional_pairs,
    describe_config,
    load_phase3_2e_config,
)

from src.phase3_2e_conditional import (
    build_phase3_2e_conditional,
    conditional_model_scores_csv_path,
    conditional_pair_scores_csv_path,
)

from src.phase3_2e_controls import (
    build_phase3_2e_controls,
    controls_json_path,
    control_runs_csv_path,
    control_pair_scores_csv_path,
)


# =========================================================
# Phase III.2E — Metrics and Gates
# Synchronized EEG–QRNG protocol version
# =========================================================
#
# This module consolidates:
#
#   - synchronized conditional CDR results;
#   - synchronized negative controls;
#   - conditional lift;
#   - subject consistency;
#   - BIC / complexity penalty;
#   - lag/window specificity;
#   - multiple-comparison guard;
#   - epsilon saturation diagnostics;
#   - final Phase III.2E interpretation.
#
# Important:
#
#   This module does NOT weaken official gates.
#   Gate sensitivity analysis for F7/F8/F12 belongs to:
#
#       src/phase3_2e_gate_sensitivity.py
#
# If no synchronized dataset exists:
#
#   status = pending_synchronized_data
#   protocol_only = True
#   metrics are generated as protocol-only outputs
#   empirical claim remains blocked
#


# =========================================================
# Constants
# =========================================================

METRICS_MODULE = "phase3_2e_metrics"
METRICS_VERSION = "phase3_2e_metrics_v1_sync_gates_protocol_aware"


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
class Phase32EFinalStatus:
    status: str
    protocol_only: bool
    interpretation: str
    passed_gates: List[str]
    failed_gates: List[str]
    warning_gates: List[str]
    not_evaluated_gates: List[str]
    can_claim_empirical_cdr: bool
    can_proceed_to_gate_sensitivity: bool
    can_proceed_to_diagnostics: bool


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
        gate=str(gate),
        name=str(name),
        passed=bool(passed),
        status=_gate_status(passed, available=available, warning=warning),
        value=float(value),
        threshold=float(threshold),
        reason=str(reason),
        details=details or {},
    )


def _cfg_float(cfg: Phase32EConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Phase32EConfig, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: Phase32EConfig, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


# =========================================================
# Output paths
# =========================================================

def gates_json_path(cfg: Phase32EConfig) -> Path:
    return Path(getattr(cfg, "gates_json", Path(cfg.results_dir) / "phase3_2e_gates.json"))


def multiple_comparison_guard_json_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "multiple_comparison_guard_json",
            Path(cfg.results_dir) / "phase3_2e_multiple_comparison_guard.json",
        )
    )


def metrics_table_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_table.csv"


def metrics_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_summary.txt"


def metrics_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_report.json"


# =========================================================
# Load Phase III.2E outputs
# =========================================================

def load_conditional_pair_scores(cfg: Phase32EConfig) -> pd.DataFrame:
    return _load_csv(conditional_pair_scores_csv_path(cfg))


def load_conditional_model_scores(cfg: Phase32EConfig) -> pd.DataFrame:
    return _load_csv(conditional_model_scores_csv_path(cfg))


def load_controls_payload(cfg: Phase32EConfig) -> Dict[str, Any]:
    return _load_json(controls_json_path(cfg))


def load_controls_summary(cfg: Phase32EConfig) -> Dict[str, Any]:
    payload = load_controls_payload(cfg)

    if not payload:
        return {
            "available": False,
            "protocol_only": False,
            "passed": False,
            "reason": "controls_json_missing",
            "status": "controls_missing",
            "result": {},
            "summary": {},
        }

    status = str(payload.get("status", ""))
    result = payload.get("result", {})
    summary = payload.get("summary", {})

    protocol_only = bool(result.get("protocol_only", False))

    if status == getattr(cfg, "status_pending_synchronized_data", "pending_synchronized_data"):
        return {
            "available": False,
            "protocol_only": True,
            "passed": False,
            "reason": "pending_synchronized_data",
            "status": status,
            "result": result,
            "summary": summary,
            "failed_control_types": result.get("failed_control_types", []),
            "required_control_types": summary.get("required_control_types", getattr(cfg, "required_control_types", [])),
        }

    if not summary:
        return {
            "available": True,
            "protocol_only": protocol_only,
            "passed": False,
            "reason": "controls_summary_empty",
            "status": status,
            "result": result,
            "summary": {},
        }

    out = dict(summary)
    out["available"] = bool(summary.get("available", True))
    out["protocol_only"] = protocol_only
    out["passed"] = bool(summary.get("passed", result.get("overall_passed", False)))
    out["reason"] = summary.get("reason", result.get("interpretation", "ok"))
    out["status"] = status
    out["result"] = result
    out["summary"] = summary
    out["failed_control_types"] = summary.get("failed_control_types", result.get("failed_control_types", []))
    out["required_control_types"] = summary.get("required_control_types", getattr(cfg, "required_control_types", []))

    return out


def load_control_runs(cfg: Phase32EConfig) -> pd.DataFrame:
    return _load_csv(control_runs_csv_path(cfg))


def load_control_pair_scores(cfg: Phase32EConfig) -> pd.DataFrame:
    return _load_csv(control_pair_scores_csv_path(cfg))


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


def exploratory_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    if "primary" not in valid.columns:
        return valid.copy()

    return valid[~_as_bool_series(valid["primary"])].copy()


def positive_lift_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty or "conditional_lift" not in valid.columns:
        return pd.DataFrame()

    return valid[_numeric_col(valid, "conditional_lift") > 0.0].copy()


def primary_positive_lift_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    primary = primary_pair_rows(pair_df)

    if primary.empty or "conditional_lift" not in primary.columns:
        return pd.DataFrame()

    return primary[_numeric_col(primary, "conditional_lift") > 0.0].copy()


def multichannel_pair_rows(pair_df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    if "multichannel" in valid.columns:
        return valid[_as_bool_series(valid["multichannel"])].copy()

    mask = pd.Series(np.zeros(len(valid), dtype=bool), index=valid.index)

    for col in ["baseline_model", "augmented_model", "family"]:
        if col not in valid.columns:
            continue

        text = valid[col].astype(str)
        mask |= text.str.contains("MCEEG", case=False, na=False)
        mask |= text.str.contains("MC_EEG", case=False, na=False)
        mask |= text.str.contains("multichannel", case=False, na=False)
        mask |= text.str.startswith("E4")
        mask |= text.str.startswith("E5")
        mask |= text.str.startswith("E6")

    return valid[mask].copy()


def strong_candidate_rows(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    required = [
        "conditional_lift",
        "augmented_eps_test",
        "fraction_positive_lift",
        "bic_improvement",
    ]

    if any(col not in valid.columns for col in required):
        return pd.DataFrame()

    strong = valid[
        (_numeric_col(valid, "conditional_lift") >= float(cfg.conditional_lift_min))
        & (_numeric_col(valid, "augmented_eps_test") >= float(cfg.strong_eps_min))
        & (_numeric_col(valid, "fraction_positive_lift") >= float(cfg.subject_effect_fraction))
        & (_numeric_col(valid, "bic_improvement") >= 0.0)
    ].copy()

    return strong


def primary_strong_candidate_rows(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    strong = strong_candidate_rows(pair_df, cfg)

    if strong.empty or "primary" not in strong.columns:
        return pd.DataFrame()

    return strong[_as_bool_series(strong["primary"])].copy()


def exploratory_lead_rows(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    positive = positive_lift_rows(pair_df)

    if positive.empty:
        return pd.DataFrame()

    strong_idx = set(strong_candidate_rows(pair_df, cfg).index.tolist())

    return positive[~positive.index.isin(strong_idx)].copy()


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
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    if model_df.empty:
        return {
            "available": False,
            "reason": "model_scores_missing_or_empty",
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

    eps_values = _numeric_col(model_df, eps_col)

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
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_pair_rows",
            "warning": False,
            "either_saturation_fraction": 0.0,
            "both_saturation_fraction": 0.0,
        }

    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 0.80)
    warn_fraction = _cfg_float(cfg, "eps_saturation_warning_fraction", 0.50)

    baseline_eps = _numeric_col(valid, "baseline_eps_test")
    augmented_eps = _numeric_col(valid, "augmented_eps_test")

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

def gate_f1_injection_recovery(cfg: Phase32EConfig) -> GateResult:
    return _gate(
        gate="F1",
        name="Injection recovery",
        passed=False,
        value=0.0,
        threshold=float(getattr(cfg, "gate_tol_abs", 0.05)),
        reason="not_evaluated_in_metrics; reserved for runner-level or synthetic injection test",
        details={
            "inj_eps_true": float(getattr(cfg, "inj_eps_true", 0.05)),
            "note": "This gate is NOT_EVALUATED here, not interpreted as a scientific failure.",
        },
        available=False,
    )


def gate_f2_controls_collapse(
    controls_summary: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> GateResult:
    if not controls_summary:
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(getattr(cfg, "required_control_fraction", 1.0)),
            reason="controls_summary_missing",
            details={},
            available=False,
        )

    if controls_summary.get("protocol_only", False):
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(getattr(cfg, "required_control_fraction", 1.0)),
            reason="pending_synchronized_data; controls cannot run without real synchronized frames",
            details=dict(controls_summary),
            available=False,
        )

    if controls_summary.get("available") is False:
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=float(getattr(cfg, "required_control_fraction", 1.0)),
            reason=str(controls_summary.get("reason", "controls_unavailable")),
            details=dict(controls_summary),
            available=False,
        )

    passed = bool(controls_summary.get("passed", False))
    failed_types = controls_summary.get("failed_control_types", [])
    control_summaries = controls_summary.get("control_summaries", [])

    fractions = [
        _safe_float(item.get("fraction_runs_collapsed_overall", item.get("fraction_runs_below_control_tol", 0.0)))
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
        threshold=float(getattr(cfg, "required_control_fraction", 1.0)),
        reason=reason,
        details={
            "failed_control_types": failed_types,
            "control_summaries": control_summaries,
            "required_control_types": controls_summary.get("required_control_types", []),
        },
    )


def gate_f3_holdout_generalization(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F3",
            name="Holdout generalization",
            passed=False,
            value=0.0,
            threshold=1.0,
            reason="no_valid_conditional_pair_rows",
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
        reason="valid synchronized ref/calib/test conditional evaluations exist" if passed else "no valid evaluations",
        details={
            "n_valid_pair_rows": n_valid,
            "n_total_pair_rows": n_total,
            "split": "reference_train -> calibration -> test_holdout",
        },
    )


def gate_f5_sensitivity_stability(cfg: Phase32EConfig) -> GateResult:
    return _gate(
        gate="F5",
        name="Sensitivity stability",
        passed=False,
        value=0.0,
        threshold=float(getattr(cfg, "sensitivity_max_delta", 0.10)),
        reason="not_evaluated_in_metrics; evaluated by phase3_2e_gate_sensitivity.py",
        details={
            "next_module": "src.phase3_2e_gate_sensitivity",
            "focus": "F7/F8/F12 official-threshold sensitivity without changing official gates",
        },
        available=False,
    )


def gate_f6_conditional_lift(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
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

    max_lift = float(_numeric_col(primary, "conditional_lift").max())
    max_eps = float(_numeric_col(primary, "augmented_eps_test").max())

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


def gate_f7_subject_consistency(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
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

    max_fraction = float(_numeric_col(primary, "fraction_positive_lift").max())
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


def gate_f8_complexity_penalty(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
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

    max_bic_improvement = float(_numeric_col(primary, "bic_improvement").max())
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


def gate_f9_lag_stability(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F9",
            name="Lag stability",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="no_valid_pair_rows",
            details={},
            available=False,
        )

    if "lag_windows" not in valid.columns:
        return _gate(
            gate="F9",
            name="Lag stability",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="lag_windows_column_missing",
            details={},
            available=False,
        )

    primary_lags = set(int(x) for x in getattr(cfg, "primary_test_lags_windows", [0]))

    lag_values = pd.to_numeric(valid["lag_windows"], errors="coerce").fillna(999999).astype(int)

    primary_lag_rows = valid[lag_values.isin(primary_lags)].copy()
    secondary_lag_rows = valid[~lag_values.isin(primary_lags)].copy()

    primary_max_lift = (
        float(_numeric_col(primary_lag_rows, "conditional_lift").max())
        if not primary_lag_rows.empty
        else 0.0
    )

    secondary_max_lift = (
        float(_numeric_col(secondary_lag_rows, "conditional_lift").max())
        if not secondary_lag_rows.empty
        else 0.0
    )

    any_meaningful_secondary = secondary_max_lift >= float(cfg.conditional_lift_min)
    primary_competitive = primary_max_lift >= (secondary_max_lift - 1e-12)

    passed = (not any_meaningful_secondary) or primary_competitive

    reason = "ok" if passed else "secondary_lag_effect_exceeds_primary_lag_effect"

    return _gate(
        gate="F9",
        name="Lag stability",
        passed=passed,
        value=primary_max_lift - secondary_max_lift,
        threshold=0.0,
        reason=reason,
        details={
            "primary_test_lags_windows": list(primary_lags),
            "primary_max_lift": primary_max_lift,
            "secondary_max_lift": secondary_max_lift,
            "any_meaningful_secondary": bool(any_meaningful_secondary),
        },
    )


def gate_f10_window_specificity(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F10",
            name="Window specificity",
            passed=False,
            value=0.0,
            threshold=float(cfg.conditional_lift_min),
            reason="no_valid_pair_rows",
            details={},
            available=False,
        )

    if "window_seconds" not in valid.columns:
        return _gate(
            gate="F10",
            name="Window specificity",
            passed=False,
            value=0.0,
            threshold=float(cfg.conditional_lift_min),
            reason="window_seconds_column_missing",
            details={},
            available=False,
        )

    primary_windows = set(int(x) for x in getattr(cfg, "primary_test_window_seconds", []))

    window_values = pd.to_numeric(valid["window_seconds"], errors="coerce").fillna(-1).astype(int)

    primary = valid[window_values.isin(primary_windows)].copy()
    secondary = valid[~window_values.isin(primary_windows)].copy()

    primary_max = (
        float(_numeric_col(primary, "conditional_lift").max())
        if not primary.empty
        else 0.0
    )

    secondary_max = (
        float(_numeric_col(secondary, "conditional_lift").max())
        if not secondary.empty
        else 0.0
    )

    passed = primary_max >= float(cfg.conditional_lift_min) or secondary_max < float(cfg.conditional_lift_min)

    reason = "ok" if passed else "only_secondary_or_exploratory_window_has_meaningful_lift"

    return _gate(
        gate="F10",
        name="Window specificity",
        passed=passed,
        value=primary_max,
        threshold=float(cfg.conditional_lift_min),
        reason=reason,
        details={
            "primary_test_window_seconds": sorted(primary_windows),
            "primary_max_lift": primary_max,
            "secondary_max_lift": secondary_max,
        },
    )


def gate_f11_multiple_comparison_guard(pair_df: pd.DataFrame, cfg: Phase32EConfig) -> GateResult:
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

    reason = "ok" if passed else "strong_candidates_exist_only_in_non_primary_tests"

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
            "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        },
    )


def gate_f12_epsilon_saturation(
    model_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
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

    if not model_summary.get("available", False) and not pair_summary.get("available", False):
        return _gate(
            gate="F12",
            name="Epsilon saturation diagnostic",
            passed=False,
            value=0.0,
            threshold=threshold,
            reason="no_epsilon_rows_available",
            details={
                "model_saturation": model_summary,
                "pair_saturation": pair_summary,
            },
            available=False,
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

def evaluate_phase3_2e_gates(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> List[GateResult]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    gates = [
        gate_f1_injection_recovery(cfg),
        gate_f2_controls_collapse(controls_summary, cfg),
        gate_f3_holdout_generalization(pair_df, cfg),
        gate_f5_sensitivity_stability(cfg),
        gate_f6_conditional_lift(pair_df, cfg),
        gate_f7_subject_consistency(pair_df, cfg),
        gate_f8_complexity_penalty(pair_df, cfg),
        gate_f9_lag_stability(pair_df, cfg),
        gate_f10_window_specificity(pair_df, cfg),
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
    cfg: Phase32EConfig,
) -> Phase32EFinalStatus:
    status_map = {gate.gate: gate for gate in gates}
    gate_summary = summarize_gate_status(gates)

    protocol_only = bool(controls_summary.get("protocol_only", False))

    if protocol_only and pair_df.empty:
        return Phase32EFinalStatus(
            status=getattr(cfg, "status_pending_synchronized_data", "pending_synchronized_data"),
            protocol_only=True,
            interpretation=(
                "No synchronized EEG–QRNG dataset is available. Phase III.2E metrics "
                "were generated in protocol-only mode. Empirical CDR remains blocked "
                "until real synchronized data are provided."
            ),
            passed_gates=gate_summary["passed"],
            failed_gates=gate_summary["failed"],
            warning_gates=gate_summary["warning"],
            not_evaluated_gates=gate_summary["not_evaluated"],
            can_claim_empirical_cdr=False,
            can_proceed_to_gate_sensitivity=True,
            can_proceed_to_diagnostics=True,
        )

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

    controls_missing_or_not_evaluated = (
        f2_status == "NOT_EVALUATED"
        or controls_summary.get("available") is False
    )

    if not strong_primary.empty and f2_pass and f6_pass and f7_pass and f8_pass and f11_pass and not f12_warn:
        status = "candidate_synchronized_signal"
        interpretation = (
            "A primary synchronized EEG–QRNG conditional candidate passed lift, "
            "subject consistency, complexity penalty, controls, saturation diagnostics "
            "and multiple-comparison safeguards. This is a candidate signal requiring "
            "diagnostic review and independent replication."
        )
        can_claim = True

    elif controls_missing_or_not_evaluated:
        if not positive_any.empty:
            status = "pending_controls_exploratory_lead"
            interpretation = (
                "Synchronized conditional CDR produced positive exploratory lift rows, "
                "but required synchronized controls are missing or not evaluated. No "
                "positive claim is allowed."
            )
        else:
            status = "pending_controls_conditional_null"
            interpretation = (
                "No positive synchronized conditional lift rows were detected, but "
                "required controls are missing or not evaluated. Clean null interpretation "
                "requires valid controls."
            )
        can_claim = False

    elif not f2_pass:
        status = "blocked_by_controls"
        interpretation = (
            "Required synchronized negative controls did not collapse sufficiently. "
            "Any positive EEG–QRNG interpretation is blocked."
        )
        can_claim = False

    elif f12_warn and not positive_any.empty:
        status = "exploratory_lead_with_saturation_warning"
        interpretation = (
            "Exploratory synchronized conditional lift appeared, but epsilon saturation "
            "diagnostics raised a warning. The result cannot be interpreted as evidence "
            "without additional diagnostics."
        )
        can_claim = False

    elif strong_any.empty and positive_any.empty:
        status = "synchronized_conditional_null_result"
        interpretation = (
            "No positive synchronized conditional EEG–QRNG lift was detected under the "
            "leakage-safe estimator. With acceptable controls, this supports a null result "
            "under the current synchronized protocol."
        )
        can_claim = False

    elif strong_any.empty and not exploratory.empty:
        status = "exploratory_lead_only"
        interpretation = (
            "Some synchronized positive conditional lift rows appeared, but they did not "
            "satisfy the full registered gate structure. They are exploratory leads only."
        )
        can_claim = False

    else:
        status = "exploratory_lead_only"
        interpretation = (
            "Conditional leads exist, but they do not satisfy the full primary gate "
            "structure. Treat them as exploratory."
        )
        can_claim = False

    return Phase32EFinalStatus(
        status=status,
        protocol_only=False,
        interpretation=interpretation,
        passed_gates=gate_summary["passed"],
        failed_gates=gate_summary["failed"],
        warning_gates=gate_summary["warning"],
        not_evaluated_gates=gate_summary["not_evaluated"],
        can_claim_empirical_cdr=bool(can_claim),
        can_proceed_to_gate_sensitivity=True,
        can_proceed_to_diagnostics=True,
    )


# =========================================================
# Multiple-comparison guard report
# =========================================================

def build_multiple_comparison_guard_report(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)
    positive = positive_lift_rows(pair_df)
    primary_positive = primary_positive_lift_rows(pair_df)
    exploratory = exploratory_lead_rows(pair_df, cfg)
    strong = strong_candidate_rows(pair_df, cfg)
    primary_strong = primary_strong_candidate_rows(pair_df, cfg)
    multichannel = multichannel_pair_rows(pair_df)

    return {
        "enabled": bool(getattr(cfg, "multiple_comparison_guard", True)),
        "allow_strong_claim_from_secondary_tests": bool(
            getattr(cfg, "allow_strong_claim_from_secondary_tests", False)
        ),
        "n_valid_tests": int(len(valid)),
        "n_positive_lift_tests": int(len(positive)),
        "n_primary_positive_lift_tests": int(len(primary_positive)),
        "n_exploratory_lift_tests": int(len(exploratory)),
        "n_strong_candidate_tests": int(len(strong)),
        "n_primary_strong_candidate_tests": int(len(primary_strong)),
        "n_multichannel_tests": int(len(multichannel)),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "primary_conditional_pairs": list(getattr(cfg, "primary_conditional_pairs", [])),
        "multichannel_conditional_pairs": list(getattr(cfg, "multichannel_conditional_pairs", [])),
        "primary_test_window_seconds": list(getattr(cfg, "primary_test_window_seconds", [])),
        "primary_test_lags_windows": list(getattr(cfg, "primary_test_lags_windows", [])),
        "positive_lift_rows": positive.to_dict(orient="records") if not positive.empty else [],
        "exploratory_lift_rows": exploratory.to_dict(orient="records") if not exploratory.empty else [],
        "strong_candidates": strong.to_dict(orient="records") if not strong.empty else [],
        "primary_strong_candidates": (
            primary_strong.to_dict(orient="records")
            if not primary_strong.empty
            else []
        ),
        "interpretation": (
            "Strong claims are allowed only from registered primary tests. "
            "Exploratory windows, exploratory lags or secondary model families can "
            "generate leads but not claims without replication."
        ),
    }


# =========================================================
# Output formatting
# =========================================================

def gates_to_dataframe(gates: Sequence[GateResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(gate) for gate in gates])


def build_metrics_summary_text(
    gates: Sequence[GateResult],
    final_status: Phase32EFinalStatus,
    mc_guard: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> str:
    lines: List[str] = []

    model_sat = model_saturation_summary(model_df, cfg)
    pair_sat = pair_saturation_summary(pair_df, cfg)

    lines.append("=" * 78)
    lines.append("Phase III.2E — Metrics and Gates Summary")
    lines.append("Synchronized EEG–QRNG protocol version")
    lines.append("=" * 78)
    lines.append("")

    lines.append("Final status")
    lines.append("-" * 78)
    lines.append(f"status: {final_status.status}")
    lines.append(f"protocol_only: {final_status.protocol_only}")
    lines.append(f"can_claim_empirical_cdr: {final_status.can_claim_empirical_cdr}")
    lines.append(f"can_proceed_to_gate_sensitivity: {final_status.can_proceed_to_gate_sensitivity}")
    lines.append(f"can_proceed_to_diagnostics: {final_status.can_proceed_to_diagnostics}")
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
    lines.append(f"n_multichannel_tests: {mc_guard.get('n_multichannel_tests')}")
    lines.append("")

    lines.append("Epsilon saturation")
    lines.append("-" * 78)
    lines.append(f"model_saturation: {model_sat}")
    lines.append(f"pair_saturation: {pair_sat}")
    lines.append("")

    lines.append("Controls")
    lines.append("-" * 78)
    lines.append(f"controls_available: {controls_summary.get('available')}")
    lines.append(f"controls_protocol_only: {controls_summary.get('protocol_only')}")
    lines.append(f"controls_passed: {controls_summary.get('passed')}")
    lines.append(f"failed_control_types: {controls_summary.get('failed_control_types')}")
    lines.append(f"required_control_types: {controls_summary.get('required_control_types')}")
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
    final_status: Phase32EFinalStatus,
    mc_guard: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    gates_df = gates_to_dataframe(gates)
    gates_df.to_csv(metrics_table_csv_path(cfg), index=False)

    payload = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": METRICS_MODULE,
        "metrics_version": METRICS_VERSION,
        "created_at": _now_str(),
        "status": final_status.status,
        "final_status": asdict(final_status),
        "gates": [asdict(gate) for gate in gates],
        "multiple_comparison_guard": dict(mc_guard),
        "epsilon_saturation": {
            "model_saturation": model_saturation_summary(model_df, cfg),
            "pair_saturation": pair_saturation_summary(pair_df, cfg),
        },
        "controls_summary": dict(controls_summary),
        "files": {
            "gates_json": str(gates_json_path(cfg)),
            "multiple_comparison_guard_json": str(multiple_comparison_guard_json_path(cfg)),
            "metrics_table_csv": str(metrics_table_csv_path(cfg)),
            "metrics_summary_txt": str(metrics_summary_txt_path(cfg)),
            "metrics_report_json": str(metrics_report_json_path(cfg)),
        },
        "config": describe_config(cfg),
    }

    save_json(gates_json_path(cfg), payload)
    save_json(multiple_comparison_guard_json_path(cfg), dict(mc_guard))
    save_json(metrics_report_json_path(cfg), payload)

    with open(metrics_summary_txt_path(cfg), "w", encoding="utf-8") as f:
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

    print(f"[Phase3.2E Metrics] Saved gates JSON: {gates_json_path(cfg)}")
    print(f"[Phase3.2E Metrics] Saved multiple-comparison guard: {multiple_comparison_guard_json_path(cfg)}")
    print(f"[Phase3.2E Metrics] Saved metrics table: {metrics_table_csv_path(cfg)}")
    print(f"[Phase3.2E Metrics] Saved metrics summary: {metrics_summary_txt_path(cfg)}")
    print(f"[Phase3.2E Metrics] Saved metrics report: {metrics_report_json_path(cfg)}")

    return payload


# =========================================================
# Main API
# =========================================================

def run_phase3_2e_metrics(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_conditional: bool = False,
    force_rebuild_controls: bool = False,
) -> Phase32EFinalStatus:
    if cfg is None:
        cfg = load_phase3_2e_config()

    if force_rebuild_conditional:
        build_phase3_2e_conditional(cfg=cfg, force_rebuild_alignment=False)

    if force_rebuild_controls:
        build_phase3_2e_controls(cfg=cfg, force_rebuild_alignment=False)

    pair_df = load_conditional_pair_scores(cfg)
    model_df = load_conditional_model_scores(cfg)
    controls_summary = load_controls_summary(cfg)

    gates = evaluate_phase3_2e_gates(
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


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III.2E metrics and official gates."
    )

    parser.add_argument(
        "--force-rebuild-conditional",
        action="store_true",
        help="Force rebuilding conditional outputs before metrics.",
    )

    parser.add_argument(
        "--force-rebuild-controls",
        action="store_true",
        help="Force rebuilding control outputs before metrics.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    final_status = run_phase3_2e_metrics(
        cfg=cfg,
        force_rebuild_conditional=bool(args.force_rebuild_conditional),
        force_rebuild_controls=bool(args.force_rebuild_controls),
    )

    print("\n" + "=" * 78)
    print("Phase III.2E metrics completed")
    print("Synchronized EEG–QRNG protocol version")
    print("=" * 78)
    print(f"status: {final_status.status}")
    print(f"protocol_only: {final_status.protocol_only}")
    print(f"can_claim_empirical_cdr: {final_status.can_claim_empirical_cdr}")
    print(f"can_proceed_to_gate_sensitivity: {final_status.can_proceed_to_gate_sensitivity}")
    print(f"can_proceed_to_diagnostics: {final_status.can_proceed_to_diagnostics}")
    print(f"interpretation: {final_status.interpretation}")
    print(f"passed_gates: {final_status.passed_gates}")
    print(f"failed_gates: {final_status.failed_gates}")
    print(f"warning_gates: {final_status.warning_gates}")
    print(f"not_evaluated_gates: {final_status.not_evaluated_gates}")
    print(f"gates_json: {gates_json_path(cfg)}")
    print(f"metrics_summary: {metrics_summary_txt_path(cfg)}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()