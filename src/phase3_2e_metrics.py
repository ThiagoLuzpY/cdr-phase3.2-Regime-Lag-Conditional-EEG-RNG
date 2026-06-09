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
    Phase32EConfig,
    active_conditional_pairs,
    describe_config,
    load_phase3_2e_config,
)

METRICS_MODULE = "phase3_2e_metrics"
METRICS_VERSION = "phase3_2e_metrics_v2_primary_lead_adaptive_f7_f12_diagnostic"


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


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    return obj.__dict__ if hasattr(obj, "__dict__") else str(obj)


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "sim", "t"}:
        return True
    if text in {"false", "0", "no", "n", "nao", "não", "f"}:
        return False
    return default


def _bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "y", "sim", "t"}
    )


def _num(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if df.empty:
        return pd.Series([], index=df.index, dtype=float)
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _bool_col(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)
    return _bool_series(df[column]).reindex(df.index).fillna(default).astype(bool)


def _cfg_float(cfg: Phase32EConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Phase32EConfig, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: Phase32EConfig, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


def _row(row: Optional[pd.Series]) -> Dict[str, Any]:
    return {} if row is None else {
        str(key): _json_default(value)
        for key, value in row.to_dict().items()
    }


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
    status = (
        "NOT_EVALUATED"
        if not available
        else "WARN"
        if warning
        else "PASS"
        if passed
        else "FAIL"
    )

    return GateResult(
        gate=gate,
        name=name,
        passed=passed,
        status=status,
        value=float(value),
        threshold=float(threshold),
        reason=reason,
        details=details or {},
    )


def conditional_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_pair_scores_csv)


def conditional_model_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_model_scores_csv)


def conditional_summary_json_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "conditional_summary_json",
            Path(cfg.results_dir) / "phase3_2e_conditional_summary.json",
        )
    )


def controls_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.controls_json)


def gates_json_path(cfg: Phase32EConfig) -> Path:
    return Path(
        getattr(
            cfg,
            "gates_json",
            Path(cfg.results_dir) / "phase3_2e_gates.json",
        )
    )


def multiple_comparison_guard_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(
        getattr(
            cfg,
            "multiple_comparison_guard_json",
            Path(cfg.results_dir)
            / "phase3_2e_multiple_comparison_guard.json",
        )
    )


def metrics_table_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_table.csv"


def metrics_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_summary.txt"


def metrics_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_metrics_report.json"


def load_conditional_pair_scores(
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    return _load_csv(conditional_pair_scores_csv_path(cfg))


def load_conditional_model_scores(
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    return _load_csv(conditional_model_scores_csv_path(cfg))


def load_controls_summary(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    payload = _load_json(controls_json_path(cfg))

    if not payload:
        return {
            "available": False,
            "passed": False,
            "protocol_only": False,
            "reason": "controls_json_missing",
            "failed_control_types": list(
                getattr(cfg, "required_control_types", ())
            ),
        }

    result = (
        payload.get("result", {})
        if isinstance(payload.get("result"), dict)
        else {}
    )

    summary = (
        payload.get("summary", {})
        if isinstance(payload.get("summary"), dict)
        else {}
    )

    out = dict(summary)

    out.update(
        {
            "available": bool(summary.get("available", True)),
            "passed": bool(
                summary.get(
                    "passed",
                    result.get("overall_passed", False),
                )
            ),
            "protocol_only": bool(
                result.get("protocol_only", False)
            ),
            "status": payload.get(
                "status",
                result.get("status"),
            ),
            "reason": summary.get(
                "reason",
                result.get("interpretation", "ok"),
            ),
            "failed_control_types": summary.get(
                "failed_control_types",
                result.get("failed_control_types", []),
            ),
            "required_control_types": summary.get(
                "required_control_types",
                getattr(cfg, "required_control_types", []),
            ),
        }
    )

    return out


def registered_windows(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    return tuple(
        int(value)
        for value in getattr(
            cfg,
            "all_window_seconds",
            (),
        )
    )


def primary_windows(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    values = getattr(
        cfg,
        "primary_test_window_seconds",
        getattr(
            cfg,
            "primary_window_seconds",
            (),
        ),
    )

    return tuple(
        int(value)
        for value in values
    )


def registered_lags(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    return tuple(
        int(value)
        for value in getattr(
            cfg,
            "all_lags_windows",
            (),
        )
    )


def primary_lags(
    cfg: Phase32EConfig,
) -> Tuple[int, ...]:
    values = getattr(
        cfg,
        "primary_test_lags_windows",
        getattr(
            cfg,
            "primary_lags_windows",
            (),
        ),
    )

    return tuple(
        int(value)
        for value in values
    )


def primary_pairs(
    cfg: Phase32EConfig,
) -> Tuple[Tuple[str, str], ...]:
    values = getattr(
        cfg,
        "primary_test_pairs",
        getattr(
            cfg,
            "primary_conditional_pairs",
            (),
        ),
    )

    return tuple(
        (
            str(baseline),
            str(augmented),
        )
        for baseline, augmented in values
    )


def valid_pair_rows(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    if "valid" not in pair_df.columns:
        return pair_df.copy()

    return pair_df[
        _bool_series(pair_df["valid"])
    ].copy()


def effective_primary_mask(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.Series:
    required = {
        "window_seconds",
        "lag_windows",
        "baseline_model",
        "augmented_model",
    }

    if pair_df.empty:
        return pd.Series(
            [],
            index=pair_df.index,
            dtype=bool,
        )

    if not required.issubset(pair_df.columns):
        return _bool_col(
            pair_df,
            "primary",
            False,
        )

    windows = (
        pd.to_numeric(
            pair_df["window_seconds"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    lags = (
        pd.to_numeric(
            pair_df["lag_windows"],
            errors="coerce",
        )
        .fillna(999999)
        .astype(int)
    )

    pairs = list(
        zip(
            pair_df["baseline_model"].astype(str),
            pair_df["augmented_model"].astype(str),
        )
    )

    primary_pair_set = set(
        primary_pairs(cfg)
    )

    pair_mask = pd.Series(
        [
            pair in primary_pair_set
            for pair in pairs
        ],
        index=pair_df.index,
    )

    return (
        windows.isin(primary_windows(cfg))
        & lags.isin(primary_lags(cfg))
        & pair_mask
    )


def primary_pair_rows(
    pair_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or load_phase3_2e_config()
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    return valid[
        effective_primary_mask(valid, cfg)
    ].copy()


def exploratory_pair_rows(
    pair_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or load_phase3_2e_config()
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    return valid[
        ~effective_primary_mask(valid, cfg)
    ].copy()


def positive_lift_rows(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    return valid[
        _num(
            valid,
            "conditional_lift",
        ) > 0.0
    ].copy()


def primary_positive_lift_rows(
    pair_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> pd.DataFrame:
    primary = primary_pair_rows(
        pair_df,
        cfg,
    )

    if primary.empty:
        return pd.DataFrame()

    return primary[
        _num(
            primary,
            "conditional_lift",
        ) > 0.0
    ].copy()


def exploratory_positive_lift_rows(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    exploratory = exploratory_pair_rows(
        pair_df,
        cfg,
    )

    if exploratory.empty:
        return pd.DataFrame()

    return exploratory[
        _num(
            exploratory,
            "conditional_lift",
        ) > 0.0
    ].copy()


def required_subject_consistency(
    n_subjects: int,
    cfg: Phase32EConfig,
) -> Tuple[int, float]:
    n_subjects = max(
        int(n_subjects),
        0,
    )

    minimum = max(
        _cfg_int(
            cfg,
            "min_positive_subjects",
            1,
        ),
        1,
    )

    if n_subjects <= 10:
        fraction = _cfg_float(
            cfg,
            "subject_effect_fraction",
            0.10,
        )
    elif n_subjects < 1000:
        fraction = _cfg_float(
            cfg,
            "large_cohort_subject_fraction_max",
            0.03,
        )
    else:
        fraction = _cfg_float(
            cfg,
            "large_cohort_subject_fraction_min",
            0.01,
        )

    required_count = max(
        minimum,
        int(
            math.ceil(
                n_subjects * fraction - 1e-12
            )
        ),
    )

    return required_count, fraction


def subject_consistency_for_row(
    row: pd.Series,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    n_evaluated = _int(
        row.get("n_subjects_evaluated")
    )

    n_positive = _int(
        row.get("n_subjects_positive_lift")
    )

    fraction = _float(
        row.get("fraction_positive_lift")
    )

    required_count, required_fraction = (
        required_subject_consistency(
            n_evaluated,
            cfg,
        )
    )

    required_count = _int(
        row.get("required_positive_subjects"),
        required_count,
    )

    required_fraction = _float(
        row.get("required_positive_fraction"),
        required_fraction,
    )

    passed = (
        _bool(
            row.get("subject_consistency_pass")
        )
        if "subject_consistency_pass" in row.index
        else (
            n_evaluated > 0
            and n_positive >= required_count
        )
    )

    return {
        "passed": passed,
        "n_subjects_evaluated": n_evaluated,
        "n_subjects_positive_lift": n_positive,
        "fraction_positive_lift": fraction,
        "required_positive_subjects": required_count,
        "required_positive_fraction": required_fraction,
    }


def subject_consistency_mask(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.Series:
    if df.empty:
        return pd.Series(
            [],
            index=df.index,
            dtype=bool,
        )

    if "subject_consistency_pass" in df.columns:
        return _bool_series(
            df["subject_consistency_pass"]
        )

    return pd.Series(
        [
            subject_consistency_for_row(
                row,
                cfg,
            )["passed"]
            for _, row in df.iterrows()
        ],
        index=df.index,
        dtype=bool,
    )


def strong_candidate_rows(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    mask = (
        (
            _num(
                valid,
                "conditional_lift",
            )
            >= _cfg_float(
                cfg,
                "conditional_lift_min",
                0.03,
            )
        )
        & (
            _num(
                valid,
                "augmented_eps_test",
            )
            >= _cfg_float(
                cfg,
                "strong_eps_min",
                0.07,
            )
        )
        & subject_consistency_mask(
            valid,
            cfg,
        )
        & (
            _num(
                valid,
                "ll_improvement",
                float("-inf"),
            ) > 0.0
        )
        & (
            _num(
                valid,
                "bic_improvement",
                float("-inf"),
            ) >= 0.0
        )
    )

    return valid[mask].copy()


def primary_strong_candidate_rows(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    strong = strong_candidate_rows(
        pair_df,
        cfg,
    )

    if strong.empty:
        return pd.DataFrame()

    return strong[
        effective_primary_mask(
            strong,
            cfg,
        )
    ].copy()


def _rank_candidates(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    work = df.copy()

    work["_subject"] = (
        subject_consistency_mask(
            work,
            cfg,
        ).astype(int)
    )

    work["_lift"] = _num(
        work,
        "conditional_lift",
    )

    work["_ll"] = _num(
        work,
        "ll_improvement",
        float("-inf"),
    )

    work["_bic"] = _num(
        work,
        "bic_improvement",
        float("-inf"),
    )

    return work.sort_values(
        [
            "_lift",
            "_subject",
            "_ll",
            "_bic",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        kind="stable",
    )


def select_primary_lead_row(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Optional[pd.Series]:
    primary = primary_pair_rows(
        pair_df,
        cfg,
    )

    if primary.empty:
        return None

    positive = primary[
        _num(
            primary,
            "conditional_lift",
        ) > 0.0
    ].copy()

    ranked = _rank_candidates(
        positive
        if not positive.empty
        else primary,
        cfg,
    )

    return ranked.iloc[0].drop(
        labels=[
            "_subject",
            "_lift",
            "_ll",
            "_bic",
        ],
        errors="ignore",
    )


def model_saturation_summary(
    model_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    if model_df.empty:
        return {
            "available": False,
            "warning": False,
            "saturation_fraction": 0.0,
        }

    eps_column = next(
        (
            column
            for column in (
                "eps_test",
                "eps_candidate",
                "eps_calib",
            )
            if column in model_df.columns
        ),
        None,
    )

    explicit = _bool_col(
        model_df,
        "eps_saturated",
        False,
    )

    threshold = _cfg_float(
        cfg,
        "eps_saturation_value",
        2.0,
    )

    warning_fraction = _cfg_float(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    numeric = (
        _num(
            model_df,
            eps_column,
        ) >= threshold
        if eps_column
        else pd.Series(
            False,
            index=model_df.index,
        )
    )

    saturated = explicit | numeric

    fraction = (
        float(saturated.mean())
        if len(saturated)
        else 0.0
    )

    return {
        "available": (
            eps_column is not None
            or "eps_saturated" in model_df.columns
        ),
        "epsilon_column_used": eps_column,
        "eps_saturation_value": threshold,
        "eps_saturation_warning_fraction": warning_fraction,
        "n_model_rows": int(len(model_df)),
        "n_saturated_rows": int(
            saturated.sum()
        ),
        "saturation_fraction": fraction,
        "warning": (
            fraction >= warning_fraction
        ),
        "diagnostic_only": True,
    }


def pair_saturation_summary(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return {
            "available": False,
            "warning": False,
            "either_saturation_fraction": 0.0,
        }

    threshold = _cfg_float(
        cfg,
        "eps_saturation_value",
        2.0,
    )

    warning_fraction = _cfg_float(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    baseline = (
        _bool_col(
            valid,
            "baseline_eps_saturated",
        )
        | (
            _num(
                valid,
                "baseline_eps_test",
            ) >= threshold
        )
    )

    augmented = (
        _bool_col(
            valid,
            "augmented_eps_saturated",
        )
        | (
            _num(
                valid,
                "augmented_eps_test",
            ) >= threshold
        )
    )

    either = baseline | augmented

    fraction = (
        float(either.mean())
        if len(either)
        else 0.0
    )

    return {
        "available": True,
        "eps_saturation_value": threshold,
        "eps_saturation_warning_fraction": warning_fraction,
        "n_valid_pair_rows": int(len(valid)),
        "n_either_saturated_rows": int(
            either.sum()
        ),
        "either_saturation_fraction": fraction,
        "warning": (
            fraction >= warning_fraction
        ),
        "diagnostic_only": True,
    }


def gate_f1_injection_recovery(
    cfg: Phase32EConfig,
) -> GateResult:
    return _gate(
        gate="F1",
        name="Injection recovery",
        passed=False,
        value=0.0,
        threshold=_cfg_float(
            cfg,
            "gate_tol_abs",
            0.05,
        ),
        reason=(
            "not_evaluated_in_metrics; "
            "reserved for runner-level injection test"
        ),
        available=False,
    )


def gate_f2_controls_collapse(
    controls: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> GateResult:
    threshold = _cfg_float(
        cfg,
        "required_control_fraction",
        0.75,
    )

    if (
        not controls
        or controls.get("available") is False
    ):
        return _gate(
            gate="F2",
            name="Controls collapse",
            passed=False,
            value=0.0,
            threshold=threshold,
            reason="controls_unavailable",
            available=False,
        )

    summaries = controls.get(
        "control_summaries",
        [],
    )

    required = set(
        controls.get(
            "required_control_types",
            [],
        )
    )

    fractions = [
        _float(
            item.get(
                "fraction_runs_collapsed_overall"
            )
        )
        for item in summaries
        if (
            isinstance(item, dict)
            and item.get("control_type") in required
        )
    ]

    value = (
        min(fractions)
        if fractions
        else 0.0
    )

    passed = bool(
        controls.get("passed", False)
    )

    return _gate(
        gate="F2",
        name="Controls collapse",
        passed=passed,
        value=value,
        threshold=threshold,
        reason=(
            "ok"
            if passed
            else (
                "failed_control_types="
                f"{controls.get('failed_control_types', [])}"
            )
        ),
        details={
            "failed_control_types": controls.get(
                "failed_control_types",
                [],
            ),
            "required_control_types": list(
                required
            ),
            "control_subject_fraction_threshold": controls.get(
                "control_subject_fraction_threshold"
            ),
            "discovery_subject_effect_fraction": controls.get(
                "discovery_subject_effect_fraction"
            ),
        },
    )


def gate_f3_holdout_generalization(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return _gate(
            gate="F3",
            name="Holdout generalization",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="no_valid_rows",
            available=False,
        )

    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    if (
        candidate is None
        or _float(
            candidate.get("conditional_lift")
        ) <= 0.0
    ):
        return _gate(
            gate="F3",
            name="Holdout generalization",
            passed=True,
            value=1.0,
            threshold=1.0,
            reason="heldout_evaluations_available",
        )

    ll_improvement = _float(
        candidate.get("ll_improvement"),
        float("-inf"),
    )

    passed = ll_improvement > 0.0

    return _gate(
        gate="F3",
        name="Holdout generalization",
        passed=passed,
        value=ll_improvement,
        threshold=0.0,
        reason=(
            "ok"
            if passed
            else (
                "selected_primary_lead_"
                "does_not_improve_heldout_LL"
            )
        ),
        details={
            "selected_primary_lead": _row(
                candidate
            )
        },
    )


def gate_f5_sensitivity_stability(
    cfg: Phase32EConfig,
) -> GateResult:
    return _gate(
        gate="F5",
        name="Sensitivity stability",
        passed=False,
        value=0.0,
        threshold=_cfg_float(
            cfg,
            "sensitivity_max_delta",
            0.10,
        ),
        reason=(
            "not_evaluated_in_metrics; "
            "use phase3_2e_gate_sensitivity.py"
        ),
        available=False,
    )


def gate_f6_conditional_lift(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    threshold = _cfg_float(
        cfg,
        "conditional_lift_min",
        0.03,
    )

    if candidate is None:
        return _gate(
            gate="F6",
            name="Conditional lift",
            passed=False,
            value=0.0,
            threshold=threshold,
            reason="no_primary_rows",
            available=False,
        )

    lift = _float(
        candidate.get("conditional_lift")
    )

    epsilon = _float(
        candidate.get("augmented_eps_test")
    )

    passed = (
        lift >= threshold
        and epsilon >= _cfg_float(
            cfg,
            "strong_eps_min",
            0.07,
        )
    )

    return _gate(
        gate="F6",
        name="Conditional lift",
        passed=passed,
        value=lift,
        threshold=threshold,
        reason=(
            "ok"
            if passed
            else (
                f"lift={lift}, "
                f"augmented_eps={epsilon}"
            )
        ),
        details={
            "selected_primary_lead": _row(
                candidate
            ),
            "augmented_eps_test": epsilon,
        },
    )


def gate_f7_subject_consistency(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    if candidate is None:
        return _gate(
            gate="F7",
            name="Adaptive subject consistency",
            passed=False,
            value=0.0,
            threshold=0.10,
            reason="no_primary_rows",
            available=False,
        )

    result = subject_consistency_for_row(
        candidate,
        cfg,
    )

    return _gate(
        gate="F7",
        name="Adaptive subject consistency",
        passed=result["passed"],
        value=result[
            "fraction_positive_lift"
        ],
        threshold=result[
            "required_positive_fraction"
        ],
        reason=(
            "ok"
            if result["passed"]
            else "adaptive_subject_threshold_not_met"
        ),
        details={
            "selected_primary_lead": _row(
                candidate
            ),
            **result,
        },
    )


def gate_f8_complexity_penalty(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    if candidate is None:
        return _gate(
            gate="F8",
            name="Complexity penalty / BIC",
            passed=False,
            value=0.0,
            threshold=0.0,
            reason="no_primary_rows",
            available=False,
        )

    bic_improvement = _float(
        candidate.get("bic_improvement"),
        float("-inf"),
    )

    passed = bic_improvement >= 0.0

    return _gate(
        gate="F8",
        name="Complexity penalty / BIC",
        passed=passed,
        value=bic_improvement,
        threshold=0.0,
        reason=(
            "ok"
            if passed
            else (
                "primary_lead_not_favored_"
                "by_official_BIC"
            )
        ),
        details={
            "selected_primary_lead": _row(
                candidate
            ),
            "ll_improvement": _float(
                candidate.get("ll_improvement")
            ),
            "aic_improvement": _float(
                candidate.get("aic_improvement")
            ),
            "next_module": (
                "phase3_2e_gate_sensitivity.py"
            ),
        },
    )


def gate_f9_lag_stability(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    if candidate is None:
        return _gate(
            gate="F9",
            name="Registered lag localization",
            passed=False,
            value=0.0,
            threshold=1.0,
            reason="no_primary_rows",
            available=False,
        )

    lag = _int(
        candidate.get("lag_windows"),
        999999,
    )

    passed = (
        lag in set(primary_lags(cfg))
        and lag in set(registered_lags(cfg))
    )

    return _gate(
        gate="F9",
        name="Registered lag localization",
        passed=passed,
        value=1.0 if passed else 0.0,
        threshold=1.0,
        reason=(
            "ok"
            if passed
            else "lead_outside_registered_primary_lags"
        ),
        details={
            "selected_lag": lag,
            "primary_lags": list(
                primary_lags(cfg)
            ),
        },
    )


def gate_f10_window_specificity(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    threshold = _cfg_float(
        cfg,
        "conditional_lift_min",
        0.03,
    )

    if candidate is None:
        return _gate(
            gate="F10",
            name="Registered window localization",
            passed=False,
            value=0.0,
            threshold=threshold,
            reason="no_primary_rows",
            available=False,
        )

    window = _int(
        candidate.get("window_seconds"),
        -1,
    )

    lift = _float(
        candidate.get("conditional_lift")
    )

    passed = (
        window in set(primary_windows(cfg))
        and lift >= threshold
    )

    return _gate(
        gate="F10",
        name="Registered window localization",
        passed=passed,
        value=lift,
        threshold=threshold,
        reason=(
            "ok"
            if passed
            else "meaningful_lift_not_in_primary_window"
        ),
        details={
            "selected_window": window,
            "primary_windows": list(
                primary_windows(cfg)
            ),
        },
    )


def gate_f11_multiple_comparison_guard(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    valid = valid_pair_rows(pair_df)
    positive = positive_lift_rows(pair_df)

    primary_positive = (
        primary_positive_lift_rows(
            pair_df,
            cfg,
        )
    )

    if valid.empty:
        return _gate(
            gate="F11",
            name="Multiple-comparison guard",
            passed=False,
            value=0.0,
            threshold=1.0,
            reason="no_valid_rows",
            available=False,
        )

    if positive.empty:
        return _gate(
            gate="F11",
            name="Multiple-comparison guard",
            passed=True,
            value=0.0,
            threshold=0.0,
            reason="no_positive_candidates",
        )

    passed = not primary_positive.empty

    return _gate(
        gate="F11",
        name=(
            "Registration-based "
            "multiple-comparison guard"
        ),
        passed=passed,
        value=float(
            len(primary_positive)
        ),
        threshold=1.0,
        reason=(
            "ok"
            if passed
            else (
                "positive_rows_exist_only_"
                "outside_primary_tests"
            )
        ),
        details={
            "n_valid_tests": int(
                len(valid)
            ),
            "n_positive_tests": int(
                len(positive)
            ),
            "n_primary_positive_tests": int(
                len(primary_positive)
            ),
        },
    )


def gate_f12_epsilon_saturation(
    model_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> GateResult:
    model_summary = model_saturation_summary(
        model_df,
        cfg,
    )

    pair_summary = pair_saturation_summary(
        pair_df,
        cfg,
    )

    warning = bool(
        model_summary.get("warning", False)
        or pair_summary.get("warning", False)
    )

    value = max(
        _float(
            model_summary.get(
                "saturation_fraction"
            )
        ),
        _float(
            pair_summary.get(
                "either_saturation_fraction"
            )
        ),
    )

    threshold = _cfg_float(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    if (
        not model_summary.get("available", False)
        and not pair_summary.get(
            "available",
            False,
        )
    ):
        return _gate(
            gate="F12",
            name="Epsilon saturation diagnostic",
            passed=False,
            value=0.0,
            threshold=threshold,
            reason="no_epsilon_rows",
            available=False,
        )

    return _gate(
        gate="F12",
        name="Epsilon saturation diagnostic",
        passed=not warning,
        value=value,
        threshold=threshold,
        reason=(
            "epsilon_saturation_warning_diagnostic_only"
            if warning
            else "ok"
        ),
        details={
            "model": model_summary,
            "pair": pair_summary,
            "diagnostic_only": True,
        },
        warning=warning,
    )


def evaluate_phase3_2e_gates(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> List[GateResult]:
    cfg = cfg or load_phase3_2e_config()

    return [
        gate_f1_injection_recovery(cfg),
        gate_f2_controls_collapse(
            controls_summary,
            cfg,
        ),
        gate_f3_holdout_generalization(
            pair_df,
            cfg,
        ),
        gate_f5_sensitivity_stability(cfg),
        gate_f6_conditional_lift(
            pair_df,
            cfg,
        ),
        gate_f7_subject_consistency(
            pair_df,
            cfg,
        ),
        gate_f8_complexity_penalty(
            pair_df,
            cfg,
        ),
        gate_f9_lag_stability(
            pair_df,
            cfg,
        ),
        gate_f10_window_specificity(
            pair_df,
            cfg,
        ),
        gate_f11_multiple_comparison_guard(
            pair_df,
            cfg,
        ),
        gate_f12_epsilon_saturation(
            model_df,
            pair_df,
            cfg,
        ),
    ]


def summarize_gate_status(
    gates: Sequence[GateResult],
) -> Dict[str, List[str]]:
    output = {
        "passed": [],
        "failed": [],
        "warning": [],
        "not_evaluated": [],
    }

    mapping = {
        "PASS": "passed",
        "FAIL": "failed",
        "WARN": "warning",
        "NOT_EVALUATED": "not_evaluated",
    }

    for gate in gates:
        output[
            mapping[gate.status]
        ].append(gate.gate)

    return output


def source_claim_allowed(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    conditional_summary: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> bool:
    values: List[bool] = []

    for frame in (
        pair_df,
        model_df,
    ):
        if (
            not frame.empty
            and "empirical_claim_allowed"
            in frame.columns
        ):
            values.append(
                bool(
                    _bool_series(
                        frame["empirical_claim_allowed"]
                    ).all()
                )
            )

    for key in (
        "can_claim_empirical_cdr",
        "source_claim_allowed",
        "empirical_claim_allowed",
    ):
        if key in conditional_summary:
            values.append(
                _bool(
                    conditional_summary.get(key)
                )
            )

    if values:
        return all(values)

    return not _cfg_bool(
        cfg,
        "require_real_synchronized_data_for_claim",
        True,
    )


def determine_final_status(
    gates: Sequence[GateResult],
    pair_df: pd.DataFrame,
    controls_summary: Mapping[str, Any],
    cfg: Phase32EConfig,
    model_df: Optional[pd.DataFrame] = None,
    conditional_summary: Optional[
        Mapping[str, Any]
    ] = None,
) -> Phase32EFinalStatus:
    model_df = (
        model_df
        if model_df is not None
        else pd.DataFrame()
    )

    conditional_summary = (
        conditional_summary
        or {}
    )

    status_map = {
        gate.gate: gate
        for gate in gates
    }

    gate_summary = summarize_gate_status(
        gates
    )

    protocol_only = bool(
        controls_summary.get(
            "protocol_only",
            False,
        )
    )

    primary_positive = (
        primary_positive_lift_rows(
            pair_df,
            cfg,
        )
    )

    exploratory_positive = (
        exploratory_positive_lift_rows(
            pair_df,
            cfg,
        )
    )

    positive = positive_lift_rows(
        pair_df
    )

    claim_source = source_claim_allowed(
        pair_df,
        model_df,
        conditional_summary,
        cfg,
    )

    if (
        protocol_only
        and pair_df.empty
    ):
        status = getattr(
            cfg,
            "status_pending_synchronized_data",
            "pending_synchronized_data",
        )

        interpretation = (
            "No synchronized EEG–QRNG dataset "
            "is available; metrics remain "
            "protocol-only."
        )

        can_claim = False

    elif (
        controls_summary.get("available")
        is False
    ):
        status = (
            "pending_controls_positive_lead"
            if not positive.empty
            else "pending_controls_null"
        )

        interpretation = (
            "Required synchronized controls are "
            "unavailable; no claim is allowed."
        )

        can_claim = False

    elif (
        status_map["F2"].status
        != "PASS"
    ):
        status = "blocked_by_controls"

        interpretation = (
            "Required negative controls failed; "
            "positive interpretation is blocked."
        )

        can_claim = False

    elif not primary_positive.empty:
        core_pass = all(
            status_map[gate].status == "PASS"
            for gate in (
                "F2",
                "F3",
                "F6",
                "F7",
                "F9",
                "F10",
                "F11",
            )
        )

        bic_pass = (
            status_map["F8"].status
            == "PASS"
        )

        if (
            core_pass
            and bic_pass
        ):
            status = (
                "candidate_synchronized_signal"
                if claim_source
                else "surrogate_primary_candidate"
            )

            interpretation = (
                "A registered primary candidate "
                "passed the official analytical gates."
                if claim_source
                else (
                    "A registered primary candidate "
                    "passed the analytical gates, but "
                    "surrogate provenance blocks an "
                    "empirical claim."
                )
            )

            can_claim = claim_source

        elif (
            core_pass
            and not bic_pass
        ):
            status = (
                "primary_lead_bic_limited"
            )

            interpretation = (
                "A registered primary lead passed "
                "controls, held-out likelihood, lift, "
                "adaptive subject consistency, "
                "lag/window registration and the "
                "multiple-comparison guard, but "
                "official BIC did not favor the "
                "augmented model."
            )

            can_claim = False

        elif (
            status_map["F7"].status
            != "PASS"
        ):
            status = (
                "primary_lead_subject_"
                "consistency_limited"
            )

            interpretation = (
                "A primary lift exists, but adaptive "
                "subject consistency was not met."
            )

            can_claim = False

        else:
            status = (
                "primary_lead_incomplete_"
                "gate_support"
            )

            interpretation = (
                "A primary positive row exists, but "
                "one or more official gates remain "
                "unsatisfied."
            )

            can_claim = False

    elif not exploratory_positive.empty:
        status = "exploratory_lead_only"

        interpretation = (
            "Positive lift appeared only outside "
            "the current registered primary "
            "test region."
        )

        can_claim = False

    else:
        status = (
            "synchronized_conditional_null_result"
        )

        interpretation = (
            "No positive synchronized conditional "
            "EEG–QRNG lift was detected."
        )

        can_claim = False

    if (
        status_map["F12"].status
        == "WARN"
    ):
        interpretation += (
            " F12 raised a diagnostic saturation "
            "warning; it does not erase the lead."
        )

    if not claim_source:
        interpretation += (
            " Source provenance blocks an empirical "
            "claim for this run."
        )

    return Phase32EFinalStatus(
        status=status,
        protocol_only=protocol_only,
        interpretation=interpretation,
        passed_gates=gate_summary["passed"],
        failed_gates=gate_summary["failed"],
        warning_gates=gate_summary["warning"],
        not_evaluated_gates=gate_summary[
            "not_evaluated"
        ],
        can_claim_empirical_cdr=can_claim,
        can_proceed_to_gate_sensitivity=True,
        can_proceed_to_diagnostics=True,
    )


def build_multiple_comparison_guard_report(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)
    positive = positive_lift_rows(pair_df)

    primary_positive = (
        primary_positive_lift_rows(
            pair_df,
            cfg,
        )
    )

    exploratory_positive = (
        exploratory_positive_lift_rows(
            pair_df,
            cfg,
        )
    )

    strong = strong_candidate_rows(
        pair_df,
        cfg,
    )

    primary_strong = (
        primary_strong_candidate_rows(
            pair_df,
            cfg,
        )
    )

    return {
        "enabled": _cfg_bool(
            cfg,
            "multiple_comparison_guard",
            True,
        ),
        "guard_type": (
            "registered_primary_region"
        ),
        "n_valid_tests": int(len(valid)),
        "n_positive_lift_tests": int(
            len(positive)
        ),
        "n_primary_positive_lift_tests": int(
            len(primary_positive)
        ),
        "n_exploratory_positive_lift_tests": int(
            len(exploratory_positive)
        ),
        "n_strong_candidate_tests": int(
            len(strong)
        ),
        "n_primary_strong_candidate_tests": int(
            len(primary_strong)
        ),
        "active_conditional_pairs": [
            list(pair)
            for pair in active_conditional_pairs(
                cfg
            )
        ],
        "primary_pairs": [
            list(pair)
            for pair in primary_pairs(cfg)
        ],
        "primary_windows": list(
            primary_windows(cfg)
        ),
        "primary_lags": list(
            primary_lags(cfg)
        ),
        "primary_positive_rows": (
            primary_positive.to_dict(
                orient="records"
            )
        ),
        "exploratory_positive_rows": (
            exploratory_positive.to_dict(
                orient="records"
            )
        ),
    }


def build_grid_audit(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_pair_rows",
        }

    derived_primary = effective_primary_mask(
        valid,
        cfg,
    )

    stored_primary = _bool_col(
        valid,
        "primary",
    )

    windows = sorted(
        pd.to_numeric(
            valid["window_seconds"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    lags = sorted(
        pd.to_numeric(
            valid["lag_windows"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    return {
        "available": True,
        "pair_rows": int(len(pair_df)),
        "valid_rows": int(len(valid)),
        "derived_primary_rows": int(
            derived_primary.sum()
        ),
        "derived_exploratory_rows": int(
            (~derived_primary).sum()
        ),
        "stored_primary_disagreements": int(
            (
                derived_primary
                != stored_primary
            ).sum()
        ),
        "windows_found": [
            int(value)
            for value in windows
        ],
        "lags_found": [
            int(value)
            for value in lags
        ],
        "primary_windows": list(
            primary_windows(cfg)
        ),
        "primary_lags": list(
            primary_lags(cfg)
        ),
        "primary_pairs": [
            list(pair)
            for pair in primary_pairs(cfg)
        ],
    }


def build_summary_text(
    gates: Sequence[GateResult],
    final: Phase32EFinalStatus,
    guard: Mapping[str, Any],
    grid: Mapping[str, Any],
    controls: Mapping[str, Any],
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> str:
    candidate = select_primary_lead_row(
        pair_df,
        cfg,
    )

    lines = [
        "=" * 78,
        "Phase III.2E — Metrics and Gates Summary",
        (
            "Evidence-guided synchronized "
            "EEG–QRNG version"
        ),
        "=" * 78,
        f"status: {final.status}",
        f"protocol_only: {final.protocol_only}",
        (
            "can_claim_empirical_cdr: "
            f"{final.can_claim_empirical_cdr}"
        ),
        (
            "interpretation: "
            f"{final.interpretation}"
        ),
        (
            "passed_gates: "
            f"{final.passed_gates}"
        ),
        (
            "failed_gates: "
            f"{final.failed_gates}"
        ),
        (
            "warning_gates: "
            f"{final.warning_gates}"
        ),
        (
            "not_evaluated_gates: "
            f"{final.not_evaluated_gates}"
        ),
        "",
        (
            "selected_primary_lead: "
            f"{_row(candidate)}"
        ),
        f"grid_audit: {dict(grid)}",
        (
            "controls_passed: "
            f"{controls.get('passed')}"
        ),
        (
            "failed_control_types: "
            f"{controls.get('failed_control_types')}"
        ),
        (
            "multiple_comparison_guard: "
            f"{dict(guard)}"
        ),
        "",
        "Gates",
        "-" * 78,
    ]

    lines.extend(
        (
            f"{gate.gate} — {gate.name}: "
            f"{gate.status} | "
            f"value={gate.value} | "
            f"threshold={gate.threshold} | "
            f"reason={gate.reason}"
        )
        for gate in gates
    )

    lines.append("=" * 78)

    return "\n".join(lines)


def save_metrics_outputs(
    gates: Sequence[GateResult],
    final: Phase32EFinalStatus,
    guard: Mapping[str, Any],
    grid: Mapping[str, Any],
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    controls: Mapping[str, Any],
    conditional_summary: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    Path(cfg.results_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            asdict(gate)
            for gate in gates
        ]
    ).to_csv(
        metrics_table_csv_path(cfg),
        index=False,
    )

    protocol = {
        "primary_windows": list(
            primary_windows(cfg)
        ),
        "primary_lags": list(
            primary_lags(cfg)
        ),
        "primary_pairs": [
            list(pair)
            for pair in primary_pairs(cfg)
        ],
        "subject_effect_fraction": _cfg_float(
            cfg,
            "subject_effect_fraction",
            0.10,
        ),
        "eps_saturation_value": _cfg_float(
            cfg,
            "eps_saturation_value",
            2.0,
        ),
    }

    protocol_fingerprint = (
        hashlib.sha256(
            json.dumps(
                protocol,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    payload = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": METRICS_MODULE,
        "metrics_version": METRICS_VERSION,
        "created_at": _now(),
        "status": final.status,
        "protocol_fingerprint": protocol_fingerprint,
        "protocol": protocol,
        "final_status": asdict(final),
        "gates": [
            asdict(gate)
            for gate in gates
        ],
        "selected_primary_lead": _row(
            select_primary_lead_row(
                pair_df,
                cfg,
            )
        ),
        "grid_audit": dict(grid),
        "multiple_comparison_guard": dict(
            guard
        ),
        "model_saturation": model_saturation_summary(
            model_df,
            cfg,
        ),
        "pair_saturation": pair_saturation_summary(
            pair_df,
            cfg,
        ),
        "controls_summary": dict(controls),
        "conditional_summary": dict(
            conditional_summary
        ),
        "config": describe_config(cfg),
    }

    _save_json(
        gates_json_path(cfg),
        payload,
    )

    _save_json(
        multiple_comparison_guard_json_path(
            cfg
        ),
        guard,
    )

    _save_json(
        metrics_report_json_path(cfg),
        payload,
    )

    metrics_summary_txt_path(
        cfg
    ).write_text(
        build_summary_text(
            gates,
            final,
            guard,
            grid,
            controls,
            pair_df,
            cfg,
        ),
        encoding="utf-8",
    )

    print(
        "[Phase3.2E Metrics] "
        f"Saved metrics table: "
        f"{metrics_table_csv_path(cfg)}"
    )

    print(
        "[Phase3.2E Metrics] "
        f"Saved gates JSON: "
        f"{gates_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Metrics] "
        f"Saved metrics report: "
        f"{metrics_report_json_path(cfg)}"
    )

    print(
        "[Phase3.2E Metrics] "
        f"Saved metrics summary: "
        f"{metrics_summary_txt_path(cfg)}"
    )

    return payload


def run_phase3_2e_metrics(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_conditional: bool = False,
    force_rebuild_controls: bool = False,
) -> Phase32EFinalStatus:
    cfg = cfg or load_phase3_2e_config()

    if (
        force_rebuild_conditional
        or force_rebuild_controls
    ):
        raise RuntimeError(
            "Metrics will not rebuild conditional "
            "or controls implicitly. Run those "
            "modules explicitly to avoid accidental "
            "multi-hour recomputation."
        )

    pair_df = load_conditional_pair_scores(
        cfg
    )

    model_df = load_conditional_model_scores(
        cfg
    )

    conditional_summary = _load_json(
        conditional_summary_json_path(cfg)
    )

    controls = load_controls_summary(cfg)

    grid = build_grid_audit(
        pair_df,
        cfg,
    )

    if pair_df.empty:
        print(
            "[Phase3.2E Metrics] ERROR: "
            f"missing "
            f"{conditional_pair_scores_csv_path(cfg)}"
        )

    if model_df.empty:
        print(
            "[Phase3.2E Metrics] WARNING: "
            f"missing "
            f"{conditional_model_scores_csv_path(cfg)}"
        )

    if grid.get(
        "stored_primary_disagreements",
        0,
    ):
        print(
            "[Phase3.2E Metrics] WARNING: "
            "stored primary flags differ from "
            "current config; current config wins."
        )

    gates = evaluate_phase3_2e_gates(
        pair_df,
        model_df,
        controls,
        cfg,
    )

    final = determine_final_status(
        gates,
        pair_df,
        controls,
        cfg,
        model_df,
        conditional_summary,
    )

    guard = (
        build_multiple_comparison_guard_report(
            pair_df,
            cfg,
        )
    )

    save_metrics_outputs(
        gates,
        final,
        guard,
        grid,
        pair_df,
        model_df,
        controls,
        conditional_summary,
        cfg,
    )

    return final


def load_or_build_phase3_2e_metrics(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    **_: Any,
) -> Phase32EFinalStatus:
    if force_rebuild:
        print(
            "[Phase3.2E Metrics] "
            "Recalculating metrics from "
            "existing outputs."
        )

    return run_phase3_2e_metrics(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase III.2E metrics "
            "from existing outputs."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-conditional",
        action="store_true",
    )

    parser.add_argument(
        "--force-rebuild-controls",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase3_2e_config()

    if (
        args.force_rebuild_conditional
        or args.force_rebuild_controls
    ):
        raise SystemExit(
            "Safety stop: metrics does not "
            "trigger conditional/control rebuilding."
        )

    if args.validate_only:
        pair_df = (
            load_conditional_pair_scores(cfg)
        )

        model_df = (
            load_conditional_model_scores(cfg)
        )

        controls = load_controls_summary(cfg)

        grid = build_grid_audit(
            pair_df,
            cfg,
        )

        result = {
            "pair_rows": int(len(pair_df)),
            "model_rows": int(len(model_df)),
            "controls_available": controls.get(
                "available"
            ),
            "controls_passed": controls.get(
                "passed"
            ),
            "grid_audit": grid,
        }

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )

        raise SystemExit(
            0
            if (
                len(pair_df)
                and controls.get("available")
            )
            else 1
        )

    final = run_phase3_2e_metrics(cfg)

    print("\n" + "=" * 78)
    print("Phase III.2E metrics completed")
    print("=" * 78)

    print(
        f"status: {final.status}"
    )

    print(
        f"protocol_only: "
        f"{final.protocol_only}"
    )

    print(
        f"can_claim_empirical_cdr: "
        f"{final.can_claim_empirical_cdr}"
    )

    print(
        f"can_proceed_to_gate_sensitivity: "
        f"{final.can_proceed_to_gate_sensitivity}"
    )

    print(
        f"can_proceed_to_diagnostics: "
        f"{final.can_proceed_to_diagnostics}"
    )

    print(
        f"interpretation: "
        f"{final.interpretation}"
    )

    print(
        f"passed_gates: "
        f"{final.passed_gates}"
    )

    print(
        f"failed_gates: "
        f"{final.failed_gates}"
    )

    print(
        f"warning_gates: "
        f"{final.warning_gates}"
    )

    print(
        f"not_evaluated_gates: "
        f"{final.not_evaluated_gates}"
    )

    print(
        f"metrics_summary: "
        f"{metrics_summary_txt_path(cfg)}"
    )

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()