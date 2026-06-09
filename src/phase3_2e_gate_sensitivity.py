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
    describe_config,
    load_phase3_2e_config,
)
from src.phase3_2e_metrics import (
    effective_primary_mask,
    model_saturation_summary,
    pair_saturation_summary,
    primary_lags,
    primary_pairs,
    primary_windows,
    select_primary_lead_row,
    subject_consistency_for_row,
)

GATE_SENSITIVITY_MODULE = "phase3_2e_gate_sensitivity"
GATE_SENSITIVITY_VERSION = (
    "phase3_2e_gate_sensitivity_v2_"
    "adaptive_f7_selected_lead_f8_f12_diagnostic"
)


@dataclass
class GateSensitivityResult:
    status: str
    protocol_only: bool
    official_f7_required_subjects: int
    official_f7_required_fraction: float
    official_f8_threshold: float
    official_f12_eps_saturation_value: float
    official_f12_warning_fraction: float
    f7_available: bool
    f8_available: bool
    f12_available: bool
    f7_official_passed: bool
    f8_official_passed: bool
    f12_official_warning: bool
    f7_selected_fraction: float
    f7_selected_positive_subjects: int
    f8_selected_bic_improvement: float
    f8_selected_aic_improvement: float
    f8_selected_ll_improvement: float
    f12_official_saturation_fraction: float
    f7_fraction_distance_to_official: float
    f7_subject_count_distance_to_official: int
    f8_distance_to_official: float
    f12_distance_to_warning: float
    can_proceed_to_diagnostics: bool
    can_claim_empirical_cdr: bool
    interpretation: str


def _now_str() -> str:
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


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        return pd.DataFrame()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_bool_series(
    series: pd.Series,
) -> pd.Series:
    if series.empty:
        return pd.Series(
            [],
            index=series.index,
            dtype=bool,
        )

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "sim",
                "t",
            }
        )
    )


def _numeric_col(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if df.empty:
        return pd.Series(
            [],
            index=df.index,
            dtype=float,
        )

    if column not in df.columns:
        return pd.Series(
            [default] * len(df),
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def _float_cfg(
    cfg: Phase32EConfig,
    name: str,
    default: float,
) -> float:
    return float(getattr(cfg, name, default))


def _int_cfg(
    cfg: Phase32EConfig,
    name: str,
    default: int,
) -> int:
    return int(getattr(cfg, name, default))


def _list_cfg(
    cfg: Phase32EConfig,
    name: str,
    default: Sequence[Any],
) -> List[Any]:
    value = getattr(cfg, name, None)
    return list(default if value is None else value)


def _row_payload(
    row: Optional[pd.Series],
) -> Dict[str, Any]:
    if row is None:
        return {}

    return {
        str(key): _json_default(value)
        for key, value in row.to_dict().items()
    }


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def conditional_pair_scores_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(
        getattr(
            cfg,
            "conditional_pair_scores_csv",
            Path(cfg.results_dir)
            / "phase3_2e_conditional_pair_scores.csv",
        )
    )


def conditional_model_scores_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(
        getattr(
            cfg,
            "conditional_model_scores_csv",
            Path(cfg.results_dir)
            / "phase3_2e_conditional_model_scores.csv",
        )
    )


def gates_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return Path(
        getattr(
            cfg,
            "gates_json",
            Path(cfg.results_dir)
            / "phase3_2e_gates.json",
        )
    )


def gate_sensitivity_report_json_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_gate_sensitivity_report.json"
    )


def gate_sensitivity_summary_txt_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_gate_sensitivity_summary.txt"
    )


def f7_sensitivity_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_f7_subject_sensitivity.csv"
    )


def f8_sensitivity_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_f8_complexity_sensitivity.csv"
    )


def f12_sensitivity_csv_path(
    cfg: Phase32EConfig,
) -> Path:
    return (
        Path(cfg.results_dir)
        / "phase3_2e_f12_epsilon_saturation_sensitivity.csv"
    )


def valid_pair_rows(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    if "valid" not in pair_df.columns:
        return pair_df.copy()

    return pair_df[
        _as_bool_series(pair_df["valid"])
    ].copy()


def primary_pair_rows(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    return valid[
        effective_primary_mask(valid, cfg)
    ].copy()


def load_official_gate_status(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    payload = _load_json(gates_json_path(cfg))

    if not payload:
        return {
            "available": False,
            "reason": "gates_json_missing",
            "gates": {},
            "final_status": {},
            "raw": {},
        }

    gate_map: Dict[str, Dict[str, Any]] = {}

    for gate in payload.get("gates", []):
        if not isinstance(gate, dict):
            continue

        gate_id = str(
            gate.get("gate", "")
        ).strip()

        if gate_id:
            gate_map[gate_id] = dict(gate)

    return {
        "available": True,
        "reason": "ok",
        "gates": gate_map,
        "final_status": payload.get(
            "final_status",
            {},
        ),
        "raw": payload,
    }


def official_gate_status(
    gates_payload: Mapping[str, Any],
    gate_id: str,
) -> str:
    gate = gates_payload.get(
        "gates",
        {},
    ).get(
        gate_id,
        {},
    )

    return str(
        gate.get(
            "status",
            "MISSING",
        )
    ).upper()


def official_gate_passed(
    gates_payload: Mapping[str, Any],
    gate_id: str,
) -> bool:
    return (
        official_gate_status(
            gates_payload,
            gate_id,
        )
        == "PASS"
    )


def official_gate_warning(
    gates_payload: Mapping[str, Any],
    gate_id: str,
) -> bool:
    return (
        official_gate_status(
            gates_payload,
            gate_id,
        )
        == "WARN"
    )


def selected_primary_lead(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Optional[pd.Series]:
    return select_primary_lead_row(
        pair_df,
        cfg,
    )


def default_f7_fraction_grid(
    official_fraction: float,
) -> List[float]:
    values = [
        0.01,
        0.02,
        0.03,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        official_fraction,
    ]

    return sorted(
        {
            float(value)
            for value in values
            if 0.0 <= float(value) <= 1.0
        }
    )


def build_f7_subject_sensitivity(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lead = selected_primary_lead(
        pair_df,
        cfg,
    )

    columns = [
        "required_fraction",
        "required_subjects",
        "n_subjects_evaluated",
        "n_subjects_positive_lift",
        "observed_fraction_positive_lift",
        "official_required_fraction",
        "official_required_subjects",
        "would_pass",
        "fraction_distance",
        "subject_count_distance",
        "is_official_scenario",
    ]

    if lead is None:
        return (
            pd.DataFrame(columns=columns),
            {
                "available": False,
                "reason": "no_primary_lead",
                "official_passed_by_data": False,
                "interpretation": (
                    "F7 sensitivity is unavailable because "
                    "no primary lead row exists."
                ),
            },
        )

    official = subject_consistency_for_row(
        lead,
        cfg,
    )

    n_subjects = int(
        official["n_subjects_evaluated"]
    )
    n_positive = int(
        official["n_subjects_positive_lift"]
    )
    observed_fraction = float(
        official["fraction_positive_lift"]
    )
    official_count = int(
        official["required_positive_subjects"]
    )
    official_fraction = float(
        official["required_positive_fraction"]
    )

    fractions = _list_cfg(
        cfg,
        "f7_subject_sensitivity_thresholds",
        default_f7_fraction_grid(
            official_fraction
        ),
    )

    fractions = sorted(
        {
            float(value)
            for value in fractions
            if 0.0 <= float(value) <= 1.0
        }
        | {official_fraction}
    )

    minimum_count = max(
        _int_cfg(
            cfg,
            "min_positive_subjects",
            1,
        ),
        1,
    )

    rows: List[Dict[str, Any]] = []

    for fraction in fractions:
        required_count = max(
            minimum_count,
            int(
                math.ceil(
                    n_subjects * fraction
                    - 1e-12
                )
            ),
        )

        rows.append(
            {
                "required_fraction": fraction,
                "required_subjects": required_count,
                "n_subjects_evaluated": n_subjects,
                "n_subjects_positive_lift": n_positive,
                "observed_fraction_positive_lift": observed_fraction,
                "official_required_fraction": official_fraction,
                "official_required_subjects": official_count,
                "would_pass": bool(
                    n_subjects > 0
                    and n_positive
                    >= required_count
                ),
                "fraction_distance": (
                    observed_fraction
                    - fraction
                ),
                "subject_count_distance": (
                    n_positive
                    - required_count
                ),
                "is_official_scenario": bool(
                    np.isclose(
                        fraction,
                        official_fraction,
                    )
                    and required_count
                    == official_count
                ),
            }
        )

    sensitivity_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    passing = sensitivity_df[
        sensitivity_df[
            "would_pass"
        ].astype(bool)
    ]

    strictest_fraction_passed = (
        float(
            passing[
                "required_fraction"
            ].max()
        )
        if not passing.empty
        else None
    )

    strictest_subject_count_passed = (
        int(
            passing[
                "required_subjects"
            ].max()
        )
        if not passing.empty
        else None
    )

    official_pass = bool(
        official["passed"]
    )

    summary = {
        "available": True,
        "reason": "ok",
        "selected_primary_lead": _row_payload(
            lead
        ),
        "n_subjects_evaluated": n_subjects,
        "n_subjects_positive_lift": n_positive,
        "observed_fraction_positive_lift": observed_fraction,
        "official_required_subjects": official_count,
        "official_required_fraction": official_fraction,
        "official_passed_by_data": official_pass,
        "fraction_distance_to_official": (
            observed_fraction
            - official_fraction
        ),
        "subject_count_distance_to_official": (
            n_positive
            - official_count
        ),
        "strictest_fraction_passed": (
            strictest_fraction_passed
        ),
        "strictest_subject_count_passed": (
            strictest_subject_count_passed
        ),
        "interpretation": (
            "The selected primary lead satisfies "
            "the adaptive F7 rule."
            if official_pass
            else (
                "The selected primary lead does not "
                "satisfy the adaptive F7 subject-count "
                "rule. Sensitivity rows are diagnostic only."
            )
        ),
    }

    return sensitivity_df, summary


def _model_row_for_lead(
    model_df: pd.DataFrame,
    lead: pd.Series,
    model_name: str,
) -> Optional[pd.Series]:
    if (
        model_df.empty
        or "model" not in model_df.columns
    ):
        return None

    mask = (
        model_df["model"].astype(str)
        == str(model_name)
    )

    for column in (
        "window_seconds",
        "lag_windows",
    ):
        if (
            column in model_df.columns
            and column in lead.index
        ):
            mask &= (
                pd.to_numeric(
                    model_df[column],
                    errors="coerce",
                )
                == _safe_float(
                    lead.get(column),
                    float("nan"),
                )
            )

    matches = model_df[mask].copy()

    return (
        matches.iloc[0]
        if not matches.empty
        else None
    )


def default_complexity_scale_grid() -> List[float]:
    return [
        0.00,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
    ]


def build_f8_complexity_sensitivity(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lead = selected_primary_lead(
        pair_df,
        cfg,
    )

    columns = [
        "scenario",
        "complexity_penalty_scale",
        "ll_improvement",
        "observed_complexity_penalty",
        "derived_bic_improvement",
        "official_bic_improvement",
        "aic_improvement",
        "would_pass_bic_threshold",
        "distance_to_official_threshold",
        "is_official_scenario",
    ]

    if lead is None:
        return (
            pd.DataFrame(columns=columns),
            {
                "available": False,
                "reason": "no_primary_lead",
                "official_passed_by_data": False,
                "interpretation": (
                    "F8 sensitivity is unavailable because "
                    "no primary lead row exists."
                ),
            },
        )

    bic_improvement = _safe_float(
        lead.get("bic_improvement"),
        float("-inf"),
    )
    aic_improvement = _safe_float(
        lead.get("aic_improvement"),
        float("-inf"),
    )
    ll_improvement = _safe_float(
        lead.get("ll_improvement"),
        float("-inf"),
    )

    observed_penalty = (
        2.0 * ll_improvement
        - bic_improvement
        if (
            np.isfinite(ll_improvement)
            and np.isfinite(bic_improvement)
        )
        else float("nan")
    )

    scales = _list_cfg(
        cfg,
        "f8_effective_complexity_scales",
        default_complexity_scale_grid(),
    )

    scales = sorted(
        {
            float(value)
            for value in scales
            if float(value) >= 0.0
        }
        | {1.0}
    )

    rows: List[Dict[str, Any]] = []

    for scale in scales:
        derived_bic = (
            2.0 * ll_improvement
            - scale * observed_penalty
        )

        rows.append(
            {
                "scenario": (
                    "official_bic"
                    if np.isclose(
                        scale,
                        1.0,
                    )
                    else "complexity_scaled_bic"
                ),
                "complexity_penalty_scale": scale,
                "ll_improvement": ll_improvement,
                "observed_complexity_penalty": observed_penalty,
                "derived_bic_improvement": derived_bic,
                "official_bic_improvement": bic_improvement,
                "aic_improvement": aic_improvement,
                "would_pass_bic_threshold": bool(
                    derived_bic >= 0.0
                ),
                "distance_to_official_threshold": derived_bic,
                "is_official_scenario": bool(
                    np.isclose(
                        scale,
                        1.0,
                    )
                ),
            }
        )

    sensitivity_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    baseline_model = str(
        lead.get(
            "baseline_model",
            "",
        )
    )
    augmented_model = str(
        lead.get(
            "augmented_model",
            "",
        )
    )

    baseline_score = _model_row_for_lead(
        model_df,
        lead,
        baseline_model,
    )
    augmented_score = _model_row_for_lead(
        model_df,
        lead,
        augmented_model,
    )

    baseline_n_params = _safe_int(
        baseline_score.get("n_params")
        if baseline_score is not None
        else None,
        0,
    )
    augmented_n_params = _safe_int(
        augmented_score.get("n_params")
        if augmented_score is not None
        else None,
        0,
    )
    baseline_n_test = _safe_int(
        baseline_score.get("n_test")
        if baseline_score is not None
        else None,
        0,
    )
    augmented_n_test = _safe_int(
        augmented_score.get("n_test")
        if augmented_score is not None
        else None,
        0,
    )

    parameter_delta = (
        augmented_n_params
        - baseline_n_params
    )

    critical_scale = None

    if (
        np.isfinite(observed_penalty)
        and observed_penalty > 0.0
        and np.isfinite(ll_improvement)
    ):
        critical_scale = float(
            2.0 * ll_improvement
            / observed_penalty
        )

    official_pass = bool(
        bic_improvement >= 0.0
    )
    aic_support = bool(
        aic_improvement > 0.0
    )
    ll_support = bool(
        ll_improvement > 0.0
    )

    if official_pass:
        interpretation = (
            "The selected primary lead passes "
            "the official F8 BIC rule."
        )
    elif ll_support or aic_support:
        interpretation = (
            "The selected primary lead fails official BIC, "
            "while held-out likelihood and/or AIC provide "
            "weaker support. Complexity-scaled scenarios "
            "are diagnostic only and do not replace "
            "official BIC."
        )
    else:
        interpretation = (
            "The selected primary lead fails official BIC "
            "and has no weaker AIC/held-out-LL support."
        )

    summary = {
        "available": True,
        "reason": "ok",
        "selected_primary_lead": _row_payload(
            lead
        ),
        "official_threshold": 0.0,
        "official_bic_improvement": bic_improvement,
        "official_passed_by_data": official_pass,
        "distance_to_official_threshold": bic_improvement,
        "aic_improvement": aic_improvement,
        "heldout_ll_improvement": ll_improvement,
        "aic_positive": aic_support,
        "heldout_ll_positive": ll_support,
        "observed_complexity_penalty": observed_penalty,
        "critical_complexity_penalty_scale": critical_scale,
        "baseline_n_params": baseline_n_params,
        "augmented_n_params": augmented_n_params,
        "parameter_delta": parameter_delta,
        "baseline_n_test": baseline_n_test,
        "augmented_n_test": augmented_n_test,
        "baseline_model_score": (
            _row_payload(baseline_score)
            if baseline_score is not None
            else {}
        ),
        "augmented_model_score": (
            _row_payload(augmented_score)
            if augmented_score is not None
            else {}
        ),
        "interpretation": interpretation,
    }

    return sensitivity_df, summary


def default_eps_saturation_values(
    cfg: Phase32EConfig,
) -> List[float]:
    official = _float_cfg(
        cfg,
        "eps_saturation_value",
        2.0,
    )

    values = [
        0.50,
        0.80,
        1.00,
        1.25,
        1.50,
        2.00,
        2.50,
        3.00,
        4.00,
        5.00,
        official,
    ]

    return sorted(
        {
            float(value)
            for value in values
            if float(value) >= 0.0
        }
    )


def default_eps_warning_fractions(
    cfg: Phase32EConfig,
) -> List[float]:
    official = _float_cfg(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    values = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        1.00,
        official,
    ]

    return sorted(
        {
            float(value)
            for value in values
            if 0.0 <= float(value) <= 1.0
        }
    )


def _model_epsilon_values(
    model_df: pd.DataFrame,
) -> pd.Series:
    if model_df.empty:
        return pd.Series([], dtype=float)

    epsilon_column = next(
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

    if epsilon_column is None:
        return pd.Series([], dtype=float)

    return (
        pd.to_numeric(
            model_df[epsilon_column],
            errors="coerce",
        )
        .dropna()
        .astype(float)
    )


def _pair_epsilon_frame(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame(
            columns=[
                "baseline_eps_test",
                "augmented_eps_test",
                "baseline_eps_saturated",
                "augmented_eps_saturated",
            ]
        )

    return pd.DataFrame(
        {
            "baseline_eps_test": _numeric_col(
                valid,
                "baseline_eps_test",
            ),
            "augmented_eps_test": _numeric_col(
                valid,
                "augmented_eps_test",
            ),
            "baseline_eps_saturated": (
                _as_bool_series(
                    valid[
                        "baseline_eps_saturated"
                    ]
                )
                if "baseline_eps_saturated"
                in valid.columns
                else pd.Series(
                    False,
                    index=valid.index,
                )
            ),
            "augmented_eps_saturated": (
                _as_bool_series(
                    valid[
                        "augmented_eps_saturated"
                    ]
                )
                if "augmented_eps_saturated"
                in valid.columns
                else pd.Series(
                    False,
                    index=valid.index,
                )
            ),
        },
        index=valid.index,
    )


def build_f12_epsilon_sensitivity(
    model_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    official_eps = _float_cfg(
        cfg,
        "eps_saturation_value",
        2.0,
    )
    official_warning_fraction = _float_cfg(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    model_values = _model_epsilon_values(
        model_df
    )

    model_explicit_saturation = (
        _as_bool_series(
            model_df["eps_saturated"]
        )
        if (
            not model_df.empty
            and "eps_saturated"
            in model_df.columns
        )
        else pd.Series(
            False,
            index=model_df.index,
        )
    )

    pair_values = _pair_epsilon_frame(
        pair_df
    )

    columns = [
        "eps_saturation_value",
        "warning_fraction",
        "model_saturation_fraction",
        "pair_either_saturation_fraction",
        "official_metric_fraction",
        "would_warn",
        "distance_to_warning_fraction",
        "n_model_values",
        "n_pair_rows",
        "is_official_scenario",
    ]

    if (
        model_values.empty
        and pair_values.empty
    ):
        return (
            pd.DataFrame(columns=columns),
            {
                "available": False,
                "reason": "no_epsilon_values_available",
                "official_warning_by_data": False,
                "interpretation": (
                    "F12 sensitivity is unavailable because "
                    "no epsilon values exist."
                ),
            },
        )

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

    eps_thresholds = sorted(
        {
            float(value)
            for value in eps_thresholds
            if float(value) >= 0.0
        }
        | {official_eps}
    )

    warning_fractions = sorted(
        {
            float(value)
            for value in warning_fractions
            if 0.0 <= float(value) <= 1.0
        }
        | {official_warning_fraction}
    )

    rows: List[Dict[str, Any]] = []

    for eps_threshold in eps_thresholds:
        if model_values.empty:
            model_fraction = 0.0
        else:
            model_saturated = (
                model_values
                >= eps_threshold
            )

            if (
                np.isclose(
                    eps_threshold,
                    official_eps,
                )
                and len(
                    model_explicit_saturation
                )
                == len(model_saturated)
            ):
                model_saturated = (
                    model_saturated
                    .reset_index(drop=True)
                    | model_explicit_saturation
                    .reset_index(drop=True)
                )

            model_fraction = float(
                model_saturated.mean()
            )

        if pair_values.empty:
            pair_fraction = 0.0
        else:
            pair_saturated = (
                (
                    pair_values[
                        "baseline_eps_test"
                    ]
                    >= eps_threshold
                )
                | (
                    pair_values[
                        "augmented_eps_test"
                    ]
                    >= eps_threshold
                )
            )

            if np.isclose(
                eps_threshold,
                official_eps,
            ):
                pair_saturated = (
                    pair_saturated
                    | pair_values[
                        "baseline_eps_saturated"
                    ].astype(bool)
                    | pair_values[
                        "augmented_eps_saturated"
                    ].astype(bool)
                )

            pair_fraction = float(
                pair_saturated.mean()
            )

        official_metric = max(
            model_fraction,
            pair_fraction,
        )

        for warning_fraction in warning_fractions:
            rows.append(
                {
                    "eps_saturation_value": eps_threshold,
                    "warning_fraction": warning_fraction,
                    "model_saturation_fraction": model_fraction,
                    "pair_either_saturation_fraction": pair_fraction,
                    "official_metric_fraction": official_metric,
                    "would_warn": bool(
                        official_metric
                        >= warning_fraction
                    ),
                    "distance_to_warning_fraction": (
                        official_metric
                        - warning_fraction
                    ),
                    "n_model_values": int(
                        len(model_values)
                    ),
                    "n_pair_rows": int(
                        len(pair_values)
                    ),
                    "is_official_scenario": bool(
                        np.isclose(
                            eps_threshold,
                            official_eps,
                        )
                        and np.isclose(
                            warning_fraction,
                            official_warning_fraction,
                        )
                    ),
                }
            )

    sensitivity_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    official_rows = sensitivity_df[
        sensitivity_df[
            "is_official_scenario"
        ].astype(bool)
    ]

    if official_rows.empty:
        official_fraction = 0.0
        official_warning = False
    else:
        official_fraction = float(
            official_rows.iloc[0][
                "official_metric_fraction"
            ]
        )
        official_warning = bool(
            official_rows.iloc[0][
                "would_warn"
            ]
        )

    metrics_model_summary = (
        model_saturation_summary(
            model_df,
            cfg,
        )
    )
    metrics_pair_summary = (
        pair_saturation_summary(
            pair_df,
            cfg,
        )
    )

    summary = {
        "available": True,
        "reason": "ok",
        "official_eps_saturation_value": official_eps,
        "official_warning_fraction": official_warning_fraction,
        "official_saturation_fraction": official_fraction,
        "official_warning_by_data": official_warning,
        "distance_to_warning_fraction": (
            official_fraction
            - official_warning_fraction
        ),
        "max_model_epsilon": (
            float(model_values.max())
            if not model_values.empty
            else 0.0
        ),
        "max_baseline_pair_epsilon": (
            float(
                pair_values[
                    "baseline_eps_test"
                ].max()
            )
            if not pair_values.empty
            else 0.0
        ),
        "max_augmented_pair_epsilon": (
            float(
                pair_values[
                    "augmented_eps_test"
                ].max()
            )
            if not pair_values.empty
            else 0.0
        ),
        "metrics_model_saturation_summary": (
            metrics_model_summary
        ),
        "metrics_pair_saturation_summary": (
            metrics_pair_summary
        ),
        "diagnostic_only": True,
        "interpretation": (
            "F12 raises an official diagnostic warning. "
            "This does not erase a primary lead, but the "
            "saturation pattern must be documented."
            if official_warning
            else (
                "F12 does not raise an official diagnostic "
                "saturation warning. Exploratory threshold "
                "scenarios remain diagnostic only."
            )
        ),
    }

    return sensitivity_df, summary


def build_gate_sensitivity_result(
    f7_summary: Mapping[str, Any],
    f8_summary: Mapping[str, Any],
    f12_summary: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> GateSensitivityResult:
    final_status = gates_payload.get(
        "final_status",
        {},
    )

    protocol_only = bool(
        final_status.get(
            "protocol_only",
            False,
        )
        if isinstance(
            final_status,
            dict,
        )
        else False
    )

    f7_available = bool(
        f7_summary.get(
            "available",
            False,
        )
    )
    f8_available = bool(
        f8_summary.get(
            "available",
            False,
        )
    )
    f12_available = bool(
        f12_summary.get(
            "available",
            False,
        )
    )

    if protocol_only:
        status = getattr(
            cfg,
            "status_pending_synchronized_data",
            "pending_synchronized_data",
        )
        interpretation = (
            "No synchronized EEG–QRNG dataset is "
            "available. Gate sensitivity remains "
            "protocol-only."
        )
    elif not any(
        (
            f7_available,
            f8_available,
            f12_available,
        )
    ):
        status = "gate_sensitivity_unavailable"
        interpretation = (
            "Gate sensitivity is unavailable because "
            "no valid pair/model outputs exist."
        )
    else:
        status = "gate_sensitivity_completed"
        interpretation = (
            "F7/F8/F12 sensitivity diagnostics completed "
            "without changing official gates. F7 uses the "
            "adaptive selected-lead rule, F8 preserves BIC "
            "as official while reporting weaker complexity "
            "sensitivities, and F12 remains diagnostic."
        )

    f7_required_subjects = _safe_int(
        f7_summary.get(
            "official_required_subjects"
        ),
        0,
    )
    f7_required_fraction = _safe_float(
        f7_summary.get(
            "official_required_fraction"
        ),
        _float_cfg(
            cfg,
            "subject_effect_fraction",
            0.10,
        ),
    )
    f7_selected_fraction = _safe_float(
        f7_summary.get(
            "observed_fraction_positive_lift"
        ),
        0.0,
    )
    f7_selected_positive = _safe_int(
        f7_summary.get(
            "n_subjects_positive_lift"
        ),
        0,
    )

    f8_bic = _safe_float(
        f8_summary.get(
            "official_bic_improvement"
        ),
        0.0,
    )
    f8_aic = _safe_float(
        f8_summary.get(
            "aic_improvement"
        ),
        0.0,
    )
    f8_ll = _safe_float(
        f8_summary.get(
            "heldout_ll_improvement"
        ),
        0.0,
    )

    f12_fraction = _safe_float(
        f12_summary.get(
            "official_saturation_fraction"
        ),
        0.0,
    )
    f12_eps = _float_cfg(
        cfg,
        "eps_saturation_value",
        2.0,
    )
    f12_warning_fraction = _float_cfg(
        cfg,
        "eps_saturation_warning_fraction",
        0.95,
    )

    return GateSensitivityResult(
        status=status,
        protocol_only=protocol_only,
        official_f7_required_subjects=(
            f7_required_subjects
        ),
        official_f7_required_fraction=(
            f7_required_fraction
        ),
        official_f8_threshold=0.0,
        official_f12_eps_saturation_value=(
            f12_eps
        ),
        official_f12_warning_fraction=(
            f12_warning_fraction
        ),
        f7_available=f7_available,
        f8_available=f8_available,
        f12_available=f12_available,
        f7_official_passed=official_gate_passed(
            gates_payload,
            "F7",
        ),
        f8_official_passed=official_gate_passed(
            gates_payload,
            "F8",
        ),
        f12_official_warning=official_gate_warning(
            gates_payload,
            "F12",
        ),
        f7_selected_fraction=f7_selected_fraction,
        f7_selected_positive_subjects=(
            f7_selected_positive
        ),
        f8_selected_bic_improvement=f8_bic,
        f8_selected_aic_improvement=f8_aic,
        f8_selected_ll_improvement=f8_ll,
        f12_official_saturation_fraction=(
            f12_fraction
        ),
        f7_fraction_distance_to_official=(
            f7_selected_fraction
            - f7_required_fraction
        ),
        f7_subject_count_distance_to_official=(
            f7_selected_positive
            - f7_required_subjects
        ),
        f8_distance_to_official=f8_bic,
        f12_distance_to_warning=(
            f12_fraction
            - f12_warning_fraction
        ),
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
    lines = [
        "=" * 78,
        "Phase III.2E — Gate Sensitivity Summary",
        (
            "Adaptive F7 / official-BIC F8 / "
            "diagnostic F12"
        ),
        "=" * 78,
        f"status: {result.status}",
        f"protocol_only: {result.protocol_only}",
        (
            "can_proceed_to_diagnostics: "
            f"{result.can_proceed_to_diagnostics}"
        ),
        (
            "can_claim_empirical_cdr: "
            f"{result.can_claim_empirical_cdr}"
        ),
        "",
        "Official gate statuses",
        "-" * 78,
        (
            "F7 official status: "
            f"{official_gate_status(gates_payload, 'F7')}"
        ),
        (
            "F8 official status: "
            f"{official_gate_status(gates_payload, 'F8')}"
        ),
        (
            "F12 official status: "
            f"{official_gate_status(gates_payload, 'F12')}"
        ),
        "",
        "F7 — Adaptive subject consistency",
        "-" * 78,
        f"available: {f7_summary.get('available')}",
        (
            "selected_primary_lead: "
            f"{f7_summary.get('selected_primary_lead')}"
        ),
        (
            "n_subjects_evaluated: "
            f"{f7_summary.get('n_subjects_evaluated')}"
        ),
        (
            "n_subjects_positive_lift: "
            f"{f7_summary.get('n_subjects_positive_lift')}"
        ),
        (
            "observed_fraction_positive_lift: "
            f"{f7_summary.get('observed_fraction_positive_lift')}"
        ),
        (
            "official_required_subjects: "
            f"{f7_summary.get('official_required_subjects')}"
        ),
        (
            "official_required_fraction: "
            f"{f7_summary.get('official_required_fraction')}"
        ),
        (
            "official_passed_by_data: "
            f"{f7_summary.get('official_passed_by_data')}"
        ),
        (
            "interpretation: "
            f"{f7_summary.get('interpretation')}"
        ),
        "",
        "F8 — Complexity penalty sensitivity",
        "-" * 78,
        f"available: {f8_summary.get('available')}",
        (
            "selected_primary_lead: "
            f"{f8_summary.get('selected_primary_lead')}"
        ),
        (
            "official_bic_improvement: "
            f"{f8_summary.get('official_bic_improvement')}"
        ),
        (
            "aic_improvement: "
            f"{f8_summary.get('aic_improvement')}"
        ),
        (
            "heldout_ll_improvement: "
            f"{f8_summary.get('heldout_ll_improvement')}"
        ),
        (
            "observed_complexity_penalty: "
            f"{f8_summary.get('observed_complexity_penalty')}"
        ),
        (
            "critical_complexity_penalty_scale: "
            f"{f8_summary.get('critical_complexity_penalty_scale')}"
        ),
        (
            "official_passed_by_data: "
            f"{f8_summary.get('official_passed_by_data')}"
        ),
        (
            "interpretation: "
            f"{f8_summary.get('interpretation')}"
        ),
        "",
        "F12 — Epsilon saturation diagnostic",
        "-" * 78,
        f"available: {f12_summary.get('available')}",
        (
            "official_eps_saturation_value: "
            f"{f12_summary.get('official_eps_saturation_value')}"
        ),
        (
            "official_warning_fraction: "
            f"{f12_summary.get('official_warning_fraction')}"
        ),
        (
            "official_saturation_fraction: "
            f"{f12_summary.get('official_saturation_fraction')}"
        ),
        (
            "official_warning_by_data: "
            f"{f12_summary.get('official_warning_by_data')}"
        ),
        (
            "interpretation: "
            f"{f12_summary.get('interpretation')}"
        ),
        "",
        "Overall interpretation",
        "-" * 78,
        result.interpretation,
        "=" * 78,
    ]

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def build_input_audit(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    gates_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    primary_count = (
        int(
            effective_primary_mask(
                valid,
                cfg,
            ).sum()
        )
        if not valid.empty
        else 0
    )

    lead = selected_primary_lead(
        pair_df,
        cfg,
    )

    payload = {
        "pair_rows": int(len(pair_df)),
        "valid_pair_rows": int(len(valid)),
        "primary_pair_rows": primary_count,
        "model_rows": int(len(model_df)),
        "gates_available": bool(
            gates_payload.get("available")
        ),
        "metrics_status": (
            gates_payload.get(
                "final_status",
                {},
            ).get("status")
            if isinstance(
                gates_payload.get(
                    "final_status",
                    {},
                ),
                dict,
            )
            else None
        ),
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
        "selected_primary_lead": (
            _row_payload(lead)
        ),
    }

    payload["fingerprint"] = _fingerprint(
        payload
    )

    return payload


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
    Path(cfg.results_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    f7_df.to_csv(
        f7_sensitivity_csv_path(cfg),
        index=False,
    )
    f8_df.to_csv(
        f8_sensitivity_csv_path(cfg),
        index=False,
    )
    f12_df.to_csv(
        f12_sensitivity_csv_path(cfg),
        index=False,
    )

    input_audit = build_input_audit(
        pair_df,
        model_df,
        gates_payload,
        cfg,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": GATE_SENSITIVITY_MODULE,
        "gate_sensitivity_version": (
            GATE_SENSITIVITY_VERSION
        ),
        "created_at": _now_str(),
        "status": result.status,
        "result": asdict(result),
        "input_audit": input_audit,
        "official_gate_statuses": {
            "F7": official_gate_status(
                gates_payload,
                "F7",
            ),
            "F8": official_gate_status(
                gates_payload,
                "F8",
            ),
            "F12": official_gate_status(
                gates_payload,
                "F12",
            ),
        },
        "f7_subject_consistency": dict(
            f7_summary
        ),
        "f8_complexity_penalty": dict(
            f8_summary
        ),
        "f12_epsilon_saturation": dict(
            f12_summary
        ),
        "files": {
            "f7_sensitivity_csv": str(
                f7_sensitivity_csv_path(cfg)
            ),
            "f8_sensitivity_csv": str(
                f8_sensitivity_csv_path(cfg)
            ),
            "f12_sensitivity_csv": str(
                f12_sensitivity_csv_path(cfg)
            ),
            "gate_sensitivity_report_json": str(
                gate_sensitivity_report_json_path(
                    cfg
                )
            ),
            "gate_sensitivity_summary_txt": str(
                gate_sensitivity_summary_txt_path(
                    cfg
                )
            ),
        },
        "official_gate_rule_preservation": {
            "official_gates_changed": False,
            "f7_rule": (
                "adaptive subject count/fraction "
                "from the selected primary lead"
            ),
            "f8_rule": (
                "official BIC improvement >= 0"
            ),
            "f12_rule": (
                "diagnostic saturation warning only"
            ),
            "note": (
                "Sensitivity scenarios explain the "
                "current result and never replace "
                "official metrics gates."
            ),
        },
        "config": describe_config(cfg),
    }

    save_json(
        gate_sensitivity_report_json_path(cfg),
        report,
    )

    write_gate_sensitivity_summary_txt(
        gate_sensitivity_summary_txt_path(cfg),
        result,
        f7_summary,
        f8_summary,
        f12_summary,
        gates_payload,
    )

    print(
        "[Phase3.2E GateSensitivity] "
        f"Saved F7 CSV: "
        f"{f7_sensitivity_csv_path(cfg)}"
    )
    print(
        "[Phase3.2E GateSensitivity] "
        f"Saved F8 CSV: "
        f"{f8_sensitivity_csv_path(cfg)}"
    )
    print(
        "[Phase3.2E GateSensitivity] "
        f"Saved F12 CSV: "
        f"{f12_sensitivity_csv_path(cfg)}"
    )
    print(
        "[Phase3.2E GateSensitivity] "
        f"Saved report: "
        f"{gate_sensitivity_report_json_path(cfg)}"
    )
    print(
        "[Phase3.2E GateSensitivity] "
        f"Saved summary: "
        f"{gate_sensitivity_summary_txt_path(cfg)}"
    )

    return report


def validate_inputs(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    pair_df = _load_csv(
        conditional_pair_scores_csv_path(cfg)
    )
    model_df = _load_csv(
        conditional_model_scores_csv_path(cfg)
    )
    gates_payload = load_official_gate_status(
        cfg
    )

    audit = build_input_audit(
        pair_df,
        model_df,
        gates_payload,
        cfg,
    )

    audit["pair_scores_file"] = str(
        conditional_pair_scores_csv_path(cfg)
    )
    audit["model_scores_file"] = str(
        conditional_model_scores_csv_path(cfg)
    )
    audit["gates_json_file"] = str(
        gates_json_path(cfg)
    )
    audit["ready"] = bool(
        len(pair_df) > 0
        and len(model_df) > 0
        and gates_payload.get(
            "available",
            False,
        )
    )

    return audit


def run_phase3_2e_gate_sensitivity(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_phase3_2e_config()

    pair_df = _load_csv(
        conditional_pair_scores_csv_path(cfg)
    )
    model_df = _load_csv(
        conditional_model_scores_csv_path(cfg)
    )
    gates_payload = load_official_gate_status(
        cfg
    )

    f7_df, f7_summary = (
        build_f7_subject_sensitivity(
            pair_df,
            cfg,
        )
    )
    f8_df, f8_summary = (
        build_f8_complexity_sensitivity(
            pair_df,
            model_df,
            cfg,
        )
    )
    f12_df, f12_summary = (
        build_f12_epsilon_sensitivity(
            model_df,
            pair_df,
            cfg,
        )
    )

    result = build_gate_sensitivity_result(
        f7_summary,
        f8_summary,
        f12_summary,
        gates_payload,
        cfg,
    )

    return save_gate_sensitivity_outputs(
        cfg,
        result,
        f7_df,
        f8_df,
        f12_df,
        f7_summary,
        f8_summary,
        f12_summary,
        gates_payload,
        pair_df,
        model_df,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase III.2E F7/F8/F12 "
            "gate-sensitivity diagnostics."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate existing conditional/metrics "
            "outputs without rebuilding anything."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase3_2e_config()

    if args.validate_only:
        audit = validate_inputs(cfg)

        print(
            json.dumps(
                audit,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )

        raise SystemExit(
            0
            if audit.get("ready")
            else 1
        )

    report = run_phase3_2e_gate_sensitivity(
        cfg
    )

    result = report.get(
        "result",
        {},
    )
    f7 = report.get(
        "f7_subject_consistency",
        {},
    )
    f8 = report.get(
        "f8_complexity_penalty",
        {},
    )
    f12 = report.get(
        "f12_epsilon_saturation",
        {},
    )

    print("\n" + "=" * 78)
    print(
        "Phase III.2E gate sensitivity completed"
    )
    print(
        "Adaptive F7 / official-BIC F8 / "
        "diagnostic F12"
    )
    print("=" * 78)

    print(
        f"status: "
        f"{result.get('status')}"
    )
    print(
        f"protocol_only: "
        f"{result.get('protocol_only')}"
    )
    print(
        "can_proceed_to_diagnostics: "
        f"{result.get('can_proceed_to_diagnostics')}"
    )
    print(
        "can_claim_empirical_cdr: "
        f"{result.get('can_claim_empirical_cdr')}"
    )

    print("")
    print(
        "F7 official status: "
        f"{report.get('official_gate_statuses', {}).get('F7')}"
    )
    print(
        "F7 selected subjects: "
        f"{f7.get('n_subjects_positive_lift')}/"
        f"{f7.get('n_subjects_evaluated')}"
    )
    print(
        "F7 required subjects: "
        f"{f7.get('official_required_subjects')}"
    )
    print(
        "F7 fraction distance: "
        f"{f7.get('fraction_distance_to_official')}"
    )

    print("")
    print(
        "F8 official status: "
        f"{report.get('official_gate_statuses', {}).get('F8')}"
    )
    print(
        "F8 selected BIC improvement: "
        f"{f8.get('official_bic_improvement')}"
    )
    print(
        "F8 selected AIC improvement: "
        f"{f8.get('aic_improvement')}"
    )
    print(
        "F8 selected held-out LL improvement: "
        f"{f8.get('heldout_ll_improvement')}"
    )
    print(
        "F8 critical complexity scale: "
        f"{f8.get('critical_complexity_penalty_scale')}"
    )

    print("")
    print(
        "F12 official status: "
        f"{report.get('official_gate_statuses', {}).get('F12')}"
    )
    print(
        "F12 official saturation fraction: "
        f"{f12.get('official_saturation_fraction')}"
    )
    print(
        "F12 distance to warning: "
        f"{f12.get('distance_to_warning_fraction')}"
    )

    print("")
    print(
        f"report_json: "
        f"{gate_sensitivity_report_json_path(cfg)}"
    )
    print(
        f"summary_txt: "
        f"{gate_sensitivity_summary_txt_path(cfg)}"
    )
    print(
        f"interpretation: "
        f"{result.get('interpretation')}"
    )

    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()