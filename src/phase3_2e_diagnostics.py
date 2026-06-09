from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    describe_config,
    load_phase3_2e_config,
)
from src.phase3_2e_metrics import (
    effective_primary_mask,
    exploratory_positive_lift_rows,
    model_saturation_summary,
    pair_saturation_summary,
    primary_lags,
    primary_pairs,
    primary_positive_lift_rows,
    primary_strong_candidate_rows,
    primary_windows,
    select_primary_lead_row,
    strong_candidate_rows,
    subject_consistency_for_row,
    valid_pair_rows,
)

DIAGNOSTICS_MODULE = "phase3_2e_diagnostics"
DIAGNOSTICS_VERSION = (
    "phase3_2e_diagnostics_v2_primary_lead_"
    "adaptive_f7_bic_limited_f12_diagnostic"
)


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


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        return payload if isinstance(payload, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def load_csv(
    path: Path,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            nrows=nrows,
        )

    except (
        OSError,
        UnicodeDecodeError,
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

        return (
            number
            if np.isfinite(number)
            else default
        )

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


def _safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "y",
        "sim",
        "t",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "n",
        "nao",
        "não",
        "f",
    }:
        return False

    return default


def _bool_series(
    series: pd.Series,
) -> pd.Series:
    if series.empty:
        return pd.Series(
            [],
            index=series.index,
            dtype=bool,
        )

    if pd.api.types.is_bool_dtype(series):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    return (
        series
        .astype(str)
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


def _num(
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

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .fillna(default)
    )


def _records(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    return (
        df
        .replace({np.nan: None})
        .to_dict(orient="records")
    )


def _row(
    row: Optional[pd.Series],
) -> Dict[str, Any]:
    if row is None:
        return {}

    return {
        str(key): _json_default(value)
        for key, value in row.to_dict().items()
    }


def _result(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    value = (
        payload.get("result", {})
        if payload
        else {}
    )

    return value if isinstance(value, dict) else {}


def _summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    value = (
        payload.get("summary", {})
        if payload
        else {}
    )

    return value if isinstance(value, dict) else {}


def _status(
    payload: Mapping[str, Any],
) -> str:
    return str(
        payload.get(
            "status",
            _result(payload).get(
                "status",
                "missing",
            ),
        )
    )


def _first_existing(
    candidates: Sequence[Path],
) -> Path:
    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def _file_status(
    path: Path,
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": (
            path.stat().st_size
            if path.exists()
            else 0
        ),
    }


def phase3_2e_paths(
    cfg: Phase32EConfig,
) -> Dict[str, Path]:
    results = Path(cfg.results_dir)
    interim = Path(cfg.interim_dir)

    diagnostics = Path(
        getattr(
            cfg,
            "diagnostics_dir",
            results / "diagnostics",
        )
    )

    sync_input = Path(
        getattr(
            cfg,
            "synchronized_input_csv",
            Path(
                "data/raw/phase3_2e/"
                "synchronized/"
                "phase3_2e_synchronized_input.csv"
            ),
        )
    )

    return {
        "sync_input": sync_input,

        "schema_report":
            results / "phase3_2e_schema_report.json",

        "sync_report":
            _first_existing(
                [
                    results
                    / "phase3_2e_sync_report.json",

                    results
                    / "phase3_2e_sync_validation_report.json",
                ]
            ),

        "loader_report":
            results / "phase3_2e_loader_report.json",

        "windows_report":
            results / "phase3_2e_windows_report.json",

        "features_report":
            results / "phase3_2e_features_report.json",

        "alignment_report":
            results / "phase3_2e_alignment_report.json",

        "conditional_report":
            results / "phase3_2e_conditional_report.json",

        "controls_json":
            Path(
                getattr(
                    cfg,
                    "controls_json",
                    results / "phase3_2e_controls.json",
                )
            ),

        "gates_json":
            Path(
                getattr(
                    cfg,
                    "gates_json",
                    results / "phase3_2e_gates.json",
                )
            ),

        "sensitivity_report":
            results
            / "phase3_2e_gate_sensitivity_report.json",

        "mc_guard":
            Path(
                getattr(
                    cfg,
                    "multiple_comparison_guard_json",
                    results
                    / "phase3_2e_multiple_comparison_guard.json",
                )
            ),

        "loaded":
            interim
            / "phase3_2e_loaded_synchronized_windows.csv",

        "windows":
            interim
            / "phase3_2e_windows.csv",

        "features":
            interim
            / "phase3_2e_features.csv",

        "feature_specs":
            interim
            / "phase3_2e_feature_specs.json",

        "alignment_inventory":
            interim
            / "phase3_2e_alignment_inventory.csv",

        "pair_scores":
            Path(cfg.conditional_pair_scores_csv),

        "model_scores":
            Path(cfg.conditional_model_scores_csv),

        "control_runs":
            Path(cfg.control_runs_csv),

        "control_pairs":
            Path(cfg.control_pair_scores_csv),

        "metrics_table":
            results
            / "phase3_2e_metrics_table.csv",

        "f7":
            results
            / "phase3_2e_f7_subject_sensitivity.csv",

        "f8":
            results
            / "phase3_2e_f8_complexity_sensitivity.csv",

        "f12":
            results
            / "phase3_2e_f12_epsilon_saturation_sensitivity.csv",

        "diagnostics": diagnostics,
    }


def primary_rows(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    return valid[
        effective_primary_mask(
            valid,
            cfg,
        )
    ].copy()


def positive_rows(
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


def multichannel_rows(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return pd.DataFrame()

    if "multichannel" in valid.columns:
        return valid[
            _bool_series(
                valid["multichannel"]
            )
        ].copy()

    mask = pd.Series(
        False,
        index=valid.index,
    )

    for column in (
        "baseline_model",
        "augmented_model",
        "family",
    ):
        if column not in valid.columns:
            continue

        text = valid[column].astype(str)

        mask |= text.str.contains(
            "MCEEG|MC_EEG|multichannel",
            case=False,
            na=False,
        )

        mask |= text.str.startswith(
            (
                "E4",
                "E5",
                "E6",
            )
        )

    return valid[mask].copy()


def best_rows_by_metric(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = valid_pair_rows(pair_df)

    rows: List[Dict[str, Any]] = []

    for metric in (
        "conditional_lift",
        "augmented_eps_test",
        "baseline_eps_test",
        "ll_improvement",
        "bic_improvement",
        "aic_improvement",
        "fraction_positive_lift",
    ):
        if metric not in valid.columns:
            continue

        numeric = pd.to_numeric(
            valid[metric],
            errors="coerce",
        )

        if numeric.notna().any():
            row = (
                valid
                .loc[numeric.idxmax()]
                .to_dict()
            )

            row["selected_by"] = metric

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_provenance(
    input_sample: pd.DataFrame,
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    conditional_payload: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    source_modes: List[str] = []
    claim_flags: List[bool] = []

    for frame in (
        input_sample,
        pair_df,
        model_df,
    ):
        for column in (
            "source_mode",
            "source_type",
            "data_mode",
            "provenance",
        ):
            if (
                not frame.empty
                and column in frame.columns
            ):
                source_modes.extend(
                    frame[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

        if (
            not frame.empty
            and "empirical_claim_allowed"
            in frame.columns
        ):
            claim_flags.extend(
                _bool_series(
                    frame["empirical_claim_allowed"]
                ).tolist()
            )

    final_status = gates_payload.get(
        "final_status",
        {},
    )

    if not isinstance(final_status, dict):
        final_status = {}

    for source in (
        conditional_payload,
        _result(conditional_payload),
        final_status,
    ):
        for key in (
            "source_mode",
            "source_type",
            "data_mode",
            "provenance",
        ):
            if source.get(key) is not None:
                source_modes.append(
                    str(source[key])
                )

        for key in (
            "empirical_claim_allowed",
            "can_claim_empirical_cdr",
            "source_claim_allowed",
        ):
            if key in source:
                claim_flags.append(
                    _safe_bool(
                        source[key]
                    )
                )

    source_modes = sorted(
        {
            value
            for value in source_modes
            if value
        }
    )

    source_text = " ".join(
        source_modes
    ).lower()

    surrogate = any(
        token in source_text
        for token in (
            "surrogate",
            "synthetic",
            "demo",
            "replication_ready",
        )
    )

    if (
        claim_flags
        and not any(claim_flags)
    ):
        surrogate = True

    metrics_claim = _safe_bool(
        final_status.get(
            "can_claim_empirical_cdr",
            False,
        )
    )

    require_real = bool(
        getattr(
            cfg,
            "require_real_synchronized_data_for_claim",
            True,
        )
    )

    can_claim = bool(
        metrics_claim
        and not surrogate
        and (
            not require_real
            or bool(
                claim_flags
                and all(claim_flags)
            )
        )
    )

    return {
        "source_class": (
            "surrogate_synchronized_input"
            if surrogate
            else (
                "real_or_external_synchronized_input"
                if source_modes
                else "provenance_unknown"
            )
        ),

        "source_modes": source_modes,

        "surrogate_detected": surrogate,

        "require_real_synchronized_data_for_claim":
            require_real,

        "can_claim_empirical_cdr": can_claim,

        "interpretation": (
            "Surrogate synchronized input: valid for "
            "pipeline validation, but not for an "
            "empirical EEG–QRNG claim."
            if surrogate
            else (
                "No surrogate marker was detected; "
                "acquisition provenance is still required."
            )
        ),
    }


def summarize_feature_audit(
    windows_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_specs: Mapping[str, Any],
    features_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    required = [
        column
        for column in (
            "subject_id",
            "session_id",
            "window_seconds",
            "window_id",
        )
        if column in windows_df.columns
    ]

    missing = [
        column
        for column in required
        if column not in features_df.columns
    ]

    passed = bool(
        len(windows_df) > 0
        and len(features_df) == len(windows_df)
        and feature_specs
        and features_payload
        and not missing
    )

    return {
        "passed": passed,

        "windows_rows":
            len(windows_df),

        "features_rows":
            len(features_df),

        "missing_identity_columns":
            missing,

        "interpretation": (
            "Feature audit passed: rebuilt features "
            "match the current windows table."
            if passed
            else (
                "Feature audit found a row-count, "
                "metadata or identity-column problem."
            )
        ),
    }


def summarize_grid(
    pair_df: pd.DataFrame,
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    if valid.empty:
        return {
            "available": False,
            "reason": "no_valid_pair_rows",
        }

    derived = effective_primary_mask(
        valid,
        cfg,
    )

    stored = (
        _bool_series(
            valid["primary"]
        )
        if "primary" in valid.columns
        else pd.Series(
            False,
            index=valid.index,
        )
    )

    windows = sorted(
        pd.to_numeric(
            valid["window_seconds"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    lags = sorted(
        pd.to_numeric(
            valid["lag_windows"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return {
        "available": True,

        "valid_rows":
            len(valid),

        "derived_primary_rows":
            int(derived.sum()),

        "derived_exploratory_rows":
            int((~derived).sum()),

        "stored_primary_disagreements":
            int((derived != stored).sum()),

        "windows_found":
            windows,

        "lags_found":
            lags,

        "primary_windows":
            list(primary_windows(cfg)),

        "primary_lags":
            list(primary_lags(cfg)),

        "primary_pairs":
            [
                list(pair)
                for pair in primary_pairs(cfg)
            ],
    }


def gate_map(
    gates_payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("gate")): item
        for item in gates_payload.get(
            "gates",
            [],
        )
        if isinstance(item, dict)
    }


def gate_passed(
    gates: Mapping[str, Mapping[str, Any]],
    gate: str,
) -> bool:
    return (
        str(
            gates
            .get(gate, {})
            .get("status", "")
        ).upper()
        == "PASS"
    )


def summarize_conditional(
    pair_df: pd.DataFrame,
    model_df: pd.DataFrame,
    conditional_payload: Mapping[str, Any],
    gates_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    valid = valid_pair_rows(pair_df)

    primary = primary_rows(
        pair_df,
        cfg,
    )

    positive = positive_rows(
        pair_df
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

    official_strong = (
        strong_candidate_rows(
            pair_df,
            cfg,
        )
    )

    official_primary_strong = (
        primary_strong_candidate_rows(
            pair_df,
            cfg,
        )
    )

    multichannel = multichannel_rows(
        pair_df
    )

    multichannel_positive = (
        multichannel[
            _num(
                multichannel,
                "conditional_lift",
            ) > 0.0
        ].copy()
        if not multichannel.empty
        else pd.DataFrame()
    )

    lead = select_primary_lead_row(
        pair_df,
        cfg,
    )

    gates = gate_map(
        gates_payload
    )

    localized_support = bool(
        lead is not None
        and not primary_positive.empty
        and all(
            gate_passed(
                gates,
                gate,
            )
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
    )

    f8_pass = gate_passed(
        gates,
        "F8",
    )

    f7 = (
        subject_consistency_for_row(
            lead,
            cfg,
        )
        if lead is not None
        else {}
    )

    f8 = {
        "official_rule":
            "bic_improvement >= 0",

        "official_passed":
            f8_pass,

        "bic_improvement": (
            _safe_float(
                lead.get(
                    "bic_improvement"
                )
            )
            if lead is not None
            else 0.0
        ),

        "aic_improvement": (
            _safe_float(
                lead.get(
                    "aic_improvement"
                )
            )
            if lead is not None
            else 0.0
        ),

        "heldout_ll_improvement": (
            _safe_float(
                lead.get(
                    "ll_improvement"
                )
            )
            if lead is not None
            else 0.0
        ),

        "does_not_negate_localized_structure":
            bool(
                localized_support
                and not f8_pass
            ),

        "future_v3_review": {
            "prospective_only": True,
            "preserve_current_v2_result": True,

            "options": [
                "effective_sample_size_BIC",
                "state_density_corrected_BIC",
                "effective_complexity",
                "subject_level_model_selection",
            ],
        },
    }

    if (
        localized_support
        and not f8_pass
    ):
        interpretation = (
            "Registered primary localized predictive "
            "support is present, but the current "
            "official global BIC gate is not satisfied."
        )

    elif not official_primary_strong.empty:
        interpretation = (
            "At least one primary row satisfies "
            "the current official strong-candidate rule."
        )

    elif not primary_positive.empty:
        interpretation = (
            "A primary positive row exists with "
            "incomplete official gate support."
        )

    elif not exploratory_positive.empty:
        interpretation = (
            "Positive rows exist only in "
            "the exploratory region."
        )

    else:
        interpretation = (
            "No positive synchronized "
            "conditional lift was detected."
        )

    return {
        "status":
            _status(conditional_payload),

        "n_model_rows":
            len(model_df),

        "n_pair_rows":
            len(pair_df),

        "n_valid_rows":
            len(valid),

        "n_primary_rows":
            len(primary),

        "n_positive_lift_rows":
            len(positive),

        "n_primary_positive_lift_rows":
            len(primary_positive),

        "n_exploratory_positive_lift_rows":
            len(exploratory_positive),

        "n_official_strong_candidate_rows":
            len(official_strong),

        "n_official_primary_strong_candidate_rows":
            len(official_primary_strong),

        "n_multichannel_rows":
            len(multichannel),

        "n_multichannel_positive_lift_rows":
            len(multichannel_positive),

        "selected_primary_lead":
            _row(lead),

        "localized_predictive_support":
            localized_support,

        "f7_adaptive_subject_consistency":
            f7,

        "f8_complexity_interpretation":
            f8,

        "model_saturation":
            model_saturation_summary(
                model_df,
                cfg,
            ),

        "pair_saturation":
            pair_saturation_summary(
                pair_df,
                cfg,
            ),

        "interpretation":
            interpretation,
    }


def summarize_controls(
    control_runs: pd.DataFrame,
    controls_payload: Mapping[str, Any],
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    result = _result(
        controls_payload
    )

    summary = _summary(
        controls_payload
    )

    passed = bool(
        summary.get(
            "passed",
            result.get(
                "overall_passed",
                False,
            ),
        )
    )

    breakdown = summary.get(
        "control_summaries",
        [],
    )

    if not isinstance(
        breakdown,
        list,
    ):
        breakdown = []

    if (
        not breakdown
        and not control_runs.empty
        and "control_type"
        in control_runs.columns
    ):
        for control_type, group in (
            control_runs.groupby(
                "control_type",
                sort=True,
            )
        ):
            valid = (
                group[
                    _bool_series(
                        group["valid"]
                    )
                ]
                if "valid" in group.columns
                else group
            )

            collapsed = (
                _bool_series(
                    valid["collapsed_overall"]
                )
                if "collapsed_overall"
                in valid.columns
                else pd.Series(
                    False,
                    index=valid.index,
                )
            )

            fraction = (
                float(
                    collapsed.mean()
                )
                if len(collapsed)
                else 0.0
            )

            breakdown.append(
                {
                    "control_type":
                        str(control_type),

                    "n_runs":
                        len(group),

                    "n_valid_runs":
                        len(valid),

                    "fraction_runs_collapsed_overall":
                        fraction,

                    "passed":
                        fraction
                        >= float(
                            getattr(
                                cfg,
                                "required_control_fraction",
                                0.75,
                            )
                        ),

                    "source":
                        "stored_collapsed_overall",
                }
            )

    return {
        "status":
            _status(controls_payload),

        "passed":
            passed,

        "n_control_runs":
            _safe_int(
                result.get(
                    "n_control_runs"
                ),
                len(control_runs),
            ),

        "n_valid_control_runs":
            _safe_int(
                result.get(
                    "n_valid_control_runs"
                ),
                (
                    int(
                        _bool_series(
                            control_runs["valid"]
                        ).sum()
                    )
                    if (
                        not control_runs.empty
                        and "valid"
                        in control_runs.columns
                    )
                    else len(control_runs)
                ),
            ),

        "n_control_types":
            _safe_int(
                result.get(
                    "n_control_types"
                ),
                len(breakdown),
            ),

        "failed_control_types":
            list(
                summary.get(
                    "failed_control_types",
                    result.get(
                        "failed_control_types",
                        [],
                    ),
                )
            ),

        "required_control_types":
            list(
                summary.get(
                    "required_control_types",
                    getattr(
                        cfg,
                        "required_control_types",
                        [],
                    ),
                )
            ),

        "control_subject_fraction_threshold":
            summary.get(
                "control_subject_fraction_threshold",
                controls_payload.get(
                    "control_subject_fraction_threshold",
                    0.60,
                ),
            ),

        "discovery_subject_effect_fraction":
            summary.get(
                "discovery_subject_effect_fraction",
                controls_payload.get(
                    "discovery_subject_effect_fraction",
                    getattr(
                        cfg,
                        "subject_effect_fraction",
                        0.10,
                    ),
                ),
            ),

        "control_breakdown":
            breakdown,

        "interpretation": (
            "All required negative controls passed "
            "under the corrected control-specific "
            "classification."
            if passed
            else (
                "One or more required "
                "negative controls failed."
            )
        ),
    }


def summarize_gates(
    gates_payload: Mapping[str, Any],
    metrics_table: pd.DataFrame,
) -> Dict[str, Any]:
    final = gates_payload.get(
        "final_status",
        {},
    )

    if not isinstance(final, dict):
        final = {}

    gates = gates_payload.get(
        "gates",
        [],
    )

    if (
        not gates
        and not metrics_table.empty
    ):
        gates = metrics_table.to_dict(
            orient="records"
        )

    grouped = {
        "PASS": [],
        "FAIL": [],
        "WARN": [],
        "NOT_EVALUATED": [],
    }

    for gate in gates:
        status = str(
            gate.get(
                "status",
                "",
            )
        ).upper()

        if status in grouped:
            grouped[status].append(
                str(
                    gate.get(
                        "gate",
                        "",
                    )
                )
            )

    return {
        "status":
            str(
                final.get(
                    "status",
                    gates_payload.get(
                        "status",
                        "missing",
                    ),
                )
            ),

        "can_claim_empirical_cdr":
            bool(
                final.get(
                    "can_claim_empirical_cdr",
                    False,
                )
            ),

        "passed_gates":
            grouped["PASS"],

        "failed_gates":
            grouped["FAIL"],

        "warning_gates":
            grouped["WARN"],

        "not_evaluated_gates":
            grouped["NOT_EVALUATED"],

        "gates":
            gates,
    }


def summarize_sensitivity(
    payload: Mapping[str, Any],
    f7: pd.DataFrame,
    f8: pd.DataFrame,
    f12: pd.DataFrame,
) -> Dict[str, Any]:
    result = _result(payload)

    return {
        "available":
            bool(payload),

        "status":
            str(
                payload.get(
                    "status",
                    result.get(
                        "status",
                        "missing",
                    ),
                )
            ),

        "official_gate_statuses":
            payload.get(
                "official_gate_statuses",
                {},
            ),

        "f7_subject_consistency":
            payload.get(
                "f7_subject_consistency",
                {},
            ),

        "f8_complexity_penalty":
            payload.get(
                "f8_complexity_penalty",
                {},
            ),

        "f12_epsilon_saturation":
            payload.get(
                "f12_epsilon_saturation",
                {},
            ),

        "f7_rows":
            len(f7),

        "f8_rows":
            len(f8),

        "f12_rows":
            len(f12),

        "interpretation":
            str(
                result.get(
                    "interpretation",
                    "",
                )
            ),
    }


def summarize_mc_guard(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if not payload:
        return {
            "available": False,
        }

    return {
        "available":
            True,

        "enabled":
            bool(
                payload.get(
                    "enabled",
                    True,
                )
            ),

        "guard_type":
            payload.get(
                "guard_type"
            ),

        "n_valid_tests":
            _safe_int(
                payload.get(
                    "n_valid_tests"
                )
            ),

        "n_positive_lift_tests":
            _safe_int(
                payload.get(
                    "n_positive_lift_tests"
                )
            ),

        "n_primary_positive_lift_tests":
            _safe_int(
                payload.get(
                    "n_primary_positive_lift_tests"
                )
            ),

        "n_exploratory_positive_lift_tests":
            _safe_int(
                payload.get(
                    "n_exploratory_positive_lift_tests",
                    payload.get(
                        "n_exploratory_lift_tests"
                    ),
                )
            ),

        "n_strong_candidate_tests":
            _safe_int(
                payload.get(
                    "n_strong_candidate_tests"
                )
            ),

        "n_primary_strong_candidate_tests":
            _safe_int(
                payload.get(
                    "n_primary_strong_candidate_tests"
                )
            ),
    }


def final_interpretation(
    provenance: Mapping[str, Any],
    conditional: Mapping[str, Any],
    controls: Mapping[str, Any],
    gates: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> Dict[str, Any]:
    failed = list(
        gates.get(
            "failed_gates",
            [],
        )
    )

    localized = bool(
        conditional.get(
            "localized_predictive_support",
            False,
        )
    )

    controls_passed = bool(
        controls.get(
            "passed",
            False,
        )
    )

    surrogate = bool(
        provenance.get(
            "surrogate_detected",
            False,
        )
    )

    f8_only = (
        set(failed)
        == {"F8"}
    )

    if (
        localized
        and controls_passed
        and f8_only
    ):
        classification = (
            "surrogate_primary_localized_"
            "predictive_lead_bic_limited"
            if surrogate
            else (
                "primary_localized_predictive_"
                "lead_bic_limited"
            )
        )

        headline = (
            "A registered primary localized predictive "
            "lead was detected; only the current official "
            "global BIC gate remains unsatisfied."
        )

        recommendation = (
            "Preserve the Phase III.2E v2 result as "
            "BIC-limited. A future v3 may prospectively "
            "re-evaluate F8 using effective sample size, "
            "effective complexity, state-density correction "
            "or subject-level selection, before any "
            "real-data analysis."
        )

    elif not controls_passed:
        classification = (
            "controls_blocked"
        )

        headline = (
            "The result is blocked "
            "by negative controls."
        )

        recommendation = (
            "Do not interpret positive rows "
            "until official controls pass."
        )

    elif conditional.get(
        "n_primary_positive_lift_rows",
        0,
    ):
        classification = (
            "primary_lead_incomplete_support"
        )

        headline = (
            "A primary positive lead exists with "
            "incomplete official gate support."
        )

        recommendation = (
            "Document the lead and "
            "the exact failed gates."
        )

    elif conditional.get(
        "n_exploratory_positive_lift_rows",
        0,
    ):
        classification = (
            "exploratory_leads_only"
        )

        headline = (
            "Positive lift exists only "
            "in the exploratory region."
        )

        recommendation = (
            "Treat exploratory rows as "
            "hypothesis-generating."
        )

    else:
        classification = (
            "conditional_null"
        )

        headline = (
            "No positive conditional "
            "lift was detected."
        )

        recommendation = (
            "Preserve the null result."
        )

    return {
        "status":
            gates.get("status"),

        "scientific_classification":
            classification,

        "headline":
            headline,

        "surrogate":
            surrogate,

        "can_claim_empirical_cdr":
            bool(
                provenance.get(
                    "can_claim_empirical_cdr",
                    False,
                )
            ),

        "failed_gates":
            failed,

        "warning_gates":
            gates.get(
                "warning_gates",
                [],
            ),

        "not_evaluated_gates":
            gates.get(
                "not_evaluated_gates",
                [],
            ),

        "f8_failure_meaning": (
            "Lack of global BIC parsimony support; "
            "not proof that the localized predictive "
            "structure is absent."
            if (
                f8_only
                and localized
            )
            else None
        ),

        "sensitivity_completed":
            bool(
                sensitivity.get(
                    "available",
                    False,
                )
            ),

        "recommendation":
            recommendation,

        "future_v3_f8_review": {
            "allowed": True,
            "prospective_only": True,
            "must_precede_real_data_analysis": True,
            "must_preserve_current_v2_result": True,
        },
    }


def build_summary_text(
    report: Mapping[str, Any],
) -> str:
    final = report[
        "final_interpretation"
    ]

    conditional = report[
        "conditional_diagnostics"
    ]

    controls = report[
        "control_diagnostics"
    ]

    gates = report[
        "gates_diagnostics"
    ]

    lines = [
        "=" * 78,
        "CDR Phase III.2E — Diagnostics Summary",
        (
            "Primary lead / adaptive F7 / "
            "BIC-limited F8 / diagnostic F12"
        ),
        "=" * 78,

        f"status: {final.get('status')}",

        (
            "classification: "
            f"{final.get('scientific_classification')}"
        ),

        (
            "headline: "
            f"{final.get('headline')}"
        ),

        (
            "surrogate: "
            f"{final.get('surrogate')}"
        ),

        (
            "can_claim_empirical_cdr: "
            f"{final.get('can_claim_empirical_cdr')}"
        ),

        (
            "recommendation: "
            f"{final.get('recommendation')}"
        ),

        "",
        "Conditional",
        "-" * 78,

        (
            "pair_rows: "
            f"{conditional.get('n_pair_rows')}"
        ),

        (
            "valid_rows: "
            f"{conditional.get('n_valid_rows')}"
        ),

        (
            "primary_rows: "
            f"{conditional.get('n_primary_rows')}"
        ),

        (
            "primary_positive_rows: "
            f"{conditional.get('n_primary_positive_lift_rows')}"
        ),

        (
            "exploratory_positive_rows: "
            f"{conditional.get('n_exploratory_positive_lift_rows')}"
        ),

        (
            "localized_predictive_support: "
            f"{conditional.get('localized_predictive_support')}"
        ),

        (
            "selected_primary_lead: "
            f"{conditional.get('selected_primary_lead')}"
        ),

        (
            "F7: "
            f"{conditional.get('f7_adaptive_subject_consistency')}"
        ),

        (
            "F8: "
            f"{conditional.get('f8_complexity_interpretation')}"
        ),

        "",
        "Controls",
        "-" * 78,

        (
            "status: "
            f"{controls.get('status')}"
        ),

        (
            "passed: "
            f"{controls.get('passed')}"
        ),

        (
            "runs: "
            f"{controls.get('n_valid_control_runs')}/"
            f"{controls.get('n_control_runs')}"
        ),

        (
            "failed_control_types: "
            f"{controls.get('failed_control_types')}"
        ),

        "",
        "Official gates",
        "-" * 78,

        (
            "passed: "
            f"{gates.get('passed_gates')}"
        ),

        (
            "failed: "
            f"{gates.get('failed_gates')}"
        ),

        (
            "warnings: "
            f"{gates.get('warning_gates')}"
        ),

        (
            "not_evaluated: "
            f"{gates.get('not_evaluated_gates')}"
        ),

        "",

        (
            "future_v3_f8_review: "
            f"{final.get('future_v3_f8_review')}"
        ),

        "=" * 78,
    ]

    return "\n".join(lines)


def validate_inputs(
    cfg: Phase32EConfig,
) -> Dict[str, Any]:
    paths = phase3_2e_paths(cfg)

    required = (
        "pair_scores",
        "model_scores",
        "controls_json",
        "gates_json",
        "sensitivity_report",
    )

    files = {
        key: _file_status(
            paths[key]
        )
        for key in required
    }

    return {
        "ready":
            all(
                item["exists"]
                and item["size_bytes"] > 0
                for item in files.values()
            ),

        "required_files":
            files,

        "note": (
            "Diagnostics is read-only and will not "
            "rebuild conditional or controls."
        ),
    }


def run_phase3_2e_diagnostics(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    cfg = (
        cfg
        or load_phase3_2e_config()
    )

    paths = phase3_2e_paths(cfg)

    diagnostics_dir = paths[
        "diagnostics"
    ]

    diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    payloads = {
        "schema":
            load_json(
                paths["schema_report"]
            ),

        "sync":
            load_json(
                paths["sync_report"]
            ),

        "loader":
            load_json(
                paths["loader_report"]
            ),

        "windows":
            load_json(
                paths["windows_report"]
            ),

        "features":
            load_json(
                paths["features_report"]
            ),

        "alignment":
            load_json(
                paths["alignment_report"]
            ),

        "conditional":
            load_json(
                paths["conditional_report"]
            ),

        "controls":
            load_json(
                paths["controls_json"]
            ),

        "gates":
            load_json(
                paths["gates_json"]
            ),

        "sensitivity":
            load_json(
                paths["sensitivity_report"]
            ),

        "mc_guard":
            load_json(
                paths["mc_guard"]
            ),
    }

    tables = {
        "loaded":
            load_csv(
                paths["loaded"]
            ),

        "windows":
            load_csv(
                paths["windows"]
            ),

        "features":
            load_csv(
                paths["features"]
            ),

        "alignment":
            load_csv(
                paths["alignment_inventory"]
            ),

        "pairs":
            load_csv(
                paths["pair_scores"]
            ),

        "models":
            load_csv(
                paths["model_scores"]
            ),

        "control_runs":
            load_csv(
                paths["control_runs"]
            ),

        "control_pairs":
            load_csv(
                paths["control_pairs"]
            ),

        "metrics":
            load_csv(
                paths["metrics_table"]
            ),

        "f7":
            load_csv(
                paths["f7"]
            ),

        "f8":
            load_csv(
                paths["f8"]
            ),

        "f12":
            load_csv(
                paths["f12"]
            ),

        "input_sample":
            load_csv(
                paths["sync_input"],
                nrows=1000,
            ),
    }

    feature_specs = load_json(
        paths["feature_specs"]
    )

    provenance = summarize_provenance(
        tables["input_sample"],
        tables["pairs"],
        tables["models"],
        payloads["conditional"],
        payloads["gates"],
        cfg,
    )

    feature_audit = summarize_feature_audit(
        tables["windows"],
        tables["features"],
        feature_specs,
        payloads["features"],
    )

    grid_audit = summarize_grid(
        tables["pairs"],
        cfg,
    )

    conditional = summarize_conditional(
        tables["pairs"],
        tables["models"],
        payloads["conditional"],
        payloads["gates"],
        cfg,
    )

    controls = summarize_controls(
        tables["control_runs"],
        payloads["controls"],
        cfg,
    )

    gates = summarize_gates(
        payloads["gates"],
        tables["metrics"],
    )

    sensitivity = summarize_sensitivity(
        payloads["sensitivity"],
        tables["f7"],
        tables["f8"],
        tables["f12"],
    )

    mc_guard = summarize_mc_guard(
        payloads["mc_guard"]
    )

    final = final_interpretation(
        provenance,
        conditional,
        controls,
        gates,
        sensitivity,
    )

    stage_status = {
        name: {
            "available": bool(payload),
            "status": _status(payload),
        }
        for name, payload in payloads.items()
        if name != "mc_guard"
    }

    report = {
        "phase":
            cfg.phase_name,

        "project_name":
            cfg.project_name,

        "module":
            DIAGNOSTICS_MODULE,

        "diagnostics_version":
            DIAGNOSTICS_VERSION,

        "created_at":
            _now(),

        "config":
            describe_config(cfg),

        "source_files": {
            key: str(value)
            for key, value in paths.items()
            if key != "diagnostics"
        },

        "file_status": {
            key: _file_status(value)
            for key, value in paths.items()
            if key != "diagnostics"
        },

        "stage_status":
            stage_status,

        "artifact_rows": {
            "loaded":
                len(tables["loaded"]),

            "windows":
                len(tables["windows"]),

            "features":
                len(tables["features"]),

            "alignment_inventory":
                len(tables["alignment"]),

            "conditional_models":
                len(tables["models"]),

            "conditional_pairs":
                len(tables["pairs"]),

            "control_runs":
                len(tables["control_runs"]),

            "control_pairs":
                len(tables["control_pairs"]),
        },

        "provenance":
            provenance,

        "feature_audit":
            feature_audit,

        "grid_audit":
            grid_audit,

        "conditional_diagnostics":
            conditional,

        "control_diagnostics":
            controls,

        "gates_diagnostics":
            gates,

        "gate_sensitivity_diagnostics":
            sensitivity,

        "multiple_comparison_guard":
            mc_guard,

        "final_interpretation":
            final,
    }

    primary = primary_rows(
        tables["pairs"],
        cfg,
    )

    positive = positive_rows(
        tables["pairs"]
    )

    primary_positive = (
        primary_positive_lift_rows(
            tables["pairs"],
            cfg,
        )
    )

    exploratory_positive = (
        exploratory_positive_lift_rows(
            tables["pairs"],
            cfg,
        )
    )

    official_strong = (
        strong_candidate_rows(
            tables["pairs"],
            cfg,
        )
    )

    official_primary_strong = (
        primary_strong_candidate_rows(
            tables["pairs"],
            cfg,
        )
    )

    multichannel = multichannel_rows(
        tables["pairs"]
    )

    multichannel_positive = (
        multichannel[
            _num(
                multichannel,
                "conditional_lift",
            ) > 0.0
        ].copy()
        if not multichannel.empty
        else pd.DataFrame()
    )

    high_eps_zero_lift = (
        valid_pair_rows(
            tables["pairs"]
        )
    )

    if not high_eps_zero_lift.empty:
        threshold = float(
            getattr(
                cfg,
                "eps_saturation_value",
                2.0,
            )
        )

        high_eps_zero_lift = (
            high_eps_zero_lift[
                (
                    _num(
                        high_eps_zero_lift,
                        "augmented_eps_test",
                    )
                    >= threshold
                )
                & (
                    _num(
                        high_eps_zero_lift,
                        "conditional_lift",
                    )
                    <= 0.0
                )
            ].copy()
        )

    report_json = (
        diagnostics_dir
        / "phase3_2e_diagnostics_report.json"
    )

    summary_txt = (
        diagnostics_dir
        / "phase3_2e_diagnostics_summary.txt"
    )

    save_json(
        report_json,
        report,
    )

    summary_txt.write_text(
        build_summary_text(report),
        encoding="utf-8",
    )

    exports = {
        "phase3_2e_primary_tests.csv":
            primary,

        "phase3_2e_best_rows.csv":
            best_rows_by_metric(
                tables["pairs"]
            ),

        "phase3_2e_positive_lift_rows.csv":
            positive,

        "phase3_2e_primary_positive_lift_rows.csv":
            primary_positive,

        "phase3_2e_exploratory_lift_rows.csv":
            exploratory_positive,

        "phase3_2e_strong_candidate_rows.csv":
            official_strong,

        "phase3_2e_primary_strong_candidate_rows.csv":
            official_primary_strong,

        "phase3_2e_high_epsilon_zero_lift_rows.csv":
            high_eps_zero_lift,

        "phase3_2e_multichannel_rows.csv":
            multichannel,

        "phase3_2e_multichannel_positive_lift_rows.csv":
            multichannel_positive,

        "phase3_2e_control_breakdown.csv":
            pd.DataFrame(
                controls.get(
                    "control_breakdown",
                    [],
                )
            ),

        "phase3_2e_control_pair_scores_copy.csv":
            tables["control_pairs"],

        "phase3_2e_gates_copy.csv":
            tables["metrics"],

        "phase3_2e_f7_subject_sensitivity_copy.csv":
            tables["f7"],

        "phase3_2e_f8_complexity_sensitivity_copy.csv":
            tables["f8"],

        "phase3_2e_f12_epsilon_saturation_sensitivity_copy.csv":
            tables["f12"],
    }

    for filename, frame in exports.items():
        frame.to_csv(
            diagnostics_dir / filename,
            index=False,
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "Phase III.2E diagnostics completed"
    )

    print(
        "=" * 78
    )

    print(
        "classification: "
        f"{final.get('scientific_classification')}"
    )

    print(
        "headline: "
        f"{final.get('headline')}"
    )

    print(
        "controls passed: "
        f"{controls.get('passed')}"
    )

    print(
        "primary positive rows: "
        f"{conditional.get('n_primary_positive_lift_rows')}"
    )

    print(
        "localized predictive support: "
        f"{conditional.get('localized_predictive_support')}"
    )

    print(
        "failed gates: "
        f"{gates.get('failed_gates')}"
    )

    print(
        "surrogate: "
        f"{provenance.get('surrogate_detected')}"
    )

    print(
        "can claim empirical CDR: "
        f"{final.get('can_claim_empirical_cdr')}"
    )

    print(
        f"report: {report_json}"
    )

    print(
        f"summary: {summary_txt}"
    )

    print(
        "=" * 78
        + "\n"
    )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Phase III.2E outputs "
            "without rebuilding heavy stages."
        )
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    if args.validate_only:
        validation = validate_inputs(cfg)

        print(
            json.dumps(
                validation,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise SystemExit(
            0
            if validation["ready"]
            else 1
        )

    report = run_phase3_2e_diagnostics(
        cfg
    )

    final = report[
        "final_interpretation"
    ]

    print(
        "Final diagnostic headline:"
    )

    print(
        final["headline"]
    )

    print(
        "\nRecommendation:"
    )

    print(
        final["recommendation"]
    )


if __name__ == "__main__":
    main()