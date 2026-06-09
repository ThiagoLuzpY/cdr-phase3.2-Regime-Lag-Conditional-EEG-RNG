from __future__ import annotations

import argparse
import hashlib
import json
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
from src.phase3_2e_alignment import (
    lagged_frame_path,
    load_or_build_phase3_2e_alignment,
)
from src.phase3_2e_conditional import (
    evaluate_conditional_models_for_frame,
)

CONTROLS_MODULE = "phase3_2e_controls"
CONTROLS_VERSION = "phase3_2e_controls_v3_primary_grid_decoupled_guard_resummarize"


@dataclass
class ControlRunResult:
    control_type: str
    replicate: int
    window_seconds: int
    lag_windows: int
    lag_seconds: int
    lag_label: str
    lag_direction: str
    alignment_scope: str
    valid: bool
    primary_alignment: bool
    n_rows: int
    n_subjects: int
    n_sessions: int
    n_pair_rows: int
    n_primary_pair_rows: int
    n_exploratory_pair_rows: int
    max_baseline_eps_test: float
    max_augmented_eps_test: float
    max_conditional_lift: float
    max_primary_conditional_lift: float
    max_exploratory_conditional_lift: float
    max_fraction_positive_lift: float
    max_bic_improvement: float
    max_ll_improvement: float
    n_positive_lift_rows: int
    n_primary_positive_lift_rows: int
    n_exploratory_positive_lift_rows: int
    n_strong_candidate_rows: int
    n_primary_strong_candidate_rows: int
    n_exploratory_strong_candidate_rows: int
    n_multichannel_pair_rows: int
    n_multichannel_positive_lift_rows: int
    n_multichannel_strong_candidate_rows: int
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
    max_primary_conditional_lift: float
    max_exploratory_conditional_lift: float
    median_max_bic_improvement: float
    max_bic_improvement: float
    median_max_ll_improvement: float
    max_ll_improvement: float
    median_eps_saturation_fraction: float
    max_eps_saturation_fraction: float
    total_pair_rows: int
    total_primary_pair_rows: int
    total_exploratory_pair_rows: int
    total_positive_lift_rows: int
    total_primary_positive_lift_rows: int
    total_exploratory_positive_lift_rows: int
    total_multichannel_pair_rows: int
    total_multichannel_positive_lift_rows: int
    total_multichannel_strong_candidate_rows: int
    fraction_runs_below_control_tol: float
    fraction_runs_collapsed_overall: float
    control_tol: float
    passed: bool
    reason: str


@dataclass
class ControlsBuildResult:
    status: str
    protocol_only: bool
    n_control_runs: int
    n_valid_control_runs: int
    n_control_pair_rows: int
    n_control_types: int
    overall_passed: bool
    failed_control_types: List[str]
    can_proceed_to_metrics: bool
    can_proceed_to_empirical_cdr: bool
    can_claim_empirical_cdr: bool
    interpretation: str


def _stable_text_offset(text: str, modulo: int = 997) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo)


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
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _cfg_float(cfg: Phase32EConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: Phase32EConfig, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))

def _control_subject_fraction_threshold(cfg: Phase32EConfig) -> float:
    """
    Negative-control persistence threshold.

    This threshold is intentionally decoupled from subject_effect_fraction.
    The latter was reduced to 0.10 for discovery leads, while the control
    battery must preserve its original broad-persistence guard.
    """
    value = _cfg_float(cfg, "control_subject_fraction_threshold", 0.60)
    return min(max(value, 0.0), 1.0)


def _as_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "sim", "t"})


def _bool_col(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)
    return _as_bool_series(df[col]).reindex(df.index).fillna(default).astype(bool)


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _as_records(items: Sequence[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "__dict__"):
            rows.append(asdict(item))
        elif isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"value": item})
    return rows


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def _existing_columns(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


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
    sort_cols = [
        c for c in [
            "subject_id", "session_id", "window_seconds", "window_start_utc",
            "window_id", "alignment_row_id", "feature_row_id",
        ] if c in df.columns
    ]
    if sort_cols:
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def _group_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ("subject_id", "session_id", "window_seconds") if c in df.columns]
    if cols:
        return cols
    return ["subject_id"] if "subject_id" in df.columns else []


def _registered_windows(cfg: Phase32EConfig) -> Tuple[int, ...]:
    values = getattr(cfg, "all_window_seconds", tuple())
    return tuple(int(v) for v in values)


def _registered_lags(cfg: Phase32EConfig) -> Tuple[int, ...]:
    values = getattr(cfg, "all_lags_windows", tuple())
    return tuple(int(v) for v in values)


def _primary_windows(cfg: Phase32EConfig) -> Tuple[int, ...]:
    values = getattr(cfg, "primary_test_window_seconds", getattr(cfg, "primary_window_seconds", tuple()))
    return tuple(int(v) for v in values)


def _primary_lags(cfg: Phase32EConfig) -> Tuple[int, ...]:
    values = getattr(cfg, "primary_test_lags_windows", getattr(cfg, "primary_lags_windows", tuple()))
    return tuple(int(v) for v in values)


def _primary_pairs(cfg: Phase32EConfig) -> Tuple[Tuple[str, str], ...]:
    values = getattr(cfg, "primary_test_pairs", getattr(cfg, "primary_conditional_pairs", tuple()))
    return tuple((str(a), str(b)) for a, b in values)


def _alignment_scope(cfg: Phase32EConfig, window_seconds: int, lag_windows: int) -> str:
    if int(window_seconds) in set(_primary_windows(cfg)) and int(lag_windows) in set(_primary_lags(cfg)):
        return "primary"
    if int(window_seconds) in set(_registered_windows(cfg)) and int(lag_windows) in set(_registered_lags(cfg)):
        return "exploratory"
    return "unregistered"


def control_runs_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.control_runs_csv)


def control_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.control_pair_scores_csv)


def controls_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.controls_json)


def controls_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.controls_summary_txt)


def controls_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_controls_report.json"


def conditional_pair_scores_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_pair_scores_csv)


def conditional_summary_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.conditional_summary_json)


def empty_control_runs_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[field for field in ControlRunResult.__dataclass_fields__])


def empty_control_pair_scores_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "control_type", "replicate", "control_seed", "window_seconds", "lag_windows",
        "lag_seconds", "lag_label", "lag_direction", "alignment_scope", "family",
        "baseline_model", "augmented_model", "valid", "primary", "multichannel",
        "conditional_lift", "baseline_eps_test", "augmented_eps_test",
        "fraction_positive_lift", "bic_improvement", "ll_improvement",
        "aic_improvement", "subject_consistency_pass", "control_collapsed_overall",
    ])


def _config_signature_payload(cfg: Phase32EConfig) -> Dict[str, Any]:
    return {
        "controls_version": CONTROLS_VERSION,
        "control_frame_policy": "primary_ready_first_fallback_to_valid_exploratory",
        "registered_windows": list(_registered_windows(cfg)),
        "primary_windows": list(_primary_windows(cfg)),
        "registered_lags": list(_registered_lags(cfg)),
        "primary_lags": list(_primary_lags(cfg)),
        "active_conditional_pairs": [list(p) for p in active_conditional_pairs(cfg)],
        "primary_test_pairs": [list(p) for p in _primary_pairs(cfg)],
        "control_types": list(cfg.control_types),
        "required_control_types": list(cfg.required_control_types),
        "n_controls": int(cfg.n_controls),
        "control_seed": int(cfg.control_seed),
        "control_tol": float(cfg.control_tol),
        "required_control_fraction": float(cfg.required_control_fraction),
        "control_subject_fraction_threshold": _control_subject_fraction_threshold(cfg),
        "discovery_subject_effect_fraction": _cfg_float(cfg, "subject_effect_fraction", 0.10),
        "min_positive_subjects": _cfg_int(cfg, "min_positive_subjects", 1),
        "conditional_lift_min": _cfg_float(cfg, "conditional_lift_min", 0.03),
        "strong_eps_min": _cfg_float(cfg, "strong_eps_min", 0.07),
        "eps_saturation_value": _cfg_float(cfg, "eps_saturation_value", 1.0),
        "timestamp_jitter_ms": _cfg_float(cfg, "timestamp_jitter_ms", 1000.0),
        "sync_break_shift_windows": _cfg_int(cfg, "sync_break_shift_windows", 10),
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_fingerprint(cfg: Phase32EConfig) -> str:
    return _fingerprint(_config_signature_payload(cfg))


def _file_signature(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {"exists": True, "path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _conditional_signature(cfg: Phase32EConfig) -> Dict[str, Any]:
    pair_path = conditional_pair_scores_csv_path(cfg)
    summary_path = conditional_summary_json_path(cfg)
    signature: Dict[str, Any] = {
        "pair_scores_file": _file_signature(pair_path),
        "summary_json_file": _file_signature(summary_path),
    }
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            signature["conditional_version"] = data.get("conditional_version") or data.get("summary", {}).get("conditional_version")
            signature["conditional_status"] = data.get("status") or data.get("summary", {}).get("status")
            signature["conditional_config_fingerprint"] = data.get("config_fingerprint") or data.get("summary", {}).get("config_fingerprint")
            signature["primary_positive_lift_rows"] = data.get("primary_positive_lift_rows") or data.get("summary", {}).get("primary_positive_lift_rows")
        except Exception as exc:
            signature["summary_read_error"] = str(exc)
    return signature


def _cache_signature(cfg: Phase32EConfig) -> Dict[str, Any]:
    return {
        "controls_version": CONTROLS_VERSION,
        "config_fingerprint": _config_fingerprint(cfg),
        "config_signature": _config_signature_payload(cfg),
        "conditional_signature": _conditional_signature(cfg),
    }


def qrng_current_columns(df: pd.DataFrame) -> List[str]:
    return _existing_columns(df, [
        "qrng_state", "qrng_info_bin", "qrng_info_score", "qrng_bit_balance",
        "qrng_transition_rate", "qrng_entropy", "qrng_run_length_mean",
        "qrng_run_length_std", "qrng_run_length_max", "qrng_compressibility_proxy",
        "qrng_surprise_index", "qrng_transition_asymmetry", "qrng_quality_score",
        "observed_joint_state", "mc_observed_joint_state", "informational_joint_state",
        "mc_informational_joint_state",
    ])


def eeg_current_columns(df: pd.DataFrame) -> List[str]:
    return _existing_columns(df, [
        "eeg_state", "eeg_info_bin", "eeg_info_score", "eeg_delta_power",
        "eeg_theta_power", "eeg_alpha_power", "eeg_beta_power", "eeg_entropy",
        "eeg_gamma_power", "eeg_hjorth_mobility", "eeg_hjorth_complexity",
        "eeg_line_length", "eeg_signal_std", "eeg_artifact_score",
        "eeg_quality_score", "mc_eeg_state", "mc_eeg_info_bin",
        "mc_eeg_info_score", "mc_activation_score", "mc_cross_channel_score",
        "eeg_primary_delta_power", "eeg_primary_alpha_power",
        "eeg_secondary_delta_power", "eeg_secondary_alpha_power",
        "interchannel_delta_ratio", "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift", "fronto_parietal_alpha_shift",
        "cross_channel_corr", "observed_joint_state", "mc_observed_joint_state",
        "informational_joint_state", "mc_informational_joint_state",
    ])


def target_columns(df: pd.DataFrame) -> List[str]:
    return _existing_columns(df, [
        "eeg_next_state", "qrng_next_state", "mc_eeg_next_state",
        "target_eeg_state", "target_qrng_state", "target_mc_eeg_state",
        "target_eeg_info_bin", "target_qrng_info_bin", "target_mc_eeg_info_bin",
        "target_observed_joint_state", "target_mc_observed_joint_state",
        "target_informational_joint_state", "target_mc_informational_joint_state",
        "observed_joint_state_next", "mc_observed_joint_state_next",
        "informational_joint_state_next", "mc_informational_joint_state_next",
    ])


def qrng_target_columns(df: pd.DataFrame) -> List[str]:
    return _existing_columns(df, ["qrng_next_state", "target_qrng_state", "target_qrng_info_bin"])


def _is_multichannel_pair_rows(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)
    mask = pd.Series(False, index=df.index)
    for col in ("baseline_model", "augmented_model", "family"):
        if col not in df.columns:
            continue
        text = df[col].astype(str)
        mask |= text.str.contains("MCEEG", case=False, na=False)
        mask |= text.str.contains("MC_EEG", case=False, na=False)
        mask |= text.str.contains("multichannel", case=False, na=False)
        mask |= text.str.startswith(("E4", "E5", "E6"))
    return mask


def _shuffle_values(values: pd.Series, rng: np.random.Generator) -> pd.Series:
    arr = values.to_numpy(copy=True)
    rng.shuffle(arr)
    return pd.Series(arr, index=values.index)


def _shuffle_columns_within_groups(
    df: pd.DataFrame, columns: Sequence[str], group_cols: Sequence[str], rng: np.random.Generator
) -> pd.DataFrame:
    out = df.copy()
    columns = [c for c in columns if c in out.columns]
    if not columns:
        return out
    if not group_cols:
        for col in columns:
            out[col] = _shuffle_values(out[col], rng).to_numpy()
        return out
    _require_columns(out, group_cols, "Phase III.2E grouped shuffle")
    for _, idx in out.groupby(list(group_cols), sort=False, dropna=False).groups.items():
        idx_list = list(idx)
        if len(idx_list) <= 1:
            continue
        for col in columns:
            out.loc[idx_list, col] = _shuffle_values(out.loc[idx_list, col], rng).to_numpy()
    return out


def _circular_shift_columns_within_groups(
    df: pd.DataFrame,
    columns: Sequence[str],
    group_cols: Sequence[str],
    rng: np.random.Generator,
    fixed_shift: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    columns = [c for c in columns if c in out.columns]
    if not columns:
        return out
    if not group_cols:
        n = len(out)
        if n <= 2:
            return out
        shift = int(fixed_shift) if fixed_shift is not None else int(rng.integers(1, n))
        shift = shift % n or 1
        for col in columns:
            out[col] = np.roll(out[col].to_numpy(copy=True), shift)
        return out
    _require_columns(out, group_cols, "Phase III.2E circular shift")
    for _, idx in out.groupby(list(group_cols), sort=False, dropna=False).groups.items():
        idx_list = list(idx)
        n = len(idx_list)
        if n <= 2:
            continue
        shift = int(fixed_shift) % n if fixed_shift is not None else int(rng.integers(1, n))
        shift = shift or 1
        for col in columns:
            out.loc[idx_list, col] = np.roll(out.loc[idx_list, col].to_numpy(copy=True), shift)
    return out


def _subject_mismatch_columns(df: pd.DataFrame, columns: Sequence[str], rng: np.random.Generator) -> pd.DataFrame:
    out = _sort_frame(df.copy())
    columns = [c for c in columns if c in out.columns]
    if not columns or "subject_id" not in out.columns:
        return out
    subjects = sorted(out["subject_id"].astype(str).unique().tolist())
    if len(subjects) < 2:
        return out
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)
    if all(a == b for a, b in zip(subjects, shuffled_subjects)):
        shuffled_subjects = shuffled_subjects[1:] + shuffled_subjects[:1]
    mapping = dict(zip(subjects, shuffled_subjects))
    source_by_subject = {
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
            source_values = source_df[col].to_numpy(copy=True)
            if len(source_values):
                out.loc[target_idx, col] = np.resize(source_values, len(target_idx))
    return out


def synchronize_target_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target_col, alias_col in [
        ("target_eeg_state", "eeg_next_state"),
        ("target_qrng_state", "qrng_next_state"),
        ("target_mc_eeg_state", "mc_eeg_next_state"),
    ]:
        if target_col in out.columns:
            out[alias_col] = pd.to_numeric(out[target_col], errors="coerce").astype("Int64")
    return out


def rebuild_control_joint_states(df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    out = df.copy()
    qrng_bins = _cfg_int(cfg, "qrng_n_bins", 3)
    info_bins = _cfg_int(cfg, "info_n_bins", 3)
    if "eeg_state" in out.columns and "qrng_state" in out.columns:
        valid = out["eeg_state"].notna() & out["qrng_state"].notna()
        out.loc[valid, "observed_joint_state"] = (
            out.loc[valid, "eeg_state"].astype(int) * qrng_bins + out.loc[valid, "qrng_state"].astype(int)
        )
    if "mc_eeg_state" in out.columns and "qrng_state" in out.columns:
        valid = out["mc_eeg_state"].notna() & out["qrng_state"].notna()
        out.loc[valid, "mc_observed_joint_state"] = (
            out.loc[valid, "mc_eeg_state"].astype(int) * qrng_bins + out.loc[valid, "qrng_state"].astype(int)
        )
    if "eeg_info_bin" in out.columns and "qrng_info_bin" in out.columns:
        valid = out["eeg_info_bin"].notna() & out["qrng_info_bin"].notna()
        out.loc[valid, "informational_joint_state"] = (
            out.loc[valid, "eeg_info_bin"].astype(int) * info_bins + out.loc[valid, "qrng_info_bin"].astype(int)
        )
    if "mc_eeg_info_bin" in out.columns and "qrng_info_bin" in out.columns:
        valid = out["mc_eeg_info_bin"].notna() & out["qrng_info_bin"].notna()
        out.loc[valid, "mc_informational_joint_state"] = (
            out.loc[valid, "mc_eeg_info_bin"].astype(int) * info_bins + out.loc[valid, "qrng_info_bin"].astype(int)
        )
    if "target_eeg_state" in out.columns and "target_qrng_state" in out.columns:
        valid = out["target_eeg_state"].notna() & out["target_qrng_state"].notna()
        out.loc[valid, "target_observed_joint_state"] = (
            out.loc[valid, "target_eeg_state"].astype(int) * qrng_bins + out.loc[valid, "target_qrng_state"].astype(int)
        )
    if "target_mc_eeg_state" in out.columns and "target_qrng_state" in out.columns:
        valid = out["target_mc_eeg_state"].notna() & out["target_qrng_state"].notna()
        out.loc[valid, "target_mc_observed_joint_state"] = (
            out.loc[valid, "target_mc_eeg_state"].astype(int) * qrng_bins
            + out.loc[valid, "target_qrng_state"].astype(int)
        )
    out = synchronize_target_aliases(out)
    for col in [
        "eeg_state", "qrng_state", "mc_eeg_state", "eeg_next_state", "qrng_next_state",
        "mc_eeg_next_state", "target_eeg_state", "target_qrng_state", "target_mc_eeg_state",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def apply_timestamp_jitter_control(df: pd.DataFrame, cfg: Phase32EConfig, rng: np.random.Generator) -> pd.DataFrame:
    out = _circular_shift_columns_within_groups(
        df=df.copy(),
        columns=qrng_current_columns(df),
        group_cols=_group_columns(df),
        rng=rng,
        fixed_shift=1,
    )
    jitter_ms = _cfg_float(cfg, "timestamp_jitter_ms", 5000.0)
    if "eeg_qrng_gap_ms" in out.columns:
        out["eeg_qrng_gap_ms"] = _cfg_float(cfg, "max_timestamp_gap_ms", 1000.0) + abs(jitter_ms)
    if "sync_row_valid" in out.columns:
        out["sync_row_valid"] = 0
    if "sync_quality" in out.columns:
        out["sync_quality"] = np.minimum(
            pd.to_numeric(out["sync_quality"], errors="coerce").fillna(1.0),
            _cfg_float(cfg, "min_sync_quality", 0.5) - 0.01,
        )
    out["timestamp_jitter_applied_ms"] = jitter_ms
    out["timestamp_jitter_control_active"] = 1
    return out


def apply_sync_break_control(df: pd.DataFrame, cfg: Phase32EConfig, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    columns = list(dict.fromkeys(qrng_current_columns(out) + qrng_target_columns(out)))
    shift_windows = max(1, _cfg_int(cfg, "sync_break_shift_windows", 10))
    out = _circular_shift_columns_within_groups(
        df=out,
        columns=columns,
        group_cols=_group_columns(out),
        rng=rng,
        fixed_shift=shift_windows,
    )
    if "sync_row_valid" in out.columns:
        out["sync_row_valid"] = 0
    if "eeg_qrng_gap_ms" in out.columns:
        out["eeg_qrng_gap_ms"] = _cfg_float(cfg, "max_timestamp_gap_ms", 1000.0) + abs(float(shift_windows)) * 1000.0
    if "sync_quality" in out.columns:
        out["sync_quality"] = np.minimum(
            pd.to_numeric(out["sync_quality"], errors="coerce").fillna(1.0),
            _cfg_float(cfg, "min_sync_quality", 0.5) - 0.01,
        )
    out["sync_break_shift_windows"] = shift_windows
    out["sync_break_control_active"] = 1
    return out


def apply_phase_randomized_eeg_control(df: pd.DataFrame, cfg: Phase32EConfig, rng: np.random.Generator) -> pd.DataFrame:
    out = _circular_shift_columns_within_groups(
        df=df.copy(),
        columns=eeg_current_columns(df),
        group_cols=_group_columns(df),
        rng=rng,
        fixed_shift=None,
    )
    out["phase_randomized_eeg_control_active"] = 1
    return out


def apply_window_permutation_control(df: pd.DataFrame, cfg: Phase32EConfig, rng: np.random.Generator) -> pd.DataFrame:
    out = _shuffle_columns_within_groups(
        df=df.copy(),
        columns=target_columns(df),
        group_cols=_group_columns(df),
        rng=rng,
    )
    out["window_permutation_control_active"] = 1
    return out


def apply_control(df: pd.DataFrame, control_type: str, cfg: Optional[Phase32EConfig] = None, seed: int = 0) -> pd.DataFrame:
    if cfg is None:
        cfg = load_phase3_2e_config()
    out = _sort_frame(df.copy())
    rng = np.random.default_rng(int(seed))
    group_cols = _group_columns(out)
    if control_type == "within_subject_qrng_shuffle":
        out = _shuffle_columns_within_groups(out, qrng_current_columns(out), group_cols, rng)
    elif control_type == "within_subject_eeg_shuffle":
        out = _shuffle_columns_within_groups(out, eeg_current_columns(out), group_cols, rng)
    elif control_type == "circular_qrng_shift":
        out = _circular_shift_columns_within_groups(out, qrng_current_columns(out), group_cols, rng)
    elif control_type == "subject_mismatch":
        out = _subject_mismatch_columns(out, qrng_current_columns(out), rng)
    elif control_type == "timestamp_jitter_control":
        out = apply_timestamp_jitter_control(out, cfg, rng)
    elif control_type == "phase_randomized_eeg_control":
        out = apply_phase_randomized_eeg_control(out, cfg, rng)
    elif control_type == "window_permutation_control":
        out = apply_window_permutation_control(out, cfg, rng)
    elif control_type == "conditional_target_shuffle":
        out = _shuffle_columns_within_groups(out, target_columns(out), group_cols, rng)
    elif control_type == "sync_break_control":
        out = apply_sync_break_control(out, cfg, rng)
    else:
        raise ValueError(f"Unknown Phase III.2E control_type: {control_type}")
    out = rebuild_control_joint_states(out, cfg)
    out["control_type"] = str(control_type)
    out["control_seed"] = int(seed)
    return _sort_frame(out)


def _valid_pair_scores(pair_scores_df: pd.DataFrame) -> pd.DataFrame:
    if pair_scores_df.empty or "valid" not in pair_scores_df.columns:
        return pd.DataFrame()
    return pair_scores_df[_as_bool_series(pair_scores_df["valid"])].copy()


def _count_eps_saturation(valid_df: pd.DataFrame, cfg: Phase32EConfig) -> Tuple[int, float]:
    if valid_df.empty:
        return 0, 0.0
    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 2.0)
    sat_mask = np.zeros(len(valid_df), dtype=bool)
    for col in ("baseline_eps_test", "augmented_eps_test"):
        if col in valid_df.columns:
            sat_mask |= (_num_col(valid_df, col, 0.0) >= eps_saturation_value).to_numpy(dtype=bool)
    return int(sat_mask.sum()), float(np.mean(sat_mask)) if len(sat_mask) else 0.0


def _subject_consistency_mask(valid_df: pd.DataFrame, cfg: Phase32EConfig) -> pd.Series:
    if valid_df.empty:
        return pd.Series([], index=valid_df.index, dtype=bool)
    fraction = _num_col(valid_df, "fraction_positive_lift", 0.0)
    return fraction >= _control_subject_fraction_threshold(cfg)


def _strong_control_candidates(valid_df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    if valid_df.empty:
        return pd.DataFrame()
    mask = (
        (_num_col(valid_df, "conditional_lift", 0.0) >= _cfg_float(cfg, "conditional_lift_min", 0.03))
        & (_num_col(valid_df, "augmented_eps_test", 0.0) >= _cfg_float(cfg, "strong_eps_min", 0.07))
        & (_num_col(valid_df, "fraction_positive_lift", 0.0) >= _control_subject_fraction_threshold(cfg))
        & (_num_col(valid_df, "bic_improvement", 0.0) >= 0.0)
    )
    return valid_df[mask].copy()


def _control_run_from_pair_scores(
    pair_scores_df: pd.DataFrame,
    control_type: str,
    replicate: int,
    window_seconds: int,
    lag_windows: int,
    lag_direction: str,
    alignment_scope: str,
    primary_alignment: bool,
    n_rows: int,
    n_subjects: int,
    n_sessions: int,
    cfg: Phase32EConfig,
) -> ControlRunResult:
    eps_saturation_value = _cfg_float(cfg, "eps_saturation_value", 2.0)
    lag_seconds = int(window_seconds) * int(lag_windows)

    def invalid(reason: str) -> ControlRunResult:
        return ControlRunResult(
            control_type=str(control_type),
            replicate=int(replicate),
            window_seconds=int(window_seconds),
            lag_windows=int(lag_windows),
            lag_seconds=int(lag_seconds),
            lag_label=_lag_label(int(lag_windows)),
            lag_direction=str(lag_direction),
            alignment_scope=str(alignment_scope),
            valid=False,
            primary_alignment=bool(primary_alignment),
            n_rows=int(n_rows),
            n_subjects=int(n_subjects),
            n_sessions=int(n_sessions),
            n_pair_rows=0,
            n_primary_pair_rows=0,
            n_exploratory_pair_rows=0,
            max_baseline_eps_test=0.0,
            max_augmented_eps_test=0.0,
            max_conditional_lift=0.0,
            max_primary_conditional_lift=0.0,
            max_exploratory_conditional_lift=0.0,
            max_fraction_positive_lift=0.0,
            max_bic_improvement=0.0,
            max_ll_improvement=0.0,
            n_positive_lift_rows=0,
            n_primary_positive_lift_rows=0,
            n_exploratory_positive_lift_rows=0,
            n_strong_candidate_rows=0,
            n_primary_strong_candidate_rows=0,
            n_exploratory_strong_candidate_rows=0,
            n_multichannel_pair_rows=0,
            n_multichannel_positive_lift_rows=0,
            n_multichannel_strong_candidate_rows=0,
            eps_saturation_value=eps_saturation_value,
            n_eps_saturated_rows=0,
            eps_saturation_fraction=0.0,
            collapsed_by_lift=False,
            collapsed_by_bic=False,
            collapsed_by_subject_fraction=False,
            collapsed_by_strong_candidates=False,
            collapsed_overall=False,
            reason=reason,
        )

    if pair_scores_df.empty:
        return invalid("empty_pair_scores")
    valid_df = _valid_pair_scores(pair_scores_df)
    if valid_df.empty:
        return invalid("no_valid_pair_scores")

    primary_mask = _bool_col(valid_df, "primary", False)
    primary_df = valid_df[primary_mask].copy()
    exploratory_df = valid_df[~primary_mask].copy()

    baseline_eps = _num_col(valid_df, "baseline_eps_test", 0.0)
    augmented_eps = _num_col(valid_df, "augmented_eps_test", 0.0)
    lift = _num_col(valid_df, "conditional_lift", 0.0)
    frac = _num_col(valid_df, "fraction_positive_lift", 0.0)
    bic = _num_col(valid_df, "bic_improvement", 0.0)
    ll = _num_col(valid_df, "ll_improvement", 0.0)

    positive_lift = valid_df[lift > 0.0].copy()
    primary_positive = primary_df[_num_col(primary_df, "conditional_lift", 0.0) > 0.0].copy()
    exploratory_positive = exploratory_df[_num_col(exploratory_df, "conditional_lift", 0.0) > 0.0].copy()

    strong_candidates = _strong_control_candidates(valid_df, cfg)
    primary_strong = _strong_control_candidates(primary_df, cfg)
    exploratory_strong = _strong_control_candidates(exploratory_df, cfg)

    multichannel_mask = _is_multichannel_pair_rows(valid_df)
    multichannel_df = valid_df[multichannel_mask].copy()
    mc_positive = multichannel_df[_num_col(multichannel_df, "conditional_lift", 0.0) > 0.0].copy()
    mc_strong = _strong_control_candidates(multichannel_df, cfg)

    n_sat, sat_frac = _count_eps_saturation(valid_df, cfg)

    max_lift = float(lift.max()) if len(lift) else 0.0
    max_primary_lift = float(_num_col(primary_df, "conditional_lift", 0.0).max()) if not primary_df.empty else 0.0
    max_exploratory_lift = float(_num_col(exploratory_df, "conditional_lift", 0.0).max()) if not exploratory_df.empty else 0.0
    max_frac = float(frac.max()) if len(frac) else 0.0
    max_bic = float(bic.max()) if len(bic) else 0.0
    max_ll = float(ll.max()) if len(ll) else 0.0

    collapsed_by_lift = max_lift <= float(cfg.control_tol)
    collapsed_by_bic = max_bic <= 0.0
    collapsed_by_subject_fraction = max_frac < _control_subject_fraction_threshold(cfg)
    collapsed_by_strong_candidates = len(strong_candidates) == 0
    collapsed_overall = bool(
        collapsed_by_lift and collapsed_by_bic and collapsed_by_subject_fraction and collapsed_by_strong_candidates
    )

    return ControlRunResult(
        control_type=str(control_type),
        replicate=int(replicate),
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_seconds=int(lag_seconds),
        lag_label=_lag_label(int(lag_windows)),
        lag_direction=str(lag_direction),
        alignment_scope=str(alignment_scope),
        valid=True,
        primary_alignment=bool(primary_alignment),
        n_rows=int(n_rows),
        n_subjects=int(n_subjects),
        n_sessions=int(n_sessions),
        n_pair_rows=int(len(valid_df)),
        n_primary_pair_rows=int(len(primary_df)),
        n_exploratory_pair_rows=int(len(exploratory_df)),
        max_baseline_eps_test=float(baseline_eps.max()) if len(baseline_eps) else 0.0,
        max_augmented_eps_test=float(augmented_eps.max()) if len(augmented_eps) else 0.0,
        max_conditional_lift=max_lift,
        max_primary_conditional_lift=max_primary_lift,
        max_exploratory_conditional_lift=max_exploratory_lift,
        max_fraction_positive_lift=max_frac,
        max_bic_improvement=max_bic,
        max_ll_improvement=max_ll,
        n_positive_lift_rows=int(len(positive_lift)),
        n_primary_positive_lift_rows=int(len(primary_positive)),
        n_exploratory_positive_lift_rows=int(len(exploratory_positive)),
        n_strong_candidate_rows=int(len(strong_candidates)),
        n_primary_strong_candidate_rows=int(len(primary_strong)),
        n_exploratory_strong_candidate_rows=int(len(exploratory_strong)),
        n_multichannel_pair_rows=int(len(multichannel_df)),
        n_multichannel_positive_lift_rows=int(len(mc_positive)),
        n_multichannel_strong_candidate_rows=int(len(mc_strong)),
        eps_saturation_value=eps_saturation_value,
        n_eps_saturated_rows=int(n_sat),
        eps_saturation_fraction=float(sat_frac),
        collapsed_by_lift=bool(collapsed_by_lift),
        collapsed_by_bic=bool(collapsed_by_bic),
        collapsed_by_subject_fraction=bool(collapsed_by_subject_fraction),
        collapsed_by_strong_candidates=bool(collapsed_by_strong_candidates),
        collapsed_overall=collapsed_overall,
        reason="ok" if collapsed_overall else "control_effect_survived",
    )


def evaluate_control_on_frame(
    df: pd.DataFrame,
    control_type: str,
    replicate: int,
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[ControlRunResult, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    window_seconds = _safe_int(df["window_seconds"].iloc[0]) if "window_seconds" in df.columns and not df.empty else 0
    lag_windows = _safe_int(df["lag_windows"].iloc[0]) if "lag_windows" in df.columns and not df.empty else 0
    lag_direction = str(df["lag_direction"].iloc[0]) if "lag_direction" in df.columns and not df.empty else _lag_direction(lag_windows)
    alignment_scope = str(df["alignment_scope"].iloc[0]) if "alignment_scope" in df.columns and not df.empty else _alignment_scope(cfg, window_seconds, lag_windows)
    primary_alignment = alignment_scope == "primary"
    if "primary_alignment" in df.columns and not df.empty:
        primary_alignment = bool(pd.to_numeric(df["primary_alignment"], errors="coerce").fillna(0).astype(int).max() == 1)

    seed = int(
        int(cfg.control_seed)
        + int(replicate) * 1009
        + abs(int(lag_windows)) * 17
        + int(window_seconds) * 31
        + _stable_text_offset(str(control_type), 997)
    )
    controlled = apply_control(df=df, control_type=control_type, cfg=cfg, seed=seed)
    _, pair_scores = evaluate_conditional_models_for_frame(
        df=controlled,
        cfg=cfg,
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
    )
    pair_scores_df = pd.DataFrame(_as_records(pair_scores))
    n_sessions = 0
    if {"subject_id", "session_id"}.issubset(controlled.columns):
        n_sessions = int(controlled[["subject_id", "session_id"]].drop_duplicates().shape[0])

    result = _control_run_from_pair_scores(
        pair_scores_df=pair_scores_df,
        control_type=str(control_type),
        replicate=int(replicate),
        window_seconds=int(window_seconds),
        lag_windows=int(lag_windows),
        lag_direction=str(lag_direction),
        alignment_scope=str(alignment_scope),
        primary_alignment=bool(primary_alignment),
        n_rows=int(len(controlled)),
        n_subjects=int(controlled["subject_id"].nunique()) if "subject_id" in controlled.columns else 0,
        n_sessions=int(n_sessions),
        cfg=cfg,
    )

    if not pair_scores_df.empty:
        pair_scores_df["control_type"] = str(control_type)
        pair_scores_df["replicate"] = int(replicate)
        pair_scores_df["control_seed"] = int(seed)
        pair_scores_df["control_collapsed_overall"] = bool(result.collapsed_overall)
        pair_scores_df["alignment_scope"] = str(alignment_scope)
        pair_scores_df["lag_seconds"] = int(window_seconds) * int(lag_windows)
        pair_scores_df["lag_label"] = _lag_label(int(lag_windows))
        pair_scores_df["lag_direction"] = str(lag_direction)
    return result, pair_scores_df


def _frame_combos_from_conditional_outputs(cfg: Phase32EConfig) -> List[Tuple[int, int]]:
    pair_path = conditional_pair_scores_csv_path(cfg)
    if not pair_path.exists():
        return []
    try:
        pairs = pd.read_csv(pair_path)
    except Exception:
        return []
    if pairs.empty or not {"window_seconds", "lag_windows"}.issubset(pairs.columns):
        return []
    if "valid" in pairs.columns:
        pairs = pairs[_as_bool_series(pairs["valid"])].copy()
    combos = (
        pairs[["window_seconds", "lag_windows"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["window_seconds", "lag_windows"], kind="stable")
    )
    return [(_safe_int(row["window_seconds"]), _safe_int(row["lag_windows"])) for _, row in combos.iterrows()]


def available_control_frames(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[List[Tuple[int, int, Path, Dict[str, Any]]], Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    inventory_df, alignment_report = load_or_build_phase3_2e_alignment(
        cfg=cfg,
        force_rebuild=bool(force_rebuild_alignment),
    )
    alignment_result = alignment_report.get("result", {})
    can_continue = bool(alignment_result.get("can_proceed_to_conditional_cdr", False))
    frames: List[Tuple[int, int, Path, Dict[str, Any]]] = []
    if inventory_df.empty or not can_continue:
        return frames, alignment_report

    work = inventory_df.copy()
    for col in ("n_valid_conditional_rows", "n_valid_mc_conditional_rows", "n_valid_primary_alignment_rows"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    valid_mask = pd.Series(False, index=work.index)
    for col in ("n_valid_conditional_rows", "n_valid_mc_conditional_rows", "n_valid_primary_alignment_rows"):
        if col in work.columns:
            valid_mask |= work[col].astype(int) > 0
    work = work[valid_mask].copy()
    if work.empty or not {"window_seconds", "lag_windows"}.issubset(work.columns):
        return frames, alignment_report

    work["window_seconds"] = pd.to_numeric(work["window_seconds"], errors="coerce")
    work["lag_windows"] = pd.to_numeric(work["lag_windows"], errors="coerce")
    work = work.dropna(subset=["window_seconds", "lag_windows"]).copy()
    work["window_seconds"] = work["window_seconds"].astype(int)
    work["lag_windows"] = work["lag_windows"].astype(int)
    work["computed_alignment_scope"] = [
        _alignment_scope(cfg, int(w), int(l))
        for w, l in zip(work["window_seconds"], work["lag_windows"])
    ]

    primary_ready = work[work["computed_alignment_scope"].eq("primary")].copy()
    selected = primary_ready if not primary_ready.empty else work
    selected = selected.drop_duplicates(subset=["window_seconds", "lag_windows"]).sort_values(
        ["window_seconds", "lag_windows"],
        kind="stable",
    )

    for _, row in selected.iterrows():
        window_seconds = int(row["window_seconds"])
        lag_windows = int(row["lag_windows"])
        raw_frame_file = str(row.get("frame_file", "")).strip()
        frame_file = Path(raw_frame_file) if raw_frame_file else lagged_frame_path(
            cfg,
            window_seconds,
            lag_windows,
        )
        if not frame_file.exists():
            frame_file = lagged_frame_path(cfg, window_seconds, lag_windows)
        if not frame_file.exists():
            continue
        scope = _alignment_scope(cfg, window_seconds, lag_windows)
        meta = row.to_dict()
        meta.update({
            "window_seconds": window_seconds,
            "lag_windows": lag_windows,
            "lag_seconds": window_seconds * lag_windows,
            "lag_label": _lag_label(lag_windows),
            "lag_direction": _lag_direction(lag_windows),
            "alignment_scope": scope,
            "primary_alignment": scope == "primary",
            "frame_file": str(frame_file),
            "control_frame_policy": (
                "primary_ready_only"
                if not primary_ready.empty
                else "fallback_to_valid_exploratory"
            ),
        })
        frames.append((window_seconds, lag_windows, frame_file, meta))

    return frames, alignment_report


def evaluate_all_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    frames, alignment_report = available_control_frames(cfg=cfg, force_rebuild_alignment=force_rebuild_alignment)
    alignment_status = str(alignment_report.get("status", ""))
    alignment_result = alignment_report.get("result", {})

    if not frames:
        context = {
            "alignment_status": alignment_status,
            "alignment_result": alignment_result,
            "protocol_only": bool(alignment_status == cfg.status_pending_synchronized_data),
            "reason": "no_alignment_frames_available_for_controls",
            "active_conditional_pairs": list(active_conditional_pairs(cfg)),
            "cache_signature": _cache_signature(cfg),
        }
        return empty_control_runs_frame(), empty_control_pair_scores_frame(), context

    run_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    print(
        "[Phase3.2E Controls] "
        f"selected_frames={len(frames)} "
        f"control_types={len(cfg.control_types)} "
        f"replicates={int(cfg.n_controls)}"
    )

    for window_seconds, lag_windows, path, frame_meta in frames:
        df = _sort_frame(pd.read_csv(path))
        if "alignment_scope" not in df.columns:
            df["alignment_scope"] = _alignment_scope(cfg, window_seconds, lag_windows)
        if "lag_direction" not in df.columns:
            df["lag_direction"] = _lag_direction(lag_windows)
        for control_type in cfg.control_types:
            for replicate in range(int(cfg.n_controls)):
                print(
                    "[Phase3.2E Controls] "
                    f"control={control_type} replicate={replicate} "
                    f"window={window_seconds}s lag={lag_windows}"
                )
                try:
                    run_result, pair_scores_df = evaluate_control_on_frame(
                        df=df,
                        control_type=str(control_type),
                        replicate=int(replicate),
                        cfg=cfg,
                    )
                    run_rows.append(asdict(run_result))
                    if not pair_scores_df.empty:
                        pair_rows.extend(pair_scores_df.to_dict(orient="records"))
                except Exception as exc:
                    n_sessions = 0
                    if {"subject_id", "session_id"}.issubset(df.columns):
                        n_sessions = int(df[["subject_id", "session_id"]].drop_duplicates().shape[0])
                    lag_seconds = int(window_seconds) * int(lag_windows)
                    run_rows.append(asdict(ControlRunResult(
                        control_type=str(control_type),
                        replicate=int(replicate),
                        window_seconds=int(window_seconds),
                        lag_windows=int(lag_windows),
                        lag_seconds=int(lag_seconds),
                        lag_label=_lag_label(int(lag_windows)),
                        lag_direction=str(frame_meta.get("lag_direction", _lag_direction(lag_windows))),
                        alignment_scope=str(frame_meta.get("alignment_scope", _alignment_scope(cfg, window_seconds, lag_windows))),
                        valid=False,
                        primary_alignment=bool(frame_meta.get("primary_alignment", False)),
                        n_rows=int(len(df)),
                        n_subjects=int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
                        n_sessions=int(n_sessions),
                        n_pair_rows=0,
                        n_primary_pair_rows=0,
                        n_exploratory_pair_rows=0,
                        max_baseline_eps_test=0.0,
                        max_augmented_eps_test=0.0,
                        max_conditional_lift=0.0,
                        max_primary_conditional_lift=0.0,
                        max_exploratory_conditional_lift=0.0,
                        max_fraction_positive_lift=0.0,
                        max_bic_improvement=0.0,
                        max_ll_improvement=0.0,
                        n_positive_lift_rows=0,
                        n_primary_positive_lift_rows=0,
                        n_exploratory_positive_lift_rows=0,
                        n_strong_candidate_rows=0,
                        n_primary_strong_candidate_rows=0,
                        n_exploratory_strong_candidate_rows=0,
                        n_multichannel_pair_rows=0,
                        n_multichannel_positive_lift_rows=0,
                        n_multichannel_strong_candidate_rows=0,
                        eps_saturation_value=_cfg_float(cfg, "eps_saturation_value", 2.0),
                        n_eps_saturated_rows=0,
                        eps_saturation_fraction=0.0,
                        collapsed_by_lift=False,
                        collapsed_by_bic=False,
                        collapsed_by_subject_fraction=False,
                        collapsed_by_strong_candidates=False,
                        collapsed_overall=False,
                        reason=str(exc),
                    )))

    runs_df = pd.DataFrame(run_rows) if run_rows else empty_control_runs_frame()
    pairs_df = pd.DataFrame(pair_rows) if pair_rows else empty_control_pair_scores_frame()

    if not runs_df.empty:
        runs_df = runs_df.sort_values(["control_type", "window_seconds", "lag_windows", "replicate"], kind="stable").reset_index(drop=True)
    if not pairs_df.empty:
        sort_cols = [
            c for c in ["control_type", "replicate", "window_seconds", "lag_windows", "family", "augmented_model"]
            if c in pairs_df.columns
        ]
        if sort_cols:
            pairs_df = pairs_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    context = {
        "alignment_status": alignment_status,
        "alignment_result": alignment_result,
        "protocol_only": False,
        "reason": "controls_completed",
        "n_selected_frames": int(len(frames)),
        "selected_frames": [meta for _, _, _, meta in frames],
        "control_frame_policy": (
            frames[0][3].get("control_frame_policy") if frames else None
        ),
        "control_subject_fraction_threshold": _control_subject_fraction_threshold(cfg),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "cache_signature": _cache_signature(cfg),
    }
    return runs_df, pairs_df, context


def reclassify_control_outputs(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    if runs_df.empty:
        return runs_df.copy(), pairs_df.copy()

    runs = runs_df.copy()
    pairs = pairs_df.copy()
    key_cols = ["control_type", "replicate", "window_seconds", "lag_windows"]
    pair_groups: Dict[Tuple[str, int, int, int], pd.DataFrame] = {}

    if not pairs.empty and all(col in pairs.columns for col in key_cols):
        for key, group in pairs.groupby(key_cols, sort=False, dropna=False):
            pair_groups[(str(key[0]), int(key[1]), int(key[2]), int(key[3]))] = group.copy()

    corrected_rows: List[Dict[str, Any]] = []
    for _, row in runs.iterrows():
        key = (
            str(row.get("control_type")),
            _safe_int(row.get("replicate")),
            _safe_int(row.get("window_seconds")),
            _safe_int(row.get("lag_windows")),
        )
        pair_group = pair_groups.get(key, pd.DataFrame())
        if pair_group.empty:
            corrected = row.to_dict()
            if bool(_as_bool_series(pd.Series([row.get("valid", False)])).iloc[0]):
                max_lift = _safe_float(row.get("max_conditional_lift"))
                max_bic = _safe_float(row.get("max_bic_improvement"))
                max_fraction = _safe_float(row.get("max_fraction_positive_lift"))
                n_strong = _safe_int(row.get("n_strong_candidate_rows"))
                corrected["collapsed_by_lift"] = max_lift <= float(cfg.control_tol)
                corrected["collapsed_by_bic"] = max_bic <= 0.0
                corrected["collapsed_by_subject_fraction"] = (
                    max_fraction < _control_subject_fraction_threshold(cfg)
                )
                corrected["collapsed_by_strong_candidates"] = n_strong == 0
                corrected["collapsed_overall"] = bool(
                    corrected["collapsed_by_lift"]
                    and corrected["collapsed_by_bic"]
                    and corrected["collapsed_by_subject_fraction"]
                    and corrected["collapsed_by_strong_candidates"]
                )
                corrected["reason"] = (
                    "ok"
                    if corrected["collapsed_overall"]
                    else "control_effect_survived"
                )
            corrected_rows.append(corrected)
            continue

        rebuilt = _control_run_from_pair_scores(
            pair_scores_df=pair_group,
            control_type=key[0],
            replicate=key[1],
            window_seconds=key[2],
            lag_windows=key[3],
            lag_direction=str(row.get("lag_direction", _lag_direction(key[3]))),
            alignment_scope=str(row.get("alignment_scope", _alignment_scope(cfg, key[2], key[3]))),
            primary_alignment=bool(
                _as_bool_series(pd.Series([row.get("primary_alignment", False)])).iloc[0]
            ),
            n_rows=_safe_int(row.get("n_rows")),
            n_subjects=_safe_int(row.get("n_subjects")),
            n_sessions=_safe_int(row.get("n_sessions")),
            cfg=cfg,
        )
        corrected_rows.append(asdict(rebuilt))

    corrected_runs = pd.DataFrame(corrected_rows)
    if not corrected_runs.empty:
        corrected_runs = corrected_runs.sort_values(
            ["control_type", "window_seconds", "lag_windows", "replicate"],
            kind="stable",
        ).reset_index(drop=True)

    if not pairs.empty:
        strong_mask = (
            (_num_col(pairs, "conditional_lift", 0.0) >= _cfg_float(cfg, "conditional_lift_min", 0.03))
            & (_num_col(pairs, "augmented_eps_test", 0.0) >= _cfg_float(cfg, "strong_eps_min", 0.07))
            & (_num_col(pairs, "fraction_positive_lift", 0.0) >= _control_subject_fraction_threshold(cfg))
            & (_num_col(pairs, "bic_improvement", 0.0) >= 0.0)
        )
        pairs["control_strong_candidate"] = strong_mask.astype(bool)

        collapse_lookup = {
            (
                str(row["control_type"]),
                int(row["replicate"]),
                int(row["window_seconds"]),
                int(row["lag_windows"]),
            ): bool(row["collapsed_overall"])
            for _, row in corrected_runs.iterrows()
        }
        pairs["control_collapsed_overall"] = [
            collapse_lookup.get(
                (
                    str(row.get("control_type")),
                    _safe_int(row.get("replicate")),
                    _safe_int(row.get("window_seconds")),
                    _safe_int(row.get("lag_windows")),
                ),
                False,
            )
            for _, row in pairs.iterrows()
        ]

    return corrected_runs, pairs

def summarize_control_runs(runs_df: pd.DataFrame, cfg: Optional[Phase32EConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    if runs_df.empty:
        return {
            "available": False,
            "reason": "empty_control_runs",
            "passed": False,
            "n_control_types": 0,
            "control_summaries": [],
            "failed_control_types": list(cfg.required_control_types),
            "control_subject_fraction_threshold": _control_subject_fraction_threshold(cfg),
            "interpretation": "No control runs are available.",
        }

    summaries: List[ControlSummary] = []
    threshold = _control_subject_fraction_threshold(cfg)

    for control_type, group in runs_df.groupby("control_type", sort=True):
        valid_group = group[_as_bool_series(group["valid"])].copy()
        if valid_group.empty:
            summaries.append(ControlSummary(
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
                max_primary_conditional_lift=0.0,
                max_exploratory_conditional_lift=0.0,
                median_max_bic_improvement=0.0,
                max_bic_improvement=0.0,
                median_max_ll_improvement=0.0,
                max_ll_improvement=0.0,
                median_eps_saturation_fraction=0.0,
                max_eps_saturation_fraction=0.0,
                total_pair_rows=0,
                total_primary_pair_rows=0,
                total_exploratory_pair_rows=0,
                total_positive_lift_rows=0,
                total_primary_positive_lift_rows=0,
                total_exploratory_positive_lift_rows=0,
                total_multichannel_pair_rows=0,
                total_multichannel_positive_lift_rows=0,
                total_multichannel_strong_candidate_rows=0,
                fraction_runs_below_control_tol=0.0,
                fraction_runs_collapsed_overall=0.0,
                control_tol=float(cfg.control_tol),
                passed=False,
                reason="no_valid_runs",
            ))
            continue

        baseline_eps = _num_col(valid_group, "max_baseline_eps_test", 0.0)
        augmented_eps = _num_col(valid_group, "max_augmented_eps_test", 0.0)
        max_lift = _num_col(valid_group, "max_conditional_lift", 0.0)
        max_primary_lift = _num_col(valid_group, "max_primary_conditional_lift", 0.0)
        max_exploratory_lift = _num_col(valid_group, "max_exploratory_conditional_lift", 0.0)
        max_bic = _num_col(valid_group, "max_bic_improvement", 0.0)
        max_ll = _num_col(valid_group, "max_ll_improvement", 0.0)
        max_frac = _num_col(valid_group, "max_fraction_positive_lift", 0.0)
        baseline_eps = _num_col(valid_group, "max_baseline_eps_test", 0.0)
        augmented_eps = _num_col(valid_group, "max_augmented_eps_test", 0.0)
        sat_frac = _num_col(valid_group, "eps_saturation_fraction", 0.0)
        n_strong = _num_col(valid_group, "n_strong_candidate_rows", 0).astype(int)

        recomputed_collapsed = (
            (max_lift <= float(cfg.control_tol))
            & (max_bic <= 0.0)
            & (max_frac < threshold)
            & (n_strong == 0)
        )
        fraction_below = float(recomputed_collapsed.mean()) if len(recomputed_collapsed) else 0.0
        fraction_collapsed = fraction_below
        passed = fraction_collapsed >= float(cfg.required_control_fraction)
        reason = "ok" if passed else f"fraction_collapsed_overall<{cfg.required_control_fraction}"

        summaries.append(ControlSummary(
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
            max_primary_conditional_lift=float(max_primary_lift.max()),
            max_exploratory_conditional_lift=float(max_exploratory_lift.max()),
            median_max_bic_improvement=float(max_bic.median()),
            max_bic_improvement=float(max_bic.max()),
            median_max_ll_improvement=float(max_ll.median()),
            max_ll_improvement=float(max_ll.max()),
            median_eps_saturation_fraction=float(sat_frac.median()),
            max_eps_saturation_fraction=float(sat_frac.max()),
            total_pair_rows=int(_num_col(valid_group, "n_pair_rows", 0).sum()),
            total_primary_pair_rows=int(_num_col(valid_group, "n_primary_pair_rows", 0).sum()),
            total_exploratory_pair_rows=int(_num_col(valid_group, "n_exploratory_pair_rows", 0).sum()),
            total_positive_lift_rows=int(_num_col(valid_group, "n_positive_lift_rows", 0).sum()),
            total_primary_positive_lift_rows=int(_num_col(valid_group, "n_primary_positive_lift_rows", 0).sum()),
            total_exploratory_positive_lift_rows=int(_num_col(valid_group, "n_exploratory_positive_lift_rows", 0).sum()),
            total_multichannel_pair_rows=int(_num_col(valid_group, "n_multichannel_pair_rows", 0).sum()),
            total_multichannel_positive_lift_rows=int(_num_col(valid_group, "n_multichannel_positive_lift_rows", 0).sum()),
            total_multichannel_strong_candidate_rows=int(_num_col(valid_group, "n_multichannel_strong_candidate_rows", 0).sum()),
            fraction_runs_below_control_tol=fraction_below,
            fraction_runs_collapsed_overall=fraction_collapsed,
            control_tol=float(cfg.control_tol),
            passed=bool(passed),
            reason=reason,
        ))

    summary_by_type = {item.control_type: item for item in summaries}
    failed_required_control_types = [
        str(control_type)
        for control_type in cfg.required_control_types
        if str(control_type) not in summary_by_type
        or not summary_by_type[str(control_type)].passed
    ]
    failed_all_control_types = [
        item.control_type
        for item in summaries
        if not item.passed
    ]
    all_required_present = all(
        str(control_type) in summary_by_type
        for control_type in cfg.required_control_types
    )
    all_required_passed = all_required_present and not failed_required_control_types

    return {
        "available": True,
        "passed": bool(all_required_passed),
        "n_control_types": int(len(summaries)),
        "n_required_control_types": int(len(cfg.required_control_types)),
        "required_control_types": list(cfg.required_control_types),
        "all_required_present": bool(all_required_present),
        "control_subject_fraction_threshold": float(threshold),
        "discovery_subject_effect_fraction": _cfg_float(cfg, "subject_effect_fraction", 0.10),
        "required_control_fraction": float(cfg.required_control_fraction),
        "control_frame_policy": "primary_ready_first_fallback_to_valid_exploratory",
        "control_summaries": [asdict(item) for item in summaries],
        "failed_control_types": sorted(set(failed_required_control_types)),
        "diagnostic_failed_optional_control_types": sorted(
            set(failed_all_control_types) - set(failed_required_control_types)
        ),
        "interpretation": (
            "The discovery subject threshold is not reused as the negative-control persistence threshold. "
            "Required controls pass when at least the configured fraction of runs jointly collapse lift, "
            "BIC support, broad subject persistence and strong-candidate survival. Epsilon saturation "
            "remains diagnostic."
        ),
    }


def build_controls_result(
    status: str,
    protocol_only: bool,
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    summary: Mapping[str, Any],
    interpretation: str,
) -> ControlsBuildResult:
    valid_runs = int(_as_bool_series(runs_df["valid"]).sum()) if not runs_df.empty and "valid" in runs_df.columns else 0
    return ControlsBuildResult(
        status=str(status),
        protocol_only=bool(protocol_only),
        n_control_runs=int(len(runs_df)),
        n_valid_control_runs=int(valid_runs),
        n_control_pair_rows=int(len(pairs_df)),
        n_control_types=int(summary.get("n_control_types", 0)),
        overall_passed=bool(summary.get("passed", False)),
        failed_control_types=list(summary.get("failed_control_types", [])),
        can_proceed_to_metrics=True,
        can_proceed_to_empirical_cdr=bool(summary.get("passed", False)),
        can_claim_empirical_cdr=False,
        interpretation=str(interpretation),
    )


def build_controls_summary_text(summary: Mapping[str, Any], result: ControlsBuildResult) -> str:
    lines = [
        "=" * 78,
        "Phase III.2E — Negative Controls Summary",
        "Synchronized EEG–QRNG protocol version",
        "=" * 78,
        "",
        f"status: {result.status}",
        f"protocol_only: {result.protocol_only}",
        f"overall_passed: {result.overall_passed}",
        f"n_control_runs: {result.n_control_runs}",
        f"n_valid_control_runs: {result.n_valid_control_runs}",
        f"n_control_pair_rows: {result.n_control_pair_rows}",
        f"n_control_types: {result.n_control_types}",
        f"failed_control_types: {result.failed_control_types}",
        f"can_proceed_to_metrics: {result.can_proceed_to_metrics}",
        f"can_proceed_to_empirical_cdr: {result.can_proceed_to_empirical_cdr}",
        f"can_claim_empirical_cdr: {result.can_claim_empirical_cdr}",
        "",
        "interpretation:",
        result.interpretation,
        "",
    ]
    if not summary.get("available"):
        lines.extend([f"Unavailable: {summary.get('reason')}", "=" * 78])
        return "\n".join(lines)

    lines.extend([
        f"required_control_types: {summary.get('required_control_types')}",
        f"all_required_present: {summary.get('all_required_present')}",
        f"control_subject_fraction_threshold: {summary.get('control_subject_fraction_threshold')}",
        f"discovery_subject_effect_fraction: {summary.get('discovery_subject_effect_fraction')}",
        f"required_control_fraction: {summary.get('required_control_fraction')}",
        "",
    ])
    for item in summary.get("control_summaries", []):
        lines.extend([
            f"Control: {item.get('control_type')}",
            "-" * 78,
            f"Runs: {item.get('n_valid_runs')}/{item.get('n_runs')}",
            f"Max lift: {item.get('max_conditional_lift')}",
            f"Max primary lift: {item.get('max_primary_conditional_lift')}",
            f"Max exploratory lift: {item.get('max_exploratory_conditional_lift')}",
            f"Max BIC improvement: {item.get('max_bic_improvement')}",
            f"Max LL improvement: {item.get('max_ll_improvement')}",
            f"Total pair rows: {item.get('total_pair_rows')}",
            f"Total primary positive lift rows: {item.get('total_primary_positive_lift_rows')}",
            f"Total exploratory positive lift rows: {item.get('total_exploratory_positive_lift_rows')}",
            f"Total multichannel positive lift rows: {item.get('total_multichannel_positive_lift_rows')}",
            f"Fraction collapsed overall: {item.get('fraction_runs_collapsed_overall')}",
            f"Passed: {item.get('passed')}",
            f"Reason: {item.get('reason')}",
            "",
        ])
    lines.append("=" * 78)
    return "\n".join(lines)


def save_control_outputs(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    context: Mapping[str, Any],
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(control_runs_csv_path(cfg), index=False)
    pairs_df.to_csv(control_pair_scores_csv_path(cfg), index=False)
    summary = summarize_control_runs(runs_df, cfg)

    if context.get("protocol_only"):
        status = cfg.status_pending_synchronized_data
        protocol_only = True
        interpretation = "No synchronized EEG–QRNG dataset is available. Controls completed in protocol-only mode."
    elif not summary.get("available"):
        status = "controls_unavailable"
        protocol_only = False
        interpretation = "Controls could not be evaluated because no valid synchronized control runs were available."
    elif summary.get("passed"):
        status = "controls_passed"
        protocol_only = False
        interpretation = (
            "Required synchronized negative controls passed. This permits metrics and gate evaluation, "
            "but does not by itself establish a positive empirical claim, especially for surrogate data."
        )
    else:
        status = "controls_failed"
        protocol_only = False
        interpretation = (
            "One or more required synchronized negative controls failed to collapse. "
            "Any positive conditional CDR interpretation is blocked until controls are revised or real data are retested."
        )

    result = build_controls_result(status, protocol_only, runs_df, pairs_df, summary, interpretation)
    cache_signature = _cache_signature(cfg)
    payload = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": CONTROLS_MODULE,
        "controls_version": CONTROLS_VERSION,
        "created_at": _now_str(),
        "status": result.status,
        "cache_signature": cache_signature,
        "config_fingerprint": cache_signature["config_fingerprint"],
        "result": asdict(result),
        "summary": summary,
        "context": dict(context),
        "runs_file": str(control_runs_csv_path(cfg)),
        "pair_scores_file": str(control_pair_scores_csv_path(cfg)),
        "controls_json": str(controls_json_path(cfg)),
        "controls_summary_txt": str(controls_summary_txt_path(cfg)),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),
        "config": describe_config(cfg),
    }
    save_json(controls_json_path(cfg), payload)
    save_json(controls_report_json_path(cfg), payload)
    controls_summary_txt_path(cfg).write_text(build_controls_summary_text(summary, result), encoding="utf-8")

    print(f"[Phase3.2E Controls] Saved runs: {control_runs_csv_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved pair scores: {control_pair_scores_csv_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved controls JSON: {controls_json_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved controls TXT: {controls_summary_txt_path(cfg)}")
    print(f"[Phase3.2E Controls] Saved report JSON: {controls_report_json_path(cfg)}")
    return payload


def build_phase3_2e_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()
    runs_df, pairs_df, context = evaluate_all_controls(
        cfg=cfg,
        force_rebuild_alignment=force_rebuild_alignment,
    )
    runs_df, pairs_df = reclassify_control_outputs(runs_df, pairs_df, cfg)
    context = dict(context)
    context["control_frame_policy"] = "primary_ready_first_fallback_to_valid_exploratory"
    context["control_subject_fraction_threshold"] = _control_subject_fraction_threshold(cfg)
    report = save_control_outputs(
        runs_df=runs_df,
        pairs_df=pairs_df,
        context=context,
        cfg=cfg,
    )
    return runs_df, pairs_df, report


def _cached_controls_compatible(report: Mapping[str, Any], cfg: Phase32EConfig) -> bool:
    if report.get("controls_version") != CONTROLS_VERSION:
        return False
    current_sig = _cache_signature(cfg)
    cached_sig = report.get("cache_signature", {})
    return bool(cached_sig == current_sig)


def resummarize_existing_controls(
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    runs_path = control_runs_csv_path(cfg)
    pairs_path = control_pair_scores_csv_path(cfg)
    report_path = controls_report_json_path(cfg)

    if not runs_path.exists():
        raise FileNotFoundError(f"Control runs CSV not found: {runs_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Control pair scores CSV not found: {pairs_path}")

    runs_df = pd.read_csv(runs_path)
    pairs_df = pd.read_csv(pairs_path)
    previous_report: Dict[str, Any] = {}

    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as file:
                previous_report = json.load(file)
        except Exception:
            previous_report = {}

    corrected_runs, corrected_pairs = reclassify_control_outputs(
        runs_df,
        pairs_df,
        cfg,
    )

    previous_context = previous_report.get("context", {})
    context = dict(previous_context) if isinstance(previous_context, Mapping) else {}
    context.update({
        "protocol_only": False,
        "reason": "controls_resummarized_without_recomputation",
        "resummarized_only": True,
        "source_controls_version": previous_report.get("controls_version"),
        "source_status": previous_report.get("status"),
        "n_selected_frames": int(
            corrected_runs[["window_seconds", "lag_windows"]].drop_duplicates().shape[0]
        ) if not corrected_runs.empty else 0,
        "control_frame_policy_for_future_rebuilds": (
            "primary_ready_first_fallback_to_valid_exploratory"
        ),
        "control_subject_fraction_threshold": _control_subject_fraction_threshold(cfg),
        "existing_run_superset_retained": True,
    })

    report = save_control_outputs(
        runs_df=corrected_runs,
        pairs_df=corrected_pairs,
        context=context,
        cfg=cfg,
    )
    return corrected_runs, corrected_pairs, report

def load_or_build_phase3_2e_controls(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    force_rebuild_alignment: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    runs_path = control_runs_csv_path(cfg)
    pairs_path = control_pair_scores_csv_path(cfg)
    report_path = controls_report_json_path(cfg)

    if runs_path.exists() and pairs_path.exists() and report_path.exists() and not force_rebuild:
        print(f"[Phase3.2E Controls] Loading cached controls: {runs_path}")
        try:
            runs_df = pd.read_csv(runs_path)
            pairs_df = pd.read_csv(pairs_path)
            with open(report_path, "r", encoding="utf-8") as file:
                report = json.load(file)
            if _cached_controls_compatible(report, cfg):
                print("[Phase3.2E Controls] Cached controls compatible.")
                return runs_df, pairs_df, report
            print("[Phase3.2E Controls] Cached controls incompatible with current config/conditional outputs. Rebuilding...")
        except Exception as exc:
            print(f"[Phase3.2E Controls] Failed to load cached controls: {exc}. Rebuilding...")

    return build_phase3_2e_controls(cfg=cfg, force_rebuild_alignment=force_rebuild_alignment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase III.2E synchronized negative controls."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the complete control battery.",
    )
    parser.add_argument(
        "--force-rebuild-alignment",
        action="store_true",
        help="Force rebuilding alignment before controls.",
    )
    parser.add_argument(
        "--resummarize-only",
        action="store_true",
        help=(
            "Reclassify and resummarize existing control CSV outputs without "
            "rerunning the expensive negative-control battery."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_phase3_2e_config()

    if args.resummarize_only:
        runs_df, pairs_df, report = resummarize_existing_controls(cfg=cfg)
    else:
        runs_df, pairs_df, report = load_or_build_phase3_2e_controls(
            cfg=cfg,
            force_rebuild=bool(args.force_rebuild),
            force_rebuild_alignment=bool(args.force_rebuild_alignment),
        )

    result = report.get("result", {})
    summary = report.get("summary", {})
    print("\n" + "=" * 78)
    print("Phase III.2E controls completed")
    print("Synchronized EEG–QRNG negative controls")
    print("=" * 78)
    print(f"status: {result.get('status')}")
    print(f"protocol_only: {result.get('protocol_only')}")
    print(f"overall_passed: {result.get('overall_passed')}")
    print(f"n_control_runs: {result.get('n_control_runs')}")
    print(f"n_valid_control_runs: {result.get('n_valid_control_runs')}")
    print(f"n_control_pair_rows: {result.get('n_control_pair_rows')}")
    print(f"n_control_types: {result.get('n_control_types')}")
    print(f"failed_control_types: {result.get('failed_control_types')}")
    print(f"required_control_types: {summary.get('required_control_types')}")
    print(
        "control_subject_fraction_threshold: "
        f"{summary.get('control_subject_fraction_threshold')}"
    )
    print(
        "discovery_subject_effect_fraction: "
        f"{summary.get('discovery_subject_effect_fraction')}"
    )
    print(f"can_proceed_to_metrics: {result.get('can_proceed_to_metrics')}")
    print(
        "can_proceed_to_empirical_cdr: "
        f"{result.get('can_proceed_to_empirical_cdr')}"
    )
    print(f"can_claim_empirical_cdr: {result.get('can_claim_empirical_cdr')}")
    print(f"runs_csv: {control_runs_csv_path(cfg)}")
    print(f"pair_scores_csv: {control_pair_scores_csv_path(cfg)}")
    print(f"controls_json: {controls_json_path(cfg)}")
    print(f"interpretation: {result.get('interpretation')}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
