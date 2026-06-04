from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2e_config import (
    Phase32EConfig,
    describe_config,
    load_phase3_2e_config,
)

from src.phase3_2e_windows import (
    load_or_build_phase3_2e_windows,
)


# =========================================================
# Phase III.2E — Feature Construction
# =========================================================
#
# Purpose:
#
#   Build the synchronized feature/state layer used by:
#
#       III.2E alignment
#       III.2E conditional CDR
#       III.2E controls
#       III.2E metrics/gates
#       III.2E F7/F8/F12 sensitivity diagnostics
#
# Important:
#
#   Phase III.2E does not reuse Sleep-EDF + ANU QRNG as if they were
#   synchronized. It only builds empirical features when the synchronized
#   CSV exists and passes schema/sync/window validation.
#
# Behavior:
#
#   If no synchronized data exists:
#       status = pending_synchronized_data
#       protocol_only = True
#       empty canonical feature outputs are saved
#
#   If windows are blocked:
#       status = features_blocked_by_windows
#       empty canonical feature outputs are saved
#
#   If valid synchronized windows exist:
#       builds:
#           eeg_state
#           eeg_next_state
#           qrng_state
#           qrng_next_state
#           mc_eeg_state
#           mc_eeg_next_state
#           eeg_info_bin
#           qrng_info_bin
#           mc_eeg_info_bin
#           valid_next_state
#           valid_mc_next_state
#           valid_conditional_row
#           valid_mc_conditional_row
#
# Outputs:
#
#   data/interim/phase3_2e/phase3_2e_features.csv
#   data/interim/phase3_2e/phase3_2e_feature_specs.json
#   results/phase3_2e/phase3_2e_features_report.json
#   results/phase3_2e/phase3_2e_features_summary.txt
#


# =========================================================
# Constants
# =========================================================

FEATURE_MODULE = "phase3_2e_features"
FEATURE_VERSION = "phase3_2e_features_v1_sync_window_state_layer"


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class BinSpec:
    name: str
    source_column: str
    n_requested_bins: int
    edges: List[float]
    effective_bins: int
    constant: bool
    fill_value: float
    train_min: float
    train_max: float
    train_mean: float
    train_std: float


@dataclass
class FeatureBuildResult:
    status: str
    protocol_only: bool
    input_rows: int
    output_rows: int
    n_subjects: int
    n_sessions: int
    n_window_sizes: int
    n_valid_feature_rows: int
    n_valid_next_state_rows: int
    n_valid_mc_next_state_rows: int
    eeg_state_n_unique: int
    qrng_state_n_unique: int
    mc_eeg_state_n_unique: int
    can_proceed_to_alignment: bool
    can_proceed_to_conditional_cdr: bool
    can_proceed_to_empirical_cdr: bool
    interpretation: str


@dataclass
class FeatureFrame:
    df: pd.DataFrame
    bin_specs: Dict[str, BinSpec]
    metadata: Dict[str, Any]


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

    if isinstance(obj, BinSpec):
        return asdict(obj)

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


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_train_index(df: pd.DataFrame, train_index: Optional[Sequence[int]]) -> np.ndarray:
    if train_index is None:
        return np.arange(len(df), dtype=int)

    arr = np.asarray(train_index, dtype=int)
    arr = arr[(arr >= 0) & (arr < len(df))]

    if arr.size == 0:
        raise RuntimeError("train_index is empty after validation.")

    return arr


def _safe_zscore(values: pd.Series, train_values: pd.Series) -> pd.Series:
    train_numeric = _safe_numeric(train_values).dropna()

    if train_numeric.empty:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)

    mean = float(train_numeric.mean())
    std = float(train_numeric.std(ddof=0))

    if not np.isfinite(std) or std <= 1e-12:
        std = 1.0

    numeric = _safe_numeric(values).fillna(mean)

    return (numeric - mean) / std


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def _value_counts_dict(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if df.empty or col not in df.columns:
        return {}

    return {
        str(k): int(v)
        for k, v in df[col].fillna("").astype(str).value_counts().to_dict().items()
    }


def _numeric_summary(df: pd.DataFrame, col: str) -> Dict[str, Optional[float]]:
    if df.empty or col not in df.columns:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    x = pd.to_numeric(df[col], errors="coerce")

    if not x.notna().any():
        return {
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        col for col in [
            "subject_id",
            "session_id",
            "window_seconds",
            "window_start_utc",
            "window_id",
            "window_row_id",
        ]
        if col in df.columns
    ]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


# =========================================================
# Output paths
# =========================================================

def features_csv_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.features_csv)


def feature_specs_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.feature_specs_json)


def features_report_json_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_features_report.json"


def features_summary_txt_path(cfg: Phase32EConfig) -> Path:
    return Path(cfg.results_dir) / "phase3_2e_features_summary.txt"


# =========================================================
# Empty output
# =========================================================

def empty_features_frame(cfg: Phase32EConfig) -> pd.DataFrame:
    columns = [
        "feature_row_id",
        "phase3_2e_window_key",
        "subject_id",
        "session_id",
        "window_id",
        "window_seconds",
        "window_scope",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
        "valid_feature_row",

        "eeg_state",
        "eeg_next_state",
        "qrng_state",
        "qrng_next_state",
        "mc_eeg_state",
        "mc_eeg_next_state",

        "eeg_info_score",
        "qrng_info_score",
        "mc_eeg_info_score",

        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",

        "valid_next_state",
        "valid_mc_next_state",
        "valid_conditional_row",
        "valid_mc_conditional_row",

        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",

        "feature_status",
    ]

    return pd.DataFrame(columns=columns)


# =========================================================
# Binning
# =========================================================

def fit_quantile_bin_spec(
    df: pd.DataFrame,
    source_column: str,
    output_name: str,
    n_bins: int,
    train_index: Optional[Sequence[int]] = None,
) -> BinSpec:
    if source_column not in df.columns:
        raise KeyError(f"Column not found for binning: {source_column}")

    train_idx = _safe_train_index(df, train_index)
    train_values = _safe_numeric(df.iloc[train_idx][source_column]).dropna()

    if train_values.empty:
        return BinSpec(
            name=output_name,
            source_column=source_column,
            n_requested_bins=int(n_bins),
            edges=[],
            effective_bins=1,
            constant=True,
            fill_value=0.0,
            train_min=0.0,
            train_max=0.0,
            train_mean=0.0,
            train_std=0.0,
        )

    train_min = float(train_values.min())
    train_max = float(train_values.max())
    train_mean = float(train_values.mean())
    train_std = float(train_values.std(ddof=0))

    if not np.isfinite(train_std):
        train_std = 0.0

    if train_values.nunique(dropna=True) <= 1 or abs(train_max - train_min) <= 1e-12:
        return BinSpec(
            name=output_name,
            source_column=source_column,
            n_requested_bins=int(n_bins),
            edges=[],
            effective_bins=1,
            constant=True,
            fill_value=train_mean,
            train_min=train_min,
            train_max=train_max,
            train_mean=train_mean,
            train_std=train_std,
        )

    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = np.quantile(train_values.to_numpy(dtype=float), quantiles)
    edges = np.asarray(edges, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(edges)

    if edges.size == 0:
        effective_bins = 1
        constant = True
    else:
        effective_bins = int(edges.size + 1)
        constant = False

    return BinSpec(
        name=output_name,
        source_column=source_column,
        n_requested_bins=int(n_bins),
        edges=[float(x) for x in edges.tolist()],
        effective_bins=effective_bins,
        constant=constant,
        fill_value=train_mean,
        train_min=train_min,
        train_max=train_max,
        train_mean=train_mean,
        train_std=train_std,
    )


def apply_bin_spec(df: pd.DataFrame, spec: BinSpec) -> pd.Series:
    if spec.source_column not in df.columns:
        raise KeyError(f"Column not found while applying bin spec: {spec.source_column}")

    values = _safe_numeric(df[spec.source_column]).fillna(spec.fill_value)

    if spec.constant or not spec.edges:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

    edges = np.asarray(spec.edges, dtype=float)
    labels = np.digitize(values.to_numpy(dtype=float), edges, right=False)
    labels = np.asarray(labels, dtype=int)

    max_label = max(spec.effective_bins - 1, 0)
    labels = np.clip(labels, 0, max_label)

    return pd.Series(labels, index=df.index, dtype=int)


def fit_and_apply_bin(
    df: pd.DataFrame,
    source_column: str,
    output_name: str,
    n_bins: int,
    train_index: Optional[Sequence[int]],
    specs: Dict[str, BinSpec],
) -> pd.DataFrame:
    spec = fit_quantile_bin_spec(
        df=df,
        source_column=source_column,
        output_name=output_name,
        n_bins=int(n_bins),
        train_index=train_index,
    )

    df[output_name] = apply_bin_spec(df, spec)
    specs[output_name] = spec

    return df


# =========================================================
# Feature scores
# =========================================================

def add_eeg_info_score(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    train_index: Optional[Sequence[int]],
) -> pd.DataFrame:
    out = df.copy()

    candidate_cols = [
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_entropy",
        "eeg_gamma_power",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
        "eeg_line_length",
        "eeg_signal_std",
    ]

    existing = [col for col in candidate_cols if col in out.columns]

    if not existing:
        out["eeg_info_score"] = 0.0
        return out

    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z_cols.append(
            _safe_zscore(
                values=out[col],
                train_values=out.iloc[train_idx][col],
            )
        )

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    out["eeg_info_score"] = np.nanmean(stacked, axis=0).astype(float)

    return out


def add_qrng_info_score(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    train_index: Optional[Sequence[int]],
) -> pd.DataFrame:
    out = df.copy()

    candidate_cols = [
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
        "qrng_run_length_mean",
        "qrng_run_length_std",
        "qrng_run_length_max",
        "qrng_compressibility_proxy",
        "qrng_surprise_index",
        "qrng_transition_asymmetry",
        "qrng_quality_score",
    ]

    existing = [col for col in candidate_cols if col in out.columns]

    if not existing:
        out["qrng_info_score"] = 0.0
        return out

    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z_cols.append(
            _safe_zscore(
                values=out[col],
                train_values=out.iloc[train_idx][col],
            )
        )

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    out["qrng_info_score"] = np.nanmean(stacked, axis=0).astype(float)

    return out


def add_multichannel_info_score(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    train_index: Optional[Sequence[int]],
) -> pd.DataFrame:
    out = df.copy()

    candidate_cols = [
        "eeg_primary_delta_power",
        "eeg_primary_alpha_power",
        "eeg_secondary_delta_power",
        "eeg_secondary_alpha_power",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
        "cross_channel_corr",
    ]

    existing = [col for col in candidate_cols if col in out.columns]

    if not existing:
        out["mc_eeg_info_score"] = 0.0
        return out

    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z_cols.append(
            _safe_zscore(
                values=out[col],
                train_values=out.iloc[train_idx][col],
            )
        )

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    out["mc_eeg_info_score"] = np.nanmean(stacked, axis=0).astype(float)

    return out


# =========================================================
# State construction
# =========================================================

def add_single_channel_state_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    specs: Dict[str, BinSpec],
    train_index: Optional[Sequence[int]],
) -> pd.DataFrame:
    out = df.copy()

    required = [
        "eeg_delta_power",
        "eeg_alpha_power",
        "qrng_state",
        "eeg_info_score",
        "qrng_info_score",
    ]

    _ensure_columns(out, required, context="Phase III.2E single-channel state construction")

    out = fit_and_apply_bin(
        df=out,
        source_column="eeg_delta_power",
        output_name="eeg_delta_bin",
        n_bins=int(cfg.eeg_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="eeg_alpha_power",
        output_name="eeg_alpha_bin",
        n_bins=int(cfg.eeg_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="eeg_info_score",
        output_name="eeg_info_bin",
        n_bins=int(cfg.info_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="qrng_info_score",
        output_name="qrng_info_bin",
        n_bins=int(cfg.info_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out["eeg_state"] = (
        out["eeg_delta_bin"].astype(int) * int(cfg.eeg_n_bins)
        + out["eeg_alpha_bin"].astype(int)
    ).astype(int)

    out["qrng_state"] = pd.to_numeric(out["qrng_state"], errors="coerce").fillna(0).astype(int)
    out["qrng_state"] = out["qrng_state"].clip(0, int(cfg.qrng_n_bins) - 1).astype(int)

    out["observed_joint_state"] = (
        out["eeg_state"].astype(int) * int(cfg.qrng_n_bins)
        + out["qrng_state"].astype(int)
    ).astype(int)

    out["informational_joint_state"] = (
        out["eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
        + out["qrng_info_bin"].astype(int)
    ).astype(int)

    return out


def add_multichannel_state_columns(
    df: pd.DataFrame,
    cfg: Phase32EConfig,
    specs: Dict[str, BinSpec],
    train_index: Optional[Sequence[int]],
) -> pd.DataFrame:
    out = df.copy()

    if not cfg.use_multichannel_eeg:
        out["mc_eeg_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_eeg_info_bin"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_observed_joint_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_informational_joint_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        return out

    required = [
        "eeg_primary_delta_power",
        "eeg_primary_alpha_power",
        "eeg_secondary_delta_power",
        "eeg_secondary_alpha_power",
        "mc_eeg_info_score",
        "qrng_state",
    ]

    missing = [col for col in required if col not in out.columns]

    if missing:
        out["mc_eeg_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_eeg_info_bin"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_observed_joint_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["mc_informational_joint_state"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["multichannel_available"] = 0
        out["multichannel_missing_columns"] = ",".join(missing)
        return out

    out["multichannel_available"] = 1
    out["multichannel_missing_columns"] = ""

    # Compact activation and cross-channel features.
    out["mc_activation_score"] = (
        _safe_numeric(out["eeg_primary_delta_power"]).fillna(0.0)
        + _safe_numeric(out["eeg_primary_alpha_power"]).fillna(0.0)
        + _safe_numeric(out["eeg_secondary_delta_power"]).fillna(0.0)
        + _safe_numeric(out["eeg_secondary_alpha_power"]).fillna(0.0)
    ) / 4.0

    cross_candidates = [
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
        "cross_channel_corr",
    ]

    existing_cross = [col for col in cross_candidates if col in out.columns]

    if existing_cross:
        train_idx = _safe_train_index(out, train_index)
        z_cols = [
            _safe_zscore(
                values=out[col],
                train_values=out.iloc[train_idx][col],
            )
            for col in existing_cross
        ]
        stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
        out["mc_cross_channel_score"] = np.nanmean(stacked, axis=0).astype(float)
    else:
        out["mc_cross_channel_score"] = 0.0

    out = fit_and_apply_bin(
        df=out,
        source_column="mc_activation_score",
        output_name="mc_activation_bin",
        n_bins=int(cfg.multichannel_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="mc_cross_channel_score",
        output_name="mc_cross_channel_bin",
        n_bins=int(cfg.multichannel_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="mc_eeg_info_score",
        output_name="mc_eeg_info_bin",
        n_bins=int(cfg.info_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out["mc_eeg_state"] = (
        out["mc_activation_bin"].astype(int) * int(cfg.multichannel_n_bins)
        + out["mc_cross_channel_bin"].astype(int)
    ).astype(int)

    out["mc_observed_joint_state"] = (
        out["mc_eeg_state"].astype(int) * int(cfg.qrng_n_bins)
        + out["qrng_state"].astype(int)
    ).astype(int)

    out["mc_informational_joint_state"] = (
        out["mc_eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
        + out["qrng_info_bin"].astype(int)
    ).astype(int)

    return out


def add_next_state_columns(df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    required = [
        "subject_id",
        "session_id",
        "window_seconds",
        "eeg_state",
        "qrng_state",
    ]

    _ensure_columns(out := df.copy(), required, context="Phase III.2E next-state construction")

    out = _sort_feature_frame(out)

    next_columns = [
        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",
        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",
    ]

    for col in next_columns:
        if col in out.columns:
            out[f"{col}_next"] = pd.NA

    out["eeg_next_state"] = pd.NA
    out["qrng_next_state"] = pd.NA
    out["mc_eeg_next_state"] = pd.NA
    out["valid_next_state"] = 0
    out["valid_mc_next_state"] = 0

    group_cols = ["subject_id", "session_id", "window_seconds"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        # The dataframe is already sorted globally, but sort again inside each group
        # to guarantee next-state does not cross subject/session/window-size.
        group = out.loc[idx_arr].copy()

        sort_cols = [
            col for col in ["window_start_utc", "window_id", "window_row_id"]
            if col in group.columns
        ]

        if sort_cols:
            group = group.sort_values(sort_cols)

        group_idx = group.index.to_numpy(dtype=int)

        for col in next_columns:
            if col in out.columns:
                out.loc[group_idx, f"{col}_next"] = group[col].shift(-1).to_numpy()

        out.loc[group_idx, "eeg_next_state"] = group["eeg_state"].shift(-1).to_numpy()
        out.loc[group_idx, "qrng_next_state"] = group["qrng_state"].shift(-1).to_numpy()

        valid = group["eeg_state"].shift(-1).notna() & group["qrng_state"].shift(-1).notna()
        out.loc[group_idx, "valid_next_state"] = valid.astype(int).to_numpy()

        if "mc_eeg_state" in group.columns:
            out.loc[group_idx, "mc_eeg_next_state"] = group["mc_eeg_state"].shift(-1).to_numpy()

            valid_mc = (
                group["mc_eeg_state"].notna()
                & group["mc_eeg_state"].shift(-1).notna()
                & group["qrng_state"].shift(-1).notna()
            )
            out.loc[group_idx, "valid_mc_next_state"] = valid_mc.astype(int).to_numpy()

    for col in [
        "eeg_next_state",
        "qrng_next_state",
        "mc_eeg_next_state",
        "eeg_state_next",
        "qrng_state_next",
        "mc_eeg_state_next",
        "eeg_info_bin_next",
        "qrng_info_bin_next",
        "mc_eeg_info_bin_next",
        "observed_joint_state_next",
        "mc_observed_joint_state_next",
        "informational_joint_state_next",
        "mc_informational_joint_state_next",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


def add_validity_columns(df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    out = df.copy()

    if "valid_for_exploratory_analysis" in out.columns:
        valid_window = pd.to_numeric(out["valid_for_exploratory_analysis"], errors="coerce").fillna(0).astype(int)
    elif "window_valid" in out.columns:
        valid_window = pd.to_numeric(out["window_valid"], errors="coerce").fillna(0).astype(int)
    else:
        valid_window = pd.Series(np.zeros(len(out), dtype=int), index=out.index)

    out["valid_feature_row"] = valid_window.astype(int)

    out["valid_conditional_row"] = (
        out["valid_feature_row"].eq(1)
        & pd.to_numeric(out.get("valid_next_state", 0), errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)

    if "mc_eeg_state" in out.columns:
        mc_available = out["mc_eeg_state"].notna().astype(int)
    else:
        mc_available = pd.Series(np.zeros(len(out), dtype=int), index=out.index)

    out["valid_mc_conditional_row"] = (
        out["valid_feature_row"].eq(1)
        & mc_available.eq(1)
        & pd.to_numeric(out.get("valid_mc_next_state", 0), errors="coerce").fillna(0).astype(int).eq(1)
    ).astype(int)

    out["feature_status"] = np.where(
        out["valid_feature_row"].eq(1),
        "valid_feature_row",
        "invalid_or_underpowered_feature_row",
    )

    return out


def enforce_feature_column_order(df: pd.DataFrame, cfg: Phase32EConfig) -> pd.DataFrame:
    if df.empty:
        return df

    preferred = [
        "feature_row_id",
        "phase3_2e_window_key",
        "subject_id",
        "session_id",
        "window_id",
        "window_seconds",
        "window_spec_name",
        "window_scope",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
        "valid_feature_row",
        "valid_next_state",
        "valid_mc_next_state",
        "valid_conditional_row",
        "valid_mc_conditional_row",

        "eeg_state",
        "eeg_next_state",
        "qrng_state",
        "qrng_next_state",
        "mc_eeg_state",
        "mc_eeg_next_state",

        "eeg_info_score",
        "qrng_info_score",
        "mc_eeg_info_score",

        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",

        "observed_joint_state",
        "observed_joint_state_next",
        "mc_observed_joint_state",
        "mc_observed_joint_state_next",
        "informational_joint_state",
        "informational_joint_state_next",
        "mc_informational_joint_state",
        "mc_informational_joint_state_next",

        "eeg_delta_bin",
        "eeg_alpha_bin",
        "mc_activation_score",
        "mc_cross_channel_score",
        "mc_activation_bin",
        "mc_cross_channel_bin",

        "window_start_utc",
        "window_end_utc",
        "window_midpoint_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_qrng_gap_ms",
        "sync_quality",
        "clock_drift_ms",
        "clock_drift_abs_ms",
        "sync_mode_effective",
        "feature_status",
    ]

    ordered = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]

    return df[ordered + remaining].copy()


# =========================================================
# Summaries
# =========================================================

def summarize_feature_frame(df: pd.DataFrame, cfg: Optional[Phase32EConfig] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }

    if df.empty:
        summary.update(
            {
                "n_subjects": 0,
                "n_sessions": 0,
                "n_window_sizes": 0,
                "n_valid_feature_rows": 0,
                "n_valid_next_state_rows": 0,
                "n_valid_mc_next_state_rows": 0,
                "n_valid_conditional_rows": 0,
                "n_valid_mc_conditional_rows": 0,
                "eeg_state_n_unique": 0,
                "qrng_state_n_unique": 0,
                "mc_eeg_state_n_unique": 0,
            }
        )
        return summary

    if "subject_id" in df.columns:
        summary["n_subjects"] = int(df["subject_id"].nunique())
        summary["subjects"] = sorted(df["subject_id"].astype(str).unique().tolist())

    if {"subject_id", "session_id"}.issubset(df.columns):
        summary["n_sessions"] = int(df[["subject_id", "session_id"]].drop_duplicates().shape[0])

    if "window_seconds" in df.columns:
        summary["n_window_sizes"] = int(pd.to_numeric(df["window_seconds"], errors="coerce").dropna().nunique())
        summary["window_seconds_counts"] = _value_counts_dict(df, "window_seconds")

    for col in [
        "valid_feature_row",
        "valid_next_state",
        "valid_mc_next_state",
        "valid_conditional_row",
        "valid_mc_conditional_row",
        "valid_for_primary_analysis",
        "valid_for_exploratory_analysis",
    ]:
        if col in df.columns:
            summary[f"{col}_rows"] = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())

    for col in [
        "eeg_state",
        "qrng_state",
        "mc_eeg_state",
        "observed_joint_state",
        "mc_observed_joint_state",
        "informational_joint_state",
        "mc_informational_joint_state",
        "eeg_info_bin",
        "qrng_info_bin",
        "mc_eeg_info_bin",
    ]:
        if col in df.columns:
            summary[f"{col}_n_unique"] = int(df[col].nunique(dropna=True))

    summary["sync_quality"] = _numeric_summary(df, "sync_quality")
    summary["eeg_qrng_gap_ms"] = _numeric_summary(df, "eeg_qrng_gap_ms")
    summary["clock_drift_abs_ms"] = _numeric_summary(df, "clock_drift_abs_ms")

    summary["feature_status_counts"] = _value_counts_dict(df, "feature_status")
    summary["window_scope_counts"] = _value_counts_dict(df, "window_scope")

    return summary


def build_feature_result(
    status: str,
    protocol_only: bool,
    df: pd.DataFrame,
    summary: Dict[str, Any],
    interpretation: str,
) -> FeatureBuildResult:
    return FeatureBuildResult(
        status=status,
        protocol_only=bool(protocol_only),
        input_rows=int(summary.get("n_rows", 0)),
        output_rows=int(summary.get("n_rows", 0)),
        n_subjects=int(summary.get("n_subjects", 0)),
        n_sessions=int(summary.get("n_sessions", 0)),
        n_window_sizes=int(summary.get("n_window_sizes", 0)),
        n_valid_feature_rows=int(summary.get("valid_feature_row_rows", 0)),
        n_valid_next_state_rows=int(summary.get("valid_next_state_rows", 0)),
        n_valid_mc_next_state_rows=int(summary.get("valid_mc_next_state_rows", 0)),
        eeg_state_n_unique=int(summary.get("eeg_state_n_unique", 0)),
        qrng_state_n_unique=int(summary.get("qrng_state_n_unique", 0)),
        mc_eeg_state_n_unique=int(summary.get("mc_eeg_state_n_unique", 0)),
        can_proceed_to_alignment=int(summary.get("valid_next_state_rows", 0)) > 0,
        can_proceed_to_conditional_cdr=int(summary.get("valid_conditional_row_rows", 0)) > 0,
        can_proceed_to_empirical_cdr=int(summary.get("valid_conditional_row_rows", 0)) > 0,
        interpretation=interpretation,
    )


def write_features_summary_txt(
    path: Path,
    result: FeatureBuildResult,
    summary: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2E — Feature Summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"status: {result.status}")
    lines.append(f"protocol_only: {result.protocol_only}")
    lines.append(f"input_rows: {result.input_rows}")
    lines.append(f"output_rows: {result.output_rows}")
    lines.append(f"n_subjects: {result.n_subjects}")
    lines.append(f"n_sessions: {result.n_sessions}")
    lines.append(f"n_window_sizes: {result.n_window_sizes}")
    lines.append("")
    lines.append(f"n_valid_feature_rows: {result.n_valid_feature_rows}")
    lines.append(f"n_valid_next_state_rows: {result.n_valid_next_state_rows}")
    lines.append(f"n_valid_mc_next_state_rows: {result.n_valid_mc_next_state_rows}")
    lines.append("")
    lines.append(f"eeg_state_n_unique: {result.eeg_state_n_unique}")
    lines.append(f"qrng_state_n_unique: {result.qrng_state_n_unique}")
    lines.append(f"mc_eeg_state_n_unique: {result.mc_eeg_state_n_unique}")
    lines.append("")
    lines.append(f"can_proceed_to_alignment: {result.can_proceed_to_alignment}")
    lines.append(f"can_proceed_to_conditional_cdr: {result.can_proceed_to_conditional_cdr}")
    lines.append(f"can_proceed_to_empirical_cdr: {result.can_proceed_to_empirical_cdr}")
    lines.append("")
    lines.append(f"window_seconds_counts: {summary.get('window_seconds_counts')}")
    lines.append(f"feature_status_counts: {summary.get('feature_status_counts')}")
    lines.append(f"window_scope_counts: {summary.get('window_scope_counts')}")
    lines.append("")
    lines.append(f"sync_quality: {summary.get('sync_quality')}")
    lines.append(f"eeg_qrng_gap_ms: {summary.get('eeg_qrng_gap_ms')}")
    lines.append(f"clock_drift_abs_ms: {summary.get('clock_drift_abs_ms')}")
    lines.append("")
    lines.append("interpretation:")
    lines.append(result.interpretation)
    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================================================
# Main feature builder
# =========================================================

def prepare_phase3_2e_features_for_modeling(
    windows_df: pd.DataFrame,
    cfg: Optional[Phase32EConfig] = None,
    train_index: Optional[Sequence[int]] = None,
) -> FeatureFrame:
    if cfg is None:
        cfg = load_phase3_2e_config()

    specs: Dict[str, BinSpec] = {}

    if windows_df.empty:
        out = empty_features_frame(cfg)
        metadata = {
            "phase": cfg.phase_name,
            "project_name": cfg.project_name,
            "feature_version": FEATURE_VERSION,
            "status": cfg.status_pending_synchronized_data,
            "protocol_only": True,
            "n_rows": 0,
            "bin_specs": {},
            "summary": summarize_feature_frame(out, cfg),
        }

        return FeatureFrame(df=out, bin_specs=specs, metadata=metadata)

    out = _sort_feature_frame(windows_df.copy())
    out["feature_row_id"] = np.arange(len(out), dtype=int)

    # Only rows valid for exploratory analysis can enter feature modeling.
    # Primary/exploratory distinction is preserved downstream.
    if "valid_for_exploratory_analysis" not in out.columns:
        raise KeyError("valid_for_exploratory_analysis missing from windows dataframe.")

    # Scores.
    out = add_eeg_info_score(out, cfg, train_index=train_index)
    out = add_qrng_info_score(out, cfg, train_index=train_index)
    out = add_multichannel_info_score(out, cfg, train_index=train_index)

    # States.
    out = add_single_channel_state_columns(out, cfg, specs, train_index=train_index)
    out = add_multichannel_state_columns(out, cfg, specs, train_index=train_index)

    # Next states and validity.
    out = add_next_state_columns(out, cfg)
    out = add_validity_columns(out, cfg)

    out = enforce_feature_column_order(out, cfg)

    summary = summarize_feature_frame(out, cfg)

    metadata = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "phase_title": cfg.phase_title,
        "config_version": cfg.version,
        "feature_module": FEATURE_MODULE,
        "feature_version": FEATURE_VERSION,
        "created_at": _now_str(),
        "train_index_used": train_index is not None,
        "n_train_rows_for_bins": int(len(_safe_train_index(out, train_index))),
        "bin_specs": {name: asdict(spec) for name, spec in specs.items()},
        "state_columns": {
            "E0_EEG_next_given_EEG": "eeg_state -> eeg_next_state",
            "E1_EEG_next_given_EEG_QRNG": "(eeg_state, qrng_state) -> eeg_next_state",
            "E2_QRNG_next_given_QRNG": "qrng_state -> qrng_next_state",
            "E3_QRNG_next_given_QRNG_EEG": "(qrng_state, eeg_state) -> qrng_next_state",
            "E4_MCEEG_next_given_MCEEG": "mc_eeg_state -> mc_eeg_next_state",
            "E5_MCEEG_next_given_MCEEG_QRNG": "(mc_eeg_state, qrng_state) -> mc_eeg_next_state",
            "E6_QRNG_next_given_QRNG_MCEEG": "(qrng_state, mc_eeg_state) -> qrng_next_state",
        },
        "summary": summary,
    }

    return FeatureFrame(df=out, bin_specs=specs, metadata=metadata)


def save_phase3_2e_features(
    feature_frame: FeatureFrame,
    cfg: Optional[Phase32EConfig] = None,
    status: Optional[str] = None,
    interpretation: Optional[str] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_phase3_2e_config()

    Path(cfg.interim_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    summary = summarize_feature_frame(feature_frame.df, cfg)

    if status is None:
        status = "features_ready" if int(summary.get("valid_conditional_row_rows", 0)) > 0 else cfg.status_pending_synchronized_data

    if interpretation is None:
        interpretation = (
            "Phase III.2E features were built successfully."
            if status == "features_ready"
            else "Phase III.2E feature construction completed without empirical rows."
        )

    result = build_feature_result(
        status=status,
        protocol_only=bool(status == cfg.status_pending_synchronized_data),
        df=feature_frame.df,
        summary=summary,
        interpretation=interpretation,
    )

    report = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "module": FEATURE_MODULE,
        "feature_version": FEATURE_VERSION,
        "status": result.status,
        "created_at": _now_str(),
        "result": asdict(result),
        "summary": summary,
        "features_csv": str(features_csv_path(cfg)),
        "feature_specs_json": str(feature_specs_json_path(cfg)),
        "features_report_json": str(features_report_json_path(cfg)),
        "features_summary_txt": str(features_summary_txt_path(cfg)),
        "metadata": feature_frame.metadata,
        "config": describe_config(cfg),
    }

    feature_frame.df.to_csv(features_csv_path(cfg), index=False)

    save_json(
        feature_specs_json_path(cfg),
        {
            "metadata": feature_frame.metadata,
            "bin_specs": {
                name: asdict(spec)
                for name, spec in feature_frame.bin_specs.items()
            },
        },
    )

    save_json(features_report_json_path(cfg), report)

    write_features_summary_txt(
        path=features_summary_txt_path(cfg),
        result=result,
        summary=summary,
    )

    print(f"[Phase3.2E Features] Saved features: {features_csv_path(cfg)}")
    print(f"[Phase3.2E Features] Saved feature specs: {feature_specs_json_path(cfg)}")
    print(f"[Phase3.2E Features] Saved report: {features_report_json_path(cfg)}")
    print(f"[Phase3.2E Features] Saved summary: {features_summary_txt_path(cfg)}")

    return report


def build_phase3_2e_features(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
    train_index: Optional[Sequence[int]] = None,
) -> FeatureFrame:
    if cfg is None:
        cfg = load_phase3_2e_config()

    print("[Phase3.2E Features] Loading Phase III.2E synchronized windows")

    windows_df, inventory_df, windows_report = load_or_build_phase3_2e_windows(
        cfg=cfg,
        force_rebuild=bool(force_rebuild_windows),
        force_rebuild_loader=bool(force_rebuild_loader),
    )

    windows_result = windows_report.get("result", {})
    windows_status = str(windows_result.get("status", windows_report.get("status", "")))

    if windows_df.empty or windows_status == cfg.status_pending_synchronized_data:
        feature_frame = prepare_phase3_2e_features_for_modeling(
            windows_df=pd.DataFrame(),
            cfg=cfg,
            train_index=train_index,
        )

        feature_frame.metadata["windows_status"] = windows_status
        feature_frame.metadata["windows_result"] = windows_result

        save_phase3_2e_features(
            feature_frame=feature_frame,
            cfg=cfg,
            status=cfg.status_pending_synchronized_data,
            interpretation=(
                "No synchronized EEG–QRNG dataset is available. Feature construction "
                "completed in protocol-only mode with empty canonical outputs."
            ),
        )

        return feature_frame

    if not bool(windows_result.get("can_proceed_to_feature_construction", False)):
        feature_frame = prepare_phase3_2e_features_for_modeling(
            windows_df=pd.DataFrame(),
            cfg=cfg,
            train_index=train_index,
        )

        feature_frame.metadata["windows_status"] = windows_status
        feature_frame.metadata["windows_result"] = windows_result

        save_phase3_2e_features(
            feature_frame=feature_frame,
            cfg=cfg,
            status="features_blocked_by_windows",
            interpretation=(
                "Synchronized windows exist, but no valid ready window configuration "
                "can proceed to feature construction."
            ),
        )

        return feature_frame

    feature_frame = prepare_phase3_2e_features_for_modeling(
        windows_df=windows_df,
        cfg=cfg,
        train_index=train_index,
    )

    summary = summarize_feature_frame(feature_frame.df, cfg)

    if int(summary.get("valid_conditional_row_rows", 0)) > 0:
        status = "features_ready"
        interpretation = (
            "Phase III.2E synchronized feature/state layer was built successfully. "
            "The dataset can proceed to lagged alignment and conditional CDR."
        )
    else:
        status = "features_no_valid_next_state"
        interpretation = (
            "Phase III.2E features were built, but no valid next-state rows are "
            "available for conditional CDR."
        )

    save_phase3_2e_features(
        feature_frame=feature_frame,
        cfg=cfg,
        status=status,
        interpretation=interpretation,
    )

    return feature_frame


def load_or_build_phase3_2e_features(
    cfg: Optional[Phase32EConfig] = None,
    force_rebuild: bool = False,
    force_rebuild_windows: bool = False,
    force_rebuild_loader: bool = False,
) -> FeatureFrame:
    if cfg is None:
        cfg = load_phase3_2e_config()

    features_path = features_csv_path(cfg)
    specs_path = feature_specs_json_path(cfg)

    if features_path.exists() and specs_path.exists() and not force_rebuild:
        print(f"[Phase3.2E Features] Loading cached features: {features_path}")

        try:
            df = pd.read_csv(features_path)

            with open(specs_path, "r", encoding="utf-8") as f:
                specs_payload = json.load(f)

            metadata = specs_payload.get("metadata", {})
            raw_specs = specs_payload.get("bin_specs", {})

            if metadata.get("feature_version") == FEATURE_VERSION:
                specs = {
                    name: BinSpec(**payload)
                    for name, payload in raw_specs.items()
                }

                return FeatureFrame(
                    df=df,
                    bin_specs=specs,
                    metadata=metadata,
                )

            print("[Phase3.2E Features] Cached feature version changed. Rebuilding...")

        except Exception as exc:
            print(f"[Phase3.2E Features] Failed to load cached features: {exc}. Rebuilding...")

    return build_phase3_2e_features(
        cfg=cfg,
        force_rebuild_windows=bool(force_rebuild_windows),
        force_rebuild_loader=bool(force_rebuild_loader),
    )


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase III.2E synchronized feature/state layer."
    )

    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding feature outputs.",
    )

    parser.add_argument(
        "--force-rebuild-windows",
        action="store_true",
        help="Force rebuilding windows before features.",
    )

    parser.add_argument(
        "--force-rebuild-loader",
        action="store_true",
        help="Force rebuilding loader and windows before features.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_phase3_2e_config()

    feature_frame = load_or_build_phase3_2e_features(
        cfg=cfg,
        force_rebuild=bool(args.force_rebuild),
        force_rebuild_windows=bool(args.force_rebuild_windows),
        force_rebuild_loader=bool(args.force_rebuild_loader),
    )

    summary = summarize_feature_frame(feature_frame.df, cfg)

    status = feature_frame.metadata.get("status", "unknown")

    print("\n" + "=" * 78)
    print("Phase III.2E feature construction completed")
    print("=" * 78)
    print(f"status: {status}")
    print(f"n_rows: {summary.get('n_rows')}")
    print(f"n_subjects: {summary.get('n_subjects')}")
    print(f"n_sessions: {summary.get('n_sessions')}")
    print(f"n_window_sizes: {summary.get('n_window_sizes')}")
    print(f"valid_feature_rows: {summary.get('valid_feature_row_rows')}")
    print(f"valid_next_state_rows: {summary.get('valid_next_state_rows')}")
    print(f"valid_mc_next_state_rows: {summary.get('valid_mc_next_state_rows')}")
    print(f"valid_conditional_rows: {summary.get('valid_conditional_row_rows')}")
    print(f"valid_mc_conditional_rows: {summary.get('valid_mc_conditional_row_rows')}")
    print(f"eeg_state_n_unique: {summary.get('eeg_state_n_unique')}")
    print(f"qrng_state_n_unique: {summary.get('qrng_state_n_unique')}")
    print(f"mc_eeg_state_n_unique: {summary.get('mc_eeg_state_n_unique')}")
    print(f"features_csv: {features_csv_path(cfg)}")
    print(f"feature_specs_json: {feature_specs_json_path(cfg)}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()