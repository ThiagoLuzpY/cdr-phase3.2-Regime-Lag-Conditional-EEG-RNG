from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import Phase32Config, load_phase3_2_config
from src.phase3_2_loader import load_or_build_phase3_2_dataset


# =========================================================
# Phase III.2 — Feature Preparation
# =========================================================
#
# This module builds the common feature/state layer used by:
#
#   III.2A — regime-aware / sleep-stage analysis
#   III.2B — lagged EEG-RNG alignment
#   III.2C — conditional CDR
#   III.2D — optional multi-channel EEG enrichment
#
# Important design principles:
#
#   1. Keep state spaces compact.
#   2. Fit discretization only on training rows when train_index is provided.
#   3. Prevent artificial next-state transitions across subjects/recordings.
#   4. Preserve the difference between:
#        - EEG-only structure
#        - RNG-only structure
#        - conditional EEG-RNG structure
#   5. Do not interpret proxy features as direct physical measurements.
#
# Output:
#
#   data/interim/phase3_2/phase3_2_modeling_features.csv
#   data/interim/phase3_2/phase3_2_feature_specs.json


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class BinSpec:
    """
    Stores the discretization rule for one continuous variable.

    The edges are fitted from the training subset when train_index is provided.
    """

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
class FeatureFrame:
    """
    Output object returned by feature preparation.
    """

    df: pd.DataFrame
    bin_specs: Dict[str, BinSpec]
    metadata: Dict[str, Any]


# =========================================================
# JSON helpers
# =========================================================

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, BinSpec):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


# =========================================================
# Generic helpers
# =========================================================

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


def _safe_zscore(
    values: pd.Series,
    train_values: pd.Series,
) -> pd.Series:
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


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


# =========================================================
# Discretization
# =========================================================

def fit_quantile_bin_spec(
    df: pd.DataFrame,
    source_column: str,
    output_name: str,
    n_bins: int,
    train_index: Optional[Sequence[int]] = None,
) -> BinSpec:
    """
    Fits quantile edges for a continuous variable.

    If the variable is constant or nearly constant, all rows are mapped to 0.
    """
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


def apply_bin_spec(
    df: pd.DataFrame,
    spec: BinSpec,
) -> pd.Series:
    """
    Applies a fitted BinSpec to a dataframe.
    """
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
        n_bins=n_bins,
        train_index=train_index,
    )

    df[output_name] = apply_bin_spec(df, spec)
    specs[output_name] = spec

    return df


# =========================================================
# Sleep-stage flags and transition markers
# =========================================================

def add_sleep_stage_flags(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["sleep_stage"], context="sleep-stage flags")

    out = df.copy()

    stage = out["sleep_stage"].astype(str)

    out["is_wake"] = (stage == "W").astype(int)
    out["is_sleep"] = stage.isin(["N1", "N2", "N3", "REM"]).astype(int)
    out["is_n1"] = (stage == "N1").astype(int)
    out["is_n2"] = (stage == "N2").astype(int)
    out["is_n3"] = (stage == "N3").astype(int)
    out["is_rem"] = (stage == "REM").astype(int)
    out["is_stable_sleep"] = stage.isin(["N2", "N3"]).astype(int)
    out["is_deep_sleep"] = (stage == "N3").astype(int)

    stage_code_map = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4,
    }

    out["sleep_stage_code"] = stage.map(stage_code_map).fillna(-1).astype(int)

    return out


def add_sleep_transition_flags(
    df: pd.DataFrame,
    transition_window_epochs: int = 2,
) -> pd.DataFrame:
    """
    Adds previous/next stage markers and transition-window flags.

    A transition epoch is defined as an epoch within ±transition_window_epochs
    around a stage change, calculated separately inside each subject/recording.
    """
    _ensure_columns(
        df,
        ["subject_id", "recording_id", "epoch_idx", "sleep_stage"],
        context="sleep-transition flags",
    )

    out = _sort_feature_frame(df.copy())

    out["stage_prev"] = None
    out["stage_next"] = None
    out["stage_changed_from_prev"] = 0
    out["stage_changed_to_next"] = 0
    out["is_transition_epoch"] = 0
    out["epoch_order_in_recording"] = 0

    group_cols = ["subject_id", "recording_id"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        group_stage = out.loc[idx_arr, "sleep_stage"].astype(str).reset_index(drop=True)

        prev_stage = group_stage.shift(1)
        next_stage = group_stage.shift(-1)

        changed_from_prev = (
            (group_stage != prev_stage)
            & prev_stage.notna()
        ).astype(int)

        changed_to_next = (
            (group_stage != next_stage)
            & next_stage.notna()
        ).astype(int)

        out.loc[idx_arr, "stage_prev"] = prev_stage.fillna("").to_numpy()
        out.loc[idx_arr, "stage_next"] = next_stage.fillna("").to_numpy()
        out.loc[idx_arr, "stage_changed_from_prev"] = changed_from_prev.to_numpy(dtype=int)
        out.loc[idx_arr, "stage_changed_to_next"] = changed_to_next.to_numpy(dtype=int)
        out.loc[idx_arr, "epoch_order_in_recording"] = np.arange(len(idx_arr), dtype=int)

        transition_positions = np.flatnonzero(changed_to_next.to_numpy(dtype=bool))
        mark = np.zeros(len(idx_arr), dtype=int)

        for pos in transition_positions:
            start = max(0, int(pos) - int(transition_window_epochs))
            end = min(len(idx_arr), int(pos) + int(transition_window_epochs) + 2)
            mark[start:end] = 1

        out.loc[idx_arr, "is_transition_epoch"] = mark

    return out


# =========================================================
# Composite proxy scores
# =========================================================

def add_eeg_informational_score(
    df: pd.DataFrame,
    train_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Adds a compact EEG informational score.

    This is a proxy score, not a direct physical measurement.
    """
    candidate_cols = [
        "spectral_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "line_length",
        "alpha_delta_ratio",
        "theta_alpha_ratio",
        "signal_std",
    ]

    existing = [c for c in candidate_cols if c in df.columns]

    if not existing:
        raise KeyError("No EEG columns available for eeg_info_score.")

    out = df.copy()
    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z = _safe_zscore(
            values=out[col],
            train_values=out.iloc[train_idx][col],
        )
        z_cols.append(z)

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    score = np.nanmean(stacked, axis=0)

    out["eeg_info_score"] = score.astype(float)

    return out


def add_rng_informational_score(
    df: pd.DataFrame,
    train_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Adds a compact RNG informational score.

    This summarizes local departures from idealized bit-level randomness.
    It is not a claim about quantum-state measurement.
    """
    candidate_cols = [
        "rng_entropy_local",
        "entropy_rate_proxy",
        "transition_rate",
        "run_length_mean",
        "run_length_std",
        "compressibility_proxy",
        "surprise_index",
        "transition_asymmetry",
        "micro_cluster_deviation",
    ]

    existing = [c for c in candidate_cols if c in df.columns]

    if not existing:
        raise KeyError("No RNG columns available for rng_info_score.")

    out = df.copy()
    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z = _safe_zscore(
            values=out[col],
            train_values=out.iloc[train_idx][col],
        )
        z_cols.append(z)

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    score = np.nanmean(stacked, axis=0)

    out["rng_info_score"] = score.astype(float)

    return out


def add_latent_proxy_score(
    df: pd.DataFrame,
    train_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Adds a compact latent proxy score combining EEG and RNG proxy layers.

    This is intentionally simple. It exists to support later diagnostic tests,
    not to impose a theoretical latent variable.
    """
    candidate_cols = [
        "eeg_info_score",
        "rng_info_score",
        "q_rng_score",
    ]

    existing = [c for c in candidate_cols if c in df.columns]

    if not existing:
        raise KeyError("No proxy columns available for latent_proxy_score.")

    out = df.copy()
    train_idx = _safe_train_index(out, train_index)

    z_cols: List[pd.Series] = []

    for col in existing:
        z = _safe_zscore(
            values=out[col],
            train_values=out.iloc[train_idx][col],
        )
        z_cols.append(z)

    stacked = np.vstack([z.to_numpy(dtype=float) for z in z_cols])
    score = np.nanmean(stacked, axis=0)

    out["latent_proxy_score"] = score.astype(float)

    return out


# =========================================================
# State construction
# =========================================================

def add_state_columns(
    df: pd.DataFrame,
    cfg: Phase32Config,
) -> pd.DataFrame:
    """
    Adds compact state columns used by conditional and joint tests.
    """
    _ensure_columns(
        df,
        ["delta_power_bin", "alpha_power_bin", "rng_bit"],
        context="state construction",
    )

    out = df.copy()

    # EEG state: 3 × 3 = 9 nominal states.
    out["eeg_state"] = (
        out["delta_power_bin"].astype(int) * int(cfg.eeg_n_bins)
        + out["alpha_power_bin"].astype(int)
    ).astype(int)

    # RNG state: binary state.
    out["rng_state"] = out["rng_bit"].astype(int)

    # Informational joint states.
    if "eeg_info_bin" in out.columns and "rng_info_bin" in out.columns:
        out["informational_joint_state"] = (
            out["eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
            + out["rng_info_bin"].astype(int)
        ).astype(int)

    # Latent/quantum-aware compact proxy state.
    if "latent_state" in out.columns and "q_rng_bin" in out.columns:
        out["latent_q_state"] = (
            out["latent_state"].astype(int) * int(cfg.q_n_bins)
            + out["q_rng_bin"].astype(int)
        ).astype(int)

    # Compact observed EEG-RNG state.
    out["observed_joint_state"] = (
        out["eeg_state"].astype(int) * int(cfg.rng_n_bins)
        + out["rng_state"].astype(int)
    ).astype(int)

    return out


def add_next_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds next-state target columns without crossing subject/recording boundaries.
    """
    _ensure_columns(
        df,
        ["subject_id", "recording_id", "epoch_idx", "eeg_state", "rng_state"],
        context="next-state construction",
    )

    out = _sort_feature_frame(df.copy())

    out["eeg_next_state"] = np.nan
    out["rng_next_state"] = np.nan
    out["valid_next_state"] = 0

    optional_next_cols = [
        "sleep_stage_code",
        "eeg_info_bin",
        "rng_info_bin",
        "q_rng_bin",
        "latent_state",
        "observed_joint_state",
        "informational_joint_state",
        "latent_q_state",
    ]

    for col in optional_next_cols:
        if col in out.columns:
            out[f"{col}_next"] = np.nan

    group_cols = ["subject_id", "recording_id"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        eeg_next = out.loc[idx_arr, "eeg_state"].shift(-1)
        rng_next = out.loc[idx_arr, "rng_state"].shift(-1)

        valid = eeg_next.notna() & rng_next.notna()

        out.loc[idx_arr, "eeg_next_state"] = eeg_next.to_numpy()
        out.loc[idx_arr, "rng_next_state"] = rng_next.to_numpy()
        out.loc[idx_arr, "valid_next_state"] = valid.astype(int).to_numpy()

        for col in optional_next_cols:
            if col in out.columns:
                out.loc[idx_arr, f"{col}_next"] = out.loc[idx_arr, col].shift(-1).to_numpy()

    out["eeg_next_state"] = out["eeg_next_state"].astype("Int64")
    out["rng_next_state"] = out["rng_next_state"].astype("Int64")

    return out


# =========================================================
# Full feature preparation
# =========================================================

def prepare_phase3_2_features_for_modeling(
    df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
    train_index: Optional[Sequence[int]] = None,
) -> FeatureFrame:
    """
    Main feature preparation entrypoint.

    Parameters
    ----------
    df:
        Base dataframe produced by phase3_2_loader.
    cfg:
        Phase III.2 configuration.
    train_index:
        Optional training row indices used to fit discretization edges.
        When provided, bins are fitted only from these rows to avoid leakage.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    out = _sort_feature_frame(df.copy())
    specs: Dict[str, BinSpec] = {}

    required_base = [
        "subject_id",
        "recording_id",
        "epoch_idx",
        "sleep_stage",
        "delta_power",
        "alpha_power",
        "rng_bit",
    ]

    _ensure_columns(out, required_base, context="Phase III.2 base features")

    # Sleep-stage flags and transition markers.
    out = add_sleep_stage_flags(out)
    out = add_sleep_transition_flags(
        out,
        transition_window_epochs=int(cfg.transition_window_epochs),
    )

    # Proxy scores.
    out = add_eeg_informational_score(out, train_index=train_index)
    out = add_rng_informational_score(out, train_index=train_index)

    if "q_rng_score" not in out.columns:
        raise KeyError("q_rng_score is required for Phase III.2 quantum-aware proxy bins.")

    out = add_latent_proxy_score(out, train_index=train_index)

    # Core bins.
    out = fit_and_apply_bin(
        df=out,
        source_column="delta_power",
        output_name="delta_power_bin",
        n_bins=int(cfg.eeg_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="alpha_power",
        output_name="alpha_power_bin",
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
        source_column="rng_info_score",
        output_name="rng_info_bin",
        n_bins=int(cfg.info_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="q_rng_score",
        output_name="q_rng_bin",
        n_bins=int(cfg.q_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="latent_proxy_score",
        output_name="latent_state",
        n_bins=int(cfg.latent_n_states),
        train_index=train_index,
        specs=specs,
    )

    # Optional multichannel bins, if features exist.
    if cfg.use_multichannel_eeg:
        optional_multichannel_sources = [
            ("pz_oz_delta_power", "pz_oz_delta_power_bin"),
            ("pz_oz_alpha_power", "pz_oz_alpha_power_bin"),
            ("delta_channel_diff", "delta_channel_diff_bin"),
            ("alpha_channel_diff", "alpha_channel_diff_bin"),
            ("cross_channel_corr", "cross_channel_corr_bin"),
        ]

        for source_col, output_col in optional_multichannel_sources:
            if source_col in out.columns:
                out = fit_and_apply_bin(
                    df=out,
                    source_column=source_col,
                    output_name=output_col,
                    n_bins=int(cfg.eeg_n_bins),
                    train_index=train_index,
                    specs=specs,
                )

    # State and next-state construction.
    out = add_state_columns(out, cfg)
    out = add_next_state_columns(out)

    metadata: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "n_rows": int(len(out)),
        "n_subjects": int(out["subject_id"].nunique()),
        "subjects": sorted(out["subject_id"].astype(str).unique().tolist()),
        "train_index_used": train_index is not None,
        "n_train_rows_for_bins": int(len(_safe_train_index(out, train_index))),
        "bin_specs": {name: asdict(spec) for name, spec in specs.items()},
        "state_columns": {
            "eeg_state": "delta_power_bin × alpha_power_bin",
            "rng_state": "rng_bit",
            "observed_joint_state": "eeg_state × rng_state",
            "informational_joint_state": "eeg_info_bin × rng_info_bin",
            "latent_q_state": "latent_state × q_rng_bin",
        },
        "valid_next_state_rows": int(out["valid_next_state"].sum()),
    }

    return FeatureFrame(
        df=out,
        bin_specs=specs,
        metadata=metadata,
    )


def save_phase3_2_modeling_features(
    feature_frame: FeatureFrame,
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    output_csv = Path(cfg.interim_dir) / "phase3_2_modeling_features.csv"
    output_json = Path(cfg.interim_dir) / "phase3_2_feature_specs.json"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    feature_frame.df.to_csv(output_csv, index=False)

    save_json(
        output_json,
        {
            "metadata": feature_frame.metadata,
            "bin_specs": {
                name: asdict(spec)
                for name, spec in feature_frame.bin_specs.items()
            },
        },
    )

    print(f"[Phase3.2 Features] Saved modeling features: {output_csv}")
    print(f"[Phase3.2 Features] Saved feature specs: {output_json}")


# =========================================================
# Diagnostics helpers
# =========================================================

def summarize_feature_frame(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }

    if "subject_id" in df.columns:
        summary["n_subjects"] = int(df["subject_id"].nunique())
        summary["subjects"] = sorted(df["subject_id"].astype(str).unique().tolist())

    if "sleep_stage" in df.columns:
        summary["sleep_stage_counts"] = (
            df.groupby(["subject_id", "sleep_stage"])
            .size()
            .reset_index(name="n_epochs")
            .to_dict(orient="records")
            if "subject_id" in df.columns
            else df["sleep_stage"].value_counts().to_dict()
        )

    for col in [
        "eeg_state",
        "rng_state",
        "observed_joint_state",
        "informational_joint_state",
        "latent_q_state",
    ]:
        if col in df.columns:
            summary[f"{col}_n_unique"] = int(df[col].nunique(dropna=True))

    if "valid_next_state" in df.columns:
        summary["valid_next_state_rows"] = int(df["valid_next_state"].sum())

    return summary


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    base_df, loader_metadata = load_or_build_phase3_2_dataset(
        cfg=cfg,
        force_rebuild=False,
    )

    feature_frame = prepare_phase3_2_features_for_modeling(
        df=base_df,
        cfg=cfg,
        train_index=None,
    )

    save_phase3_2_modeling_features(feature_frame, cfg)

    summary = summarize_feature_frame(feature_frame.df)

    print("\n============================================================")
    print("Phase III.2 feature preparation completed")
    print("============================================================")
    print(f"Rows: {summary.get('n_rows')}")
    print(f"Subjects: {summary.get('subjects')}")
    print(f"Valid next-state rows: {summary.get('valid_next_state_rows')}")
    print(f"EEG states: {summary.get('eeg_state_n_unique')}")
    print(f"RNG states: {summary.get('rng_state_n_unique')}")
    print(f"Observed joint states: {summary.get('observed_joint_state_n_unique')}")
    print(f"Informational joint states: {summary.get('informational_joint_state_n_unique')}")
    print(f"Latent-Q states: {summary.get('latent_q_state_n_unique')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()