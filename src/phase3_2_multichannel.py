from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import Phase32Config, load_phase3_2_config


# =========================================================
# Phase III.2D — Multichannel EEG Enrichment
# =========================================================
#
# Purpose:
#
#   Build a compact multichannel EEG representation from Sleep-EDF channels:
#
#       primary:   EEG Fpz-Cz
#       secondary: EEG Pz-Oz
#
#   The goal is not to inflate the state space. The goal is to improve the EEG
#   state representation while keeping a compact 3 × 3 = 9-state structure.
#
# Outputs:
#
#   data/interim/phase3_2/phase3_2d_multichannel_features.csv
#   data/interim/phase3_2/phase3_2d_multichannel_inventory.json
#   results/phase3_2/phase3_2d_multichannel_summary.json
#
# Important:
#
#   This module does not claim EEG-RNG coupling. It only prepares a richer
#   neural state for Phase III.2D conditional testing.
#


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class MultichannelBinSpec:
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
class MultichannelFeatureResult:
    df: pd.DataFrame
    bin_specs: Dict[str, MultichannelBinSpec]
    inventory: Dict[str, Any]


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

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, MultichannelBinSpec):
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


def _sort_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = _safe_numeric(numerator).astype(float)
    den = _safe_numeric(denominator).astype(float)

    num = num.replace([np.inf, -np.inf], np.nan)
    den = den.replace([np.inf, -np.inf], np.nan)

    num_fill = float(num.dropna().median()) if not num.dropna().empty else 0.0
    den_fill = float(den.dropna().median()) if not den.dropna().empty else 1.0

    num = num.fillna(num_fill)
    den = den.fillna(den_fill)

    # Power-like values should be non-negative, but we keep this robust.
    num_safe = np.log1p(np.abs(num.to_numpy(dtype=float)))
    den_safe = np.log1p(np.abs(den.to_numpy(dtype=float))) + 1e-9

    return pd.Series(num_safe / den_safe, index=numerator.index, dtype=float)


def _resolve_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    return None


# =========================================================
# Discretization
# =========================================================

def fit_quantile_bin_spec(
    df: pd.DataFrame,
    source_column: str,
    output_name: str,
    n_bins: int,
    train_index: Optional[Sequence[int]] = None,
) -> MultichannelBinSpec:
    if source_column not in df.columns:
        raise KeyError(f"Column not found for multichannel binning: {source_column}")

    train_idx = _safe_train_index(df, train_index)
    train_values = _safe_numeric(df.iloc[train_idx][source_column]).dropna()

    if train_values.empty:
        return MultichannelBinSpec(
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
        return MultichannelBinSpec(
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
        return MultichannelBinSpec(
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

    return MultichannelBinSpec(
        name=output_name,
        source_column=source_column,
        n_requested_bins=int(n_bins),
        edges=[float(x) for x in edges.tolist()],
        effective_bins=int(edges.size + 1),
        constant=False,
        fill_value=train_mean,
        train_min=train_min,
        train_max=train_max,
        train_mean=train_mean,
        train_std=train_std,
    )


def apply_bin_spec(
    df: pd.DataFrame,
    spec: MultichannelBinSpec,
) -> pd.Series:
    if spec.source_column not in df.columns:
        raise KeyError(f"Column not found while applying multichannel bin spec: {spec.source_column}")

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
    specs: Dict[str, MultichannelBinSpec],
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
# Secondary-channel resolution
# =========================================================

def resolve_secondary_delta_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_first_existing_column(
        df,
        [
            "delta_power_secondary",
            "pz_oz_delta_power",
            "delta_power_pz_oz",
            "delta_power_Pz_Oz",
            "EEG_Pz_Oz_delta_power",
            "eeg_pz_oz_delta_power",
        ],
    )


def resolve_secondary_alpha_column(df: pd.DataFrame) -> Optional[str]:
    return _resolve_first_existing_column(
        df,
        [
            "alpha_power_secondary",
            "pz_oz_alpha_power",
            "alpha_power_pz_oz",
            "alpha_power_Pz_Oz",
            "EEG_Pz_Oz_alpha_power",
            "eeg_pz_oz_alpha_power",
        ],
    )


def normalize_secondary_channel_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ensures canonical Phase III.2D columns exist:

        delta_power_secondary
        alpha_power_secondary

    If true secondary columns are unavailable, it falls back to primary columns
    and marks multichannel_available=False in the inventory.
    """
    _ensure_columns(
        df,
        ["delta_power", "alpha_power"],
        context="multichannel canonical columns",
    )

    out = df.copy()

    delta_source = resolve_secondary_delta_column(out)
    alpha_source = resolve_secondary_alpha_column(out)

    true_delta_available = delta_source is not None
    true_alpha_available = alpha_source is not None

    if delta_source is None:
        delta_source = "delta_power"

    if alpha_source is None:
        alpha_source = "alpha_power"

    out["delta_power_secondary"] = _safe_numeric(out[delta_source])
    out["alpha_power_secondary"] = _safe_numeric(out[alpha_source])

    inventory = {
        "delta_secondary_source_column": delta_source,
        "alpha_secondary_source_column": alpha_source,
        "true_delta_secondary_available": bool(true_delta_available),
        "true_alpha_secondary_available": bool(true_alpha_available),
        "multichannel_available": bool(true_delta_available and true_alpha_available),
    }

    return out, inventory


# =========================================================
# Multichannel features
# =========================================================

def add_interchannel_features(
    df: pd.DataFrame,
    train_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    _ensure_columns(
        df,
        [
            "delta_power",
            "alpha_power",
            "delta_power_secondary",
            "alpha_power_secondary",
        ],
        context="interchannel feature construction",
    )

    out = df.copy()
    train_idx = _safe_train_index(out, train_index)

    out["interchannel_delta_ratio"] = _safe_ratio(
        out["delta_power_secondary"],
        out["delta_power"],
    )

    out["interchannel_alpha_ratio"] = _safe_ratio(
        out["alpha_power_secondary"],
        out["alpha_power"],
    )

    z_delta_primary = _safe_zscore(out["delta_power"], out.iloc[train_idx]["delta_power"])
    z_alpha_primary = _safe_zscore(out["alpha_power"], out.iloc[train_idx]["alpha_power"])

    z_delta_secondary = _safe_zscore(
        out["delta_power_secondary"],
        out.iloc[train_idx]["delta_power_secondary"],
    )

    z_alpha_secondary = _safe_zscore(
        out["alpha_power_secondary"],
        out.iloc[train_idx]["alpha_power_secondary"],
    )

    out["fronto_parietal_delta_shift"] = z_delta_secondary - z_delta_primary
    out["fronto_parietal_alpha_shift"] = z_alpha_secondary - z_alpha_primary

    activation_stack = np.vstack(
        [
            z_delta_primary.to_numpy(dtype=float),
            z_alpha_primary.to_numpy(dtype=float),
            z_delta_secondary.to_numpy(dtype=float),
            z_alpha_secondary.to_numpy(dtype=float),
        ]
    )

    cross_stack = np.vstack(
        [
            _safe_zscore(out["interchannel_delta_ratio"], out.iloc[train_idx]["interchannel_delta_ratio"]).to_numpy(dtype=float),
            _safe_zscore(out["interchannel_alpha_ratio"], out.iloc[train_idx]["interchannel_alpha_ratio"]).to_numpy(dtype=float),
            _safe_zscore(out["fronto_parietal_delta_shift"], out.iloc[train_idx]["fronto_parietal_delta_shift"]).to_numpy(dtype=float),
            _safe_zscore(out["fronto_parietal_alpha_shift"], out.iloc[train_idx]["fronto_parietal_alpha_shift"]).to_numpy(dtype=float),
        ]
    )

    out["multichannel_activation_score"] = np.nanmean(activation_stack, axis=0)
    out["multichannel_cross_channel_score"] = np.nanmean(cross_stack, axis=0)

    combined_stack = np.vstack(
        [
            out["multichannel_activation_score"].to_numpy(dtype=float),
            out["multichannel_cross_channel_score"].to_numpy(dtype=float),
        ]
    )

    out["multichannel_info_score"] = np.nanmean(combined_stack, axis=0)

    return out


def add_multichannel_bins_and_states(
    df: pd.DataFrame,
    cfg: Phase32Config,
    train_index: Optional[Sequence[int]],
    specs: Dict[str, MultichannelBinSpec],
) -> pd.DataFrame:
    out = df.copy()

    # Individual bins for diagnostics.
    diagnostic_bins = [
        ("delta_power_secondary", "delta_power_secondary_bin"),
        ("alpha_power_secondary", "alpha_power_secondary_bin"),
        ("interchannel_delta_ratio", "interchannel_delta_ratio_bin"),
        ("interchannel_alpha_ratio", "interchannel_alpha_ratio_bin"),
        ("fronto_parietal_delta_shift", "fronto_parietal_delta_shift_bin"),
        ("fronto_parietal_alpha_shift", "fronto_parietal_alpha_shift_bin"),
    ]

    for source_col, output_col in diagnostic_bins:
        if source_col in out.columns:
            out = fit_and_apply_bin(
                df=out,
                source_column=source_col,
                output_name=output_col,
                n_bins=int(cfg.multichannel_n_bins),
                train_index=train_index,
                specs=specs,
            )

    # Compact 3 × 3 multichannel state.
    out = fit_and_apply_bin(
        df=out,
        source_column="multichannel_activation_score",
        output_name="multichannel_activation_bin",
        n_bins=int(cfg.multichannel_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="multichannel_cross_channel_score",
        output_name="multichannel_cross_channel_bin",
        n_bins=int(cfg.multichannel_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out = fit_and_apply_bin(
        df=out,
        source_column="multichannel_info_score",
        output_name=str(cfg.multichannel_info_bin_column),
        n_bins=int(cfg.multichannel_n_bins),
        train_index=train_index,
        specs=specs,
    )

    out[str(cfg.multichannel_state_column)] = (
        out["multichannel_activation_bin"].astype(int) * int(cfg.multichannel_n_bins)
        + out["multichannel_cross_channel_bin"].astype(int)
    ).astype(int)

    # Safety clamp: expected 0..8.
    out[str(cfg.multichannel_state_column)] = np.clip(
        out[str(cfg.multichannel_state_column)].astype(int),
        0,
        int(cfg.multichannel_n_states) - 1,
    )

    return out


def add_multichannel_next_state_columns(
    df: pd.DataFrame,
    cfg: Phase32Config,
) -> pd.DataFrame:
    _ensure_columns(
        df,
        [
            "subject_id",
            "recording_id",
            "epoch_idx",
            str(cfg.multichannel_state_column),
            str(cfg.multichannel_info_bin_column),
        ],
        context="multichannel next-state construction",
    )

    out = _sort_feature_frame(df.copy())

    state_col = str(cfg.multichannel_state_column)
    next_col = str(cfg.multichannel_next_state_column)
    info_col = str(cfg.multichannel_info_bin_column)

    out[next_col] = np.nan
    out[f"{info_col}_next"] = np.nan
    out["valid_multichannel_next_state"] = 0

    group_cols = ["subject_id", "recording_id"]

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        state_next = out.loc[idx_arr, state_col].shift(-1)
        info_next = out.loc[idx_arr, info_col].shift(-1)

        valid = state_next.notna()

        out.loc[idx_arr, next_col] = state_next.to_numpy()
        out.loc[idx_arr, f"{info_col}_next"] = info_next.to_numpy()
        out.loc[idx_arr, "valid_multichannel_next_state"] = valid.astype(int).to_numpy()

    out[next_col] = out[next_col].astype("Int64")
    out[f"{info_col}_next"] = out[f"{info_col}_next"].astype("Int64")

    return out


def add_multichannel_validity_flags(
    df: pd.DataFrame,
    cfg: Phase32Config,
) -> pd.DataFrame:
    out = df.copy()

    next_col = str(cfg.multichannel_next_state_column)

    if "valid_next_state" in out.columns:
        base_valid = pd.to_numeric(out["valid_next_state"], errors="coerce").fillna(0).astype(int)
    else:
        base_valid = pd.Series(np.ones(len(out), dtype=int), index=out.index)

    mc_valid = pd.to_numeric(
        out.get("valid_multichannel_next_state", 0),
        errors="coerce",
    ).fillna(0).astype(int)

    has_next = out[next_col].notna().astype(int) if next_col in out.columns else 0

    out["valid_multichannel_conditional_row"] = (
        (base_valid == 1)
        & (mc_valid == 1)
        & (has_next == 1)
    ).astype(int)

    return out


# =========================================================
# Inventory / diagnostics
# =========================================================

def build_multichannel_inventory(
    df: pd.DataFrame,
    cfg: Phase32Config,
    source_inventory: Dict[str, Any],
    bin_specs: Dict[str, MultichannelBinSpec],
) -> Dict[str, Any]:
    state_col = str(cfg.multichannel_state_column)
    next_col = str(cfg.multichannel_next_state_column)
    info_col = str(cfg.multichannel_info_bin_column)

    inventory: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "module": "Phase_III_2D_multichannel_EEG_enrichment",
        "version": cfg.version,
        "enabled": bool(cfg.run_multichannel_analysis and cfg.use_multichannel_eeg),
        "source_columns": source_inventory,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "subjects": sorted(df["subject_id"].astype(str).unique().tolist()) if "subject_id" in df.columns else [],
        "state_column": state_col,
        "next_state_column": next_col,
        "info_bin_column": info_col,
        "state_n_unique": int(df[state_col].nunique(dropna=True)) if state_col in df.columns else 0,
        "next_state_n_unique": int(df[next_col].nunique(dropna=True)) if next_col in df.columns else 0,
        "info_bin_n_unique": int(df[info_col].nunique(dropna=True)) if info_col in df.columns else 0,
        "valid_multichannel_next_state_rows": int(df["valid_multichannel_next_state"].sum()) if "valid_multichannel_next_state" in df.columns else 0,
        "valid_multichannel_conditional_rows": int(df["valid_multichannel_conditional_row"].sum()) if "valid_multichannel_conditional_row" in df.columns else 0,
        "bin_specs": {name: asdict(spec) for name, spec in bin_specs.items()},
        "density": {},
        "warnings": [],
    }

    if state_col in df.columns:
        valid_rows = df[df.get("valid_multichannel_conditional_row", 0) == 1].copy()

        n_valid = int(len(valid_rows))
        n_states = int(valid_rows[state_col].nunique(dropna=True)) if not valid_rows.empty else 0

        transitions_per_state = float(n_valid / max(n_states, 1)) if n_states > 0 else 0.0

        inventory["density"] = {
            "n_valid_rows": n_valid,
            "n_states": n_states,
            "transitions_per_state": transitions_per_state,
            "minimum_required": float(cfg.multichannel_density_min_transitions_per_state),
            "recommended": float(cfg.multichannel_density_recommended_transitions_per_state),
        }

        if transitions_per_state < float(cfg.multichannel_density_min_transitions_per_state):
            inventory["warnings"].append(
                "multichannel_state_density_below_minimum"
            )

    if not source_inventory.get("multichannel_available", False):
        inventory["warnings"].append(
            "true_secondary_channel_features_not_detected_using_primary_fallback"
        )

    if inventory["state_n_unique"] > int(cfg.multichannel_state_max_unique_values):
        inventory["warnings"].append(
            "multichannel_state_exceeds_max_unique_values"
        )

    return inventory


def summarize_multichannel_features(df: pd.DataFrame, cfg: Phase32Config) -> Dict[str, Any]:
    state_col = str(cfg.multichannel_state_column)
    next_col = str(cfg.multichannel_next_state_column)
    info_col = str(cfg.multichannel_info_bin_column)

    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "state_column": state_col,
        "next_state_column": next_col,
        "info_bin_column": info_col,
    }

    if "subject_id" in df.columns:
        summary["n_subjects"] = int(df["subject_id"].nunique())
        summary["subjects"] = sorted(df["subject_id"].astype(str).unique().tolist())

    for col in [
        "delta_power_secondary",
        "alpha_power_secondary",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
        "multichannel_activation_score",
        "multichannel_cross_channel_score",
        "multichannel_info_score",
        state_col,
        next_col,
        info_col,
        "valid_multichannel_next_state",
        "valid_multichannel_conditional_row",
    ]:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = {
                    "min": float(pd.to_numeric(df[col], errors="coerce").min()),
                    "max": float(pd.to_numeric(df[col], errors="coerce").max()),
                    "mean": float(pd.to_numeric(df[col], errors="coerce").mean()),
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "non_null": int(df[col].notna().sum()),
                }
            else:
                summary[col] = {
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "non_null": int(df[col].notna().sum()),
                }

    if "sleep_stage" in df.columns and "subject_id" in df.columns:
        summary["sleep_stage_counts"] = (
            df.groupby(["subject_id", "sleep_stage"])
            .size()
            .reset_index(name="n_epochs")
            .to_dict(orient="records")
        )

    return summary


# =========================================================
# Main API
# =========================================================

def add_phase3_2d_multichannel_features(
    df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
    train_index: Optional[Sequence[int]] = None,
) -> MultichannelFeatureResult:
    if cfg is None:
        cfg = load_phase3_2_config()

    out = _sort_feature_frame(df.copy())

    required = [
        "subject_id",
        "recording_id",
        "epoch_idx",
        "delta_power",
        "alpha_power",
    ]

    _ensure_columns(out, required, context="Phase III.2D multichannel base input")

    specs: Dict[str, MultichannelBinSpec] = {}

    out, source_inventory = normalize_secondary_channel_columns(out)

    out = add_interchannel_features(
        df=out,
        train_index=train_index,
    )

    out = add_multichannel_bins_and_states(
        df=out,
        cfg=cfg,
        train_index=train_index,
        specs=specs,
    )

    out = add_multichannel_next_state_columns(
        df=out,
        cfg=cfg,
    )

    out = add_multichannel_validity_flags(
        df=out,
        cfg=cfg,
    )

    inventory = build_multichannel_inventory(
        df=out,
        cfg=cfg,
        source_inventory=source_inventory,
        bin_specs=specs,
    )

    return MultichannelFeatureResult(
        df=out,
        bin_specs=specs,
        inventory=inventory,
    )


def save_multichannel_outputs(
    result: MultichannelFeatureResult,
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    features_path = Path(cfg.multichannel_features_file)
    inventory_path = Path(cfg.multichannel_inventory_json)
    summary_path = Path(cfg.multichannel_summary_json)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.df.to_csv(features_path, index=False)

    save_json(inventory_path, result.inventory)

    save_json(
        summary_path,
        {
            "phase": cfg.phase_name,
            "module": "Phase_III_2D_multichannel_EEG_enrichment",
            "summary": summarize_multichannel_features(result.df, cfg),
            "inventory_file": str(inventory_path),
            "features_file": str(features_path),
        },
    )

    print(f"[Phase3.2D Multichannel] Saved features: {features_path}")
    print(f"[Phase3.2D Multichannel] Saved inventory: {inventory_path}")
    print(f"[Phase3.2D Multichannel] Saved summary: {summary_path}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    input_path = Path(cfg.interim_dir) / "phase3_2_modeling_features.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            "Modeling features file not found. Run first:\n"
            "python -m src.phase3_2_features"
        )

    df = pd.read_csv(input_path)

    result = add_phase3_2d_multichannel_features(
        df=df,
        cfg=cfg,
        train_index=None,
    )

    save_multichannel_outputs(result, cfg)

    summary = summarize_multichannel_features(result.df, cfg)
    inventory = result.inventory

    print("\n============================================================")
    print("Phase III.2D multichannel feature enrichment completed")
    print("============================================================")
    print(f"Rows: {summary.get('n_rows')}")
    print(f"Subjects: {summary.get('subjects')}")
    print(f"Multichannel available: {inventory.get('source_columns', {}).get('multichannel_available')}")
    print(f"Secondary delta source: {inventory.get('source_columns', {}).get('delta_secondary_source_column')}")
    print(f"Secondary alpha source: {inventory.get('source_columns', {}).get('alpha_secondary_source_column')}")
    print(f"MC state unique: {inventory.get('state_n_unique')}")
    print(f"MC next-state valid rows: {inventory.get('valid_multichannel_next_state_rows')}")
    print(f"MC conditional valid rows: {inventory.get('valid_multichannel_conditional_rows')}")
    print(f"Warnings: {inventory.get('warnings')}")
    print("============================================================\n")


if __name__ == "__main__":
    main()