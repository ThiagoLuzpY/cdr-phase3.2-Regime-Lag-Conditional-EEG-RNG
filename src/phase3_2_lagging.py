from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import Phase32Config, RegimeSpec, load_phase3_2_config
from src.phase3_2_regimes import (
    RegimeFrame,
    build_all_regime_frames,
    build_regime_inventory,
    load_or_build_modeling_features,
    save_json,
)


# =========================================================
# Phase III.2B — Lagged EEG-RNG alignment
# =========================================================
#
# Convention:
#
#   lag > 0:
#       EEG_t aligned with RNG_{t+lag}
#
#   lag < 0:
#       EEG_t aligned with RNG_{t-|lag|}
#
#   lag = 0:
#       EEG_t aligned with RNG_t
#
# Interpretation:
#
#   This is temporal alignment sensitivity.
#   It is NOT a causal claim.
#
# This module never allows lagging to cross subject_id or recording_id.


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class LaggedFrame:
    """
    Container for one regime + lag dataframe.
    """

    regime_name: str
    lag_epochs: int
    lag_label: str
    df: pd.DataFrame
    valid: bool
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

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def _sort_frame(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in df.columns]

    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)

    return df.reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default

        return int(value)

    except Exception:
        return default


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


def lag_label(lag_epochs: int) -> str:
    """
    Converts lag integer to stable filename-safe label.
    """
    lag_epochs = int(lag_epochs)

    if lag_epochs < 0:
        return f"lag_m{abs(lag_epochs)}"

    if lag_epochs > 0:
        return f"lag_p{lag_epochs}"

    return "lag_0"


# =========================================================
# Lag columns
# =========================================================

def rng_alignment_source_columns(df: pd.DataFrame) -> List[str]:
    """
    RNG-side columns that can be shifted/aligned to EEG epochs.

    We shift only RNG/proxy-side variables. EEG-side columns stay fixed.
    """
    candidates = [
        # Core RNG state
        "rng_state",
        "rng_bit",
        "rng_next_bit_raw",

        # RNG informational / proxy bins
        "rng_info_bin",
        "q_rng_bin",

        # RNG raw alignment fields
        "rng_window_id",
        "rng_window_start",
        "rng_window_end",

        # RNG local metrics
        "bit_balance_local",
        "transition_rate",
        "rng_entropy_local",
        "entropy_rate_proxy",
        "run_length_mean",
        "run_length_max",
        "run_length_std",
        "compressibility_proxy",
        "surprise_index",
        "transition_asymmetry",
        "micro_cluster_deviation",
        "q_rng_score",

        # Joint/proxy states that depend partly on RNG
        "informational_joint_state",
        "latent_q_state",
    ]

    return [c for c in candidates if c in df.columns]


def aligned_column_name(source_col: str) -> str:
    """
    Standard suffix for lag-aligned RNG-side columns.
    """
    return f"{source_col}_aligned"


# =========================================================
# Core lagging logic
# =========================================================

def apply_rng_lag_to_frame(
    df: pd.DataFrame,
    lag_epochs: int,
    cfg: Optional[Phase32Config] = None,
) -> pd.DataFrame:
    """
    Applies RNG-side lag alignment inside each subject/recording.

    The EEG timeline remains fixed. RNG-side variables are shifted according
    to lag_epochs and written to *_aligned columns.

    Important:
        pandas shift convention:
            shift(-lag) gives row t value from row t+lag.

        Therefore:
            lag = +1 -> row t receives RNG_{t+1}
            lag = -1 -> row t receives RNG_{t-1}
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    required = [
        "subject_id",
        "recording_id",
        "epoch_idx",
        "eeg_state",
        "rng_state",
        "eeg_next_state",
        "valid_next_state",
    ]

    _require_columns(df, required, context="lagged alignment")

    out = _sort_frame(df.copy())

    lag_epochs = int(lag_epochs)
    shift_periods = -lag_epochs

    rng_cols = rng_alignment_source_columns(out)

    if "rng_state" not in rng_cols:
        raise KeyError("rng_state is required for lag alignment.")

    group_cols = ["subject_id", "recording_id"]

    for col in rng_cols:
        aligned_col = aligned_column_name(col)
        out[aligned_col] = np.nan

    out["lag_epochs"] = lag_epochs
    out["lag_label"] = lag_label(lag_epochs)
    out["valid_lag_alignment"] = 0

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        valid_mask = np.ones(len(idx_arr), dtype=bool)

        for col in rng_cols:
            aligned_col = aligned_column_name(col)

            shifted = out.loc[idx_arr, col].shift(shift_periods)
            out.loc[idx_arr, aligned_col] = shifted.to_numpy()

            valid_mask &= shifted.notna().to_numpy()

        out.loc[idx_arr, "valid_lag_alignment"] = valid_mask.astype(int)

    # Standard model-facing aliases.
    out["rng_state_model"] = out["rng_state_aligned"]
    out["rng_bit_model"] = out["rng_bit_aligned"] if "rng_bit_aligned" in out.columns else out["rng_state_aligned"]

    if "rng_info_bin_aligned" in out.columns:
        out["rng_info_bin_model"] = out["rng_info_bin_aligned"]

    if "q_rng_bin_aligned" in out.columns:
        out["q_rng_bin_model"] = out["q_rng_bin_aligned"]

    # Rebuild lag-aware joint state using EEG_t and RNG_{t+lag}.
    out["observed_joint_state_lagged"] = np.nan

    valid_joint = out["rng_state_model"].notna()

    out.loc[valid_joint, "observed_joint_state_lagged"] = (
        out.loc[valid_joint, "eeg_state"].astype(int) * int(cfg.rng_n_bins)
        + out.loc[valid_joint, "rng_state_model"].astype(int)
    )

    if "eeg_info_bin" in out.columns and "rng_info_bin_model" in out.columns:
        out["informational_joint_state_lagged"] = np.nan

        valid_info = out["rng_info_bin_model"].notna()

        out.loc[valid_info, "informational_joint_state_lagged"] = (
            out.loc[valid_info, "eeg_info_bin"].astype(int) * int(cfg.info_n_bins)
            + out.loc[valid_info, "rng_info_bin_model"].astype(int)
        )

    if "latent_state" in out.columns and "q_rng_bin_model" in out.columns:
        out["latent_q_state_lagged"] = np.nan

        valid_q = out["q_rng_bin_model"].notna()

        out.loc[valid_q, "latent_q_state_lagged"] = (
            out.loc[valid_q, "latent_state"].astype(int) * int(cfg.q_n_bins)
            + out.loc[valid_q, "q_rng_bin_model"].astype(int)
        )

    # Build RNG next-state target after alignment, without crossing boundaries.
    out["rng_next_state_model"] = np.nan
    out["valid_rng_next_state_model"] = 0

    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        idx_arr = np.asarray(list(idx), dtype=int)

        shifted_next = out.loc[idx_arr, "rng_state_model"].shift(-1)
        valid_next = shifted_next.notna() & out.loc[idx_arr, "rng_state_model"].notna()

        out.loc[idx_arr, "rng_next_state_model"] = shifted_next.to_numpy()
        out.loc[idx_arr, "valid_rng_next_state_model"] = valid_next.astype(int).to_numpy()

    # Validity flags for later conditional models.
    out["valid_eeg_conditional_row"] = (
        (out["valid_next_state"].astype(int) == 1)
        & (out["valid_lag_alignment"].astype(int) == 1)
        & out["eeg_next_state"].notna()
        & out["rng_state_model"].notna()
    ).astype(int)

    out["valid_rng_conditional_row"] = (
        (out["valid_lag_alignment"].astype(int) == 1)
        & (out["valid_rng_next_state_model"].astype(int) == 1)
        & out["rng_next_state_model"].notna()
        & out["rng_state_model"].notna()
    ).astype(int)

    out["valid_conditional_row"] = (
        (out["valid_eeg_conditional_row"].astype(int) == 1)
        & (out["valid_rng_conditional_row"].astype(int) == 1)
    ).astype(int)

    # Cast common state columns.
    for col in [
        "rng_state_model",
        "rng_bit_model",
        "rng_next_state_model",
        "observed_joint_state_lagged",
        "informational_joint_state_lagged",
        "latent_q_state_lagged",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return _sort_frame(out)


# =========================================================
# Validation and metadata
# =========================================================

def subject_lag_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts valid lag/conditional rows by subject.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "n_rows",
                "valid_lag_alignment",
                "valid_eeg_conditional_row",
                "valid_rng_conditional_row",
                "valid_conditional_row",
            ]
        )

    rows: List[Dict[str, Any]] = []

    for subject_id, group in df.groupby("subject_id", sort=True):
        row: Dict[str, Any] = {
            "subject_id": str(subject_id),
            "n_rows": int(len(group)),
            "valid_lag_alignment": int(pd.to_numeric(group["valid_lag_alignment"], errors="coerce").fillna(0).sum()),
            "valid_eeg_conditional_row": int(pd.to_numeric(group["valid_eeg_conditional_row"], errors="coerce").fillna(0).sum()),
            "valid_rng_conditional_row": int(pd.to_numeric(group["valid_rng_conditional_row"], errors="coerce").fillna(0).sum()),
            "valid_conditional_row": int(pd.to_numeric(group["valid_conditional_row"], errors="coerce").fillna(0).sum()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def validate_lagged_frame(
    df: pd.DataFrame,
    regime_spec: RegimeSpec,
    lag_epochs: int,
    cfg: Optional[Phase32Config] = None,
) -> Dict[str, Any]:
    """
    Validates one lagged frame.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    if df.empty:
        return {
            "valid": False,
            "reason": "empty_lagged_frame",
            "lag_epochs": int(lag_epochs),
            "total_rows": 0,
            "valid_lag_alignment_rows": 0,
            "valid_eeg_conditional_rows": 0,
            "valid_rng_conditional_rows": 0,
            "valid_conditional_rows": 0,
            "n_subjects": 0,
            "n_valid_subjects": 0,
            "subject_counts": [],
        }

    counts = subject_lag_counts(df)

    min_rows_per_subject = max(
        10,
        int(min(regime_spec.min_epochs_per_subject, 50) * 0.5),
    )

    valid_subjects_df = counts[
        counts["valid_eeg_conditional_row"].astype(int) >= int(min_rows_per_subject)
    ].copy()

    n_subjects = int(counts["subject_id"].nunique())
    n_valid_subjects = int(valid_subjects_df["subject_id"].nunique())

    total_rows = int(len(df))
    valid_lag_alignment_rows = int(df["valid_lag_alignment"].astype(int).sum())
    valid_eeg_conditional_rows = int(df["valid_eeg_conditional_row"].astype(int).sum())
    valid_rng_conditional_rows = int(df["valid_rng_conditional_row"].astype(int).sum())
    valid_conditional_rows = int(df["valid_conditional_row"].astype(int).sum())

    failures: List[str] = []

    if n_valid_subjects < int(max(cfg.min_subjects_for_loso, 3)):
        failures.append(f"n_valid_subjects<{max(cfg.min_subjects_for_loso, 3)}")

    if valid_eeg_conditional_rows < int(cfg.min_transitions_total):
        failures.append(f"valid_eeg_conditional_rows<{cfg.min_transitions_total}")

    if valid_rng_conditional_rows < int(cfg.min_transitions_total):
        failures.append(f"valid_rng_conditional_rows<{cfg.min_transitions_total}")

    valid = len(failures) == 0

    metadata: Dict[str, Any] = {
        "valid": valid,
        "reason": "ok" if valid else ";".join(failures),
        "lag_epochs": int(lag_epochs),
        "lag_label": lag_label(int(lag_epochs)),
        "total_rows": total_rows,
        "valid_lag_alignment_rows": valid_lag_alignment_rows,
        "valid_eeg_conditional_rows": valid_eeg_conditional_rows,
        "valid_rng_conditional_rows": valid_rng_conditional_rows,
        "valid_conditional_rows": valid_conditional_rows,
        "n_subjects": n_subjects,
        "n_valid_subjects": n_valid_subjects,
        "min_rows_per_subject": int(min_rows_per_subject),
        "valid_subjects": sorted(valid_subjects_df["subject_id"].astype(str).tolist()),
        "excluded_subjects": sorted(
            set(counts["subject_id"].astype(str).tolist())
            - set(valid_subjects_df["subject_id"].astype(str).tolist())
        ),
        "subject_counts": counts.to_dict(orient="records"),
    }

    for col in [
        "observed_joint_state_lagged",
        "informational_joint_state_lagged",
        "latent_q_state_lagged",
        "rng_state_model",
    ]:
        if col in df.columns:
            metadata[f"{col}_n_unique"] = int(df[col].nunique(dropna=True))

    return metadata


def keep_valid_lag_subjects_only(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    valid_subjects = metadata.get("valid_subjects", [])

    if not valid_subjects:
        return df.iloc[0:0].copy()

    out = df[df["subject_id"].astype(str).isin(set(valid_subjects))].copy()
    return _sort_frame(out)


# =========================================================
# Building lagged frames
# =========================================================

def build_lagged_frame(
    regime_frame: RegimeFrame,
    lag_epochs: int,
    cfg: Optional[Phase32Config] = None,
    keep_only_valid_subjects: bool = True,
) -> LaggedFrame:
    """
    Builds one lagged frame for one regime and lag.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    lagged = apply_rng_lag_to_frame(
        df=regime_frame.df,
        lag_epochs=int(lag_epochs),
        cfg=cfg,
    )

    metadata = validate_lagged_frame(
        df=lagged,
        regime_spec=regime_frame.spec,
        lag_epochs=int(lag_epochs),
        cfg=cfg,
    )

    if keep_only_valid_subjects and metadata.get("valid_subjects"):
        lagged_valid = keep_valid_lag_subjects_only(
            df=lagged,
            metadata=metadata,
        )

        metadata_after = validate_lagged_frame(
            df=lagged_valid,
            regime_spec=regime_frame.spec,
            lag_epochs=int(lag_epochs),
            cfg=cfg,
        )

        metadata_after["before_subject_filter"] = metadata
        metadata = metadata_after
        lagged = lagged_valid

    metadata["regime_name"] = regime_frame.name
    metadata["regime_valid"] = bool(regime_frame.valid)
    metadata["regime_primary"] = bool(regime_frame.spec.primary)
    metadata["regime_spec"] = asdict(regime_frame.spec)

    return LaggedFrame(
        regime_name=regime_frame.name,
        lag_epochs=int(lag_epochs),
        lag_label=lag_label(int(lag_epochs)),
        df=lagged,
        valid=bool(metadata.get("valid", False)),
        metadata=metadata,
    )


def build_lagged_frames_for_regime(
    regime_frame: RegimeFrame,
    cfg: Optional[Phase32Config] = None,
    keep_only_valid_subjects: bool = True,
) -> Dict[int, LaggedFrame]:
    """
    Builds all configured lags for one regime.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    frames: Dict[int, LaggedFrame] = {}

    for lag_value in cfg.lags_epochs:
        print(
            f"[Phase3.2 Lagging] Building lagged frame: "
            f"regime={regime_frame.name} lag={lag_value}"
        )

        frame = build_lagged_frame(
            regime_frame=regime_frame,
            lag_epochs=int(lag_value),
            cfg=cfg,
            keep_only_valid_subjects=keep_only_valid_subjects,
        )

        frames[int(lag_value)] = frame

        print(
            f"[Phase3.2 Lagging] regime={regime_frame.name} "
            f"lag={lag_value} valid={frame.valid} "
            f"rows={len(frame.df)} "
            f"subjects={frame.metadata.get('n_valid_subjects')} "
            f"reason={frame.metadata.get('reason')}"
        )

    return frames


def build_all_lagged_frames(
    regime_frames: Dict[str, RegimeFrame],
    cfg: Optional[Phase32Config] = None,
    keep_only_valid_subjects: bool = True,
    valid_regimes_only: bool = True,
) -> Dict[str, Dict[int, LaggedFrame]]:
    """
    Builds all lagged frames for all regimes.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    all_frames: Dict[str, Dict[int, LaggedFrame]] = {}

    for regime_name, regime_frame in regime_frames.items():
        if valid_regimes_only and not regime_frame.valid:
            print(f"[Phase3.2 Lagging] Skipping invalid regime: {regime_name}")
            continue

        all_frames[regime_name] = build_lagged_frames_for_regime(
            regime_frame=regime_frame,
            cfg=cfg,
            keep_only_valid_subjects=keep_only_valid_subjects,
        )

    return all_frames


# =========================================================
# Inventory and output
# =========================================================

def lag_inventory_row(frame: LaggedFrame) -> Dict[str, Any]:
    m = frame.metadata

    return {
        "regime": frame.regime_name,
        "lag_epochs": int(frame.lag_epochs),
        "lag_label": frame.lag_label,
        "valid": bool(m.get("valid", False)),
        "reason": str(m.get("reason", "")),
        "regime_primary": bool(m.get("regime_primary", False)),
        "total_rows": _safe_int(m.get("total_rows")),
        "valid_lag_alignment_rows": _safe_int(m.get("valid_lag_alignment_rows")),
        "valid_eeg_conditional_rows": _safe_int(m.get("valid_eeg_conditional_rows")),
        "valid_rng_conditional_rows": _safe_int(m.get("valid_rng_conditional_rows")),
        "valid_conditional_rows": _safe_int(m.get("valid_conditional_rows")),
        "n_subjects": _safe_int(m.get("n_subjects")),
        "n_valid_subjects": _safe_int(m.get("n_valid_subjects")),
        "observed_joint_state_lagged_n_unique": _safe_int(m.get("observed_joint_state_lagged_n_unique")),
        "informational_joint_state_lagged_n_unique": _safe_int(m.get("informational_joint_state_lagged_n_unique")),
        "latent_q_state_lagged_n_unique": _safe_int(m.get("latent_q_state_lagged_n_unique")),
        "rng_state_model_n_unique": _safe_int(m.get("rng_state_model_n_unique")),
        "valid_subjects": ",".join(m.get("valid_subjects", [])),
        "excluded_subjects": ",".join(m.get("excluded_subjects", [])),
    }


def build_lag_inventory(
    lagged_frames: Dict[str, Dict[int, LaggedFrame]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for regime_frames in lagged_frames.values():
        for frame in regime_frames.values():
            rows.append(lag_inventory_row(frame))

    if not rows:
        return pd.DataFrame()

    inventory = pd.DataFrame(rows)
    inventory = inventory.sort_values(["regime", "lag_epochs"]).reset_index(drop=True)

    return inventory


def save_lagged_outputs(
    lagged_frames: Dict[str, Dict[int, LaggedFrame]],
    cfg: Optional[Phase32Config] = None,
    save_valid_csvs: bool = True,
) -> None:
    """
    Saves:
        - lag inventory CSV
        - lag summary JSON
        - valid lagged CSVs by regime/lag
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    output_dir = Path(cfg.interim_dir) / "lagged"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_lag_inventory(lagged_frames)

    inventory_csv = Path(cfg.interim_dir) / "phase3_2_lag_inventory.csv"
    summary_json = Path(cfg.interim_dir) / "phase3_2_lag_summary.json"

    inventory.to_csv(inventory_csv, index=False)

    summary_payload: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "lags_epochs": list(cfg.lags_epochs),
        "n_lagged_frames": int(len(inventory)),
        "valid_frames": (
            inventory[inventory["valid"] == True][["regime", "lag_epochs", "lag_label"]]
            .to_dict(orient="records")
            if not inventory.empty and "valid" in inventory.columns
            else []
        ),
        "invalid_frames": (
            inventory[inventory["valid"] == False][["regime", "lag_epochs", "lag_label", "reason"]]
            .to_dict(orient="records")
            if not inventory.empty and "valid" in inventory.columns
            else []
        ),
        "inventory_file": str(inventory_csv),
        "frames": {
            regime: {
                str(lag): frame.metadata
                for lag, frame in frames.items()
            }
            for regime, frames in lagged_frames.items()
        },
    }

    save_json(summary_json, summary_payload)

    if save_valid_csvs:
        for regime, frames in lagged_frames.items():
            for lag_value, frame in frames.items():
                if frame.valid:
                    output_file = output_dir / (
                        f"phase3_2b_{regime}_{frame.lag_label}.csv"
                    )
                    frame.df.to_csv(output_file, index=False)

    print(f"[Phase3.2 Lagging] Saved lag inventory: {inventory_csv}")
    print(f"[Phase3.2 Lagging] Saved lag summary: {summary_json}")
    print(f"[Phase3.2 Lagging] Saved valid lagged CSVs in: {output_dir}")


# =========================================================
# Summary text
# =========================================================

def build_lag_summary_text(
    lagged_frames: Dict[str, Dict[int, LaggedFrame]],
) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2B — Lagged EEG-RNG alignment")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Convention:")
    lines.append("  lag > 0: EEG_t aligned with RNG_{t+lag}")
    lines.append("  lag < 0: EEG_t aligned with RNG_{t-|lag|}")
    lines.append("  lag = 0: EEG_t aligned with RNG_t")
    lines.append("")
    lines.append("Interpretation: temporal alignment sensitivity, not causality.")
    lines.append("")

    for regime, frames in lagged_frames.items():
        lines.append(f"Regime: {regime}")
        lines.append("-" * 78)

        for lag_value in sorted(frames.keys()):
            frame = frames[lag_value]
            m = frame.metadata

            lines.append(
                f"{frame.lag_label}: "
                f"valid={frame.valid} | "
                f"rows={m.get('total_rows')} | "
                f"valid_eeg_rows={m.get('valid_eeg_conditional_rows')} | "
                f"valid_rng_rows={m.get('valid_rng_conditional_rows')} | "
                f"subjects={m.get('n_valid_subjects')} | "
                f"reason={m.get('reason')}"
            )

        lines.append("")

    lines.append("=" * 78)

    return "\n".join(lines)


def save_lag_summary_text(
    lagged_frames: Dict[str, Dict[int, LaggedFrame]],
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    path = Path(cfg.interim_dir) / "phase3_2_lag_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(build_lag_summary_text(lagged_frames))

    print(f"[Phase3.2 Lagging] Saved lag text summary: {path}")


# =========================================================
# Convenience loaders for downstream modules
# =========================================================

def lagged_csv_path(
    regime_name: str,
    lag_epochs: int,
    cfg: Optional[Phase32Config] = None,
) -> Path:
    if cfg is None:
        cfg = load_phase3_2_config()

    return Path(cfg.interim_dir) / "lagged" / (
        f"phase3_2b_{regime_name}_{lag_label(lag_epochs)}.csv"
    )


def load_lagged_frame_csv(
    regime_name: str,
    lag_epochs: int,
    cfg: Optional[Phase32Config] = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_phase3_2_config()

    path = lagged_csv_path(regime_name, lag_epochs, cfg)

    if not path.exists():
        raise FileNotFoundError(f"Lagged CSV not found: {path}")

    return _sort_frame(pd.read_csv(path))


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    df = load_or_build_modeling_features(
        cfg=cfg,
        force_rebuild=False,
    )

    regime_frames = build_all_regime_frames(
        df=df,
        cfg=cfg,
        keep_only_valid_subjects=True,
    )

    lagged_frames = build_all_lagged_frames(
        regime_frames=regime_frames,
        cfg=cfg,
        keep_only_valid_subjects=True,
        valid_regimes_only=True,
    )

    save_lagged_outputs(
        lagged_frames=lagged_frames,
        cfg=cfg,
        save_valid_csvs=True,
    )

    save_lag_summary_text(
        lagged_frames=lagged_frames,
        cfg=cfg,
    )

    inventory = build_lag_inventory(lagged_frames)

    print("\n============================================================")
    print("Phase III.2B lagged alignment completed")
    print("============================================================")

    if inventory.empty:
        print("No lagged frames generated.")
    else:
        display_cols = [
            "regime",
            "lag_epochs",
            "valid",
            "total_rows",
            "valid_eeg_conditional_rows",
            "valid_rng_conditional_rows",
            "n_valid_subjects",
            "reason",
        ]
        print(inventory[display_cols])

    print("============================================================\n")


if __name__ == "__main__":
    main()