from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config.phase3_2_config import (
    Phase32Config,
    RegimeSpec,
    load_phase3_2_config,
    regime_map,
)

from src.phase3_2_features import (
    prepare_phase3_2_features_for_modeling,
    save_json,
)
from src.phase3_2_loader import load_or_build_phase3_2_dataset


# =========================================================
# Phase III.2A — Regime-aware / sleep-stage analysis
# =========================================================
#
# This module defines and validates the Phase III.2 sleep-stage regimes:
#
#   full
#   no_wake
#   stable_sleep = N2 + N3
#   deep_sleep = N3
#   rem_only = REM
#   transition_epochs
#
# It does not estimate CDR yet.
# It prepares clean regime-specific dataframes and metadata for:
#
#   III.2B — lagged EEG-RNG alignment
#   III.2C — conditional CDR
#   III.2D — multi-channel enrichment


# =========================================================
# Dataclasses
# =========================================================

@dataclass
class RegimeFrame:
    """
    Container for one regime-specific dataframe and its validation metadata.
    """

    name: str
    spec: RegimeSpec
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

    if isinstance(obj, RegimeSpec):
        return asdict(obj)

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


# =========================================================
# Loading modeling features
# =========================================================

def modeling_features_path(cfg: Phase32Config) -> Path:
    return Path(cfg.interim_dir) / "phase3_2_modeling_features.csv"


def feature_specs_path(cfg: Phase32Config) -> Path:
    return Path(cfg.interim_dir) / "phase3_2_feature_specs.json"


def load_or_build_modeling_features(
    cfg: Optional[Phase32Config] = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Loads phase3_2_modeling_features.csv if available.

    If missing, it rebuilds:
        loader -> feature preparation -> save modeling features
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    path = modeling_features_path(cfg)

    if path.exists() and not force_rebuild:
        print(f"[Phase3.2 Regimes] Loading modeling features: {path}")
        return _sort_frame(pd.read_csv(path))

    print("[Phase3.2 Regimes] Modeling features not found or force_rebuild=True.")
    print("[Phase3.2 Regimes] Rebuilding from loader + features...")

    base_df, _ = load_or_build_phase3_2_dataset(
        cfg=cfg,
        force_rebuild=False,
    )

    feature_frame = prepare_phase3_2_features_for_modeling(
        df=base_df,
        cfg=cfg,
        train_index=None,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.df.to_csv(path, index=False)

    save_json(
        feature_specs_path(cfg),
        {
            "metadata": feature_frame.metadata,
            "bin_specs": {
                name: asdict(spec)
                for name, spec in feature_frame.bin_specs.items()
            },
        },
    )

    return _sort_frame(feature_frame.df)


# =========================================================
# Regime filtering
# =========================================================

def filter_regular_regime(
    df: pd.DataFrame,
    spec: RegimeSpec,
) -> pd.DataFrame:
    """
    Filters a dataframe by sleep_stage membership.
    """
    _require_columns(df, ["sleep_stage"], context=f"regime {spec.name}")

    if not spec.stages:
        return df.iloc[0:0].copy()

    out = df[df["sleep_stage"].astype(str).isin(spec.stages)].copy()
    return _sort_frame(out)


def filter_transition_regime(
    df: pd.DataFrame,
    cfg: Phase32Config,
    spec: RegimeSpec,
) -> pd.DataFrame:
    """
    Filters epochs marked as transition epochs.

    The flag is generated in phase3_2_features.py as:
        is_transition_epoch
    """
    _require_columns(df, ["is_transition_epoch"], context="transition regime")

    out = df[df["is_transition_epoch"].astype(int) == 1].copy()
    return _sort_frame(out)


def filter_regime(
    df: pd.DataFrame,
    regime_name: str,
    cfg: Optional[Phase32Config] = None,
) -> pd.DataFrame:
    """
    Main regime filtering entrypoint.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    regimes = regime_map(cfg)

    if regime_name not in regimes:
        raise KeyError(
            f"Unknown regime: {regime_name}. "
            f"Available regimes: {list(regimes.keys())}"
        )

    spec = regimes[regime_name]

    if regime_name == cfg.transition_regime_name:
        return filter_transition_regime(df, cfg, spec)

    return filter_regular_regime(df, spec)


# =========================================================
# Regime validation
# =========================================================

def subject_epoch_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns number of rows and valid-next-state rows by subject.
    """
    _require_columns(df, ["subject_id"], context="subject epoch counts")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "n_epochs",
                "n_valid_next_state",
                "n_recordings",
            ]
        )

    rows: List[Dict[str, Any]] = []

    for subject_id, group in df.groupby("subject_id", sort=True):
        row: Dict[str, Any] = {
            "subject_id": str(subject_id),
            "n_epochs": int(len(group)),
            "n_recordings": int(group["recording_id"].nunique()) if "recording_id" in group.columns else 0,
        }

        if "valid_next_state" in group.columns:
            row["n_valid_next_state"] = int(pd.to_numeric(group["valid_next_state"], errors="coerce").fillna(0).sum())
        else:
            row["n_valid_next_state"] = max(int(len(group)) - 1, 0)

        rows.append(row)

    return pd.DataFrame(rows)


def sleep_stage_distribution(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns sleep-stage counts by subject and globally.
    """
    if df.empty or "sleep_stage" not in df.columns:
        return []

    counts = (
        df.groupby(["subject_id", "sleep_stage"])
        .size()
        .reset_index(name="n_epochs")
        .sort_values(["subject_id", "sleep_stage"])
        .reset_index(drop=True)
    )

    return counts.to_dict(orient="records")


def validate_regime_frame(
    df: pd.DataFrame,
    spec: RegimeSpec,
) -> Dict[str, Any]:
    """
    Validates a regime against minimum sample-size criteria.
    """
    total_epochs = int(len(df))

    if df.empty:
        return {
            "valid": False,
            "reason": "empty_regime",
            "total_epochs": 0,
            "total_valid_next_state": 0,
            "n_subjects": 0,
            "n_valid_subjects": 0,
            "subject_counts": [],
        }

    counts = subject_epoch_counts(df)

    if counts.empty:
        return {
            "valid": False,
            "reason": "no_subject_counts",
            "total_epochs": total_epochs,
            "total_valid_next_state": 0,
            "n_subjects": 0,
            "n_valid_subjects": 0,
            "subject_counts": [],
        }

    valid_subjects_df = counts[counts["n_epochs"].astype(int) >= int(spec.min_epochs_per_subject)].copy()

    n_subjects = int(counts["subject_id"].nunique())
    n_valid_subjects = int(valid_subjects_df["subject_id"].nunique())

    total_valid_next_state = int(counts["n_valid_next_state"].sum())

    failures: List[str] = []

    if total_epochs < int(spec.min_total_epochs):
        failures.append(
            f"total_epochs<{spec.min_total_epochs}"
        )

    if n_valid_subjects < int(spec.min_subjects):
        failures.append(
            f"n_valid_subjects<{spec.min_subjects}"
        )

    if total_valid_next_state < max(int(spec.min_total_epochs) - n_valid_subjects, 1):
        failures.append(
            "insufficient_valid_next_state_rows"
        )

    valid = len(failures) == 0

    metadata: Dict[str, Any] = {
        "valid": valid,
        "reason": "ok" if valid else ";".join(failures),
        "total_epochs": total_epochs,
        "total_valid_next_state": total_valid_next_state,
        "n_subjects": n_subjects,
        "n_valid_subjects": n_valid_subjects,
        "min_subjects_required": int(spec.min_subjects),
        "min_total_epochs_required": int(spec.min_total_epochs),
        "min_epochs_per_subject_required": int(spec.min_epochs_per_subject),
        "valid_subjects": sorted(valid_subjects_df["subject_id"].astype(str).tolist()),
        "excluded_subjects": sorted(
            set(counts["subject_id"].astype(str).tolist())
            - set(valid_subjects_df["subject_id"].astype(str).tolist())
        ),
        "subject_counts": counts.to_dict(orient="records"),
        "sleep_stage_distribution": sleep_stage_distribution(df),
    }

    if "eeg_state" in df.columns:
        metadata["eeg_state_n_unique"] = int(df["eeg_state"].nunique(dropna=True))

    if "rng_state" in df.columns:
        metadata["rng_state_n_unique"] = int(df["rng_state"].nunique(dropna=True))

    if "observed_joint_state" in df.columns:
        metadata["observed_joint_state_n_unique"] = int(df["observed_joint_state"].nunique(dropna=True))

    if "informational_joint_state" in df.columns:
        metadata["informational_joint_state_n_unique"] = int(df["informational_joint_state"].nunique(dropna=True))

    if "latent_q_state" in df.columns:
        metadata["latent_q_state_n_unique"] = int(df["latent_q_state"].nunique(dropna=True))

    return metadata


def keep_valid_subjects_only(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    """
    Keeps only subjects that satisfy min_epochs_per_subject for the regime.
    """
    valid_subjects = metadata.get("valid_subjects", [])

    if not valid_subjects:
        return df.iloc[0:0].copy()

    out = df[df["subject_id"].astype(str).isin(set(valid_subjects))].copy()
    return _sort_frame(out)


# =========================================================
# Building all regimes
# =========================================================

def build_regime_frame(
    df: pd.DataFrame,
    regime_name: str,
    cfg: Optional[Phase32Config] = None,
    keep_only_valid_subjects: bool = True,
) -> RegimeFrame:
    """
    Builds and validates one RegimeFrame.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    regimes = regime_map(cfg)

    if regime_name not in regimes:
        raise KeyError(
            f"Unknown regime: {regime_name}. "
            f"Available regimes: {list(regimes.keys())}"
        )

    spec = regimes[regime_name]

    filtered = filter_regime(
        df=df,
        regime_name=regime_name,
        cfg=cfg,
    )

    metadata = validate_regime_frame(
        df=filtered,
        spec=spec,
    )

    if keep_only_valid_subjects and metadata.get("valid_subjects"):
        filtered_valid = keep_valid_subjects_only(filtered, metadata)

        # Recompute metadata after dropping subjects below minimum.
        metadata_after = validate_regime_frame(
            df=filtered_valid,
            spec=spec,
        )

        metadata_after["before_subject_filter"] = metadata
        metadata = metadata_after
        filtered = filtered_valid

    metadata["regime_name"] = regime_name
    metadata["regime_spec"] = asdict(spec)
    metadata["primary"] = bool(spec.primary)

    return RegimeFrame(
        name=regime_name,
        spec=spec,
        df=filtered,
        valid=bool(metadata.get("valid", False)),
        metadata=metadata,
    )


def build_all_regime_frames(
    df: pd.DataFrame,
    cfg: Optional[Phase32Config] = None,
    keep_only_valid_subjects: bool = True,
) -> Dict[str, RegimeFrame]:
    """
    Builds all configured regime frames.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    regimes = regime_map(cfg)
    frames: Dict[str, RegimeFrame] = {}

    for name in regimes.keys():
        print(f"[Phase3.2 Regimes] Building regime: {name}")

        frame = build_regime_frame(
            df=df,
            regime_name=name,
            cfg=cfg,
            keep_only_valid_subjects=keep_only_valid_subjects,
        )

        frames[name] = frame

        print(
            f"[Phase3.2 Regimes] {name}: "
            f"valid={frame.valid} | "
            f"rows={len(frame.df)} | "
            f"subjects={frame.metadata.get('n_valid_subjects')} | "
            f"reason={frame.metadata.get('reason')}"
        )

    return frames


# =========================================================
# Inventory and output
# =========================================================

def regime_inventory_row(frame: RegimeFrame) -> Dict[str, Any]:
    m = frame.metadata

    row: Dict[str, Any] = {
        "regime": frame.name,
        "primary": bool(m.get("primary", False)),
        "valid": bool(m.get("valid", False)),
        "reason": str(m.get("reason", "")),
        "total_epochs": _safe_int(m.get("total_epochs")),
        "total_valid_next_state": _safe_int(m.get("total_valid_next_state")),
        "n_subjects": _safe_int(m.get("n_subjects")),
        "n_valid_subjects": _safe_int(m.get("n_valid_subjects")),
        "min_subjects_required": _safe_int(m.get("min_subjects_required")),
        "min_total_epochs_required": _safe_int(m.get("min_total_epochs_required")),
        "min_epochs_per_subject_required": _safe_int(m.get("min_epochs_per_subject_required")),
        "eeg_state_n_unique": _safe_int(m.get("eeg_state_n_unique")),
        "rng_state_n_unique": _safe_int(m.get("rng_state_n_unique")),
        "observed_joint_state_n_unique": _safe_int(m.get("observed_joint_state_n_unique")),
        "informational_joint_state_n_unique": _safe_int(m.get("informational_joint_state_n_unique")),
        "latent_q_state_n_unique": _safe_int(m.get("latent_q_state_n_unique")),
        "valid_subjects": ",".join(m.get("valid_subjects", [])),
        "excluded_subjects": ",".join(m.get("excluded_subjects", [])),
    }

    return row


def build_regime_inventory(frames: Dict[str, RegimeFrame]) -> pd.DataFrame:
    rows = [regime_inventory_row(frame) for frame in frames.values()]
    return pd.DataFrame(rows)


def save_regime_outputs(
    frames: Dict[str, RegimeFrame],
    cfg: Optional[Phase32Config] = None,
) -> None:
    """
    Saves:
        - regime inventory CSV
        - regime summary JSON
        - individual valid regime CSV files
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    output_dir = Path(cfg.interim_dir) / "regimes"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_regime_inventory(frames)

    inventory_csv = Path(cfg.interim_dir) / "phase3_2_regime_inventory.csv"
    summary_json = Path(cfg.interim_dir) / "phase3_2_regime_summary.json"

    inventory.to_csv(inventory_csv, index=False)

    summary_payload: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "n_regimes": len(frames),
        "valid_regimes": [name for name, frame in frames.items() if frame.valid],
        "invalid_regimes": [name for name, frame in frames.items() if not frame.valid],
        "inventory_file": str(inventory_csv),
        "regimes": {
            name: frame.metadata
            for name, frame in frames.items()
        },
    }

    save_json(summary_json, summary_payload)

    for name, frame in frames.items():
        regime_csv = output_dir / f"phase3_2a_regime_{name}.csv"

        if frame.valid:
            frame.df.to_csv(regime_csv, index=False)

    print(f"[Phase3.2 Regimes] Saved inventory: {inventory_csv}")
    print(f"[Phase3.2 Regimes] Saved summary: {summary_json}")
    print(f"[Phase3.2 Regimes] Saved valid regime CSVs in: {output_dir}")


# =========================================================
# Summary text
# =========================================================

def build_regime_summary_text(frames: Dict[str, RegimeFrame]) -> str:
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("Phase III.2A — Regime-aware / sleep-stage analysis")
    lines.append("=" * 78)
    lines.append("")

    for name, frame in frames.items():
        m = frame.metadata

        lines.append(f"Regime: {name}")
        lines.append("-" * 78)
        lines.append(f"Description: {frame.spec.description}")
        lines.append(f"Primary: {frame.spec.primary}")
        lines.append(f"Valid: {frame.valid}")
        lines.append(f"Reason: {m.get('reason')}")
        lines.append(f"Total epochs: {m.get('total_epochs')}")
        lines.append(f"Valid next-state rows: {m.get('total_valid_next_state')}")
        lines.append(f"Valid subjects: {m.get('n_valid_subjects')}/{m.get('n_subjects')}")
        lines.append(f"Subjects: {', '.join(m.get('valid_subjects', []))}")

        if m.get("excluded_subjects"):
            lines.append(f"Excluded subjects: {', '.join(m.get('excluded_subjects', []))}")

        lines.append(f"EEG states: {m.get('eeg_state_n_unique')}")
        lines.append(f"RNG states: {m.get('rng_state_n_unique')}")
        lines.append(f"Observed joint states: {m.get('observed_joint_state_n_unique')}")
        lines.append(f"Informational joint states: {m.get('informational_joint_state_n_unique')}")
        lines.append(f"Latent-Q states: {m.get('latent_q_state_n_unique')}")
        lines.append("")

    lines.append("=" * 78)

    return "\n".join(lines)


def save_regime_summary_text(
    frames: Dict[str, RegimeFrame],
    cfg: Optional[Phase32Config] = None,
) -> None:
    if cfg is None:
        cfg = load_phase3_2_config()

    path = Path(cfg.interim_dir) / "phase3_2_regime_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(build_regime_summary_text(frames))

    print(f"[Phase3.2 Regimes] Saved text summary: {path}")


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    df = load_or_build_modeling_features(
        cfg=cfg,
        force_rebuild=False,
    )

    frames = build_all_regime_frames(
        df=df,
        cfg=cfg,
        keep_only_valid_subjects=True,
    )

    save_regime_outputs(frames, cfg)
    save_regime_summary_text(frames, cfg)

    inventory = build_regime_inventory(frames)

    print("\n============================================================")
    print("Phase III.2A regime validation completed")
    print("============================================================")
    print(inventory[["regime", "primary", "valid", "total_epochs", "n_valid_subjects", "reason"]])
    print("============================================================\n")


if __name__ == "__main__":
    main()