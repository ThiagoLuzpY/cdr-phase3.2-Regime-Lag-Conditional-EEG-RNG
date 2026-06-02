from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from config.phase3_2_config import Phase32Config, describe_config, load_phase3_2_config


# =========================================================
# Phase III.2 — Loader
# =========================================================
#
# This loader builds the base dataframe used by all Phase III.2 modules:
#
#   III.2A — regime-aware analysis
#   III.2B — lagged alignment
#   III.2C — conditional CDR
#   III.2D — optional multi-channel enrichment
#
# It intentionally creates a shared, auditable feature table:
#
#   data/interim/phase3_2/phase3_2_combined_features.csv
#
# The feature-layer construction, regime filtering, lagging and conditional
# state building are handled in later modules.


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

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if hasattr(obj, "__dict__"):
        return obj.__dict__

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)

        if not np.isfinite(x):
            return default

        return x

    except Exception:
        return default


def _subject_key_from_filename(path: Path) -> Optional[str]:
    """
    Extracts Sleep-EDF subject/session key from names such as:

        SC4001E0-PSG.edf
        SC4001EC-Hypnogram.edf
        SC4011EH-Hypnogram.edf

    We use the first SC#### block as subject key:

        SC4001
        SC4011
    """
    name = Path(path).name
    match = re.search(r"(SC\d{4})", name, flags=re.IGNORECASE)

    if not match:
        return None

    return match.group(1).upper()


def _recording_id_from_filename(path: Path) -> str:
    """
    Extracts recording identifier before the first dash.

    Example:
        SC4001E0-PSG.edf -> SC4001E0
    """
    name = Path(path).name
    return name.split("-")[0]


def _select_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    """
    Selects an EEG channel robustly.

    First tries exact match, then normalized lowercase match.
    """
    if requested in raw.ch_names:
        return requested

    normalized_requested = requested.lower().replace(" ", "").replace("_", "").replace("-", "")

    for ch in raw.ch_names:
        normalized_ch = ch.lower().replace(" ", "").replace("_", "").replace("-", "")
        if normalized_ch == normalized_requested:
            return ch

    # fallback: contains FpzCz or PzOz style fragments
    if "fpzcz" in normalized_requested:
        for ch in raw.ch_names:
            normalized_ch = ch.lower().replace(" ", "").replace("_", "").replace("-", "")
            if "fpzcz" in normalized_ch:
                return ch

    if "pzoz" in normalized_requested:
        for ch in raw.ch_names:
            normalized_ch = ch.lower().replace(" ", "").replace("_", "").replace("-", "")
            if "pzoz" in normalized_ch:
                return ch

    raise ValueError(
        f"Requested channel not found: {requested}. "
        f"Available channels: {raw.ch_names}"
    )


# =========================================================
# EEG file pairing
# =========================================================

def discover_eeg_pairs(cfg: Phase32Config) -> List[Dict[str, Any]]:
    """
    Discovers PSG + Hypnogram pairs in data/raw/eeg/.

    Pairing is based on the Sleep-EDF subject key SC####.
    """
    eeg_dir = Path(cfg.raw_eeg_dir)

    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG raw directory not found: {eeg_dir}")

    psg_files = sorted(eeg_dir.glob(cfg.eeg_psg_pattern))
    hypnogram_files = sorted(eeg_dir.glob(cfg.eeg_hypnogram_pattern))

    if not psg_files:
        raise FileNotFoundError(f"No PSG files found in {eeg_dir}")

    if not hypnogram_files:
        raise FileNotFoundError(f"No Hypnogram files found in {eeg_dir}")

    hyp_by_key: Dict[str, Path] = {}

    for hyp in hypnogram_files:
        key = _subject_key_from_filename(hyp)

        if key is not None:
            hyp_by_key[key] = hyp

    pairs: List[Dict[str, Any]] = []

    for psg in psg_files:
        key = _subject_key_from_filename(psg)

        if key is None:
            continue

        hyp = hyp_by_key.get(key)

        if hyp is None:
            continue

        pairs.append(
            {
                "subject_id": key,
                "recording_id": _recording_id_from_filename(psg),
                "psg_file": psg,
                "hypnogram_file": hyp,
            }
        )

    pairs = sorted(pairs, key=lambda x: str(x["subject_id"]))

    if cfg.max_subjects is not None:
        pairs = pairs[: int(cfg.max_subjects)]

    if not pairs:
        raise RuntimeError("No valid PSG/Hypnogram pairs found.")

    return pairs


# =========================================================
# Sleep-stage handling
# =========================================================

def normalize_sleep_stage(description: str) -> Optional[str]:
    """
    Maps Sleep-EDF annotations to compact stage labels.

    Sleep-EDF commonly contains:
        Sleep stage W
        Sleep stage 1
        Sleep stage 2
        Sleep stage 3
        Sleep stage 4
        Sleep stage R
        Sleep stage ?
        Movement time

    We map:
        stage 1 -> N1
        stage 2 -> N2
        stage 3/4 -> N3
        stage R -> REM
        stage W -> W
    """
    desc = str(description).strip().lower()

    if desc in {"w", "wake"} or "stage w" in desc:
        return "W"

    if desc in {"r", "rem"} or "stage r" in desc:
        return "REM"

    if "stage 1" in desc or desc == "n1":
        return "N1"

    if "stage 2" in desc or desc == "n2":
        return "N2"

    if "stage 3" in desc or "stage 4" in desc or desc in {"n3", "n4"}:
        return "N3"

    if "movement" in desc:
        return "MOVEMENT"

    if "?" in desc or "unknown" in desc:
        return None

    return None


def build_epoch_stage_table(
    annotations: mne.Annotations,
    n_epochs: int,
    epoch_seconds: int,
) -> List[Optional[str]]:
    """
    Assigns one sleep stage to each fixed-length epoch.

    The stage is assigned based on epoch midpoint.
    """
    epoch_stages: List[Optional[str]] = []

    ann_rows: List[Tuple[float, float, Optional[str]]] = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        stage = normalize_sleep_stage(description)

        if stage is None:
            continue

        start = float(onset)
        end = float(onset + duration)
        ann_rows.append((start, end, stage))

    for epoch_idx in range(n_epochs):
        midpoint = (epoch_idx * epoch_seconds) + (epoch_seconds / 2.0)
        stage_for_epoch: Optional[str] = None

        for start, end, stage in ann_rows:
            if start <= midpoint < end:
                stage_for_epoch = stage
                break

        epoch_stages.append(stage_for_epoch)

    return epoch_stages


# =========================================================
# EEG feature extraction
# =========================================================

def _band_power(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)

    if not np.any(mask):
        return 0.0

    x = freqs[mask]
    y = power[mask]

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


def _spectral_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    total = float(np.sum(p))

    if total <= 0:
        return 0.0

    p = p / total
    p = p[p > 0]

    if len(p) == 0:
        return 0.0

    entropy = -float(np.sum(p * np.log2(p)))

    if len(power) <= 1:
        return entropy

    return entropy / math.log2(len(power))


def _hjorth_parameters(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)

    if len(x) < 3:
        return 0.0, 0.0

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))

    if var_x <= 0 or var_dx <= 0:
        return 0.0, 0.0

    mobility = math.sqrt(var_dx / var_x)
    complexity = math.sqrt(var_ddx / var_dx) / mobility if mobility > 0 else 0.0

    return float(mobility), float(complexity)


def _line_length(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        return 0.0

    return float(np.sum(np.abs(np.diff(x))))


def extract_epoch_features(
    signal: np.ndarray,
    sfreq: float,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Extracts compact EEG features from one epoch.

    Features are intentionally standard and interpretable:
        - band powers
        - band ratios
        - spectral entropy
        - Hjorth parameters
        - line length
    """
    x = np.asarray(signal, dtype=float)

    if len(x) == 0:
        return {}

    x = x - np.nanmean(x)

    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    nperseg = min(len(x), int(sfreq * 4))

    if nperseg < 8:
        nperseg = len(x)

    freqs, power = welch(
        x,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=0,
        scaling="density",
    )

    delta = _band_power(freqs, power, 0.5, 4.0)
    theta = _band_power(freqs, power, 4.0, 8.0)
    alpha = _band_power(freqs, power, 8.0, 13.0)
    beta = _band_power(freqs, power, 13.0, 30.0)
    total = delta + theta + alpha + beta

    hjorth_mobility, hjorth_complexity = _hjorth_parameters(x)

    eps = 1e-12

    return {
        f"{prefix}delta_power": float(delta),
        f"{prefix}theta_power": float(theta),
        f"{prefix}alpha_power": float(alpha),
        f"{prefix}beta_power": float(beta),
        f"{prefix}total_power": float(total),
        f"{prefix}alpha_delta_ratio": float(alpha / (delta + eps)),
        f"{prefix}theta_alpha_ratio": float(theta / (alpha + eps)),
        f"{prefix}beta_alpha_ratio": float(beta / (alpha + eps)),
        f"{prefix}spectral_entropy": float(_spectral_entropy(power)),
        f"{prefix}hjorth_mobility": float(hjorth_mobility),
        f"{prefix}hjorth_complexity": float(hjorth_complexity),
        f"{prefix}line_length": float(_line_length(x)),
        f"{prefix}signal_std": float(np.std(x)),
        f"{prefix}signal_mean_abs": float(np.mean(np.abs(x))),
    }


def load_one_eeg_recording(
    pair: Dict[str, Any],
    cfg: Phase32Config,
) -> pd.DataFrame:
    """
    Loads one Sleep-EDF PSG/Hypnogram pair and returns epoch-level EEG features.
    """
    subject_id = str(pair["subject_id"])
    recording_id = str(pair["recording_id"])
    psg_file = Path(pair["psg_file"])
    hypnogram_file = Path(pair["hypnogram_file"])

    print(f"[Phase3.2 Loader] Loading EEG subject={subject_id} recording={recording_id}")

    raw = mne.io.read_raw_edf(
        psg_file,
        preload=True,
        verbose="ERROR",
    )

    annotations = mne.read_annotations(hypnogram_file)
    raw.set_annotations(annotations)

    primary_channel = _select_channel(raw, cfg.primary_eeg_channel)
    primary_data = raw.get_data(picks=[primary_channel])[0]
    sfreq = float(raw.info["sfreq"])

    secondary_data: Optional[np.ndarray] = None
    secondary_channel: Optional[str] = None

    if cfg.use_multichannel_eeg:
        try:
            secondary_channel = _select_channel(raw, cfg.secondary_eeg_channel)
            secondary_data = raw.get_data(picks=[secondary_channel])[0]
        except Exception:
            secondary_channel = None
            secondary_data = None

    samples_per_epoch = int(round(sfreq * cfg.eeg_epoch_seconds))
    n_epochs = int(len(primary_data) // samples_per_epoch)

    stages = build_epoch_stage_table(
        annotations=annotations,
        n_epochs=n_epochs,
        epoch_seconds=cfg.eeg_epoch_seconds,
    )

    rows: List[Dict[str, Any]] = []

    for epoch_idx in range(n_epochs):
        stage = stages[epoch_idx]

        if stage is None:
            continue

        if stage == "MOVEMENT":
            continue

        start = epoch_idx * samples_per_epoch
        end = start + samples_per_epoch

        primary_epoch = primary_data[start:end]

        row: Dict[str, Any] = {
            "subject_id": subject_id,
            "recording_id": recording_id,
            "epoch_idx": int(epoch_idx),
            "epoch_start_sec": float(epoch_idx * cfg.eeg_epoch_seconds),
            "epoch_end_sec": float((epoch_idx + 1) * cfg.eeg_epoch_seconds),
            "sleep_stage": stage,
            "primary_channel": primary_channel,
            "sfreq": sfreq,
        }

        row.update(extract_epoch_features(primary_epoch, sfreq, prefix=""))

        if secondary_data is not None:
            secondary_epoch = secondary_data[start:end]
            row["secondary_channel"] = secondary_channel

            sec_features = extract_epoch_features(
                secondary_epoch,
                sfreq,
                prefix="pz_oz_",
            )
            row.update(sec_features)

            if "delta_power" in row and "pz_oz_delta_power" in row:
                row["delta_channel_diff"] = float(row["delta_power"] - row["pz_oz_delta_power"])

            if "alpha_power" in row and "pz_oz_alpha_power" in row:
                row["alpha_channel_diff"] = float(row["alpha_power"] - row["pz_oz_alpha_power"])

            if len(primary_epoch) == len(secondary_epoch) and len(primary_epoch) > 2:
                corr = np.corrcoef(primary_epoch, secondary_epoch)[0, 1]
                row["cross_channel_corr"] = float(corr) if np.isfinite(corr) else 0.0

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No valid EEG epochs extracted for subject={subject_id}")

    return df


# =========================================================
# RNG handling
# =========================================================

def load_rng_uint8_values(rng_file: Path) -> np.ndarray:
    """
    Loads ANU QRNG JSON sample.

    Supported formats:
        {"data": [uint8...]}
        [uint8...]
    """
    rng_file = Path(rng_file)

    if not rng_file.exists():
        raise FileNotFoundError(f"RNG file not found: {rng_file}")

    with open(rng_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "data" not in payload:
            raise KeyError(f"RNG JSON does not contain 'data': {rng_file}")

        values = payload["data"]

    elif isinstance(payload, list):
        values = payload

    else:
        raise TypeError(f"Unsupported RNG JSON structure: {type(payload)}")

    arr = np.asarray(values, dtype=np.uint8)

    if arr.size == 0:
        raise RuntimeError("RNG uint8 array is empty.")

    return arr


def uint8_to_bits(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    bits = np.unpackbits(values)
    return bits.astype(np.int8)


def _binary_entropy(p: float) -> float:
    p = float(p)

    if p <= 0.0 or p >= 1.0:
        return 0.0

    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _run_lengths(bits: np.ndarray) -> List[int]:
    bits = np.asarray(bits, dtype=int)

    if bits.size == 0:
        return []

    runs: List[int] = []
    current = int(bits[0])
    length = 1

    for value in bits[1:]:
        value = int(value)

        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1

    runs.append(length)

    return runs


def compute_rng_window_metrics(bits: np.ndarray) -> Dict[str, float]:
    bits = np.asarray(bits, dtype=int)

    if bits.size == 0:
        return {
            "bit_balance_local": 0.0,
            "transition_rate": 0.0,
            "rng_entropy_local": 0.0,
            "entropy_rate_proxy": 0.0,
            "run_length_mean": 0.0,
            "run_length_max": 0.0,
            "run_length_std": 0.0,
            "compressibility_proxy": 0.0,
            "surprise_index": 0.0,
            "transition_asymmetry": 0.0,
            "micro_cluster_deviation": 0.0,
            "q_rng_score": 0.0,
        }

    p1 = float(np.mean(bits))
    entropy = _binary_entropy(p1)

    if bits.size > 1:
        transitions = bits[1:] != bits[:-1]
        transition_rate = float(np.mean(transitions))
    else:
        transition_rate = 0.0

    runs = _run_lengths(bits)
    run_mean = float(np.mean(runs)) if runs else 0.0
    run_max = float(np.max(runs)) if runs else 0.0
    run_std = float(np.std(runs)) if runs else 0.0

    # Simple interpretable proxies.
    compressibility_proxy = float(abs(run_mean - 2.0))
    surprise_index = float(abs(p1 - 0.5))
    transition_asymmetry = float(abs(transition_rate - 0.5))
    micro_cluster_deviation = float(abs(run_max - math.log2(max(bits.size, 2))))

    # The q_rng_score is not a quantum measurement.
    # It is a proxy summarizing local deviations from idealized balanced randomness.
    q_rng_score = float(
        surprise_index
        + transition_asymmetry
        + min(compressibility_proxy / 10.0, 1.0)
        + min(micro_cluster_deviation / 10.0, 1.0)
    )

    return {
        "bit_balance_local": float(p1),
        "transition_rate": transition_rate,
        "rng_entropy_local": float(entropy),
        "entropy_rate_proxy": float(entropy * transition_rate),
        "run_length_mean": run_mean,
        "run_length_max": run_max,
        "run_length_std": run_std,
        "compressibility_proxy": compressibility_proxy,
        "surprise_index": surprise_index,
        "transition_asymmetry": transition_asymmetry,
        "micro_cluster_deviation": micro_cluster_deviation,
        "q_rng_score": q_rng_score,
    }


def align_rng_to_subject_epochs(
    df: pd.DataFrame,
    bits: np.ndarray,
    cfg: Phase32Config,
) -> pd.DataFrame:
    """
    Aligns deterministic RNG windows to each subject's EEG epochs.

    The same RNG sample is used for each subject, but window positions are
    resampled deterministically across the subject-specific epoch axis.

    This mirrors the Phase III.1 logic while keeping Phase III.2 fully auditable.
    """
    bits = np.asarray(bits, dtype=int)

    if bits.size < cfg.rng_window_size + 2:
        raise RuntimeError(
            f"RNG bit sequence too short: {bits.size} bits, "
            f"window={cfg.rng_window_size}"
        )

    out_frames: List[pd.DataFrame] = []

    max_start = int(bits.size - cfg.rng_window_size)

    for subject_id, group in df.groupby("subject_id", sort=True):
        g = group.copy().reset_index(drop=True)
        n = len(g)

        if n <= 1:
            starts = np.zeros(n, dtype=int)
        else:
            starts = np.round(np.linspace(0, max_start, n)).astype(int)

        rng_rows: List[Dict[str, Any]] = []

        for i, start in enumerate(starts):
            start = int(start)
            end = int(start + cfg.rng_window_size)

            window = bits[start:end]
            metrics = compute_rng_window_metrics(window)

            row: Dict[str, Any] = {
                "rng_window_id": int(i),
                "rng_window_start": start,
                "rng_window_end": end,
                "rng_bit": int(bits[start]),
                "rng_next_bit_raw": int(bits[min(start + 1, bits.size - 1)]),
            }

            row.update(metrics)
            rng_rows.append(row)

        rng_df = pd.DataFrame(rng_rows)
        g = pd.concat([g, rng_df], axis=1)

        out_frames.append(g)

    aligned = pd.concat(out_frames, ignore_index=True)

    return aligned


# =========================================================
# Full dataset construction
# =========================================================

def build_phase3_2_dataset(cfg: Optional[Phase32Config] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2_config()

    pairs = discover_eeg_pairs(cfg)

    print(f"[Phase3.2 Loader] EEG pairs discovered: {len(pairs)}")

    eeg_frames: List[pd.DataFrame] = []
    errors: List[Dict[str, Any]] = []

    for pair in pairs:
        try:
            eeg_df = load_one_eeg_recording(pair, cfg)
            eeg_frames.append(eeg_df)

            print(
                f"[Phase3.2 Loader] Loaded {pair['subject_id']} | "
                f"rows={len(eeg_df)}"
            )

        except Exception as exc:
            errors.append(
                {
                    "subject_id": str(pair.get("subject_id")),
                    "recording_id": str(pair.get("recording_id")),
                    "psg_file": str(pair.get("psg_file")),
                    "hypnogram_file": str(pair.get("hypnogram_file")),
                    "error": str(exc),
                }
            )
            print(f"[Phase3.2 Loader] ERROR loading {pair.get('subject_id')}: {exc}")

    if not eeg_frames:
        raise RuntimeError(f"No EEG recordings loaded. Errors: {errors}")

    eeg_all = pd.concat(eeg_frames, ignore_index=True)

    eeg_all = eeg_all.sort_values(
        [c for c in ["subject_id", "recording_id", "epoch_idx"] if c in eeg_all.columns]
    ).reset_index(drop=True)

    rng_uint8 = load_rng_uint8_values(cfg.rng_file)
    rng_bits = uint8_to_bits(rng_uint8)

    print(
        f"[Phase3.2 Loader] RNG loaded: "
        f"{len(rng_uint8)} uint8 -> {len(rng_bits)} bits"
    )

    combined = align_rng_to_subject_epochs(
        df=eeg_all,
        bits=rng_bits,
        cfg=cfg,
    )

    metadata: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "config": describe_config(cfg),
        "n_pairs_discovered": len(pairs),
        "n_subjects_loaded": int(combined["subject_id"].nunique()),
        "subjects_loaded": sorted(combined["subject_id"].astype(str).unique().tolist()),
        "total_rows": int(len(combined)),
        "eeg_rows_before_rng_alignment": int(len(eeg_all)),
        "rng_uint8_count": int(len(rng_uint8)),
        "rng_bit_count": int(len(rng_bits)),
        "rng_window_size": int(cfg.rng_window_size),
        "rng_alignment_mode": cfg.rng_alignment_mode,
        "errors": errors,
    }

    if "sleep_stage" in combined.columns:
        stage_counts = (
            combined.groupby(["subject_id", "sleep_stage"])
            .size()
            .reset_index(name="n_epochs")
            .to_dict(orient="records")
        )
        metadata["sleep_stage_counts"] = stage_counts

    return combined, metadata


def save_phase3_2_dataset(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    cfg: Phase32Config,
) -> None:
    Path(cfg.interim_dir).mkdir(parents=True, exist_ok=True)

    df.to_csv(cfg.combined_features_file, index=False)
    save_json(cfg.loader_metadata_file, metadata)

    print(f"[Phase3.2 Loader] Saved features: {cfg.combined_features_file}")
    print(f"[Phase3.2 Loader] Saved metadata: {cfg.loader_metadata_file}")


def load_or_build_phase3_2_dataset(
    cfg: Optional[Phase32Config] = None,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cfg is None:
        cfg = load_phase3_2_config()

    if cfg.combined_features_file.exists() and cfg.loader_metadata_file.exists() and not force_rebuild:
        print(f"[Phase3.2 Loader] Loading cached features: {cfg.combined_features_file}")

        df = pd.read_csv(cfg.combined_features_file)

        with open(cfg.loader_metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return df, metadata

    df, metadata = build_phase3_2_dataset(cfg)
    save_phase3_2_dataset(df, metadata, cfg)

    return df, metadata


# =========================================================
# CLI
# =========================================================

def main() -> None:
    cfg = load_phase3_2_config()

    df, metadata = load_or_build_phase3_2_dataset(
        cfg=cfg,
        force_rebuild=True,
    )

    print("\n============================================================")
    print("Phase III.2 loader completed")
    print("============================================================")
    print(f"Rows: {len(df)}")
    print(f"Subjects: {metadata.get('subjects_loaded')}")
    print(f"RNG bits: {metadata.get('rng_bit_count')}")
    print(f"Output CSV: {cfg.combined_features_file}")
    print("============================================================\n")


if __name__ == "__main__":
    main()