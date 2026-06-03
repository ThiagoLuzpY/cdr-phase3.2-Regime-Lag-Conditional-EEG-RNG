from __future__ import annotations

import json
import math
import re
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
#   III.2D — multichannel EEG enrichment
#   III.2E — future synchronized EEG + QRNG protocol
#
# Main Phase III.2D patch:
#
#   - Extracts primary channel:   EEG Fpz-Cz
#   - Extracts secondary channel: EEG Pz-Oz
#   - Saves secondary features with two compatible naming conventions:
#
#       pz_oz_delta_power
#       pz_oz_alpha_power
#
#     and canonical III.2D columns:
#
#       delta_power_secondary
#       alpha_power_secondary
#
#   - Detects stale cached loader output and rebuilds automatically when
#     multichannel features are required but missing.
#
# Output:
#
#   data/interim/phase3_2/phase3_2_combined_features.csv
#   data/interim/phase3_2/phase3_2_loader_metadata.json


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


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y) or len(x) < 3:
        return 0.0

    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0

    corr = np.corrcoef(x, y)[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return float(corr)


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


def _normalized_channel_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
    )


def _select_channel(raw: mne.io.BaseRaw, requested: str) -> str:
    """
    Selects an EEG channel robustly.

    First tries exact match, then normalized lowercase match, then canonical
    fragments such as FpzCz and PzOz.
    """
    if requested in raw.ch_names:
        return requested

    normalized_requested = _normalized_channel_name(requested)

    for ch in raw.ch_names:
        normalized_ch = _normalized_channel_name(ch)

        if normalized_ch == normalized_requested:
            return ch

    if "fpzcz" in normalized_requested:
        for ch in raw.ch_names:
            normalized_ch = _normalized_channel_name(ch)

            if "fpzcz" in normalized_ch:
                return ch

    if "pzoz" in normalized_requested:
        for ch in raw.ch_names:
            normalized_ch = _normalized_channel_name(ch)

            if "pzoz" in normalized_ch:
                return ch

    raise ValueError(
        f"Requested channel not found: {requested}. "
        f"Available channels: {raw.ch_names}"
    )


def _has_required_multichannel_columns(df: pd.DataFrame) -> bool:
    required = [
        "delta_power_secondary",
        "alpha_power_secondary",
        "secondary_channel",
        "has_secondary_channel",
    ]

    return all(col in df.columns for col in required)


def _cached_dataset_is_compatible(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    cfg: Phase32Config,
) -> Tuple[bool, str]:
    """
    Prevents reuse of old cached loader output after III.2D activation.

    Before III.2D, the cached file may not contain true secondary-channel
    features. If multichannel EEG is enabled and those columns are absent,
    the loader should rebuild automatically.
    """
    if df.empty:
        return False, "cached_dataframe_empty"

    if cfg.use_multichannel_eeg or cfg.run_multichannel_analysis:
        if not _has_required_multichannel_columns(df):
            return False, "multichannel_required_columns_missing"

        if "has_secondary_channel" in df.columns:
            has_secondary = pd.to_numeric(
                df["has_secondary_channel"],
                errors="coerce",
            ).fillna(0).astype(int)

            if int(has_secondary.sum()) == 0:
                return False, "cached_dataframe_has_no_secondary_channel_rows"

    metadata_version = str(metadata.get("config", {}).get("version", metadata.get("version", "")))

    if metadata_version and metadata_version != str(cfg.version):
        # Version mismatch alone does not force rebuild unless critical columns
        # are missing. We keep this as compatible to avoid unnecessary rebuilds
        # after purely metadata-level changes.
        return True, f"compatible_with_version_note_cached={metadata_version}_current={cfg.version}"

    return True, "compatible"


# =========================================================
# EEG file pairing
# =========================================================

def discover_eeg_pairs(cfg: Phase32Config) -> List[Dict[str, Any]]:
    """
    Discovers PSG + Hypnogram pairs in data/raw/eeg/.

    Pairing is based on the Sleep-EDF subject key SC####. This preserves the
    earlier Phase III.1/III.2 behavior while staying compatible with files such
    as SC4001E0-PSG.edf and SC4001EC-Hypnogram.edf.
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

    hyp_by_key: Dict[str, List[Path]] = {}

    for hyp in hypnogram_files:
        key = _subject_key_from_filename(hyp)

        if key is not None:
            hyp_by_key.setdefault(key, []).append(hyp)

    pairs: List[Dict[str, Any]] = []

    for psg in psg_files:
        key = _subject_key_from_filename(psg)

        if key is None:
            continue

        candidate_hyps = hyp_by_key.get(key, [])

        if not candidate_hyps:
            continue

        # Sleep-EDF often uses SC4001E0-PSG with SC4001EC-Hypnogram.
        # We keep the old SC#### pairing rule but choose the first sorted
        # candidate deterministically.
        hyp = sorted(candidate_hyps)[0]

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


def add_secondary_features_to_row(
    row: Dict[str, Any],
    secondary_epoch: np.ndarray,
    primary_epoch: np.ndarray,
    sfreq: float,
    secondary_channel: str,
) -> Dict[str, Any]:
    """
    Adds secondary-channel features using both legacy pz_oz_* names and
    canonical *_secondary names required by Phase III.2D.
    """
    row["secondary_channel"] = str(secondary_channel)
    row["has_secondary_channel"] = 1

    secondary_features = extract_epoch_features(
        secondary_epoch,
        sfreq,
        prefix="",
    )

    for name, value in secondary_features.items():
        # Canonical III.2D columns:
        #   delta_power_secondary
        #   alpha_power_secondary
        row[f"{name}_secondary"] = value

        # Backward-compatible columns:
        #   pz_oz_delta_power
        #   pz_oz_alpha_power
        row[f"pz_oz_{name}"] = value

    if "delta_power" in row and "delta_power_secondary" in row:
        row["delta_channel_diff"] = float(
            _safe_float(row["delta_power"]) - _safe_float(row["delta_power_secondary"])
        )

    if "alpha_power" in row and "alpha_power_secondary" in row:
        row["alpha_channel_diff"] = float(
            _safe_float(row["alpha_power"]) - _safe_float(row["alpha_power_secondary"])
        )

    row["cross_channel_corr"] = _safe_corr(primary_epoch, secondary_epoch)

    return row


def add_missing_secondary_placeholders(
    row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Adds explicit placeholders when the secondary channel is unavailable.

    This makes the absence auditable and prevents silent fallback ambiguity.
    """
    row["secondary_channel"] = ""
    row["has_secondary_channel"] = 0

    secondary_columns = [
        "delta_power_secondary",
        "theta_power_secondary",
        "alpha_power_secondary",
        "beta_power_secondary",
        "total_power_secondary",
        "alpha_delta_ratio_secondary",
        "theta_alpha_ratio_secondary",
        "beta_alpha_ratio_secondary",
        "spectral_entropy_secondary",
        "hjorth_mobility_secondary",
        "hjorth_complexity_secondary",
        "line_length_secondary",
        "signal_std_secondary",
        "signal_mean_abs_secondary",
        "pz_oz_delta_power",
        "pz_oz_theta_power",
        "pz_oz_alpha_power",
        "pz_oz_beta_power",
        "pz_oz_total_power",
        "pz_oz_alpha_delta_ratio",
        "pz_oz_theta_alpha_ratio",
        "pz_oz_beta_alpha_ratio",
        "pz_oz_spectral_entropy",
        "pz_oz_hjorth_mobility",
        "pz_oz_hjorth_complexity",
        "pz_oz_line_length",
        "pz_oz_signal_std",
        "pz_oz_signal_mean_abs",
        "delta_channel_diff",
        "alpha_channel_diff",
        "cross_channel_corr",
    ]

    for col in secondary_columns:
        row[col] = np.nan

    return row


def load_one_eeg_recording(
    pair: Dict[str, Any],
    cfg: Phase32Config,
) -> pd.DataFrame:
    """
    Loads one Sleep-EDF PSG/Hypnogram pair and returns epoch-level EEG features.

    Phase III.2D update:
        If use_multichannel_eeg=True, this extracts both Fpz-Cz and Pz-Oz
        features when Pz-Oz is present.
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
    secondary_error: Optional[str] = None

    if cfg.use_multichannel_eeg or cfg.run_multichannel_analysis:
        try:
            secondary_channel = _select_channel(raw, cfg.secondary_eeg_channel)
            secondary_data = raw.get_data(picks=[secondary_channel])[0]

            if len(secondary_data) != len(primary_data):
                secondary_error = (
                    f"Secondary channel length mismatch: primary={len(primary_data)}, "
                    f"secondary={len(secondary_data)}"
                )
                secondary_channel = None
                secondary_data = None

        except Exception as exc:
            secondary_error = str(exc)
            secondary_channel = None
            secondary_data = None

    if secondary_channel:
        print(
            f"[Phase3.2 Loader] Channels subject={subject_id}: "
            f"primary={primary_channel}, secondary={secondary_channel}"
        )
    else:
        print(
            f"[Phase3.2 Loader] WARNING subject={subject_id}: "
            f"secondary channel unavailable. requested={cfg.secondary_eeg_channel}; "
            f"error={secondary_error}"
        )

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
            "secondary_channel_requested": cfg.secondary_eeg_channel,
            "secondary_channel_error": secondary_error or "",
            "sfreq": sfreq,
        }

        row.update(extract_epoch_features(primary_epoch, sfreq, prefix=""))

        if secondary_data is not None and secondary_channel is not None:
            secondary_epoch = secondary_data[start:end]

            row = add_secondary_features_to_row(
                row=row,
                secondary_epoch=secondary_epoch,
                primary_epoch=primary_epoch,
                sfreq=sfreq,
                secondary_channel=secondary_channel,
            )
        else:
            row = add_missing_secondary_placeholders(row)

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

    compressibility_proxy = float(abs(run_mean - 2.0))
    surprise_index = float(abs(p1 - 0.5))
    transition_asymmetry = float(abs(transition_rate - 0.5))
    micro_cluster_deviation = float(abs(run_max - math.log2(max(bits.size, 2))))

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

    This keeps Phase III.2 auditable while preserving the public-data limitation:
    Sleep-EDF and ANU QRNG are not truly co-acquired.
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

            has_secondary_rows = (
                int(pd.to_numeric(eeg_df.get("has_secondary_channel", 0), errors="coerce").fillna(0).sum())
                if "has_secondary_channel" in eeg_df.columns
                else 0
            )

            print(
                f"[Phase3.2 Loader] Loaded {pair['subject_id']} | "
                f"rows={len(eeg_df)} | secondary_rows={has_secondary_rows}"
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

    has_secondary_total = (
        int(pd.to_numeric(combined.get("has_secondary_channel", 0), errors="coerce").fillna(0).sum())
        if "has_secondary_channel" in combined.columns
        else 0
    )

    secondary_subjects: List[str] = []

    if "has_secondary_channel" in combined.columns and "subject_id" in combined.columns:
        sec_mask = pd.to_numeric(
            combined["has_secondary_channel"],
            errors="coerce",
        ).fillna(0).astype(int) == 1

        secondary_subjects = sorted(
            combined.loc[sec_mask, "subject_id"].astype(str).unique().tolist()
        )

    metadata: Dict[str, Any] = {
        "phase": cfg.phase_name,
        "project_name": cfg.project_name,
        "version": cfg.version,
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
        "primary_eeg_channel_requested": cfg.primary_eeg_channel,
        "secondary_eeg_channel_requested": cfg.secondary_eeg_channel,
        "use_multichannel_eeg": bool(cfg.use_multichannel_eeg),
        "run_multichannel_analysis": bool(cfg.run_multichannel_analysis),
        "has_secondary_channel_rows": has_secondary_total,
        "secondary_channel_subjects": secondary_subjects,
        "multichannel_loader_available": bool(has_secondary_total > 0),
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

    if "secondary_channel" in combined.columns:
        secondary_channel_counts = (
            combined["secondary_channel"]
            .fillna("")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        metadata["secondary_channel_counts"] = secondary_channel_counts

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

        compatible, reason = _cached_dataset_is_compatible(df, metadata, cfg)

        if compatible:
            print(f"[Phase3.2 Loader] Cached dataset compatible: {reason}")
            return df, metadata

        print(
            "[Phase3.2 Loader] Cached dataset is not compatible with current config. "
            f"Reason: {reason}. Rebuilding..."
        )

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
    print(f"Multichannel loader available: {metadata.get('multichannel_loader_available')}")
    print(f"Secondary-channel rows: {metadata.get('has_secondary_channel_rows')}")
    print(f"Secondary-channel subjects: {metadata.get('secondary_channel_subjects')}")
    print(f"Output CSV: {cfg.combined_features_file}")
    print("============================================================\n")


if __name__ == "__main__":
    main()