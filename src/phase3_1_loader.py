from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.phase3_1_features import (
    compute_phase3_1_feature_frame,
    convert_uint8_to_bits,
)


# =========================================================
# Optional MNE support
# =========================================================

try:
    import mne

    MNE_AVAILABLE = True
except Exception:
    mne = None
    MNE_AVAILABLE = False


# =========================================================
# Data containers
# =========================================================

@dataclass(frozen=True)
class EEGFilePair:
    """
    Represents one PSG/Hypnogram pair from Sleep-EDF.
    """

    subject_id: str
    recording_id: str
    psg_file: Path
    hypnogram_file: Path


@dataclass(frozen=True)
class RNGSequence:
    """
    Container for the RNG sequence used in Phase III.1.
    """

    raw_uint8: np.ndarray
    bits: np.ndarray
    source: str
    n_uint8: int
    n_bits: int


@dataclass(frozen=True)
class LoadedSubject:
    """
    Container for one loaded EEG subject/recording.
    """

    subject_id: str
    recording_id: str
    channel_used: str
    sfreq: float
    n_epochs: int
    feature_frame: pd.DataFrame
    psg_file: Path
    hypnogram_file: Path


@dataclass(frozen=True)
class Phase3_1Dataset:
    """
    Full dataset returned by the Phase III.1 loader.
    """

    df: pd.DataFrame
    subjects: Tuple[LoadedSubject, ...]
    rng: RNGSequence
    metadata: Dict[str, Any]


# =========================================================
# Logging helper
# =========================================================

def _log(cfg: Any, message: str) -> None:
    if getattr(cfg, "verbose", 1):
        print(message)


# =========================================================
# RNG loading
# =========================================================

def load_rng_sequence(cfg: Any) -> RNGSequence:
    """
    Loads a single ANU RNG JSON file and converts it to bits.

    This intentionally does not concatenate multiple RNG requests.
    """
    rng_file = Path(cfg.rng_file)

    if not rng_file.exists():
        raise FileNotFoundError(f"RNG file not found: {rng_file}")

    _log(cfg, f"[Phase3.1-Loader] Loading RNG JSON: {rng_file}")

    with open(rng_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw.get("success", False):
        raise ValueError(f"RNG JSON indicates failure: {raw}")

    if "data" not in raw:
        raise ValueError("RNG JSON missing required field: 'data'")

    seq_uint8 = np.asarray(raw["data"], dtype=np.uint8).reshape(-1)

    if cfg.rng_sequence_length is not None:
        seq_uint8 = seq_uint8[: int(cfg.rng_sequence_length)]

    if len(seq_uint8) < 10:
        raise RuntimeError(
            f"RNG sequence too short after truncation: len={len(seq_uint8)}"
        )

    if getattr(cfg, "allow_multiple_rng_requests", False):
        raise RuntimeError(
            "Multiple RNG requests are not allowed in the locked Phase III.1 design."
        )

    if getattr(cfg, "rng_use_bits", True):
        bits = convert_uint8_to_bits(seq_uint8)
    else:
        # Fallback mode: binarize uint8 values around midpoint.
        # Not recommended for final Phase III.1.
        bits = (seq_uint8 >= 128).astype(np.uint8)

    if len(bits) < cfg.rng_window_size:
        raise RuntimeError(
            f"RNG bit sequence shorter than rng_window_size: "
            f"n_bits={len(bits)}, window={cfg.rng_window_size}"
        )

    rng = RNGSequence(
        raw_uint8=seq_uint8,
        bits=bits.astype(np.uint8),
        source=str(getattr(cfg, "rng_source_name", "ANU")),
        n_uint8=int(len(seq_uint8)),
        n_bits=int(len(bits)),
    )

    _log(
        cfg,
        "[Phase3.1-Loader] RNG loaded: "
        f"{rng.n_uint8} uint8 values → {rng.n_bits} bits",
    )

    return rng


# =========================================================
# Sleep-EDF file discovery
# =========================================================

def _edf_files(root: Path) -> List[Path]:
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"EEG raw directory not found: {root}")

    files = []
    for path in root.rglob("*.edf"):
        if path.is_file():
            files.append(path)

    for path in root.rglob("*.EDF"):
        if path.is_file():
            files.append(path)

    # Deduplicate and sort.
    files = sorted(set(files), key=lambda p: str(p).lower())

    return files


def _is_psg_file(path: Path, cfg: Any) -> bool:
    name = path.name.lower()
    suffix = str(cfg.psg_suffix).lower()
    return name.endswith(suffix.lower())


def _is_hypnogram_file(path: Path, cfg: Any) -> bool:
    name = path.name.lower()
    suffix = str(cfg.hypnogram_suffix).lower()
    return name.endswith(suffix.lower())


def _recording_id_from_psg(path: Path) -> str:
    name = path.name
    name = re.sub(r"-PSG\.edf$", "", name, flags=re.IGNORECASE)
    return name


def _subject_key_from_recording_id(recording_id: str) -> str:
    """
    Sleep-EDF SC files often follow forms such as:
        SC4001E0-PSG.edf
        SC4001EC-Hypnogram.edf

    The first six characters identify the subject/session pair sufficiently
    for matching PSG and hypnogram in the common Sleep-EDF layout.
    """
    if len(recording_id) >= 6:
        return recording_id[:6]
    return recording_id


def discover_sleep_edf_pairs(cfg: Any) -> Tuple[EEGFilePair, ...]:
    """
    Discovers PSG/Hypnogram pairs under cfg.eeg_raw_dir.

    Matching strategy:
        - PSG files end with cfg.psg_suffix
        - Hypnogram files end with cfg.hypnogram_suffix
        - Pair using the first six characters of the PSG recording id
    """
    root = Path(cfg.eeg_raw_dir)
    all_files = _edf_files(root)

    psg_files = [p for p in all_files if _is_psg_file(p, cfg)]
    hyp_files = [p for p in all_files if _is_hypnogram_file(p, cfg)]

    if not psg_files:
        raise FileNotFoundError(
            f"No PSG files found in {root} with suffix {cfg.psg_suffix}"
        )

    if not hyp_files:
        raise FileNotFoundError(
            f"No Hypnogram files found in {root} with suffix {cfg.hypnogram_suffix}"
        )

    hyp_by_key: Dict[str, List[Path]] = {}

    for hyp in hyp_files:
        hyp_base = re.sub(
            r"-Hypnogram\.edf$",
            "",
            hyp.name,
            flags=re.IGNORECASE,
        )
        key = _subject_key_from_recording_id(hyp_base)
        hyp_by_key.setdefault(key, []).append(hyp)

    pairs: List[EEGFilePair] = []

    requested = tuple(getattr(cfg, "subject_ids", tuple()) or tuple())
    requested_set = {str(x) for x in requested}

    for psg in psg_files:
        recording_id = _recording_id_from_psg(psg)
        key = _subject_key_from_recording_id(recording_id)

        # Allow filtering by recording_id, subject key, or exact PSG stem.
        if requested_set:
            if (
                recording_id not in requested_set
                and key not in requested_set
                and psg.stem not in requested_set
            ):
                continue

        candidates = hyp_by_key.get(key, [])

        if not candidates:
            continue

        # Choose the closest lexical match. For SC4001E0, SC4001EC is normally first.
        hyp = sorted(candidates, key=lambda p: str(p).lower())[0]

        pair = EEGFilePair(
            subject_id=key,
            recording_id=recording_id,
            psg_file=psg,
            hypnogram_file=hyp,
        )
        pairs.append(pair)

    pairs = sorted(pairs, key=lambda x: x.recording_id)

    max_subjects = int(getattr(cfg, "max_subjects", len(pairs)))
    if max_subjects > 0:
        pairs = pairs[:max_subjects]

    if not pairs:
        raise RuntimeError(
            "No matched Sleep-EDF PSG/Hypnogram pairs were found after filtering."
        )

    return tuple(pairs)


# =========================================================
# Sleep stage handling
# =========================================================

def normalize_sleep_stage(description: str, cfg: Any) -> str:
    """
    Normalizes Sleep-EDF annotation descriptions to compact stage labels.
    """
    desc = str(description).strip().lower()

    desc = desc.replace("sleep stage", "").strip()
    desc = desc.replace("stage", "").strip()

    if desc in {"w", "wake", "awake"}:
        return "W"

    if desc in {"1", "n1", "s1"}:
        return "N1"

    if desc in {"2", "n2", "s2"}:
        return "N2"

    if desc in {"3", "n3", "s3"}:
        return "N3"

    if desc in {"4", "n4", "s4"}:
        return "N3" if getattr(cfg, "merge_n3_n4", True) else "N4"

    if desc in {"r", "rem"}:
        return "REM"

    if "movement" in desc:
        return "UNKNOWN"

    if desc in {"?", "unknown"}:
        return "UNKNOWN"

    return "UNKNOWN"


def _annotation_table(raw: Any, cfg: Any) -> pd.DataFrame:
    """
    Extracts normalized annotation intervals from an MNE Raw object.
    """
    rows: List[Dict[str, Any]] = []

    for ann in raw.annotations:
        onset = float(ann["onset"])
        duration = float(ann["duration"])
        description = str(ann["description"])

        stage = normalize_sleep_stage(description, cfg)

        rows.append(
            {
                "onset": onset,
                "end": onset + duration,
                "duration": duration,
                "description": description,
                "stage": stage,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["onset", "end", "duration", "description", "stage"])

    table = pd.DataFrame(rows)
    table = table.sort_values("onset").reset_index(drop=True)

    return table


def _stage_for_time(t: float, annotations_df: pd.DataFrame) -> str:
    if annotations_df.empty:
        return "UNKNOWN"

    mask = (annotations_df["onset"] <= t) & (t < annotations_df["end"])

    if not mask.any():
        return "UNKNOWN"

    return str(annotations_df.loc[mask, "stage"].iloc[0])


def epoch_sleep_stages(
    raw: Any,
    n_epochs: int,
    epoch_seconds: int,
    cfg: Any,
) -> List[str]:
    """
    Assigns one sleep stage to each fixed-length EEG epoch using epoch midpoint.
    """
    annotations_df = _annotation_table(raw, cfg)

    labels: List[str] = []

    for epoch_idx in range(n_epochs):
        midpoint = epoch_idx * epoch_seconds + (epoch_seconds / 2.0)
        labels.append(_stage_for_time(midpoint, annotations_df))

    return labels


def _filter_epoch_rows_by_stage(
    epochs: np.ndarray,
    stages: Sequence[str],
    cfg: Any,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Filters epochs according to cfg.keep_sleep_stages and cfg.drop_unknown_stages.
    """
    keep_stages = set(str(x) for x in getattr(cfg, "keep_sleep_stages", tuple()))

    kept_indices: List[int] = []
    kept_stages: List[str] = []

    for i, stage in enumerate(stages):
        stage = str(stage)

        if stage == "UNKNOWN" and getattr(cfg, "drop_unknown_stages", True):
            continue

        if keep_stages and stage not in keep_stages:
            continue

        kept_indices.append(i)
        kept_stages.append(stage)

    if not kept_indices:
        raise RuntimeError("No EEG epochs remained after sleep-stage filtering")

    idx = np.asarray(kept_indices, dtype=int)
    return epochs[idx], kept_stages, idx


# =========================================================
# EEG loading
# =========================================================

def _require_mne() -> None:
    if not MNE_AVAILABLE:
        raise ImportError(
            "MNE is required to load EDF files. Install it in your active venv:\n"
            "pip install mne"
        )


def _select_eeg_channel(raw: Any, cfg: Any) -> str:
    channels = list(raw.ch_names)
    preferred = tuple(getattr(cfg, "eeg_channels_preferred", tuple()))

    # Exact match first.
    for wanted in preferred:
        for ch in channels:
            if ch == wanted:
                return ch

    # Case-insensitive exact match.
    lower_map = {ch.lower(): ch for ch in channels}
    for wanted in preferred:
        match = lower_map.get(str(wanted).lower())
        if match is not None:
            return match

    # Contains match.
    for wanted in preferred:
        wanted_l = str(wanted).lower()
        for ch in channels:
            if wanted_l in ch.lower():
                return ch

    # Fallback to first EEG-like channel.
    for ch in channels:
        if "eeg" in ch.lower():
            return ch

    raise RuntimeError(
        "No EEG channel found. Available channels: " + ", ".join(channels)
    )


def _load_raw_with_annotations(pair: EEGFilePair, cfg: Any) -> Tuple[Any, str]:
    """
    Loads one Sleep-EDF PSG file and attaches hypnogram annotations.
    """
    _require_mne()

    raw = mne.io.read_raw_edf(
        str(pair.psg_file),
        preload=True,
        verbose=False,
    )

    annotations = mne.read_annotations(str(pair.hypnogram_file))
    raw.set_annotations(annotations, emit_warning=False)

    channel = _select_eeg_channel(raw, cfg)

    # Pick only the EEG channel used for feature extraction.
    raw.pick_channels([channel], ordered=True)

    # Filtering before feature extraction.
    l_freq = getattr(cfg, "eeg_filter_l_freq", None)
    h_freq = getattr(cfg, "eeg_filter_h_freq", None)

    if l_freq is not None or h_freq is not None:
        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design="firwin",
            verbose=False,
        )

    # Resampling to reduce computational load and normalize subject files.
    target_sfreq = getattr(cfg, "eeg_resample_hz", None)
    if target_sfreq is not None:
        if abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), npad="auto", verbose=False)

    return raw, channel


def _raw_to_fixed_epochs(raw: Any, cfg: Any) -> Tuple[np.ndarray, float, List[str]]:
    """
    Converts continuous one-channel Raw data into fixed-length epochs.
    """
    sfreq = float(raw.info["sfreq"])
    epoch_seconds = int(cfg.eeg_epoch_seconds)
    samples_per_epoch = int(round(epoch_seconds * sfreq))

    if samples_per_epoch < 2:
        raise RuntimeError(
            f"samples_per_epoch too small: {samples_per_epoch}"
        )

    data = raw.get_data()

    if data.ndim != 2 or data.shape[0] != 1:
        raise RuntimeError(
            f"Expected one selected EEG channel, got data shape: {data.shape}"
        )

    signal = np.asarray(data[0], dtype=float)

    n_epochs_total = len(signal) // samples_per_epoch

    if n_epochs_total < 1:
        raise RuntimeError("No full EEG epochs could be created")

    usable = n_epochs_total * samples_per_epoch
    signal = signal[:usable]

    epochs = signal.reshape(n_epochs_total, samples_per_epoch)

    stages = epoch_sleep_stages(
        raw=raw,
        n_epochs=n_epochs_total,
        epoch_seconds=epoch_seconds,
        cfg=cfg,
    )

    epochs, stages, kept_idx = _filter_epoch_rows_by_stage(
        epochs=epochs,
        stages=stages,
        cfg=cfg,
    )

    max_epochs = getattr(cfg, "max_epochs_per_subject", None)
    if max_epochs is not None:
        max_epochs = int(max_epochs)
        epochs = epochs[:max_epochs]
        stages = stages[:max_epochs]

    if len(epochs) < int(cfg.min_epochs_per_subject):
        raise RuntimeError(
            f"Subject has too few usable epochs after filtering: "
            f"{len(epochs)} < {cfg.min_epochs_per_subject}"
        )

    return epochs, sfreq, list(stages)


def load_subject_feature_frame(
    pair: EEGFilePair,
    rng: RNGSequence,
    cfg: Any,
) -> LoadedSubject:
    """
    Loads one subject/recording, extracts EEG + RNG features, and returns a frame.
    """
    _log(
        cfg,
        f"[Phase3.1-Loader] Loading subject={pair.subject_id} "
        f"recording={pair.recording_id}",
    )

    raw, channel = _load_raw_with_annotations(pair, cfg)

    epochs, sfreq, sleep_stages = _raw_to_fixed_epochs(raw, cfg)

    feature_frame = compute_phase3_1_feature_frame(
        eeg_epochs=epochs,
        sfreq=sfreq,
        rng_bits=rng.bits,
        cfg=cfg,
        subject_id=pair.subject_id,
        sleep_stages=sleep_stages,
    )

    feature_frame["recording_id"] = pair.recording_id
    feature_frame["channel_used"] = channel
    feature_frame["sfreq"] = float(sfreq)
    feature_frame["psg_file"] = str(pair.psg_file)
    feature_frame["hypnogram_file"] = str(pair.hypnogram_file)

    # Put identity columns first.
    identity_cols = [
        "subject_id",
        "recording_id",
        "epoch_idx",
        "sleep_stage",
        "channel_used",
        "sfreq",
    ]
    existing_identity = [c for c in identity_cols if c in feature_frame.columns]
    other_cols = [c for c in feature_frame.columns if c not in existing_identity]
    feature_frame = feature_frame[existing_identity + other_cols]

    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.dropna().reset_index(drop=True)

    loaded = LoadedSubject(
        subject_id=pair.subject_id,
        recording_id=pair.recording_id,
        channel_used=channel,
        sfreq=float(sfreq),
        n_epochs=int(len(feature_frame)),
        feature_frame=feature_frame,
        psg_file=pair.psg_file,
        hypnogram_file=pair.hypnogram_file,
    )

    _log(
        cfg,
        f"[Phase3.1-Loader] Loaded {loaded.n_epochs} epochs "
        f"for subject={loaded.subject_id}, channel={loaded.channel_used}",
    )

    return loaded


# =========================================================
# Full Phase III.1 dataset loader
# =========================================================

def load_phase3_1_dataset(cfg: Any) -> Phase3_1Dataset:
    """
    Main dataset loading function for Phase III.1.

    Steps:
        1. Discover Sleep-EDF PSG/Hypnogram pairs.
        2. Load single legitimate RNG sample.
        3. Convert RNG uint8 to bits.
        4. For each EEG subject:
            - load PSG + hypnogram
            - build 30s epochs
            - extract EEG observed/informational features
            - extract RNG informational and quantum-aware proxies
            - align RNG feature windows to EEG epochs
        5. Concatenate subject frames.
    """
    pairs = discover_sleep_edf_pairs(cfg)
    rng = load_rng_sequence(cfg)

    _log(cfg, f"[Phase3.1-Loader] Matched EEG pairs: {len(pairs)}")

    subjects: List[LoadedSubject] = []
    errors: List[Dict[str, str]] = []

    for pair in pairs:
        try:
            loaded = load_subject_feature_frame(
                pair=pair,
                rng=rng,
                cfg=cfg,
            )
            subjects.append(loaded)

            if getattr(cfg, "save_intermediate_features", True):
                cfg.interim_dir.mkdir(parents=True, exist_ok=True)
                out_file = cfg.interim_dir / f"{pair.recording_id}_features.csv"
                loaded.feature_frame.to_csv(out_file, index=False)

        except Exception as exc:
            errors.append(
                {
                    "subject_id": pair.subject_id,
                    "recording_id": pair.recording_id,
                    "error": str(exc),
                }
            )
            _log(
                cfg,
                f"[Phase3.1-Loader] WARNING: failed subject={pair.subject_id} "
                f"recording={pair.recording_id}: {exc}",
            )

    if not subjects:
        raise RuntimeError(
            "No subjects were successfully loaded. "
            f"Errors: {errors}"
        )

    df = pd.concat([s.feature_frame for s in subjects], axis=0, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 10:
        raise RuntimeError("Combined Phase III.1 dataset has too few rows")

    n_subjects = len(subjects)

    if (
        getattr(cfg, "holdout_mode", "") == "leave_one_subject_out"
        and n_subjects < int(cfg.min_subjects_for_loso)
    ):
        _log(
            cfg,
            "[Phase3.1-Loader] WARNING: insufficient subjects for LOSO holdout. "
            f"Loaded={n_subjects}, required={cfg.min_subjects_for_loso}. "
            f"Runner should fallback to {cfg.fallback_holdout_mode}.",
        )

    metadata: Dict[str, Any] = {
        "project_name": getattr(cfg, "project_name", "Phase3.1-Advanced-EEG-RNG"),
        "phase_name": getattr(cfg, "phase_name", "Phase III.1"),
        "n_subjects_loaded": int(n_subjects),
        "subject_ids": [s.subject_id for s in subjects],
        "recording_ids": [s.recording_id for s in subjects],
        "total_rows": int(len(df)),
        "rng_source": rng.source,
        "rng_n_uint8": rng.n_uint8,
        "rng_n_bits": rng.n_bits,
        "rng_use_bits": bool(getattr(cfg, "rng_use_bits", True)),
        "rng_window_size": int(getattr(cfg, "rng_window_size", 64)),
        "rng_alignment_mode": str(getattr(cfg, "rng_alignment_mode", "")),
        "eeg_epoch_seconds": int(getattr(cfg, "eeg_epoch_seconds", 30)),
        "channels_used": {s.recording_id: s.channel_used for s in subjects},
        "errors": errors,
    }

    if getattr(cfg, "save_intermediate_features", True):
        cfg.interim_dir.mkdir(parents=True, exist_ok=True)
        combined_file = cfg.interim_dir / "phase3_1_combined_features.csv"
        df.to_csv(combined_file, index=False)

        metadata_file = cfg.interim_dir / "phase3_1_loader_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    _log(
        cfg,
        f"[Phase3.1-Loader] Final combined dataset: rows={len(df)}, "
        f"subjects={n_subjects}",
    )

    return Phase3_1Dataset(
        df=df,
        subjects=tuple(subjects),
        rng=rng,
        metadata=metadata,
    )


# =========================================================
# Convenience function
# =========================================================

def load_phase3_1(cfg: Any) -> pd.DataFrame:
    """
    Convenience loader returning only the combined dataframe.
    """
    dataset = load_phase3_1_dataset(cfg)
    return dataset.df


# =========================================================
# Optional diagnostic entrypoint
# =========================================================

if __name__ == "__main__":
    from config.phase3_1_config import load_phase3_1_config

    cfg = load_phase3_1_config()
    dataset = load_phase3_1_dataset(cfg)

    print("\n==============================")
    print("Phase III.1 Loader Diagnostic")
    print("==============================")
    print(f"Rows: {len(dataset.df)}")
    print(f"Subjects: {dataset.metadata['n_subjects_loaded']}")
    print(f"RNG uint8: {dataset.rng.n_uint8}")
    print(f"RNG bits: {dataset.rng.n_bits}")
    print("Columns:")
    print(list(dataset.df.columns))