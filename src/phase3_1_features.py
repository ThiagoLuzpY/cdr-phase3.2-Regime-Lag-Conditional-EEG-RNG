from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Optional sklearn support
# =========================================================

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    KMeans = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False


# =========================================================
# Data containers
# =========================================================

@dataclass(frozen=True)
class ScoreTransformSpec:
    """
    Stores train-fitted normalization and binning metadata for a compact score.

    Used for:
        - eeg_info_bin
        - rng_info_bin
        - q_rng_bin
    """

    name: str
    source_features: Tuple[str, ...]
    means: Dict[str, float]
    stds: Dict[str, float]
    edges: Tuple[float, ...]
    n_bins: int


@dataclass(frozen=True)
class LatentTransformSpec:
    """
    Stores train-fitted metadata for latent state assignment.
    """

    method: str
    features: Tuple[str, ...]
    k: int
    means: Dict[str, float]
    stds: Dict[str, float]
    centers: np.ndarray


@dataclass(frozen=True)
class FeatureLayerArtifacts:
    """
    Train-fitted artifacts needed to transform features without leakage.
    """

    eeg_info_spec: Optional[ScoreTransformSpec]
    rng_info_spec: Optional[ScoreTransformSpec]
    q_rng_spec: Optional[ScoreTransformSpec]
    latent_spec: Optional[LatentTransformSpec]


@dataclass(frozen=True)
class FeatureLayerResult:
    """
    Output from fitting/applying Phase III.1 feature layers.
    """

    df: pd.DataFrame
    artifacts: FeatureLayerArtifacts


@dataclass(frozen=True)
class StateComponentSpec:
    """
    Stores discretization metadata for a single component of a state model.
    """

    component: str
    n_bins: int
    pre_binned: bool
    edges: Tuple[float, ...]
    sensitivity: bool = False


@dataclass(frozen=True)
class StateComponentResult:
    """
    Output for one state model's component matrix.
    """

    components_df: pd.DataFrame
    specs: Tuple[StateComponentSpec, ...]
    n_states: int
    transitions_per_state: float
    observed_states: int
    transitions_per_observed_state: float


# =========================================================
# Numeric helpers
# =========================================================

_EPS = 1e-12


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        value = float(x)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _safe_log2(x: np.ndarray | float) -> np.ndarray | float:
    return np.log2(np.maximum(x, _EPS))


def _binary_entropy(p: float) -> float:
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))


def _normalize_probability_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.maximum(values, 0.0)

    total = values.sum()
    if total <= 0:
        return np.ones_like(values, dtype=float) / max(len(values), 1)

    return values / total


def _safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return 0.0

    return float(np.std(values, ddof=0))


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def _replace_nonfinite_df(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(fill_value)
    return out


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = fill_value

    return out


# =========================================================
# EEG feature extraction
# =========================================================

def _periodogram_power(epoch: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight FFT periodogram without scipy dependency.
    """
    epoch = _as_float_array(epoch)

    if len(epoch) < 4:
        return np.array([0.0]), np.array([0.0])

    epoch = epoch - np.mean(epoch)
    window = np.hanning(len(epoch))
    signal = epoch * window

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / float(sfreq))
    power = np.abs(np.fft.rfft(signal)) ** 2

    return freqs, power


def _band_power(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)

    if not np.any(mask):
        return 0.0

    y = power[mask]
    x = freqs[mask]

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


def _spectral_entropy_from_bandpowers(bandpowers: Dict[str, float]) -> float:
    values = np.array(list(bandpowers.values()), dtype=float)
    probs = _normalize_probability_vector(values)

    entropy = -float(np.sum(probs * _safe_log2(probs)))
    max_entropy = math.log2(max(len(probs), 2))

    return float(entropy / max_entropy)


def _permutation_entropy(
    values: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Computes normalized permutation entropy for one epoch.

    This implementation is dependency-free and intentionally conservative.
    """
    x = _as_float_array(values)

    if order < 2:
        raise ValueError("order must be >= 2")

    if delay < 1:
        raise ValueError("delay must be >= 1")

    n_required = delay * (order - 1) + 1

    if len(x) < n_required + 1:
        return 0.0

    patterns = list(permutations(range(order)))
    pattern_to_idx = {p: i for i, p in enumerate(patterns)}
    counts = np.zeros(len(patterns), dtype=float)

    n_vectors = len(x) - delay * (order - 1)

    for i in range(n_vectors):
        window = x[i : i + delay * order : delay]

        if len(window) != order:
            continue

        ranks = tuple(np.argsort(window, kind="mergesort"))

        idx = pattern_to_idx.get(ranks)
        if idx is not None:
            counts[idx] += 1.0

    probs = _normalize_probability_vector(counts)
    entropy = -float(np.sum(probs * _safe_log2(probs)))

    if normalize:
        max_entropy = math.log2(math.factorial(order))
        if max_entropy <= 0:
            return 0.0
        entropy /= max_entropy

    return float(entropy)


def _hjorth_parameters(values: np.ndarray) -> Tuple[float, float]:
    """
    Returns Hjorth mobility and complexity.
    """
    x = _as_float_array(values)

    if len(x) < 4:
        return 0.0, 0.0

    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    if var_x <= _EPS:
        mobility = 0.0
    else:
        mobility = math.sqrt(max(var_dx, 0.0) / max(var_x, _EPS))

    if var_dx <= _EPS:
        complexity = 0.0
    else:
        mobility_dx = math.sqrt(max(var_ddx, 0.0) / max(var_dx, _EPS))
        complexity = mobility_dx / max(mobility, _EPS)

    return float(mobility), float(complexity)


def _line_length(values: np.ndarray) -> float:
    x = _as_float_array(values)

    if len(x) < 2:
        return 0.0

    return float(np.sum(np.abs(np.diff(x))) / max(len(x) - 1, 1))


def compute_eeg_epoch_features(
    epochs: np.ndarray,
    sfreq: float,
    cfg: Any,
    sleep_stages: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Computes observed and informational EEG features per epoch.

    Parameters
    ----------
    epochs:
        Array with shape (n_epochs, n_samples).
    sfreq:
        Sampling frequency in Hz.
    cfg:
        Phase3_1Config instance.
    sleep_stages:
        Optional sleep-stage labels aligned to epochs.

    Returns
    -------
    pd.DataFrame
        One row per EEG epoch.
    """
    arr = np.asarray(epochs, dtype=float)

    if arr.ndim != 2:
        raise ValueError("epochs must have shape (n_epochs, n_samples)")

    n_epochs = arr.shape[0]

    rows: List[Dict[str, float | str | int]] = []

    for epoch_idx in range(n_epochs):
        epoch = arr[epoch_idx]

        freqs, power = _periodogram_power(epoch, sfreq)

        bandpowers: Dict[str, float] = {}
        for band_name, (lo, hi) in cfg.eeg_bands.items():
            bandpowers[f"{band_name}_power"] = _band_power(freqs, power, lo, hi)

        delta_power = bandpowers.get("delta_power", 0.0)
        theta_power = bandpowers.get("theta_power", 0.0)
        alpha_power = bandpowers.get("alpha_power", 0.0)
        beta_power = bandpowers.get("beta_power", 0.0)

        alpha_delta_ratio = alpha_power / max(delta_power, _EPS)
        delta_alpha_balance = (delta_power - alpha_power) / max(delta_power + alpha_power, _EPS)

        spectral_entropy = _spectral_entropy_from_bandpowers(bandpowers)
        permutation_entropy = _permutation_entropy(
            epoch,
            order=cfg.permutation_entropy_order,
            delay=cfg.permutation_entropy_delay,
            normalize=True,
        )
        hjorth_mobility, hjorth_complexity = _hjorth_parameters(epoch)
        line_length = _line_length(epoch)

        row: Dict[str, float | str | int] = {
            "epoch_idx": int(epoch_idx),
            "delta_power": float(delta_power),
            "theta_power": float(theta_power),
            "alpha_power": float(alpha_power),
            "beta_power": float(beta_power),
            "alpha_delta_ratio": float(alpha_delta_ratio),
            "delta_alpha_balance": float(delta_alpha_balance),
            "spectral_entropy": float(spectral_entropy),
            "permutation_entropy": float(permutation_entropy),
            "hjorth_mobility": float(hjorth_mobility),
            "hjorth_complexity": float(hjorth_complexity),
            "line_length": float(line_length),
        }

        if sleep_stages is not None:
            row["sleep_stage"] = str(sleep_stages[epoch_idx])

        rows.append(row)

    df = pd.DataFrame(rows)

    df = add_eeg_temporal_features(df, cfg)
    df = _replace_nonfinite_df(df)

    return df


def add_eeg_temporal_features(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """
    Adds rolling volatility and state-instability EEG features.
    """
    out = df.copy()

    window = int(cfg.eeg_info_rolling_window_epochs)
    window = max(window, 2)

    if "delta_power" not in out.columns:
        out["delta_power"] = 0.0

    if "alpha_power" not in out.columns:
        out["alpha_power"] = 0.0

    if "delta_alpha_balance" not in out.columns:
        out["delta_alpha_balance"] = (
            out["delta_power"] - out["alpha_power"]
        ) / (out["delta_power"] + out["alpha_power"] + _EPS)

    # Rolling volatility of neural band organization.
    out["bandpower_volatility"] = (
        out["delta_alpha_balance"]
        .astype(float)
        .rolling(window=window, min_periods=2)
        .std()
        .fillna(0.0)
    )

    # Local instability proxy: rolling average absolute change in balance.
    balance_diff = out["delta_alpha_balance"].astype(float).diff().abs().fillna(0.0)

    out["state_instability"] = (
        balance_diff.rolling(window=window, min_periods=2).mean().fillna(0.0)
    )

    return out


# =========================================================
# RNG feature extraction
# =========================================================

def convert_uint8_to_bits(sequence_uint8: np.ndarray) -> np.ndarray:
    """
    Converts uint8 values to binary bits.

    This preserves a single RNG request and does not concatenate independent samples.
    """
    seq = np.asarray(sequence_uint8, dtype=np.uint8).reshape(-1)
    return np.unpackbits(seq).astype(np.uint8)


def _window_starts_for_alignment(n_bits: int, n_rows: int, window_size: int) -> np.ndarray:
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")

    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    window_size = max(1, min(int(window_size), int(n_bits)))

    max_start = max(n_bits - window_size, 0)

    if n_rows == 1:
        return np.array([0], dtype=int)

    starts = np.linspace(0, max_start, num=n_rows)
    return np.rint(starts).astype(int)


def _pair_frequencies(bits: np.ndarray) -> Dict[str, float]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    result = {
        "pair_frequency_00": 0.0,
        "pair_frequency_01": 0.0,
        "pair_frequency_10": 0.0,
        "pair_frequency_11": 0.0,
    }

    if len(bits) < 2:
        return result

    pairs = bits[:-1].astype(int) * 2 + bits[1:].astype(int)
    counts = np.bincount(pairs, minlength=4).astype(float)
    probs = counts / max(counts.sum(), 1.0)

    result["pair_frequency_00"] = float(probs[0])
    result["pair_frequency_01"] = float(probs[1])
    result["pair_frequency_10"] = float(probs[2])
    result["pair_frequency_11"] = float(probs[3])

    return result


def _transition_rate(bits: np.ndarray) -> float:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if len(bits) < 2:
        return 0.0

    return float(np.mean(bits[1:] != bits[:-1]))


def _run_lengths(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if len(bits) == 0:
        return np.array([], dtype=float)

    lengths: List[int] = []
    current = int(bits[0])
    count = 1

    for value in bits[1:]:
        value = int(value)

        if value == current:
            count += 1
        else:
            lengths.append(count)
            current = value
            count = 1

    lengths.append(count)

    return np.asarray(lengths, dtype=float)


def _entropy_rate_proxy_from_pairs(pair_probs: Dict[str, float]) -> float:
    probs = np.array(
        [
            pair_probs.get("pair_frequency_00", 0.0),
            pair_probs.get("pair_frequency_01", 0.0),
            pair_probs.get("pair_frequency_10", 0.0),
            pair_probs.get("pair_frequency_11", 0.0),
        ],
        dtype=float,
    )

    probs = _normalize_probability_vector(probs)

    # Entropy of 2-bit patterns, normalized to [0, 1].
    entropy = -float(np.sum(probs * _safe_log2(probs)))
    return float(entropy / 2.0)


def _surprise_index_from_pairs(pair_probs: Dict[str, float]) -> float:
    probs = np.array(
        [
            pair_probs.get("pair_frequency_00", 0.0),
            pair_probs.get("pair_frequency_01", 0.0),
            pair_probs.get("pair_frequency_10", 0.0),
            pair_probs.get("pair_frequency_11", 0.0),
        ],
        dtype=float,
    )

    probs = _normalize_probability_vector(probs)

    # Mean local surprise of pair distribution.
    surprise = -float(np.sum(probs * _safe_log2(probs)))

    # Normalize by maximum pair surprise.
    return float(surprise / 2.0)


def _micro_cluster_deviation(pair_probs: Dict[str, float]) -> float:
    probs = np.array(
        [
            pair_probs.get("pair_frequency_00", 0.0),
            pair_probs.get("pair_frequency_01", 0.0),
            pair_probs.get("pair_frequency_10", 0.0),
            pair_probs.get("pair_frequency_11", 0.0),
        ],
        dtype=float,
    )

    return float(np.max(np.abs(probs - 0.25)))


def compute_rng_epoch_features(
    bits: np.ndarray,
    n_rows: int,
    cfg: Any,
) -> pd.DataFrame:
    """
    Computes RNG local features aligned to the number of EEG epochs/rows.

    The RNG sequence is not repeated as raw bits. Instead, deterministic local
    windows are placed across the single legitimate RNG sample and transformed
    into local informational features.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if len(bits) < 2:
        raise ValueError("RNG bit sequence must contain at least 2 bits")

    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    window_size = int(cfg.rng_window_size)
    starts = _window_starts_for_alignment(len(bits), n_rows, window_size)

    rows: List[Dict[str, float | int]] = []

    for row_idx, start in enumerate(starts):
        start = int(start)
        end = min(start + window_size, len(bits))
        window = bits[start:end]

        if len(window) == 0:
            window = bits[-1:]

        center_idx = start + len(window) // 2
        center_idx = min(max(center_idx, 0), len(bits) - 1)

        rng_bit = int(bits[center_idx])
        bit_balance_local = float(np.mean(window))
        transition_rate = _transition_rate(window)

        pair_probs = _pair_frequencies(window)
        entropy_rate_proxy = _entropy_rate_proxy_from_pairs(pair_probs)

        runs = _run_lengths(window)
        run_length_mean = _safe_mean(runs)
        run_length_max = float(np.max(runs)) if len(runs) else 0.0
        run_length_std = _safe_std(runs)

        # More long runs → more compressible. Alternation → less compressible.
        n_runs = len(runs)
        compressibility_proxy = 1.0 - (n_runs / max(len(window), 1))
        compressibility_proxy = float(np.clip(compressibility_proxy, 0.0, 1.0))

        surprise_index = _surprise_index_from_pairs(pair_probs)
        transition_asymmetry = float(
            pair_probs["pair_frequency_01"] - pair_probs["pair_frequency_10"]
        )

        micro_cluster_deviation = _micro_cluster_deviation(pair_probs)

        rng_entropy_local = _binary_entropy(bit_balance_local)

        row: Dict[str, float | int] = {
            "rng_window_id": int(row_idx),
            "rng_window_start": int(start),
            "rng_window_end": int(end),
            "rng_bit": int(rng_bit),
            "bit_balance_local": float(bit_balance_local),
            "transition_rate": float(transition_rate),
            "rng_entropy_local": float(rng_entropy_local),
            "entropy_rate_proxy": float(entropy_rate_proxy),
            "run_length_mean": float(run_length_mean),
            "run_length_max": float(run_length_max),
            "run_length_std": float(run_length_std),
            "compressibility_proxy": float(compressibility_proxy),
            "surprise_index": float(surprise_index),
            "transition_asymmetry": float(transition_asymmetry),
            "micro_cluster_deviation": float(micro_cluster_deviation),
        }

        row.update(pair_probs)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_quantum_aware_rng_proxies(df, cfg)
    df = _replace_nonfinite_df(df)

    return df


def add_quantum_aware_rng_proxies(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """
    Adds Q_t-style proxy features derived from local RNG structure.

    These are not claims about direct quantum-state measurement.
    They are operational proxies computed from observed RNG output.
    """
    out = df.copy()

    window = max(3, int(getattr(cfg, "eeg_info_rolling_window_epochs", 5)))

    if "entropy_rate_proxy" not in out.columns:
        out["entropy_rate_proxy"] = 0.0

    if "transition_asymmetry" not in out.columns:
        out["transition_asymmetry"] = 0.0

    if "run_length_std" not in out.columns:
        out["run_length_std"] = 0.0

    if "surprise_index" not in out.columns:
        out["surprise_index"] = 0.0

    if "micro_cluster_deviation" not in out.columns:
        out["micro_cluster_deviation"] = 0.0

    if "bit_balance_local" not in out.columns:
        out["bit_balance_local"] = 0.5

    entropy_roll = (
        out["entropy_rate_proxy"]
        .astype(float)
        .rolling(window=window, min_periods=2)
        .mean()
        .fillna(out["entropy_rate_proxy"].astype(float))
    )

    surprise_roll = (
        out["surprise_index"]
        .astype(float)
        .rolling(window=window, min_periods=2)
        .mean()
        .fillna(out["surprise_index"].astype(float))
    )

    run_roll = (
        out["run_length_std"]
        .astype(float)
        .rolling(window=window, min_periods=2)
        .mean()
        .fillna(out["run_length_std"].astype(float))
    )

    out["q_entropy_rate_drift"] = (
        out["entropy_rate_proxy"].astype(float) - entropy_roll
    ).abs()

    out["q_transition_asymmetry"] = out["transition_asymmetry"].astype(float).abs()

    out["q_run_instability"] = (
        out["run_length_std"].astype(float) - run_roll
    ).abs()

    out["q_surprise_burst"] = (
        out["surprise_index"].astype(float) - surprise_roll
    ).abs()

    out["q_micro_cluster_deviation"] = out["micro_cluster_deviation"].astype(float)

    out["q_local_balance_deviation"] = (
        out["bit_balance_local"].astype(float) - 0.5
    ).abs()

    return out


# =========================================================
# Combined EEG + RNG frame construction
# =========================================================

def combine_eeg_rng_features(
    eeg_df: pd.DataFrame,
    rng_df: pd.DataFrame,
    subject_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Combines already-computed EEG and RNG features row-wise.

    Both frames must have the same number of rows.
    """
    if len(eeg_df) != len(rng_df):
        raise ValueError(
            f"EEG and RNG frames must have the same length. "
            f"Got EEG={len(eeg_df)}, RNG={len(rng_df)}"
        )

    eeg = eeg_df.reset_index(drop=True).copy()
    rng = rng_df.reset_index(drop=True).copy()

    # Avoid accidental duplicate columns.
    duplicate_cols = set(eeg.columns).intersection(set(rng.columns))
    duplicate_cols.discard("epoch_idx")

    if duplicate_cols:
        rng = rng.rename(columns={col: f"rng_{col}" for col in duplicate_cols})

    out = pd.concat([eeg, rng], axis=1)

    if subject_id is not None:
        out.insert(0, "subject_id", str(subject_id))

    out = _replace_nonfinite_df(out)

    return out


def compute_phase3_1_feature_frame(
    eeg_epochs: np.ndarray,
    sfreq: float,
    rng_bits: np.ndarray,
    cfg: Any,
    subject_id: Optional[str] = None,
    sleep_stages: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Full feature extraction for one subject:
        EEG epochs → EEG features
        RNG bits   → local RNG features aligned to EEG rows
        Combined subject-level feature frame
    """
    eeg_df = compute_eeg_epoch_features(
        epochs=eeg_epochs,
        sfreq=sfreq,
        cfg=cfg,
        sleep_stages=sleep_stages,
    )

    rng_df = compute_rng_epoch_features(
        bits=rng_bits,
        n_rows=len(eeg_df),
        cfg=cfg,
    )

    return combine_eeg_rng_features(
        eeg_df=eeg_df,
        rng_df=rng_df,
        subject_id=subject_id,
    )


# =========================================================
# Train-only scoring helpers for I_t and Q_t
# =========================================================

def _normalize_train_index(train_index: Optional[Sequence[int]], n: int) -> np.ndarray:
    if train_index is None:
        return np.arange(n, dtype=int)

    idx = np.asarray(train_index, dtype=int).reshape(-1)

    if len(idx) == 0:
        raise ValueError("train_index cannot be empty")

    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("train_index contains out-of-range indices")

    return idx


def _fit_score_spec(
    df: pd.DataFrame,
    name: str,
    source_features: Sequence[str],
    train_index: Sequence[int],
    n_bins: int,
    quantiles: Tuple[float, ...],
) -> ScoreTransformSpec:
    features = tuple(source_features)

    work = _ensure_columns(df, features, fill_value=0.0)
    train = work.iloc[np.asarray(train_index, dtype=int)]

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    z_train_parts: List[np.ndarray] = []

    for col in features:
        values = train[col].astype(float).to_numpy()
        mean = _safe_mean(values)
        std = _safe_std(values)

        if std <= _EPS:
            std = 1.0

        means[col] = float(mean)
        stds[col] = float(std)

        z_train_parts.append((values - mean) / std)

    if len(z_train_parts) == 0:
        score_train = np.zeros(len(train), dtype=float)
    else:
        score_train = np.mean(np.vstack(z_train_parts), axis=0)

    edges = _fit_quantile_edges(score_train, quantiles)

    return ScoreTransformSpec(
        name=name,
        source_features=features,
        means=means,
        stds=stds,
        edges=tuple(float(e) for e in edges),
        n_bins=int(n_bins),
    )


def _apply_score_spec(df: pd.DataFrame, spec: ScoreTransformSpec) -> Tuple[np.ndarray, np.ndarray]:
    work = _ensure_columns(df, spec.source_features, fill_value=0.0)

    z_parts: List[np.ndarray] = []

    for col in spec.source_features:
        values = work[col].astype(float).to_numpy()
        mean = spec.means.get(col, 0.0)
        std = spec.stds.get(col, 1.0)

        if std <= _EPS:
            std = 1.0

        z_parts.append((values - mean) / std)

    if len(z_parts) == 0:
        score = np.zeros(len(df), dtype=float)
    else:
        score = np.mean(np.vstack(z_parts), axis=0)

    bins = _apply_edges(score, spec.edges, spec.n_bins)

    return score.astype(float), bins.astype(int)


# =========================================================
# Latent Z_t fitting/applying
# =========================================================

def _fit_latent_spec(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Sequence[int],
) -> LatentTransformSpec:
    features = tuple(cfg.latent_features)
    k = int(cfg.latent_k)

    work = _ensure_columns(df, features, fill_value=0.0)
    train = work.iloc[np.asarray(train_index, dtype=int)].copy()

    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    x_parts: List[np.ndarray] = []

    for col in features:
        values = train[col].astype(float).to_numpy()
        mean = _safe_mean(values)
        std = _safe_std(values)

        if std <= _EPS:
            std = 1.0

        means[col] = float(mean)
        stds[col] = float(std)
        x_parts.append((values - mean) / std)

    if len(x_parts) == 0:
        x_train = np.zeros((len(train), 1), dtype=float)
    else:
        x_train = np.vstack(x_parts).T

    if len(x_train) < k:
        raise RuntimeError(
            f"Cannot fit latent model with k={k}; only {len(x_train)} training rows"
        )

    if SKLEARN_AVAILABLE:
        model = KMeans(
            n_clusters=k,
            random_state=int(cfg.latent_random_state),
            n_init=10,
        )
        model.fit(x_train)
        centers = np.asarray(model.cluster_centers_, dtype=float)
    else:
        warnings.warn(
            "scikit-learn not available. Falling back to deterministic simple k-means.",
            RuntimeWarning,
        )
        centers = _simple_kmeans_centers(
            x_train,
            k=k,
            seed=int(cfg.latent_random_state),
            max_iter=100,
        )

    return LatentTransformSpec(
        method=str(cfg.latent_method),
        features=features,
        k=k,
        means=means,
        stds=stds,
        centers=centers,
    )


def _apply_latent_spec(df: pd.DataFrame, spec: LatentTransformSpec) -> np.ndarray:
    work = _ensure_columns(df, spec.features, fill_value=0.0)

    x_parts: List[np.ndarray] = []

    for col in spec.features:
        values = work[col].astype(float).to_numpy()
        mean = spec.means.get(col, 0.0)
        std = spec.stds.get(col, 1.0)

        if std <= _EPS:
            std = 1.0

        x_parts.append((values - mean) / std)

    if len(x_parts) == 0:
        x = np.zeros((len(df), 1), dtype=float)
    else:
        x = np.vstack(x_parts).T

    centers = np.asarray(spec.centers, dtype=float)

    # Assign nearest center.
    distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(distances, axis=1)

    return labels.astype(int)


def _simple_kmeans_centers(
    x: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Small deterministic fallback k-means implementation.
    """
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=float)

    if len(x) < k:
        raise ValueError("not enough samples for k-means fallback")

    initial_idx = rng.choice(len(x), size=k, replace=False)
    centers = x[initial_idx].copy()

    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()

        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = x[mask].mean(axis=0)

        if np.allclose(new_centers, centers):
            break

        centers = new_centers

    return centers


# =========================================================
# Feature layer fitting/applying
# =========================================================

def fit_transform_phase3_1_layers(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
) -> FeatureLayerResult:
    """
    Fits I_t, Z_t and Q_t layer transformations on train rows only,
    then applies them to the full dataframe.

    If train_index is None, all rows are used. For final validation,
    the runner should pass a train index to avoid leakage.
    """
    out = df.copy().reset_index(drop=True)
    out = _replace_nonfinite_df(out)

    idx_train = _normalize_train_index(train_index, len(out))

    eeg_info_spec: Optional[ScoreTransformSpec] = None
    rng_info_spec: Optional[ScoreTransformSpec] = None
    q_rng_spec: Optional[ScoreTransformSpec] = None
    latent_spec: Optional[LatentTransformSpec] = None

    q3 = tuple(cfg.quantiles_by_bins[3])

    # ------------------------------
    # EEG informational compact score/bin
    # ------------------------------
    if getattr(cfg, "use_informational_layer", True):
        eeg_info_spec = _fit_score_spec(
            df=out,
            name=str(cfg.eeg_info_component_name),
            source_features=cfg.eeg_info_source_features,
            train_index=idx_train,
            n_bins=3,
            quantiles=q3,
        )
        score, bins = _apply_score_spec(out, eeg_info_spec)
        out["eeg_info_score"] = score
        out[str(cfg.eeg_info_component_name)] = bins

        rng_info_spec = _fit_score_spec(
            df=out,
            name=str(cfg.rng_info_component_name),
            source_features=cfg.rng_info_source_features,
            train_index=idx_train,
            n_bins=3,
            quantiles=q3,
        )
        score, bins = _apply_score_spec(out, rng_info_spec)
        out["rng_info_score"] = score
        out[str(cfg.rng_info_component_name)] = bins

    # ------------------------------
    # Quantum-aware compact score/bin
    # ------------------------------
    if getattr(cfg, "use_quantum_proxy_layer", True):
        q_rng_spec = _fit_score_spec(
            df=out,
            name=str(cfg.q_component_name),
            source_features=cfg.q_source_features,
            train_index=idx_train,
            n_bins=3,
            quantiles=q3,
        )
        score, bins = _apply_score_spec(out, q_rng_spec)
        out["q_rng_score"] = score
        out[str(cfg.q_component_name)] = bins

    # ------------------------------
    # Latent state Z_t
    # ------------------------------
    if getattr(cfg, "use_latent_layer", True):
        latent_spec = _fit_latent_spec(
            df=out,
            cfg=cfg,
            train_index=idx_train,
        )
        labels = _apply_latent_spec(out, latent_spec)
        out[str(cfg.latent_component_name)] = labels.astype(int)

    out = _replace_nonfinite_df(out)

    artifacts = FeatureLayerArtifacts(
        eeg_info_spec=eeg_info_spec,
        rng_info_spec=rng_info_spec,
        q_rng_spec=q_rng_spec,
        latent_spec=latent_spec,
    )

    return FeatureLayerResult(df=out, artifacts=artifacts)


def transform_phase3_1_layers(
    df: pd.DataFrame,
    artifacts: FeatureLayerArtifacts,
    cfg: Any,
) -> pd.DataFrame:
    """
    Applies previously fitted feature-layer artifacts to a new dataframe.
    """
    out = df.copy().reset_index(drop=True)
    out = _replace_nonfinite_df(out)

    if artifacts.eeg_info_spec is not None:
        score, bins = _apply_score_spec(out, artifacts.eeg_info_spec)
        out["eeg_info_score"] = score
        out[str(cfg.eeg_info_component_name)] = bins

    if artifacts.rng_info_spec is not None:
        score, bins = _apply_score_spec(out, artifacts.rng_info_spec)
        out["rng_info_score"] = score
        out[str(cfg.rng_info_component_name)] = bins

    if artifacts.q_rng_spec is not None:
        score, bins = _apply_score_spec(out, artifacts.q_rng_spec)
        out["q_rng_score"] = score
        out[str(cfg.q_component_name)] = bins

    if artifacts.latent_spec is not None:
        labels = _apply_latent_spec(out, artifacts.latent_spec)
        out[str(cfg.latent_component_name)] = labels.astype(int)

    out = _replace_nonfinite_df(out)

    return out


# =========================================================
# State-model discretization
# =========================================================

def _fit_quantile_edges(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.asarray([0.0 for _ in quantiles], dtype=float)

    edges = np.quantile(values, quantiles)

    # Ensure monotonic edges even when data contain many repeated values.
    edges = np.asarray(edges, dtype=float)

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    return edges


def _apply_edges(values: np.ndarray, edges: Sequence[float], n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)

    bins = np.digitize(values, bins=np.asarray(edges, dtype=float), right=False)
    bins = np.clip(bins, 0, int(n_bins) - 1)

    return bins.astype(int)


def _is_pre_binned_component(values: np.ndarray, n_bins: int, component_name: str) -> bool:
    values = np.asarray(values)

    if component_name in {"rng_bit", "latent_state"}:
        candidate = True
    elif component_name.endswith("_bin"):
        candidate = True
    else:
        candidate = False

    if not candidate:
        return False

    finite = values[np.isfinite(values.astype(float))]

    if len(finite) == 0:
        return False

    as_float = finite.astype(float)

    if not np.allclose(as_float, np.round(as_float)):
        return False

    as_int = np.round(as_float).astype(int)

    return bool(as_int.min() >= 0 and as_int.max() < int(n_bins))


def fit_state_components(
    df: pd.DataFrame,
    model: Any,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
    sensitivity: bool = False,
) -> StateComponentResult:
    """
    Converts a model's component columns into integer state components.

    Important:
        - Already-binned components are preserved.
        - Continuous components are discretized using train-fitted quantile edges.
        - Per-component bin counts are supported.
    """
    if len(model.components) != len(model.bins):
        raise ValueError(f"Model {model.name} has mismatched components and bins")

    work = df.copy().reset_index(drop=True)
    work = _replace_nonfinite_df(work)

    idx_train = _normalize_train_index(train_index, len(work))

    components_out: Dict[str, np.ndarray] = {}
    specs: List[StateComponentSpec] = []

    quantile_map = (
        cfg.sensitivity_quantiles_by_bins if sensitivity else cfg.quantiles_by_bins
    )

    for component, n_bins in zip(model.components, model.bins):
        if component not in work.columns:
            raise KeyError(
                f"State model {model.name} requires missing component column: {component}"
            )

        values = work[component].to_numpy()
        n_bins = int(n_bins)

        if _is_pre_binned_component(values, n_bins=n_bins, component_name=component):
            comp_values = np.round(values.astype(float)).astype(int)
            comp_values = np.clip(comp_values, 0, n_bins - 1)

            components_out[component] = comp_values
            specs.append(
                StateComponentSpec(
                    component=component,
                    n_bins=n_bins,
                    pre_binned=True,
                    edges=tuple(),
                    sensitivity=sensitivity,
                )
            )
            continue

        quantiles = tuple(quantile_map[n_bins])
        train_values = work.iloc[idx_train][component].astype(float).to_numpy()

        edges = _fit_quantile_edges(train_values, quantiles)
        comp_values = _apply_edges(values.astype(float), edges, n_bins)

        components_out[component] = comp_values
        specs.append(
            StateComponentSpec(
                component=component,
                n_bins=n_bins,
                pre_binned=False,
                edges=tuple(float(e) for e in edges),
                sensitivity=sensitivity,
            )
        )

    components_df = pd.DataFrame(components_out).astype(int)

    n_states = int(model.n_states)

    n_transitions = max(len(components_df) - 1, 0)
    transitions_per_state = float(n_transitions / max(n_states, 1))

    if len(components_df) == 0:
        observed_states = 0
    else:
        observed_states = int(len(components_df.drop_duplicates()))

    transitions_per_observed_state = float(
        n_transitions / max(observed_states, 1)
    )

    return StateComponentResult(
        components_df=components_df,
        specs=tuple(specs),
        n_states=n_states,
        transitions_per_state=transitions_per_state,
        observed_states=observed_states,
        transitions_per_observed_state=transitions_per_observed_state,
    )


def apply_state_component_specs(
    df: pd.DataFrame,
    specs: Sequence[StateComponentSpec],
) -> pd.DataFrame:
    """
    Applies previously fitted component specs to a dataframe.
    """
    work = df.copy().reset_index(drop=True)
    work = _replace_nonfinite_df(work)

    out: Dict[str, np.ndarray] = {}

    for spec in specs:
        if spec.component not in work.columns:
            raise KeyError(f"Missing component column: {spec.component}")

        values = work[spec.component].to_numpy()

        if spec.pre_binned:
            comp_values = np.round(values.astype(float)).astype(int)
            comp_values = np.clip(comp_values, 0, spec.n_bins - 1)
        else:
            comp_values = _apply_edges(values.astype(float), spec.edges, spec.n_bins)

        out[spec.component] = comp_values.astype(int)

    return pd.DataFrame(out).astype(int)


def build_state_component_matrix(
    df: pd.DataFrame,
    model: Any,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
    sensitivity: bool = False,
) -> Tuple[np.ndarray, StateComponentResult]:
    """
    Convenience wrapper returning both matrix and metadata.
    """
    result = fit_state_components(
        df=df,
        model=model,
        cfg=cfg,
        train_index=train_index,
        sensitivity=sensitivity,
    )

    matrix = result.components_df.to_numpy(dtype=int)

    return matrix, result


# =========================================================
# Model / feature utilities
# =========================================================

def get_active_state_models(cfg: Any) -> Tuple[Any, ...]:
    return tuple(m for m in cfg.state_models if getattr(m, "active", True))


def get_state_model_by_name(cfg: Any, name: str) -> Any:
    for model in cfg.state_models:
        if model.name == name:
            return model

    raise KeyError(f"State model not found: {name}")


def get_required_feature_columns_for_models(cfg: Any) -> Tuple[str, ...]:
    columns: List[str] = []

    for model in cfg.state_models:
        for component in model.components:
            if component not in columns:
                columns.append(component)

    return tuple(columns)


def validate_feature_frame_for_models(df: pd.DataFrame, cfg: Any) -> None:
    missing: List[str] = []

    for col in get_required_feature_columns_for_models(cfg):
        if col not in df.columns:
            missing.append(col)

    if missing:
        raise KeyError(
            "Feature frame is missing required state-model columns: "
            + ", ".join(missing)
        )


def compute_density_report(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Computes state-space density diagnostics for all active models.
    """
    rows: List[Dict[str, Any]] = []

    for model in get_active_state_models(cfg):
        result = fit_state_components(
            df=df,
            model=model,
            cfg=cfg,
            train_index=train_index,
            sensitivity=False,
        )

        rows.append(
            {
                "model": model.name,
                "label": model.label,
                "n_components": model.n_components,
                "n_states": result.n_states,
                "observed_states": result.observed_states,
                "transitions_per_state": result.transitions_per_state,
                "transitions_per_observed_state": result.transitions_per_observed_state,
                "passes_min_transitions_per_state": (
                    result.transitions_per_state >= cfg.min_transitions_per_state
                ),
                "passes_min_transitions_per_observed_state": (
                    result.transitions_per_observed_state
                    >= cfg.min_transitions_per_observed_state
                ),
            }
        )

    return pd.DataFrame(rows)


def classify_injectability_from_curve(
    injection_curve: Dict[float, float],
    tol_abs: float = 0.05,
) -> str:
    """
    Simple descriptive injectability classification from injection curve.

    Parameters
    ----------
    injection_curve:
        Mapping eps_true -> eps_hat.
    tol_abs:
        Recovery tolerance.

    Returns
    -------
    str
        high, moderate, low, or invalid.
    """
    if not injection_curve:
        return "invalid"

    recoveries = []

    for eps_true, eps_hat in injection_curve.items():
        eps_true = float(eps_true)
        eps_hat = float(eps_hat)

        if eps_true <= 0:
            continue

        err = abs(eps_hat - eps_true)
        recoveries.append(err <= tol_abs)

    if not recoveries:
        return "invalid"

    frac = float(np.mean(recoveries))

    if frac >= 0.80:
        return "high"

    if frac >= 0.50:
        return "moderate"

    return "low"


# =========================================================
# Compact end-to-end helper for loader/runner
# =========================================================

def prepare_phase3_1_features_for_modeling(
    df: pd.DataFrame,
    cfg: Any,
    train_index: Optional[Sequence[int]] = None,
) -> FeatureLayerResult:
    """
    Main preparation function for the runner.

    Input:
        Raw combined feature frame containing observed EEG/RNG and local metrics.

    Output:
        Feature frame with:
            - eeg_info_score / eeg_info_bin
            - rng_info_score / rng_info_bin
            - q_rng_score / q_rng_bin
            - latent_state
    """
    result = fit_transform_phase3_1_layers(
        df=df,
        cfg=cfg,
        train_index=train_index,
    )

    validate_feature_frame_for_models(result.df, cfg)

    return result