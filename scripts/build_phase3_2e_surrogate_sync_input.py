from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Phase III.2E — Surrogate synchronized input builder
# =========================================================
#
# Purpose:
#   Build a surrogate synchronized EEG + QRNG CSV for Phase III.2E using
#   already-processed Phase III.2 / III.2D artifacts.
#
# Output:
#   data/raw/phase3_2e/synchronized/phase3_2e_synchronized_input.csv
#
# Scientific guardrail:
#   This is NOT empirical synchronized EEG + QRNG acquisition.
#   It is a surrogate/demo dataset for pipeline testing only.
#
#   The generated CSV explicitly records:
#       source_mode = surrogate_from_phase3_2d
#       sync_mode = software_timestamp_aligned
#       empirical_claim_allowed = False
#
# Optional:
#   --inject-signal creates a controlled synthetic dependency from EEG(t)
#   into QRNG(t+1) to test whether the pipeline can detect a known structure.
#   This remains methodological only and cannot support empirical claims.
# =========================================================

SCRIPT_VERSION = "phase3_2e_surrogate_sync_builder_v2_schema_safe"
SOURCE_MODE = "surrogate_from_phase3_2d"
SYNC_MODE = "software_timestamp_aligned"
EMPIRICAL_CLAIM_ALLOWED = False

DEFAULT_WINDOW_SECONDS = 30
DEFAULT_START_UTC = "2026-01-01T00:00:00Z"
DEFAULT_SEED = 2026032

OUTPUT_CSV = Path("data/raw/phase3_2e/synchronized/phase3_2e_synchronized_input.csv")
OUTPUT_MANIFEST = Path("results/phase3_2e/phase3_2e_surrogate_sync_manifest.json")
OUTPUT_SUMMARY = Path("results/phase3_2e/phase3_2e_surrogate_sync_summary.txt")

SOURCE_CANDIDATES = [
    Path("data/interim/phase3_2/phase3_2d_multichannel_features.csv"),
    Path("results/phase3_2/phase3_2d_multichannel_results.csv"),
    Path("data/interim/phase3_2/phase3_2_modeling_features.csv"),
    Path("data/interim/phase3_2/phase3_2_modeling_feature.csv"),
    Path("data/interim/phase3_2/phase3_2_combined_features.csv"),
    Path("data/interim/phase3_2/phase3_2_features.csv"),
]

REQUIRED_PHASE32E_COLUMNS = [
    "subject_id",
    "session_id",
    "window_id",
    "window_start_utc",
    "window_end_utc",
    "eeg_timestamp_utc",
    "qrng_timestamp_utc",
    "sync_quality",
    "clock_drift_ms",
    "eeg_channel",
    "eeg_delta_power",
    "eeg_theta_power",
    "eeg_alpha_power",
    "eeg_beta_power",
    "eeg_entropy",
    "qrng_state",
    "qrng_bit_balance",
    "qrng_transition_rate",
    "qrng_entropy",
    "eeg_primary_channel",
    "eeg_secondary_channel",
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

OPTIONAL_PHASE32E_COLUMNS = [
    "sync_mode",
    "source_mode",
    "empirical_claim_allowed",
    "surrogate_builder_version",
    "surrogate_injected_signal",
    "surrogate_injection_strength",
    "source_file",
    "source_row_index",
    "eeg_state",
    "mc_eeg_state",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path | str) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    return project_root() / p


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return str(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=json_default)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def numeric_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default: float = 0.0,
) -> pd.Series:
    col = first_existing_col(df, candidates)

    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=float)

    out = pd.to_numeric(df[col], errors="coerce")

    if out.dropna().empty:
        return pd.Series([default] * len(df), index=df.index, dtype=float)

    median = float(out.median(skipna=True)) if not out.dropna().empty else default

    return out.fillna(median).astype(float)


def string_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
    default: str,
) -> pd.Series:
    col = first_existing_col(df, candidates)

    if col is None:
        return pd.Series([default] * len(df), index=df.index, dtype=str)

    out = df[col].astype(str)
    out = out.replace({"nan": default, "None": default, "": default})

    return out.fillna(default).astype(str)


def ensure_nonnegative_power(series: pd.Series) -> pd.Series:
    """
    Schema-safe converter for power-like features.
    If source features were z-scored and contain negative values, shift them
    above zero while preserving relative variation.
    """
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(skipna=True) if not values.dropna().empty else 0.0)
    values = values.astype(float)

    min_value = float(values.min()) if len(values) else 0.0

    if min_value < 0.0:
        values = values - min_value

    return values.clip(lower=0.0)


def normalize_01(series: pd.Series) -> pd.Series:
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    lo = float(values.min()) if len(values) else 0.0
    hi = float(values.max()) if len(values) else 0.0

    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series([0.5] * len(values), index=values.index, dtype=float)

    return ((values - lo) / (hi - lo)).clip(0.0, 1.0)


def safe_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    a = (
        pd.to_numeric(a, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    b = (
        pd.to_numeric(b, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    out = a / (b.abs() + eps)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def binary_entropy_from_probability(p: pd.Series) -> pd.Series:
    p = pd.to_numeric(p, errors="coerce").fillna(0.5).clip(1e-6, 1.0 - 1e-6)

    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    return entropy.clip(0.0, 1.0)


def spectral_entropy_from_bands(
    delta: pd.Series,
    theta: pd.Series,
    alpha: pd.Series,
    beta: pd.Series,
) -> pd.Series:
    arr = np.vstack(
        [
            ensure_nonnegative_power(delta).to_numpy(dtype=float),
            ensure_nonnegative_power(theta).to_numpy(dtype=float),
            ensure_nonnegative_power(alpha).to_numpy(dtype=float),
            ensure_nonnegative_power(beta).to_numpy(dtype=float),
        ]
    ).T

    total = arr.sum(axis=1, keepdims=True)
    total[total <= 1e-12] = 1.0

    p = arr / total

    entropy = -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    entropy = entropy / math.log2(arr.shape[1])

    return pd.Series(entropy, index=delta.index, dtype=float).clip(0.0, 1.0)


def tertile_state_from_two_features(a: pd.Series, b: pd.Series) -> pd.Series:
    a01 = normalize_01(a)
    b01 = normalize_01(b)

    a_bin = pd.cut(
        a01,
        bins=[-0.001, 1 / 3, 2 / 3, 1.001],
        labels=[0, 1, 2],
    ).astype(int)

    b_bin = pd.cut(
        b01,
        bins=[-0.001, 1 / 3, 2 / 3, 1.001],
        labels=[0, 1, 2],
    ).astype(int)

    return (a_bin * 3 + b_bin).astype(int).clip(0, 8)


def infer_binary_state_from_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")

    if s.dropna().empty:
        return pd.Series([0] * len(series), index=series.index, dtype=int)

    unique_values = set(sorted(s.dropna().unique().tolist()))

    if unique_values.issubset({0, 1}):
        return s.fillna(0).astype(int).clip(0, 1)

    threshold = float(s.median(skipna=True))

    return (s.fillna(threshold) > threshold).astype(int)


def ensure_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def iso_z(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def find_source_file(explicit_source: Optional[str] = None) -> Path:
    if explicit_source:
        path = resolve_project_path(explicit_source)

        if not path.exists():
            raise FileNotFoundError(f"Explicit source file not found: {path}")

        return path

    for candidate in SOURCE_CANDIDATES:
        path = resolve_project_path(candidate)

        if path.exists():
            return path

    available_msg = "\n".join(str(resolve_project_path(p)) for p in SOURCE_CANDIDATES)

    raise FileNotFoundError(
        "No Phase III.2/III.2D processed source file was found. Expected one of:\n"
        f"{available_msg}"
    )


def load_source_dataframe(source_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(source_path)

    if df.empty:
        raise ValueError(f"Source file is empty: {source_path}")

    if max_rows is not None and int(max_rows) > 0:
        df = df.head(int(max_rows)).copy()

    df = df.reset_index(drop=True)
    df["source_row_index"] = np.arange(len(df), dtype=int)

    return df


def build_subject_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["subject_id"] = string_series(
        df,
        candidates=[
            "subject_id",
            "subject",
            "eeg_subject",
            "patient_id",
            "recording_subject",
        ],
        default="SURR_SUBJECT_000",
    )

    session_col = first_existing_col(
        df,
        [
            "session_id",
            "recording",
            "recording_id",
            "night",
            "file_id",
        ],
    )

    if session_col:
        out["session_id"] = (
            df[session_col]
            .astype(str)
            .replace({"nan": "session_0", "None": "session_0", "": "session_0"})
        )
    else:
        out["session_id"] = out["subject_id"].astype(str) + "_SURR_SESSION_0"

    return out


def extract_eeg_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    delta = ensure_nonnegative_power(
        numeric_series(
            df,
            [
                "eeg_delta_power",
                "delta_power",
                "delta_power_primary",
                "eeg_primary_delta_power",
                "primary_delta_power",
                "delta",
            ],
            default=0.0,
        )
    )

    alpha = ensure_nonnegative_power(
        numeric_series(
            df,
            [
                "eeg_alpha_power",
                "alpha_power",
                "alpha_power_primary",
                "eeg_primary_alpha_power",
                "primary_alpha_power",
                "alpha",
            ],
            default=0.0,
        )
    )

    theta_col = first_existing_col(
        df,
        [
            "eeg_theta_power",
            "theta_power",
            "theta_power_primary",
            "primary_theta_power",
            "theta",
        ],
    )

    if theta_col:
        theta = ensure_nonnegative_power(pd.to_numeric(df[theta_col], errors="coerce"))
    else:
        theta = ensure_nonnegative_power(
            0.45 * delta + 0.25 * alpha + 0.05 * normalize_01(delta)
        )

    beta_col = first_existing_col(
        df,
        [
            "eeg_beta_power",
            "beta_power",
            "beta_power_primary",
            "primary_beta_power",
            "beta",
        ],
    )

    if beta_col:
        beta = ensure_nonnegative_power(pd.to_numeric(df[beta_col], errors="coerce"))
    else:
        beta = ensure_nonnegative_power(
            0.25 * alpha + 0.10 * normalize_01(alpha) + 0.02 * normalize_01(delta)
        )

    entropy_col = first_existing_col(
        df,
        [
            "eeg_entropy",
            "entropy",
            "spectral_entropy",
            "eeg_spectral_entropy",
        ],
    )

    if entropy_col:
        entropy = normalize_01(pd.to_numeric(df[entropy_col], errors="coerce"))
    else:
        entropy = spectral_entropy_from_bands(delta, theta, alpha, beta)

    secondary_delta_col = first_existing_col(
        df,
        [
            "eeg_secondary_delta_power",
            "delta_power_secondary",
            "secondary_delta_power",
            "pzoz_delta_power",
        ],
    )

    if secondary_delta_col:
        secondary_delta = ensure_nonnegative_power(
            pd.to_numeric(df[secondary_delta_col], errors="coerce")
        )
    else:
        modulation = 0.92 + 0.08 * np.sin(np.arange(len(df)) / 17.0)
        secondary_delta = ensure_nonnegative_power(
            pd.Series(delta.to_numpy() * modulation, index=df.index, dtype=float)
        )

    secondary_alpha_col = first_existing_col(
        df,
        [
            "eeg_secondary_alpha_power",
            "alpha_power_secondary",
            "secondary_alpha_power",
            "pzoz_alpha_power",
        ],
    )

    if secondary_alpha_col:
        secondary_alpha = ensure_nonnegative_power(
            pd.to_numeric(df[secondary_alpha_col], errors="coerce")
        )
    else:
        modulation = 1.05 + 0.06 * np.cos(np.arange(len(df)) / 19.0)
        secondary_alpha = ensure_nonnegative_power(
            pd.Series(alpha.to_numpy() * modulation, index=df.index, dtype=float)
        )

    eeg_state_col = first_existing_col(
        df,
        [
            "eeg_state",
            "EEG_state",
            "state_eeg",
        ],
    )

    if eeg_state_col:
        eeg_state = (
            pd.to_numeric(df[eeg_state_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 8)
        )
    else:
        eeg_state = tertile_state_from_two_features(delta, alpha)

    mc_state_col = first_existing_col(
        df,
        [
            "mc_eeg_state",
            "eeg_multichannel_state",
            "multichannel_eeg_state",
        ],
    )

    if mc_state_col:
        mc_eeg_state = (
            pd.to_numeric(df[mc_state_col], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 8)
        )
    else:
        mc_eeg_state = tertile_state_from_two_features(secondary_delta, secondary_alpha)

    inter_delta_col = first_existing_col(
        df,
        [
            "interchannel_delta_ratio",
            "delta_interchannel_ratio",
        ],
    )

    if inter_delta_col:
        inter_delta = (
            pd.to_numeric(df[inter_delta_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    else:
        inter_delta = safe_ratio(delta, secondary_delta)

    inter_alpha_col = first_existing_col(
        df,
        [
            "interchannel_alpha_ratio",
            "alpha_interchannel_ratio",
        ],
    )

    if inter_alpha_col:
        inter_alpha = (
            pd.to_numeric(df[inter_alpha_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    else:
        inter_alpha = safe_ratio(alpha, secondary_alpha)

    delta_shift_col = first_existing_col(
        df,
        [
            "fronto_parietal_delta_shift",
            "delta_shift",
        ],
    )

    if delta_shift_col:
        delta_shift = (
            pd.to_numeric(df[delta_shift_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    else:
        delta_shift = delta - secondary_delta

    alpha_shift_col = first_existing_col(
        df,
        [
            "fronto_parietal_alpha_shift",
            "alpha_shift",
        ],
    )

    if alpha_shift_col:
        alpha_shift = (
            pd.to_numeric(df[alpha_shift_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
    else:
        alpha_shift = alpha - secondary_alpha

    corr_col = first_existing_col(
        df,
        [
            "cross_channel_corr",
            "interchannel_corr",
            "primary_secondary_corr",
        ],
    )

    if corr_col:
        corr = (
            pd.to_numeric(df[corr_col], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .clip(-1.0, 1.0)
        )
    else:
        corr = 1.0 - 0.5 * (
            (normalize_01(delta) - normalize_01(secondary_delta)).abs()
            + (normalize_01(alpha) - normalize_01(secondary_alpha)).abs()
        )
        corr = corr.clip(-1.0, 1.0)

    return {
        "eeg_delta_power": delta,
        "eeg_theta_power": theta,
        "eeg_alpha_power": alpha,
        "eeg_beta_power": beta,
        "eeg_entropy": entropy.clip(0.0, 1.0),
        "eeg_primary_delta_power": delta,
        "eeg_primary_alpha_power": alpha,
        "eeg_secondary_delta_power": secondary_delta,
        "eeg_secondary_alpha_power": secondary_alpha,
        "interchannel_delta_ratio": inter_delta.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "interchannel_alpha_ratio": inter_alpha.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "fronto_parietal_delta_shift": delta_shift.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "fronto_parietal_alpha_shift": alpha_shift.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "cross_channel_corr": corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0),
        "eeg_state": eeg_state,
        "mc_eeg_state": mc_eeg_state,
    }


def extract_qrng_features(
    df: pd.DataFrame,
    subject_session: pd.DataFrame,
    eeg_state: pd.Series,
    inject_signal: bool,
    inject_strength: float,
    rng: np.random.Generator,
) -> Dict[str, pd.Series]:
    state_col = first_existing_col(
        df,
        [
            "qrng_state",
            "rng_state",
            "RNG_state",
            "bit_state",
            "rng_current_state",
            "rng_bin",
            "rng_next_state",
            "target_rng_state",
            "qrng_next_state",
        ],
    )

    if state_col:
        qrng_state = infer_binary_state_from_series(df[state_col])
    else:
        balance_source = numeric_series(
            df,
            [
                "qrng_bit_balance",
                "rng_bit_balance",
                "bit_balance",
                "rng_window_mean",
                "rng_mean",
            ],
            default=0.5,
        )
        qrng_state = infer_binary_state_from_series(balance_source)

    qrng_state = qrng_state.astype(int).clip(0, 1)

    if inject_signal:
        strength = float(np.clip(inject_strength, 0.0, 1.0))
        injected = qrng_state.copy().astype(int)

        group_keys = (
            subject_session["subject_id"].astype(str)
            + "||"
            + subject_session["session_id"].astype(str)
        )

        for _, idx in group_keys.groupby(group_keys, sort=False).groups.items():
            idx_list = list(idx)

            if len(idx_list) < 2:
                continue

            for pos in range(1, len(idx_list)):
                prev_idx = idx_list[pos - 1]
                cur_idx = idx_list[pos]

                if rng.random() < strength:
                    injected.iloc[cur_idx] = int(eeg_state.iloc[prev_idx]) % 2

        qrng_state = injected.astype(int).clip(0, 1)

    balance_col = first_existing_col(
        df,
        [
            "qrng_bit_balance",
            "rng_bit_balance",
            "bit_balance",
            "rng_window_mean",
            "rng_mean",
        ],
    )

    if balance_col:
        bit_balance = normalize_01(pd.to_numeric(df[balance_col], errors="coerce"))
    else:
        jitter = rng.normal(loc=0.0, scale=0.015, size=len(df))
        bit_balance = pd.Series(
            0.48 + 0.04 * qrng_state.to_numpy(dtype=float) + jitter,
            index=df.index,
            dtype=float,
        ).clip(0.0, 1.0)

    entropy_col = first_existing_col(
        df,
        [
            "qrng_entropy",
            "rng_entropy",
            "rng_window_entropy",
            "random_entropy",
        ],
    )

    if entropy_col:
        qrng_entropy = normalize_01(pd.to_numeric(df[entropy_col], errors="coerce"))
    else:
        qrng_entropy = binary_entropy_from_probability(bit_balance)

    transition_col = first_existing_col(
        df,
        [
            "qrng_transition_rate",
            "rng_transition_rate",
            "transition_rate",
        ],
    )

    if transition_col:
        transition_rate = normalize_01(pd.to_numeric(df[transition_col], errors="coerce"))
    else:
        transition_rate = pd.Series([0.0] * len(df), index=df.index, dtype=float)

        group_keys = (
            subject_session["subject_id"].astype(str)
            + "||"
            + subject_session["session_id"].astype(str)
        )

        for _, idx in group_keys.groupby(group_keys, sort=False).groups.items():
            idx_list = list(idx)

            if len(idx_list) <= 1:
                continue

            states = qrng_state.iloc[idx_list].to_numpy(dtype=int)
            changed = np.zeros(len(states), dtype=float)
            changed[1:] = (states[1:] != states[:-1]).astype(float)

            rate = pd.Series(changed).rolling(window=5, min_periods=1).mean().to_numpy(dtype=float)
            transition_rate.iloc[idx_list] = rate

    return {
        "qrng_state": qrng_state.astype(int).clip(0, 1),
        "qrng_bit_balance": bit_balance.astype(float).clip(0.0, 1.0),
        "qrng_transition_rate": transition_rate.astype(float).clip(0.0, 1.0),
        "qrng_entropy": qrng_entropy.astype(float).clip(0.0, 1.0),
    }


def build_time_columns(
    subject_session: pd.DataFrame,
    window_seconds: int,
    start_utc: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create schema-safe synchronized timestamps.

    Fix applied here:
    - timestamps are placed near the middle of each window;
    - QRNG timestamp is allowed to drift by a few milliseconds;
    - both EEG and QRNG timestamps are hard-clamped inside the window.
    """
    out = pd.DataFrame(index=subject_session.index)
    base_ts = ensure_utc_timestamp(start_utc)

    subject_values = sorted(subject_session["subject_id"].astype(str).unique().tolist())
    subject_day_offset = {subj: i for i, subj in enumerate(subject_values)}

    group_keys = (
        subject_session["subject_id"].astype(str)
        + "||"
        + subject_session["session_id"].astype(str)
    )

    window_id = pd.Series([0] * len(subject_session), index=subject_session.index, dtype=int)
    window_start: List[Optional[pd.Timestamp]] = [None] * len(subject_session)
    window_end: List[Optional[pd.Timestamp]] = [None] * len(subject_session)
    eeg_ts: List[Optional[pd.Timestamp]] = [None] * len(subject_session)
    qrng_ts: List[Optional[pd.Timestamp]] = [None] * len(subject_session)
    drift_ms_values = pd.Series([0.0] * len(subject_session), index=subject_session.index, dtype=float)

    for group_number, (_, idx) in enumerate(group_keys.groupby(group_keys, sort=False).groups.items()):
        idx_list = list(idx)

        if not idx_list:
            continue

        subject = str(subject_session.loc[idx_list[0], "subject_id"])
        day_offset = subject_day_offset.get(subject, 0)

        session_start = (
            base_ts
            + pd.Timedelta(days=int(day_offset))
            + pd.Timedelta(hours=int(group_number % 3) * 8)
        )

        for position, row_idx in enumerate(idx_list):
            wid = int(position)

            start = session_start + pd.Timedelta(seconds=wid * int(window_seconds))
            end = start + pd.Timedelta(seconds=int(window_seconds))

            drift_ms = float(rng.normal(loc=0.0, scale=2.5))
            drift_ms = float(np.clip(drift_ms, -12.0, 12.0))

            midpoint = start + pd.Timedelta(seconds=float(window_seconds) / 2.0)

            eeg_time = midpoint
            qrng_time = midpoint + pd.Timedelta(milliseconds=drift_ms)

            min_allowed = start + pd.Timedelta(milliseconds=1)
            max_allowed = end - pd.Timedelta(milliseconds=1)

            if eeg_time < min_allowed:
                eeg_time = min_allowed

            if eeg_time > max_allowed:
                eeg_time = max_allowed

            if qrng_time < min_allowed:
                qrng_time = min_allowed

            if qrng_time > max_allowed:
                qrng_time = max_allowed

            window_id.loc[row_idx] = wid
            window_start[row_idx] = start
            window_end[row_idx] = end
            eeg_ts[row_idx] = eeg_time
            qrng_ts[row_idx] = qrng_time
            drift_ms_values.loc[row_idx] = drift_ms

    out["window_id"] = window_id.astype(int)
    out["window_start_utc"] = [iso_z(ts) for ts in window_start]
    out["window_end_utc"] = [iso_z(ts) for ts in window_end]
    out["eeg_timestamp_utc"] = [iso_z(ts) for ts in eeg_ts]
    out["qrng_timestamp_utc"] = [iso_z(ts) for ts in qrng_ts]
    out["clock_drift_ms"] = drift_ms_values.astype(float)

    sync_quality = 0.995 - (drift_ms_values.abs() / 1000.0)
    out["sync_quality"] = sync_quality.clip(0.950, 0.999).astype(float)

    return out


def build_surrogate_dataframe(
    source_df: pd.DataFrame,
    source_path: Path,
    window_seconds: int,
    start_utc: str,
    inject_signal: bool,
    inject_strength: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    source_df = source_df.copy().reset_index(drop=True)

    subject_session = build_subject_session_columns(source_df)
    eeg = extract_eeg_features(source_df)

    qrng = extract_qrng_features(
        df=source_df,
        subject_session=subject_session,
        eeg_state=eeg["eeg_state"],
        inject_signal=bool(inject_signal),
        inject_strength=float(inject_strength),
        rng=rng,
    )

    time_cols = build_time_columns(
        subject_session=subject_session,
        window_seconds=int(window_seconds),
        start_utc=str(start_utc),
        rng=rng,
    )

    out = pd.DataFrame(index=source_df.index)

    out["subject_id"] = subject_session["subject_id"].astype(str)
    out["session_id"] = subject_session["session_id"].astype(str)
    out["window_id"] = time_cols["window_id"].astype(int)
    out["window_start_utc"] = time_cols["window_start_utc"].astype(str)
    out["window_end_utc"] = time_cols["window_end_utc"].astype(str)
    out["eeg_timestamp_utc"] = time_cols["eeg_timestamp_utc"].astype(str)
    out["qrng_timestamp_utc"] = time_cols["qrng_timestamp_utc"].astype(str)
    out["sync_quality"] = time_cols["sync_quality"].astype(float)
    out["clock_drift_ms"] = time_cols["clock_drift_ms"].astype(float)

    out["eeg_channel"] = string_series(
        source_df,
        ["eeg_channel", "channel", "primary_eeg_channel"],
        default="EEG Fpz-Cz",
    )

    out["eeg_delta_power"] = eeg["eeg_delta_power"].astype(float)
    out["eeg_theta_power"] = eeg["eeg_theta_power"].astype(float)
    out["eeg_alpha_power"] = eeg["eeg_alpha_power"].astype(float)
    out["eeg_beta_power"] = eeg["eeg_beta_power"].astype(float)
    out["eeg_entropy"] = eeg["eeg_entropy"].astype(float).clip(0.0, 1.0)

    out["qrng_state"] = qrng["qrng_state"].astype(int).clip(0, 1)
    out["qrng_bit_balance"] = qrng["qrng_bit_balance"].astype(float).clip(0.0, 1.0)
    out["qrng_transition_rate"] = qrng["qrng_transition_rate"].astype(float).clip(0.0, 1.0)
    out["qrng_entropy"] = qrng["qrng_entropy"].astype(float).clip(0.0, 1.0)

    out["eeg_primary_channel"] = string_series(
        source_df,
        ["eeg_primary_channel", "primary_eeg_channel"],
        default="EEG Fpz-Cz",
    )

    out["eeg_secondary_channel"] = string_series(
        source_df,
        ["eeg_secondary_channel", "secondary_eeg_channel"],
        default="EEG Pz-Oz",
    )

    out["eeg_primary_delta_power"] = eeg["eeg_primary_delta_power"].astype(float)
    out["eeg_primary_alpha_power"] = eeg["eeg_primary_alpha_power"].astype(float)
    out["eeg_secondary_delta_power"] = eeg["eeg_secondary_delta_power"].astype(float)
    out["eeg_secondary_alpha_power"] = eeg["eeg_secondary_alpha_power"].astype(float)
    out["interchannel_delta_ratio"] = eeg["interchannel_delta_ratio"].astype(float)
    out["interchannel_alpha_ratio"] = eeg["interchannel_alpha_ratio"].astype(float)
    out["fronto_parietal_delta_shift"] = eeg["fronto_parietal_delta_shift"].astype(float)
    out["fronto_parietal_alpha_shift"] = eeg["fronto_parietal_alpha_shift"].astype(float)
    out["cross_channel_corr"] = eeg["cross_channel_corr"].astype(float).clip(-1.0, 1.0)

    out["eeg_state"] = eeg["eeg_state"].astype(int).clip(0, 8)
    out["mc_eeg_state"] = eeg["mc_eeg_state"].astype(int).clip(0, 8)

    out["sync_mode"] = SYNC_MODE
    out["source_mode"] = SOURCE_MODE
    out["empirical_claim_allowed"] = bool(EMPIRICAL_CLAIM_ALLOWED)
    out["surrogate_builder_version"] = SCRIPT_VERSION
    out["surrogate_injected_signal"] = bool(inject_signal)
    out["surrogate_injection_strength"] = float(inject_strength if inject_signal else 0.0)
    out["source_file"] = str(source_path).replace("\\", "/")
    out["source_row_index"] = source_df["source_row_index"].astype(int)

    ordered_cols = REQUIRED_PHASE32E_COLUMNS + [
        col for col in OPTIONAL_PHASE32E_COLUMNS if col in out.columns
    ]
    ordered_cols += [col for col in out.columns if col not in ordered_cols]

    out = out[ordered_cols].copy()

    for col in REQUIRED_PHASE32E_COLUMNS:
        if col not in out.columns:
            raise RuntimeError(f"Internal error: missing required output column {col}")

    numeric_required = [
        "window_id",
        "sync_quality",
        "clock_drift_ms",
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_entropy",
        "qrng_state",
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
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

    for col in numeric_required:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    for col in [
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_primary_delta_power",
        "eeg_primary_alpha_power",
        "eeg_secondary_delta_power",
        "eeg_secondary_alpha_power",
    ]:
        out[col] = ensure_nonnegative_power(out[col])

    out["window_id"] = out["window_id"].astype(int)
    out["qrng_state"] = out["qrng_state"].astype(int).clip(0, 1)
    out["sync_quality"] = out["sync_quality"].astype(float).clip(0.950, 0.999)
    out["clock_drift_ms"] = out["clock_drift_ms"].astype(float).clip(-50.0, 50.0)
    out["qrng_bit_balance"] = out["qrng_bit_balance"].astype(float).clip(0.0, 1.0)
    out["qrng_transition_rate"] = out["qrng_transition_rate"].astype(float).clip(0.0, 1.0)
    out["qrng_entropy"] = out["qrng_entropy"].astype(float).clip(0.0, 1.0)
    out["eeg_entropy"] = out["eeg_entropy"].astype(float).clip(0.0, 1.0)
    out["cross_channel_corr"] = out["cross_channel_corr"].astype(float).clip(-1.0, 1.0)

    string_required = [
        "subject_id",
        "session_id",
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "eeg_channel",
        "eeg_primary_channel",
        "eeg_secondary_channel",
    ]

    for col in string_required:
        out[col] = out[col].astype(str).replace({"nan": "", "None": ""}).fillna("")

    missing_counts = {
        col: int(out[col].isna().sum())
        for col in REQUIRED_PHASE32E_COLUMNS
    }

    manifest = {
        "created_at": now_str(),
        "script_version": SCRIPT_VERSION,
        "source_mode": SOURCE_MODE,
        "sync_mode": SYNC_MODE,
        "empirical_claim_allowed": EMPIRICAL_CLAIM_ALLOWED,
        "source_file": str(source_path),
        "n_rows": int(len(out)),
        "n_subjects": int(out["subject_id"].nunique()),
        "n_sessions": int(out[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "window_seconds": int(window_seconds),
        "start_utc": str(start_utc),
        "inject_signal": bool(inject_signal),
        "inject_strength": float(inject_strength if inject_signal else 0.0),
        "seed": int(seed),
        "required_columns": REQUIRED_PHASE32E_COLUMNS,
        "optional_columns": OPTIONAL_PHASE32E_COLUMNS,
        "missing_required_value_counts": missing_counts,
        "output_columns": list(out.columns),
        "source_columns": list(source_df.columns),
        "guardrail": (
            "This CSV is a surrogate synchronized input for testing the Phase III.2E "
            "pipeline only. It must not be interpreted as empirical synchronized "
            "EEG-QRNG evidence."
        ),
    }

    return out, manifest


def write_summary(path: Path, manifest: Dict[str, Any], output_csv: Path) -> None:
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 78,
        "Phase III.2E — Surrogate synchronized input summary",
        "=" * 78,
        "",
        f"created_at: {manifest.get('created_at')}",
        f"script_version: {manifest.get('script_version')}",
        f"source_mode: {manifest.get('source_mode')}",
        f"sync_mode: {manifest.get('sync_mode')}",
        f"empirical_claim_allowed: {manifest.get('empirical_claim_allowed')}",
        "",
        f"source_file: {manifest.get('source_file')}",
        f"output_csv: {output_csv}",
        "",
        f"n_rows: {manifest.get('n_rows')}",
        f"n_subjects: {manifest.get('n_subjects')}",
        f"n_sessions: {manifest.get('n_sessions')}",
        f"window_seconds: {manifest.get('window_seconds')}",
        f"start_utc: {manifest.get('start_utc')}",
        "",
        f"inject_signal: {manifest.get('inject_signal')}",
        f"inject_strength: {manifest.get('inject_strength')}",
        f"seed: {manifest.get('seed')}",
        "",
        "Guardrail:",
        str(manifest.get("guardrail")),
        "",
        "Missing required values:",
    ]

    for col, count in manifest.get("missing_required_value_counts", {}).items():
        lines.append(f"{col}: {count}")

    lines.append("")
    lines.append("=" * 78)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_and_save_surrogate(
    source: Optional[str],
    output_csv: Path,
    manifest_json: Path,
    summary_txt: Path,
    window_seconds: int,
    start_utc: str,
    seed: int,
    max_rows: Optional[int],
    force: bool,
    inject_signal: bool,
    inject_strength: float,
) -> Dict[str, Any]:
    root = project_root()

    output_csv_abs = resolve_project_path(output_csv)
    manifest_json_abs = resolve_project_path(manifest_json)
    summary_txt_abs = resolve_project_path(summary_txt)

    if output_csv_abs.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output_csv_abs}\n"
            "Use --force to overwrite it."
        )

    source_path = find_source_file(source)
    source_df = load_source_dataframe(source_path, max_rows=max_rows)

    surrogate_df, manifest = build_surrogate_dataframe(
        source_df=source_df,
        source_path=source_path,
        window_seconds=int(window_seconds),
        start_utc=str(start_utc),
        inject_signal=bool(inject_signal),
        inject_strength=float(inject_strength),
        seed=int(seed),
    )

    output_csv_abs.parent.mkdir(parents=True, exist_ok=True)
    manifest_json_abs.parent.mkdir(parents=True, exist_ok=True)
    summary_txt_abs.parent.mkdir(parents=True, exist_ok=True)

    surrogate_df.to_csv(output_csv_abs, index=False)

    manifest["project_root"] = str(root)
    manifest["output_csv"] = str(output_csv_abs)
    manifest["manifest_json"] = str(manifest_json_abs)
    manifest["summary_txt"] = str(summary_txt_abs)

    save_json(manifest_json, manifest)
    write_summary(summary_txt, manifest, output_csv_abs)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Phase III.2E surrogate synchronized EEG-QRNG input CSV."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional explicit source CSV.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(OUTPUT_CSV),
        help="Output synchronized input CSV path.",
    )

    parser.add_argument(
        "--manifest-json",
        type=str,
        default=str(OUTPUT_MANIFEST),
        help="Output manifest JSON path.",
    )

    parser.add_argument(
        "--summary-txt",
        type=str,
        default=str(OUTPUT_SUMMARY),
        help="Output summary TXT path.",
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help="Synthetic synchronized window size in seconds.",
    )

    parser.add_argument(
        "--start-utc",
        type=str,
        default=DEFAULT_START_UTC,
        help="Synthetic start timestamp in UTC.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic surrogate timestamps/noise.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick tests.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output CSV.",
    )

    parser.add_argument(
        "--inject-signal",
        action="store_true",
        help="Inject EEG(t)->QRNG(t+1) dependency for positive-control testing.",
    )

    parser.add_argument(
        "--inject-strength",
        type=float,
        default=0.20,
        help="Injection probability when --inject-signal is enabled.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = build_and_save_surrogate(
        source=args.source,
        output_csv=Path(args.output_csv),
        manifest_json=Path(args.manifest_json),
        summary_txt=Path(args.summary_txt),
        window_seconds=int(args.window_seconds),
        start_utc=str(args.start_utc),
        seed=int(args.seed),
        max_rows=args.max_rows,
        force=bool(args.force),
        inject_signal=bool(args.inject_signal),
        inject_strength=float(args.inject_strength),
    )

    print("\n" + "=" * 78)
    print("Phase III.2E surrogate synchronized input completed")
    print("=" * 78)
    print(f"source_mode: {manifest.get('source_mode')}")
    print(f"sync_mode: {manifest.get('sync_mode')}")
    print(f"empirical_claim_allowed: {manifest.get('empirical_claim_allowed')}")
    print(f"source_file: {manifest.get('source_file')}")
    print(f"output_csv: {manifest.get('output_csv')}")
    print(f"manifest_json: {manifest.get('manifest_json')}")
    print(f"summary_txt: {manifest.get('summary_txt')}")
    print(f"n_rows: {manifest.get('n_rows')}")
    print(f"n_subjects: {manifest.get('n_subjects')}")
    print(f"n_sessions: {manifest.get('n_sessions')}")
    print(f"inject_signal: {manifest.get('inject_signal')}")
    print(f"inject_strength: {manifest.get('inject_strength')}")
    print("")
    print("Scientific guardrail:")
    print(manifest.get("guardrail"))
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()