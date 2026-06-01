from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Data containers
# =========================================================

@dataclass(frozen=True)
class ControlFrame:
    """
    One controlled/surrogate dataframe.

    The dataframe must preserve the same number of rows and the same
    modeling columns as the original feature frame.
    """

    name: str
    control_type: str
    df: pd.DataFrame
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ControlRunResult:
    """
    Output of estimating epsilon across generated controls.
    """

    eps_controls: Tuple[float, ...]
    control_names: Tuple[str, ...]
    control_details: Tuple[Dict[str, Any], ...]
    summary: Dict[str, Any]


@dataclass(frozen=True)
class AblationFrame:
    """
    One proxy-ablation dataframe for F9.

    Used to test whether I_t, Z_t, Q_t actually contribute to the final signal.
    """

    name: str
    ablation_type: str
    df: pd.DataFrame
    metadata: Dict[str, Any]


# =========================================================
# Generic helpers
# =========================================================

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().reset_index(drop=True)


def _dedupe_keep_order(items: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []

    for item in items:
        item = str(item)
        if item not in seen:
            seen.add(item)
            out.append(item)

    return tuple(out)


def _existing(df: pd.DataFrame, columns: Iterable[str]) -> Tuple[str, ...]:
    return tuple(c for c in _dedupe_keep_order(columns) if c in df.columns)


def _safe_group_columns(df: pd.DataFrame, wanted: Sequence[str]) -> Tuple[str, ...]:
    return tuple(c for c in wanted if c in df.columns)


def _group_indices(df: pd.DataFrame, group_cols: Sequence[str]) -> List[np.ndarray]:
    group_cols = _safe_group_columns(df, group_cols)

    if not group_cols:
        return [np.arange(len(df), dtype=int)]

    groups: List[np.ndarray] = []

    for _, idx in df.groupby(list(group_cols), sort=False).groups.items():
        groups.append(np.asarray(list(idx), dtype=int))

    return groups


def _resample_rows(values: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """
    Deterministically resamples a source frame to exactly n_rows.

    This is used for subject mismatch controls when subjects have different
    numbers of epochs.
    """
    if len(values) == 0:
        raise ValueError("cannot resample empty dataframe")

    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")

    if len(values) == n_rows:
        return values.reset_index(drop=True).copy()

    positions = np.linspace(0, len(values) - 1, num=n_rows)
    positions = np.rint(positions).astype(int)

    return values.iloc[positions].reset_index(drop=True).copy()


def _shuffle_block_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    group_cols: Sequence[str] = ("subject_id",),
) -> pd.DataFrame:
    """
    Shuffles selected columns as a block, preserving row-wise relationships
    among those selected columns.

    If group_cols exist, shuffling is performed independently within groups.
    """
    out = _copy(df)
    columns = _existing(out, columns)

    if not columns:
        return out

    rng = _rng(seed)

    for idx in _group_indices(out, group_cols):
        if len(idx) < 2:
            continue

        perm = idx.copy()
        rng.shuffle(perm)

        out.loc[idx, list(columns)] = out.loc[perm, list(columns)].to_numpy()

    return out


def _circular_shift_block_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    seed: int,
    min_shift: int = 1,
    group_cols: Sequence[str] = ("subject_id",),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Circularly shifts selected columns as a block.

    Preserves local temporal structure within the shifted block while breaking
    alignment with non-shifted variables.
    """
    out = _copy(df)
    columns = _existing(out, columns)

    shifts: Dict[str, int] = {}

    if not columns:
        return out, shifts

    rng = _rng(seed)

    group_cols_existing = _safe_group_columns(out, group_cols)

    if group_cols_existing:
        grouped = out.groupby(list(group_cols_existing), sort=False).groups.items()
    else:
        grouped = [("all", np.arange(len(out), dtype=int))]

    for group_key, idx in grouped:
        idx_arr = np.asarray(list(idx), dtype=int)

        if len(idx_arr) < 2:
            shifts[str(group_key)] = 0
            continue

        low = min(int(min_shift), len(idx_arr) - 1)
        high = len(idx_arr)

        if low >= high:
            shift = 1
        else:
            shift = int(rng.integers(low=low, high=high))

        values = out.loc[idx_arr, list(columns)].to_numpy()
        shifted = np.roll(values, shift=shift, axis=0)

        out.loc[idx_arr, list(columns)] = shifted
        shifts[str(group_key)] = int(shift)

    return out, shifts


def _constant_replace_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    value: float | int = 0,
) -> pd.DataFrame:
    out = _copy(df)

    for col in _existing(out, columns):
        out[col] = value

    return out


# =========================================================
# Column groups
# =========================================================

def eeg_feature_columns(df: pd.DataFrame, cfg: Any) -> Tuple[str, ...]:
    """
    EEG-side columns that can be shuffled/ablated without touching identity columns.
    """
    columns: List[str] = []

    columns.extend(getattr(cfg, "eeg_observed_features", tuple()))
    columns.extend(getattr(cfg, "eeg_informational_features", tuple()))

    columns.extend(
        [
            "eeg_info_score",
            str(getattr(cfg, "eeg_info_component_name", "eeg_info_bin")),
        ]
    )

    # Add common EEG feature names defensively.
    columns.extend(
        [
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "alpha_delta_ratio",
            "delta_alpha_balance",
            "spectral_entropy",
            "permutation_entropy",
            "hjorth_mobility",
            "hjorth_complexity",
            "line_length",
            "bandpower_volatility",
            "state_instability",
        ]
    )

    return _existing(df, columns)


def rng_feature_columns(df: pd.DataFrame, cfg: Any) -> Tuple[str, ...]:
    """
    RNG-side columns that can be shuffled/shifted as a block.
    """
    columns: List[str] = []

    columns.extend(getattr(cfg, "rng_basic_features", tuple()))
    columns.extend(getattr(cfg, "rng_informational_features", tuple()))
    columns.extend(getattr(cfg, "rng_quantum_proxy_features", tuple()))

    columns.extend(
        [
            "rng_info_score",
            str(getattr(cfg, "rng_info_component_name", "rng_info_bin")),
            "q_rng_score",
            str(getattr(cfg, "q_component_name", "q_rng_bin")),
            "rng_window_id",
            "rng_window_start",
            "rng_window_end",
        ]
    )

    # Add common RNG/Q feature names defensively.
    columns.extend(
        [
            "rng_bit",
            "bit_balance_local",
            "transition_rate",
            "pair_frequency_00",
            "pair_frequency_01",
            "pair_frequency_10",
            "pair_frequency_11",
            "rng_entropy_local",
            "entropy_rate_proxy",
            "run_length_mean",
            "run_length_max",
            "run_length_std",
            "compressibility_proxy",
            "surprise_index",
            "transition_asymmetry",
            "micro_cluster_deviation",
            "q_entropy_rate_drift",
            "q_transition_asymmetry",
            "q_run_instability",
            "q_surprise_burst",
            "q_micro_cluster_deviation",
            "q_local_balance_deviation",
        ]
    )

    return _existing(df, columns)


def informational_proxy_columns(df: pd.DataFrame, cfg: Any) -> Tuple[str, ...]:
    columns = [
        "eeg_info_score",
        "rng_info_score",
        str(getattr(cfg, "eeg_info_component_name", "eeg_info_bin")),
        str(getattr(cfg, "rng_info_component_name", "rng_info_bin")),
    ]

    return _existing(df, columns)


def latent_columns(df: pd.DataFrame, cfg: Any) -> Tuple[str, ...]:
    columns = [
        str(getattr(cfg, "latent_component_name", "latent_state")),
    ]

    return _existing(df, columns)


def quantum_proxy_columns(df: pd.DataFrame, cfg: Any) -> Tuple[str, ...]:
    columns: List[str] = []

    columns.extend(getattr(cfg, "rng_quantum_proxy_features", tuple()))
    columns.extend(
        [
            "q_rng_score",
            str(getattr(cfg, "q_component_name", "q_rng_bin")),
        ]
    )

    columns.extend(
        [
            "q_entropy_rate_drift",
            "q_transition_asymmetry",
            "q_run_instability",
            "q_surprise_burst",
            "q_micro_cluster_deviation",
            "q_local_balance_deviation",
        ]
    )

    return _existing(df, columns)


def model_identity_columns(df: pd.DataFrame) -> Tuple[str, ...]:
    columns = [
        "subject_id",
        "recording_id",
        "epoch_idx",
        "sleep_stage",
        "channel_used",
        "sfreq",
        "psg_file",
        "hypnogram_file",
    ]

    return _existing(df, columns)


# =========================================================
# Control 1 — RNG shuffle
# =========================================================

def control_rng_shuffle(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Shuffles RNG feature rows as a block within subject.

    Purpose:
        Destroy RNG temporal order and EEG-RNG alignment while preserving the
        marginal distribution of RNG-derived features.
    """
    cols = rng_feature_columns(df, cfg)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=seed,
        group_cols=("subject_id",),
    )

    return ControlFrame(
        name="rng_shuffle",
        control_type="rng",
        df=controlled,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": "within_subject",
            "purpose": "destroy_rng_temporal_order_and_alignment",
        },
    )


# =========================================================
# Control 2 — RNG circular shift
# =========================================================

def control_rng_circular_shift(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Circularly shifts RNG features within subject.

    Purpose:
        Preserve local RNG feature order while breaking EEG-RNG alignment.
    """
    cols = rng_feature_columns(df, cfg)

    controlled, shifts = _circular_shift_block_columns(
        df=df,
        columns=cols,
        seed=seed,
        min_shift=int(getattr(cfg, "rng_circular_shift_min", 64)),
        group_cols=("subject_id",),
    )

    return ControlFrame(
        name="rng_circular_shift",
        control_type="rng",
        df=controlled,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": "within_subject",
            "shifts": shifts,
            "purpose": "preserve_rng_local_structure_but_break_alignment",
        },
    )


# =========================================================
# Control 3 — EEG epoch shuffle
# =========================================================

def control_eeg_epoch_shuffle(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Shuffles EEG feature rows as a block within subject.

    Purpose:
        Destroy EEG temporal order while preserving EEG marginal feature
        distribution within each subject.
    """
    cols = eeg_feature_columns(df, cfg)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=seed,
        group_cols=("subject_id",),
    )

    return ControlFrame(
        name="eeg_epoch_shuffle",
        control_type="eeg",
        df=controlled,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": "within_subject",
            "purpose": "destroy_eeg_temporal_order",
        },
    )


# =========================================================
# Control 4 — EEG stage-stratified shuffle
# =========================================================

def control_eeg_stage_stratified_shuffle(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Shuffles EEG features within subject and sleep stage.

    Purpose:
        Preserve stage distribution while breaking fine-grained EEG dynamics.
    """
    cols = eeg_feature_columns(df, cfg)

    group_cols = ("subject_id", "sleep_stage") if "sleep_stage" in df.columns else ("subject_id",)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=seed,
        group_cols=group_cols,
    )

    return ControlFrame(
        name="eeg_stage_stratified_shuffle",
        control_type="eeg",
        df=controlled,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": list(_safe_group_columns(df, group_cols)),
            "purpose": "preserve_sleep_stage_distribution_but_break_fine_eeg_dynamics",
        },
    )


# =========================================================
# Control 5 — Joint pairing shuffle
# =========================================================

def control_joint_pairing_shuffle(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Shuffles RNG block globally relative to EEG.

    Purpose:
        Keep EEG and RNG marginal structures but destroy their row-wise pairing.
    """
    out = _copy(df)
    cols = rng_feature_columns(out, cfg)

    if cols:
        rng = _rng(seed)
        idx = np.arange(len(out), dtype=int)
        perm = idx.copy()
        rng.shuffle(perm)
        out.loc[idx, list(cols)] = out.loc[perm, list(cols)].to_numpy()

    return ControlFrame(
        name="joint_pairing_shuffle",
        control_type="joint",
        df=out,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": "global",
            "purpose": "destroy_rowwise_eeg_rng_pairing",
        },
    )


# =========================================================
# Control 6 — Subject mismatch
# =========================================================

def control_subject_mismatch(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Replaces EEG feature blocks with EEG features from a different subject.

    Purpose:
        Test whether joint signal depends on correct subject-level structure.
    """
    out = _copy(df)

    if "subject_id" not in out.columns:
        return ControlFrame(
            name="subject_mismatch",
            control_type="subject",
            df=out,
            metadata={
                "seed": int(seed),
                "skipped": True,
                "reason": "subject_id column not found",
            },
        )

    eeg_cols = eeg_feature_columns(out, cfg)

    if not eeg_cols:
        return ControlFrame(
            name="subject_mismatch",
            control_type="subject",
            df=out,
            metadata={
                "seed": int(seed),
                "skipped": True,
                "reason": "no EEG columns found",
            },
        )

    subject_ids = [str(x) for x in out["subject_id"].dropna().unique().tolist()]
    subject_ids = sorted(subject_ids)

    if len(subject_ids) < 2:
        return ControlFrame(
            name="subject_mismatch",
            control_type="subject",
            df=out,
            metadata={
                "seed": int(seed),
                "skipped": True,
                "reason": "requires at least two subjects",
                "n_subjects": len(subject_ids),
            },
        )

    rng = _rng(seed)
    shift = int(rng.integers(low=1, high=len(subject_ids)))
    source_map = {
        target: subject_ids[(i + shift) % len(subject_ids)]
        for i, target in enumerate(subject_ids)
    }

    original = out.copy()

    for target_subject, source_subject in source_map.items():
        target_idx = np.flatnonzero(out["subject_id"].astype(str).to_numpy() == target_subject)
        source_rows = original[original["subject_id"].astype(str) == source_subject]

        if len(target_idx) == 0 or len(source_rows) == 0:
            continue

        replacement = _resample_rows(source_rows[list(eeg_cols)], len(target_idx))
        out.loc[target_idx, list(eeg_cols)] = replacement.to_numpy()

    return ControlFrame(
        name="subject_mismatch",
        control_type="subject",
        df=out,
        metadata={
            "seed": int(seed),
            "columns": list(eeg_cols),
            "source_map": source_map,
            "shift": shift,
            "purpose": "replace_eeg_features_with_mismatched_subject_structure",
        },
    )


# =========================================================
# Control 7 — Latent control shuffle
# =========================================================

def control_latent_shuffle(
    df: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> ControlFrame:
    """
    Shuffles latent state Z_t within subject.

    Purpose:
        Test whether the latent layer is creating artificial signal.
    """
    out = _copy(df)
    cols = latent_columns(out, cfg)

    if cols:
        out = _shuffle_block_columns(
            df=out,
            columns=cols,
            seed=seed,
            group_cols=("subject_id",),
        )

    return ControlFrame(
        name="latent_shuffle",
        control_type="latent",
        df=out,
        metadata={
            "seed": int(seed),
            "columns": list(cols),
            "grouping": "within_subject",
            "purpose": "destroy_latent_temporal_alignment",
        },
    )


# =========================================================
# Proxy ablation frames
# =========================================================

def ablation_remove_I_t(
    df: pd.DataFrame,
    cfg: Any,
) -> AblationFrame:
    """
    Ablates the informational proxy layer I_t via within-subject shuffle.

    Important:
        The previous implementation replaced I_t columns with constant zero.
        That created degenerate super-states and could artificially inflate epsilon.

        This version preserves the marginal distribution of I_t while destroying
        its temporal/structural alignment within each subject.
    """
    cols = informational_proxy_columns(df, cfg)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=99991,
        group_cols=("subject_id",),
    )

    return AblationFrame(
        name="remove_I_t",
        ablation_type="informational",
        df=controlled,
        metadata={
            "columns": list(cols),
            "replacement": "within_subject_shuffle",
            "seed": 99991,
            "purpose": "ablate_informational_proxy_layer_without_state_degeneracy",
        },
    )


def ablation_remove_Z_t(
    df: pd.DataFrame,
    cfg: Any,
) -> AblationFrame:
    """
    Ablates the latent inferred state Z_t via within-subject shuffle.

    This preserves the empirical distribution of latent states while destroying
    their temporal/structural alignment.
    """
    cols = latent_columns(df, cfg)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=99992,
        group_cols=("subject_id",),
    )

    return AblationFrame(
        name="remove_Z_t",
        ablation_type="latent",
        df=controlled,
        metadata={
            "columns": list(cols),
            "replacement": "within_subject_shuffle",
            "seed": 99992,
            "purpose": "ablate_latent_state_layer_without_state_degeneracy",
        },
    )


def ablation_remove_Q_t(
    df: pd.DataFrame,
    cfg: Any,
) -> AblationFrame:
    """
    Ablates the quantum-aware proxy layer Q_t via within-subject shuffle.

    Important:
        The previous implementation replaced Q_t columns with constant zero.
        That could collapse the proxy state space and create artificial structure.

        This version preserves the marginal distribution of Q_t while destroying
        its temporal/structural alignment within each subject.
    """
    cols = quantum_proxy_columns(df, cfg)

    controlled = _shuffle_block_columns(
        df=df,
        columns=cols,
        seed=99993,
        group_cols=("subject_id",),
    )

    return AblationFrame(
        name="remove_Q_t",
        ablation_type="quantum_proxy",
        df=controlled,
        metadata={
            "columns": list(cols),
            "replacement": "within_subject_shuffle",
            "seed": 99993,
            "purpose": "ablate_quantum_aware_proxy_layer_without_state_degeneracy",
        },
    )

def generate_proxy_ablation_frames(
    df: pd.DataFrame,
    cfg: Any,
) -> Tuple[AblationFrame, ...]:
    """
    Generates the proxy ablation frames used by F9.
    """
    frames: List[AblationFrame] = []

    if getattr(cfg, "use_informational_layer", True):
        frames.append(ablation_remove_I_t(df, cfg))

    if getattr(cfg, "use_latent_layer", True):
        frames.append(ablation_remove_Z_t(df, cfg))

    if getattr(cfg, "use_quantum_proxy_layer", True):
        frames.append(ablation_remove_Q_t(df, cfg))

    return tuple(frames)


# =========================================================
# Control generation
# =========================================================

def generate_phase3_1_control_frames(
    df: pd.DataFrame,
    cfg: Any,
    seed: Optional[int] = None,
) -> Tuple[ControlFrame, ...]:
    """
    Generates all enabled Phase III.1 control frames.

    The number of returned frames may exceed cfg.n_controls because each
    enabled control type is scientifically distinct.
    """
    base_seed = int(seed if seed is not None else getattr(cfg, "random_seed", 42))

    controls: List[ControlFrame] = []

    if getattr(cfg, "enable_control_rng_shuffle", True):
        controls.append(control_rng_shuffle(df, cfg, seed=base_seed + 101))

    if getattr(cfg, "enable_control_rng_circular_shift", True):
        controls.append(control_rng_circular_shift(df, cfg, seed=base_seed + 102))

    if getattr(cfg, "enable_control_eeg_epoch_shuffle", True):
        controls.append(control_eeg_epoch_shuffle(df, cfg, seed=base_seed + 201))

    if getattr(cfg, "enable_control_eeg_stage_stratified_shuffle", True):
        controls.append(control_eeg_stage_stratified_shuffle(df, cfg, seed=base_seed + 202))

    if getattr(cfg, "enable_control_joint_pairing_shuffle", True):
        controls.append(control_joint_pairing_shuffle(df, cfg, seed=base_seed + 301))

    if getattr(cfg, "enable_control_subject_mismatch", True):
        controls.append(control_subject_mismatch(df, cfg, seed=base_seed + 401))

    if getattr(cfg, "enable_control_latent_shuffle", True):
        controls.append(control_latent_shuffle(df, cfg, seed=base_seed + 501))

    # If cfg.n_controls is larger than the number of distinct control types,
    # generate additional RNG/joint shuffled replicates with different seeds.
    requested = int(getattr(cfg, "n_controls", len(controls)))

    replicate_id = 0
    while len(controls) < requested:
        replicate_seed = base_seed + 1000 + replicate_id

        if replicate_id % 2 == 0:
            frame = control_joint_pairing_shuffle(df, cfg, seed=replicate_seed)
            frame = ControlFrame(
                name=f"joint_pairing_shuffle_rep{replicate_id}",
                control_type=frame.control_type,
                df=frame.df,
                metadata={**frame.metadata, "replicate_id": replicate_id},
            )
        else:
            frame = control_rng_shuffle(df, cfg, seed=replicate_seed)
            frame = ControlFrame(
                name=f"rng_shuffle_rep{replicate_id}",
                control_type=frame.control_type,
                df=frame.df,
                metadata={**frame.metadata, "replicate_id": replicate_id},
            )

        controls.append(frame)
        replicate_id += 1

    return tuple(controls)


# =========================================================
# Control estimation wrapper
# =========================================================

def summarize_control_eps(
    eps_controls: Sequence[float],
    tol: float,
    required_fraction: float,
) -> Dict[str, Any]:
    eps = np.asarray(eps_controls, dtype=float)
    eps = eps[np.isfinite(eps)]

    if len(eps) == 0:
        return {
            "n_controls": 0,
            "median_eps_controls": 0.0,
            "mean_eps_controls": 0.0,
            "max_eps_controls": 0.0,
            "fraction_below_tol": 0.0,
            "tol": float(tol),
            "required_fraction": float(required_fraction),
            "passed": False,
        }

    below = eps <= float(tol)
    fraction = float(np.mean(below))

    return {
        "n_controls": int(len(eps)),
        "median_eps_controls": float(np.median(eps)),
        "mean_eps_controls": float(np.mean(eps)),
        "max_eps_controls": float(np.max(eps)),
        "fraction_below_tol": fraction,
        "tol": float(tol),
        "required_fraction": float(required_fraction),
        "passed": bool(fraction >= float(required_fraction)),
    }


def run_phase3_1_controls(
    df: pd.DataFrame,
    cfg: Any,
    estimate_fn: Callable[[pd.DataFrame, str], float],
    seed: Optional[int] = None,
) -> ControlRunResult:
    """
    Generates controls and estimates epsilon for each one.

    Parameters
    ----------
    df:
        Fully prepared feature dataframe.
    cfg:
        Phase III.1 config.
    estimate_fn:
        Callable receiving (controlled_df, control_name) and returning eps_hat.
        The runner supplies this function because only the runner knows the
        active state model, split, kernel, and epsilon grid.
    seed:
        Optional seed override.

    Returns
    -------
    ControlRunResult
    """
    controls = generate_phase3_1_control_frames(
        df=df,
        cfg=cfg,
        seed=seed,
    )

    eps_values: List[float] = []
    details: List[Dict[str, Any]] = []
    names: List[str] = []

    for control in controls:
        eps_hat = float(estimate_fn(control.df, control.name))

        eps_values.append(eps_hat)
        names.append(control.name)

        details.append(
            {
                "name": control.name,
                "control_type": control.control_type,
                "eps_hat": eps_hat,
                "metadata": control.metadata,
            }
        )

    summary = summarize_control_eps(
        eps_controls=eps_values,
        tol=float(getattr(cfg, "control_tol", 0.05)),
        required_fraction=float(getattr(cfg, "control_fraction", 0.75)),
    )

    return ControlRunResult(
        eps_controls=tuple(float(x) for x in eps_values),
        control_names=tuple(names),
        control_details=tuple(details),
        summary=summary,
    )


# =========================================================
# Diagnostic utilities
# =========================================================

def control_frame_metadata_table(controls: Sequence[ControlFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for c in controls:
        row = {
            "name": c.name,
            "control_type": c.control_type,
            "n_rows": len(c.df),
        }
        row.update({f"meta_{k}": v for k, v in c.metadata.items() if not isinstance(v, (list, dict))})
        rows.append(row)

    return pd.DataFrame(rows)


def validate_control_frame_shapes(
    original: pd.DataFrame,
    controls: Sequence[ControlFrame],
) -> None:
    """
    Ensures all controls preserve row count and columns.
    """
    original_cols = list(original.columns)
    original_len = len(original)

    for control in controls:
        if len(control.df) != original_len:
            raise RuntimeError(
                f"Control {control.name} changed row count: "
                f"{len(control.df)} != {original_len}"
            )

        if list(control.df.columns) != original_cols:
            raise RuntimeError(
                f"Control {control.name} changed dataframe columns"
            )


def validate_ablation_frame_shapes(
    original: pd.DataFrame,
    ablations: Sequence[AblationFrame],
) -> None:
    original_cols = list(original.columns)
    original_len = len(original)

    for ablation in ablations:
        if len(ablation.df) != original_len:
            raise RuntimeError(
                f"Ablation {ablation.name} changed row count: "
                f"{len(ablation.df)} != {original_len}"
            )

        if list(ablation.df.columns) != original_cols:
            raise RuntimeError(
                f"Ablation {ablation.name} changed dataframe columns"
            )