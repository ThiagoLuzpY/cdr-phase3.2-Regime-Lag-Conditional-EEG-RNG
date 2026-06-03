from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================================================
# Phase III.2 — Configuration
# =========================================================
#
# Project:
#   Phase III.2 — Regime-Aware, Lagged, Conditional and Multichannel
#   EEG-RNG Validation
#
# Purpose:
#   Test whether the clean null_result from Phase III.1 was caused by
#   limitations of global joint-state modeling by evaluating:
#
#   III.2A — sleep-stage / regime-aware structure
#   III.2B — lagged EEG-RNG alignment
#   III.2C — conditional CDR
#   III.2D — multi-channel EEG enrichment
#   III.2E — future synchronized EEG + QRNG protocol
#
# This config intentionally keeps all submodules under a single Phase III.2
# namespace to avoid duplicated phase-specific files.
#
# Current activation:
#   III.2A/B/C — implemented and audit-bundled
#   III.2D     — active from v1.1
#   III.2E     — protocol-only, not executed on Sleep-EDF + ANU QRNG
#


# =========================================================
# Dataclasses
# =========================================================

@dataclass(frozen=True)
class RegimeSpec:
    """
    Sleep-stage / regime specification used by Phase III.2A.

    Parameters
    ----------
    name:
        Internal regime name.
    stages:
        Sleep stages included in this regime.
    description:
        Human-readable explanation.
    primary:
        Whether this regime is considered part of the primary hypothesis set.
        Primary regimes can support stronger interpretation if all gates pass.
    min_subjects:
        Minimum number of subjects required after filtering.
    min_total_epochs:
        Minimum total number of epochs required after filtering.
    min_epochs_per_subject:
        Minimum number of epochs per subject required for that subject to
        remain valid in this regime.
    """

    name: str
    stages: Tuple[str, ...]
    description: str
    primary: bool = False
    min_subjects: int = 5
    min_total_epochs: int = 500
    min_epochs_per_subject: int = 50


@dataclass(frozen=True)
class ConditionalModelSpec:
    """
    Conditional CDR model specification used by Phase III.2C/D.

    The model compares whether an augmented current-state representation improves
    the prediction of a next-state target beyond a baseline representation.

    Examples
    --------
    C0:
        current_state = EEG_t
        target_next = EEG_{t+1}

    C1:
        current_state = EEG_t × RNG_t
        target_next = EEG_{t+1}

    D0:
        current_state = multichannel EEG_t
        target_next = multichannel EEG_{t+1}
    """

    name: str
    label: str
    family: str
    current_components: Tuple[str, ...]
    target_component: str
    baseline_model: Optional[str] = None
    primary: bool = False
    description: str = ""


@dataclass(frozen=True)
class Phase32Config:
    """
    Central configuration object for Phase III.2.
    """

    # -----------------------------------------------------
    # Project metadata
    # -----------------------------------------------------
    project_name: str = "Phase3.2-Regime-Lag-Conditional-EEG-RNG"
    phase_name: str = "Phase III.2"
    phase_title: str = "Regime-Aware, Lagged, Conditional and Multichannel EEG-RNG Validation"
    version: str = "v1.1-phase3_2d"

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    root_dir: Path = Path(".")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    raw_eeg_dir: Path = Path("data/raw/eeg")
    raw_rng_dir: Path = Path("data/raw/rng")
    interim_dir: Path = Path("data/interim/phase3_2")
    processed_dir: Path = Path("data/processed")
    results_dir: Path = Path("results/phase3_2")
    diagnostics_dir: Path = Path("results/phase3_2/diagnostics")

    # -----------------------------------------------------
    # Data source settings
    # -----------------------------------------------------
    eeg_source_name: str = "Sleep-EDF Expanded"
    eeg_subset_name: str = "sleep-cassette"
    rng_source_name: str = "ANU Quantum Random Number Generator"

    eeg_epoch_seconds: int = 30
    max_subjects: int = 10
    min_subjects_for_loso: int = 3

    eeg_psg_pattern: str = "*PSG.edf"
    eeg_hypnogram_pattern: str = "*Hypnogram.edf"

    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    rng_use_bits: bool = True
    rng_n_uint8_expected: int = 1024
    rng_n_bits_expected: int = 8192
    rng_window_size: int = 64
    rng_alignment_mode: str = "resample_rng_metrics_to_eeg_epochs"

    # -----------------------------------------------------
    # EEG channel settings
    # -----------------------------------------------------
    primary_eeg_channel: str = "EEG Fpz-Cz"
    secondary_eeg_channel: str = "EEG Pz-Oz"

    # Phase III.2D activation:
    #   use both Sleep-EDF EEG channels when available.
    #   This does not claim synchronization improvement by itself; it only
    #   enriches the neural state representation before EEG-RNG conditional tests.
    use_multichannel_eeg: bool = True

    eeg_channels: Tuple[str, ...] = (
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
    )

    multichannel_eeg_channels: Tuple[str, ...] = (
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
    )

    multichannel_primary_channel: str = "EEG Fpz-Cz"
    multichannel_secondary_channel: str = "EEG Pz-Oz"

    multichannel_state_column: str = "eeg_multichannel_state"
    multichannel_next_state_column: str = "eeg_multichannel_next_state"
    multichannel_info_bin_column: str = "eeg_multichannel_info_bin"

    multichannel_n_states: int = 9
    multichannel_n_bins: int = 3

    multichannel_feature_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
        "delta_power_secondary",
        "alpha_power_secondary",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
    )

    multichannel_bin_columns: Tuple[str, ...] = (
        "delta_power_bin",
        "alpha_power_bin",
        "delta_power_secondary_bin",
        "alpha_power_secondary_bin",
        "interchannel_delta_ratio_bin",
        "interchannel_alpha_ratio_bin",
    )

    # -----------------------------------------------------
    # Module toggles
    # -----------------------------------------------------
    run_regime_analysis: bool = True
    run_lagged_analysis: bool = True
    run_conditional_analysis: bool = True

    # Phase III.2D — active from v1.1.
    run_multichannel_analysis: bool = True

    # Phase III.2E — protocol-ready, not empirical execution yet.
    # It requires truly synchronized EEG + QRNG acquisition.
    run_synchronized_protocol_analysis: bool = False
    phase3_2e_protocol_only: bool = True

    run_diagnostics_after_runner: bool = True

    # -----------------------------------------------------
    # Phase III.2A — Regime-aware analysis
    # -----------------------------------------------------
    regime_specs: Tuple[RegimeSpec, ...] = field(
        default_factory=lambda: (
            RegimeSpec(
                name="full",
                stages=("W", "N1", "N2", "N3", "REM"),
                description="Full dataset including Wake and all sleep stages.",
                primary=False,
                min_subjects=10,
                min_total_epochs=1000,
                min_epochs_per_subject=100,
            ),
            RegimeSpec(
                name="no_wake",
                stages=("N1", "N2", "N3", "REM"),
                description="All sleep stages excluding Wake.",
                primary=True,
                min_subjects=8,
                min_total_epochs=1000,
                min_epochs_per_subject=80,
            ),
            RegimeSpec(
                name="stable_sleep",
                stages=("N2", "N3"),
                description="Stable non-REM sleep: N2 + N3.",
                primary=True,
                min_subjects=8,
                min_total_epochs=700,
                min_epochs_per_subject=60,
            ),
            RegimeSpec(
                name="deep_sleep",
                stages=("N3",),
                description="Deep sleep only.",
                primary=False,
                min_subjects=5,
                min_total_epochs=300,
                min_epochs_per_subject=30,
            ),
            RegimeSpec(
                name="rem_only",
                stages=("REM",),
                description="REM sleep only.",
                primary=False,
                min_subjects=5,
                min_total_epochs=300,
                min_epochs_per_subject=30,
            ),
        )
    )

    include_transition_regime: bool = True
    transition_regime_name: str = "transition_epochs"
    transition_window_epochs: int = 2
    transition_min_subjects: int = 5
    transition_min_total_epochs: int = 250
    transition_min_epochs_per_subject: int = 20

    # -----------------------------------------------------
    # Phase III.2B — Lagged analysis
    # -----------------------------------------------------
    lags_epochs: Tuple[int, ...] = (-5, -3, -1, 0, 1, 3, 5)
    primary_lags_epochs: Tuple[int, ...] = (0,)
    secondary_lags_epochs: Tuple[int, ...] = (-5, -3, -1, 1, 3, 5)

    # Positive lag convention:
    #   lag > 0: EEG_t aligned with RNG_{t+lag}
    #   lag < 0: RNG_t aligned with EEG_{t+abs(lag)}
    prevent_lag_cross_subject: bool = True
    prevent_lag_cross_recording: bool = True

    # -----------------------------------------------------
    # Phase III.2C / III.2D — Conditional CDR
    # -----------------------------------------------------
    conditional_models: Tuple[ConditionalModelSpec, ...] = field(
        default_factory=lambda: (
            # -------------------------------------------------
            # Phase III.2C — single-channel conditional models
            # -------------------------------------------------
            ConditionalModelSpec(
                name="C0_EEG_next_given_EEG",
                label="EEG self-transition baseline",
                family="EEG_next",
                current_components=("eeg_state",),
                target_component="eeg_next_state",
                baseline_model=None,
                primary=True,
                description="Baseline model: P(EEG_{t+1} | EEG_t).",
            ),
            ConditionalModelSpec(
                name="C1_EEG_next_given_EEG_RNG",
                label="EEG transition conditioned by RNG",
                family="EEG_next",
                current_components=("eeg_state", "rng_state"),
                target_component="eeg_next_state",
                baseline_model="C0_EEG_next_given_EEG",
                primary=True,
                description="Augmented model: P(EEG_{t+1} | EEG_t, RNG_t).",
            ),
            ConditionalModelSpec(
                name="C2_RNG_next_given_RNG",
                label="RNG self-transition baseline",
                family="RNG_next",
                current_components=("rng_state",),
                target_component="rng_next_state",
                baseline_model=None,
                primary=True,
                description="Baseline model: P(RNG_{t+1} | RNG_t).",
            ),
            ConditionalModelSpec(
                name="C3_RNG_next_given_RNG_EEG",
                label="RNG transition conditioned by EEG",
                family="RNG_next",
                current_components=("rng_state", "eeg_state"),
                target_component="rng_next_state",
                baseline_model="C2_RNG_next_given_RNG",
                primary=True,
                description="Augmented model: P(RNG_{t+1} | RNG_t, EEG_t).",
            ),

            # -------------------------------------------------
            # Phase III.2D — multichannel EEG enrichment models
            # -------------------------------------------------
            ConditionalModelSpec(
                name="D0_MCEEG_next_given_MCEEG",
                label="Multichannel EEG self-transition baseline",
                family="MCEEG_next",
                current_components=("eeg_multichannel_state",),
                target_component="eeg_multichannel_next_state",
                baseline_model=None,
                primary=False,
                description=(
                    "Phase III.2D baseline: "
                    "P(MC-EEG_{t+1} | MC-EEG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="D1_MCEEG_next_given_MCEEG_RNG",
                label="Multichannel EEG transition conditioned by RNG",
                family="MCEEG_next",
                current_components=("eeg_multichannel_state", "rng_state"),
                target_component="eeg_multichannel_next_state",
                baseline_model="D0_MCEEG_next_given_MCEEG",
                primary=False,
                description=(
                    "Phase III.2D augmented model: "
                    "P(MC-EEG_{t+1} | MC-EEG_t, RNG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="D2_RNG_next_given_RNG_MCEEG",
                label="RNG transition conditioned by multichannel EEG",
                family="RNG_next",
                current_components=("rng_state", "eeg_multichannel_state"),
                target_component="rng_next_state",
                baseline_model="C2_RNG_next_given_RNG",
                primary=False,
                description=(
                    "Phase III.2D augmented model: "
                    "P(RNG_{t+1} | RNG_t, MC-EEG_t)."
                ),
            ),
        )
    )

    primary_conditional_pairs: Tuple[Tuple[str, str], ...] = (
        ("C0_EEG_next_given_EEG", "C1_EEG_next_given_EEG_RNG"),
        ("C2_RNG_next_given_RNG", "C3_RNG_next_given_RNG_EEG"),
    )

    # -----------------------------------------------------
    # Phase III.2D — multichannel conditional pairs
    # -----------------------------------------------------
    # These are intentionally not part of the primary A/B/C claim set.
    # They are active exploratory extensions designed to test whether a richer
    # EEG representation improves conditional EEG-RNG detection.
    multichannel_conditional_pairs: Tuple[Tuple[str, str], ...] = (
        ("D0_MCEEG_next_given_MCEEG", "D1_MCEEG_next_given_MCEEG_RNG"),
        ("C2_RNG_next_given_RNG", "D2_RNG_next_given_RNG_MCEEG"),
    )

    # Phase III.2E is not executed on Sleep-EDF + ANU public QRNG.
    # It remains a protocol-ready future experiment requiring synchronized
    # EEG and QRNG acquisition.
    synchronized_protocol_required_pairs: Tuple[Tuple[str, str], ...] = (
        ("D0_MCEEG_next_given_MCEEG", "D1_MCEEG_next_given_MCEEG_RNG"),
        ("C2_RNG_next_given_RNG", "D2_RNG_next_given_RNG_MCEEG"),
    )

    # -----------------------------------------------------
    # Discretization / state construction
    # -----------------------------------------------------
    eeg_n_bins: int = 3
    rng_n_bins: int = 2
    info_n_bins: int = 3
    q_n_bins: int = 3
    latent_n_states: int = 3

    eeg_state_columns: Tuple[str, ...] = (
        "delta_power_bin",
        "alpha_power_bin",
    )

    rng_state_columns: Tuple[str, ...] = (
        "rng_bit",
    )

    informational_columns: Tuple[str, ...] = (
        "eeg_info_bin",
        "rng_info_bin",
    )

    quantum_proxy_columns: Tuple[str, ...] = (
        "q_rng_bin",
    )

    # -----------------------------------------------------
    # Phase III.2D — compact multichannel EEG state
    # -----------------------------------------------------
    # The multichannel state must stay compact to avoid repeating the
    # high-dimensional state-space dilution observed in earlier joint tests.
    multichannel_state_input_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
        "delta_power_secondary",
        "alpha_power_secondary",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
    )

    multichannel_state_required_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
        "delta_power_secondary",
        "alpha_power_secondary",
    )

    multichannel_state_max_unique_values: int = 9
    multichannel_min_valid_subjects: int = 8
    multichannel_min_total_epochs: int = 1000
    multichannel_min_epochs_per_subject: int = 80

    multichannel_density_min_transitions_per_state: float = 30.0
    multichannel_density_recommended_transitions_per_state: float = 50.0

    # -----------------------------------------------------
    # Split / validation settings
    # -----------------------------------------------------
    train_ratio: float = 0.70
    lag: int = 1

    # -----------------------------------------------------
    # Conditional CDR leakage-safe split settings
    # -----------------------------------------------------
    # Phase III.2C/D uses three chronological partitions:
    #
    #   reference_train -> fits P0
    #   calibration     -> estimates Δχ and selects ε
    #   test_holdout    -> evaluates selected ε
    #
    # This prevents the final test fold from being used to construct the
    # reweighting direction.
    conditional_split_ref_ratio: float = 0.50
    conditional_split_calib_ratio: float = 0.20
    conditional_split_test_ratio: float = 0.30

    conditional_min_ref_rows: int = 20
    conditional_min_calib_rows: int = 10
    conditional_min_test_rows: int = 10

    conditional_min_calib_ll_gain: float = 1e-6
    conditional_min_test_ll_gain: float = 1e-6

    eps_saturation_value: float = 0.80
    eps_saturation_warning_fraction: float = 0.50

    use_loso_diagnostics: bool = True
    use_within_subject_diagnostics: bool = True

    # -----------------------------------------------------
    # Injection / epsilon grid
    # -----------------------------------------------------
    inj_eps_true: float = 0.05
    injection_eps_grid: Tuple[float, ...] = (0.00, 0.01, 0.03, 0.05)
    eps_grid: Tuple[float, ...] = tuple(round(x * 0.01, 2) for x in range(0, 81))

    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    required_control_fraction: float = 0.75

    # -----------------------------------------------------
    # Gates / success criteria
    # -----------------------------------------------------
    holdout_max_delta: float = 0.10
    sensitivity_max_delta: float = 0.12

    conditional_lift_min: float = 0.03
    strong_eps_min: float = 0.07
    subject_effect_fraction: float = 0.60
    moderate_subject_effect_fraction: float = 0.30

    min_test_ll_improvement: float = 0.0
    use_bic_complexity_gate: bool = True
    use_aic_secondary: bool = True

    # F11 — multiple-comparison guard
    enable_multiple_comparison_guard: bool = True
    primary_regimes: Tuple[str, ...] = ("no_wake", "stable_sleep")
    secondary_regimes: Tuple[str, ...] = ("deep_sleep", "rem_only", "transition_epochs")
    allow_strong_claim_from_secondary_tests: bool = False

    # -----------------------------------------------------
    # Density / sparsity safeguards
    # -----------------------------------------------------
    min_transitions_total: int = 100
    min_transitions_per_state: float = 30.0
    recommended_transitions_per_state: float = 50.0
    ideal_transitions_per_state: float = 100.0

    max_nominal_states_soft: int = 64
    enforce_state_density_warning: bool = True

    # -----------------------------------------------------
    # Controls
    # -----------------------------------------------------
    n_controls: int = 12
    control_seed: int = 424242

    control_types: Tuple[str, ...] = (
        "within_subject_rng_shuffle",
        "within_subject_eeg_shuffle",
        "circular_rng_shift",
        "subject_mismatch",
        "stage_preserving_rng_shuffle",
        "conditional_target_shuffle",
    )

    # -----------------------------------------------------
    # Phase III.2E — synchronized EEG + QRNG protocol gate
    # -----------------------------------------------------
    # This is intentionally disabled for the current public-data experiment.
    # It exists so that the codebase can validate a future synchronized dataset
    # without pretending that ANU QRNG and Sleep-EDF were temporally co-acquired.
    phase3_2e_requires_real_time_sync: bool = True
    phase3_2e_min_sync_quality: float = 0.95
    phase3_2e_max_clock_drift_ms: float = 50.0
    phase3_2e_required_timestamp_columns: Tuple[str, ...] = (
        "subject_id",
        "recording_id",
        "eeg_timestamp_utc",
        "rng_timestamp_utc",
        "sync_quality",
    )

    phase3_2e_protocol_doc: Path = Path(
        "docs/phase3_2e_synchronized_eeg_qrng_protocol.md"
    )

    # -----------------------------------------------------
    # Output files
    # -----------------------------------------------------
    combined_features_file: Path = Path("data/interim/phase3_2/phase3_2_combined_features.csv")
    loader_metadata_file: Path = Path("data/interim/phase3_2/phase3_2_loader_metadata.json")

    results_json: Path = Path("results/phase3_2/phase3_2_results.json")
    summary_txt: Path = Path("results/phase3_2/phase3_2_summary.txt")

    regime_results_csv: Path = Path("results/phase3_2/phase3_2a_regime_results.csv")
    regime_summary_json: Path = Path("results/phase3_2/phase3_2a_regime_summary.json")

    lag_results_csv: Path = Path("results/phase3_2/phase3_2b_lag_results.csv")
    lag_summary_json: Path = Path("results/phase3_2/phase3_2b_lag_summary.json")

    conditional_results_csv: Path = Path("results/phase3_2/phase3_2c_conditional_results.csv")
    conditional_summary_json: Path = Path("results/phase3_2/phase3_2c_conditional_summary.json")

    multichannel_features_file: Path = Path(
        "data/interim/phase3_2/phase3_2d_multichannel_features.csv"
    )
    multichannel_inventory_json: Path = Path(
        "data/interim/phase3_2/phase3_2d_multichannel_inventory.json"
    )
    multichannel_summary_json: Path = Path(
        "results/phase3_2/phase3_2d_multichannel_summary.json"
    )
    multichannel_results_csv: Path = Path(
        "results/phase3_2/phase3_2d_multichannel_results.csv"
    )

    synchronized_protocol_validation_json: Path = Path(
        "results/phase3_2/phase3_2e_synchronized_protocol_validation.json"
    )

    controls_json: Path = Path("results/phase3_2/phase3_2_controls.json")
    gates_json: Path = Path("results/phase3_2/phase3_2_gates.json")
    multiple_comparison_guard_json: Path = Path(
        "results/phase3_2/phase3_2_multiple_comparison_guard.json"
    )

    diagnostics_report_json: Path = Path(
        "results/phase3_2/diagnostics/phase3_2_diagnostics_report.json"
    )
    diagnostics_summary_txt: Path = Path(
        "results/phase3_2/diagnostics/phase3_2_diagnostics_summary.txt"
    )

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------
    random_seed: int = 20260427
    numpy_seed: int = 20260427

    save_intermediate_files: bool = True
    overwrite_existing_outputs: bool = True
    verbose: int = 1


# =========================================================
# Helper functions
# =========================================================

def load_phase3_2_config() -> Phase32Config:
    """
    Returns the Phase III.2 configuration object and ensures that
    required output folders exist.
    """
    cfg = Phase32Config()

    required_dirs = [
        cfg.raw_eeg_dir,
        cfg.raw_rng_dir,
        cfg.interim_dir,
        cfg.processed_dir,
        cfg.results_dir,
        cfg.diagnostics_dir,
        cfg.phase3_2e_protocol_doc.parent,
    ]

    for path in required_dirs:
        Path(path).mkdir(parents=True, exist_ok=True)

    return cfg


def regime_map(cfg: Optional[Phase32Config] = None) -> Dict[str, RegimeSpec]:
    """
    Returns regime specifications indexed by regime name.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    regimes = {spec.name: spec for spec in cfg.regime_specs}

    if cfg.include_transition_regime:
        regimes[cfg.transition_regime_name] = RegimeSpec(
            name=cfg.transition_regime_name,
            stages=tuple(),
            description="Epochs around sleep-stage transitions.",
            primary=False,
            min_subjects=cfg.transition_min_subjects,
            min_total_epochs=cfg.transition_min_total_epochs,
            min_epochs_per_subject=cfg.transition_min_epochs_per_subject,
        )

    return regimes


def conditional_model_map(
    cfg: Optional[Phase32Config] = None,
) -> Dict[str, ConditionalModelSpec]:
    """
    Returns conditional model specifications indexed by model name.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    return {spec.name: spec for spec in cfg.conditional_models}


def active_conditional_pairs(
    cfg: Optional[Phase32Config] = None,
) -> Tuple[Tuple[str, str], ...]:
    """
    Returns the conditional model pairs active for the current Phase III.2 run.

    Phase III.2A/B/C pairs remain the primary registered set.
    Phase III.2D pairs are appended only when multichannel analysis is enabled.
    Phase III.2E remains protocol-only unless explicitly activated in a future
    synchronized EEG + QRNG experiment.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    pairs: List[Tuple[str, str]] = list(cfg.primary_conditional_pairs)

    if cfg.run_multichannel_analysis and cfg.use_multichannel_eeg:
        pairs.extend(list(cfg.multichannel_conditional_pairs))

    return tuple(pairs)


def primary_test_definitions(cfg: Optional[Phase32Config] = None) -> List[Dict[str, object]]:
    """
    Defines the primary Phase III.2 tests.

    These are the tests that can support stronger interpretation if all gates pass.
    Secondary regimes/lags and Phase III.2D multichannel tests can generate leads,
    but not strong claims without replication.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    tests: List[Dict[str, object]] = []

    for regime in cfg.primary_regimes:
        for lag_value in cfg.primary_lags_epochs:
            for baseline, augmented in cfg.primary_conditional_pairs:
                tests.append(
                    {
                        "regime": regime,
                        "lag_epochs": lag_value,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "primary": True,
                    }
                )

    return tests


def describe_config(cfg: Optional[Phase32Config] = None) -> Dict[str, object]:
    """
    Compact serializable description used in metadata outputs.
    """
    if cfg is None:
        cfg = load_phase3_2_config()

    return {
        "project_name": cfg.project_name,
        "phase_name": cfg.phase_name,
        "phase_title": cfg.phase_title,
        "version": cfg.version,

        "max_subjects": cfg.max_subjects,
        "eeg_epoch_seconds": cfg.eeg_epoch_seconds,

        "primary_eeg_channel": cfg.primary_eeg_channel,
        "secondary_eeg_channel": cfg.secondary_eeg_channel,
        "use_multichannel_eeg": cfg.use_multichannel_eeg,
        "run_multichannel_analysis": cfg.run_multichannel_analysis,
        "multichannel_state_column": cfg.multichannel_state_column,
        "multichannel_next_state_column": cfg.multichannel_next_state_column,
        "multichannel_n_states": cfg.multichannel_n_states,
        "multichannel_feature_columns": list(cfg.multichannel_feature_columns),

        "run_synchronized_protocol_analysis": cfg.run_synchronized_protocol_analysis,
        "phase3_2e_protocol_only": cfg.phase3_2e_protocol_only,
        "phase3_2e_requires_real_time_sync": cfg.phase3_2e_requires_real_time_sync,

        "rng_source_name": cfg.rng_source_name,
        "rng_n_uint8_expected": cfg.rng_n_uint8_expected,
        "rng_n_bits_expected": cfg.rng_n_bits_expected,
        "rng_window_size": cfg.rng_window_size,

        "regimes": [spec.name for spec in cfg.regime_specs],
        "include_transition_regime": cfg.include_transition_regime,
        "lags_epochs": list(cfg.lags_epochs),
        "primary_regimes": list(cfg.primary_regimes),
        "primary_lags_epochs": list(cfg.primary_lags_epochs),

        "conditional_models": [spec.name for spec in cfg.conditional_models],
        "primary_conditional_pairs": list(cfg.primary_conditional_pairs),
        "multichannel_conditional_pairs": list(cfg.multichannel_conditional_pairs),
        "active_conditional_pairs": list(active_conditional_pairs(cfg)),

        "inj_eps_true": cfg.inj_eps_true,
        "conditional_lift_min": cfg.conditional_lift_min,
        "strong_eps_min": cfg.strong_eps_min,
        "subject_effect_fraction": cfg.subject_effect_fraction,
        "multiple_comparison_guard": cfg.enable_multiple_comparison_guard,
    }


if __name__ == "__main__":
    config = load_phase3_2_config()
    print("Phase III.2 config loaded successfully.")
    print(describe_config(config))