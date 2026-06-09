from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WindowSpec:
    """
    Multi-resolution synchronized window specification.

    Parameters
    ----------
    name:
        Internal window name.
    seconds:
        Window duration in seconds.
    primary:
        Whether this window is part of the primary test set.
    min_windows_total:
        Minimum number of valid synchronized windows.
    min_windows_per_subject:
        Minimum number of valid windows per subject.
    description:
        Human-readable explanation.
    """

    name: str
    seconds: int
    primary: bool
    min_windows_total: int
    min_windows_per_subject: int
    description: str


@dataclass(frozen=True)
class SyncModeSpec:
    """
    Synchronization mode specification.

    A dataset can be confirmatory only if it uses an accepted synchronization
    mode and passes quality/drift thresholds.
    """

    name: str
    accepted_for_confirmatory: bool
    description: str


@dataclass(frozen=True)
class ConditionalModelSpec:
    """
    Conditional CDR model specification for Phase III.2E.
    """

    name: str
    label: str
    family: str
    current_components: Tuple[str, ...]
    target_component: str
    baseline_model: Optional[str] = None
    primary: bool = False
    multichannel: bool = False
    description: str = ""


@dataclass(frozen=True)
class GateSpec:
    """
    Gate specification for the evidence-guided replication protocol.

    Gate-sensitivity diagnostics explain threshold behavior without changing
    negative-control, leakage, or synchronization safeguards.
    """

    name: str
    label: str
    required_for_candidate: bool
    description: str


@dataclass(frozen=True)
class GateSensitivitySpec:
    """
    Diagnostic specification for F7/F8/F12.

    These diagnostics preserve the negative-control, leakage, holdout and
    synchronization safeguards.
    """

    name: str
    target_gate: str
    enabled: bool
    primary_diagnostic: bool
    description: str


@dataclass(frozen=True)
class Phase32EConfig:
    """
    Central configuration object for Phase III.2E.
    """

    # =====================================================
    # Project metadata
    # =====================================================

    project_name: str = "Phase3.2-Regime-Lag-Conditional-EEG-RNG"
    phase_name: str = "Phase III.2E"
    phase_title: str = (
        "Synchronized EEG–QRNG Protocol and Gate-Sensitivity Validation"
    )

    version: str = (
        "v1.1-phase3_2e_evidence_guided_surrogate_replication"
    )

    protocol_status: str = "protocol_ready"
    empirical_execution_status: str = "surrogate_replication_ready"

    # =====================================================
    # Paths
    # =====================================================

    root_dir: Path = Path(".")
    data_dir: Path = Path("data")

    raw_dir: Path = Path("data/raw")
    raw_phase3_2e_dir: Path = Path("data/raw/phase3_2e")

    raw_synchronized_dir: Path = Path(
        "data/raw/phase3_2e/synchronized"
    )

    raw_eeg_dir: Path = Path("data/raw/phase3_2e/eeg")
    raw_qrng_dir: Path = Path("data/raw/phase3_2e/qrng")

    interim_dir: Path = Path("data/interim/phase3_2e")
    processed_dir: Path = Path("data/processed/phase3_2e")

    results_dir: Path = Path("results/phase3_2e")
    diagnostics_dir: Path = Path("results/phase3_2e/diagnostics")
    audit_bundle_dir: Path = Path("results/phase3_2e/audit_bundle")

    docs_dir: Path = Path("docs")

    protocol_doc: Path = Path(
        "docs/phase3_2e_synchronized_eeg_qrng_protocol.md"
    )

    # =====================================================
    # Data source settings
    # =====================================================

    eeg_source_name: str = "Synchronized EEG source"
    qrng_source_name: str = "Synchronized QRNG source"

    dataset_name: str = "phase3_2e_synchronized_eeg_qrng"

    synchronized_input_csv: Path = Path(
        "data/raw/phase3_2e/synchronized/"
        "phase3_2e_synchronized_input.csv"
    )

    allow_protocol_only_without_data: bool = True

    # Permite a execução metodológica com o surrogate.
    allow_synthetic_demo_data: bool = True

    # Continua impedindo que o surrogate seja descrito como aquisição real.
    require_real_synchronized_data_for_claim: bool = True

    # =====================================================
    # Required schema columns
    # =====================================================

    identity_columns: Tuple[str, ...] = (
        "subject_id",
        "session_id",
        "window_id",
    )

    timing_columns: Tuple[str, ...] = (
        "window_start_utc",
        "window_end_utc",
        "eeg_timestamp_utc",
        "qrng_timestamp_utc",
        "sync_quality",
        "clock_drift_ms",
    )

    required_eeg_columns: Tuple[str, ...] = (
        "eeg_channel",
        "eeg_delta_power",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
        "eeg_entropy",
    )

    optional_eeg_columns: Tuple[str, ...] = (
        "eeg_gamma_power",
        "eeg_hjorth_mobility",
        "eeg_hjorth_complexity",
        "eeg_line_length",
        "eeg_signal_std",
        "eeg_artifact_score",
        "eeg_quality_score",
    )

    required_multichannel_columns: Tuple[str, ...] = (
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
    )

    required_qrng_columns: Tuple[str, ...] = (
        "qrng_state",
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
    )

    optional_qrng_columns: Tuple[str, ...] = (
        "qrng_run_length_mean",
        "qrng_run_length_std",
        "qrng_run_length_max",
        "qrng_compressibility_proxy",
        "qrng_surprise_index",
        "qrng_transition_asymmetry",
        "qrng_quality_score",
    )

    # =====================================================
    # Synchronization validation
    # =====================================================

    require_utc_timestamps: bool = True
    require_monotonic_timestamps: bool = True
    require_non_overlapping_windows: bool = True

    require_no_cross_subject_leakage: bool = True
    require_no_cross_session_leakage: bool = True
    require_no_missing_timestamps: bool = True

    min_sync_quality: float = 0.95
    max_clock_drift_ms: float = 50.0
    max_timestamp_gap_ms: float = 250.0
    max_window_overlap_fraction: float = 0.05

    accepted_sync_modes: Tuple[str, ...] = (
        "hardware_shared_clock",
        "software_timestamp_aligned",
        "posthoc_clock_drift_corrected",
        "external_sync_marker",
    )

    rejected_sync_modes: Tuple[str, ...] = (
        "unknown_clock_relation",
        "manual_alignment_without_log",
        "random_pairing",
        "posthoc_alignment_to_maximize_signal",
    )

    sync_mode_column: str = "sync_mode"
    default_sync_mode_if_missing: str = "unknown_clock_relation"

    sync_mode_specs: Tuple[SyncModeSpec, ...] = field(
        default_factory=lambda: (
            SyncModeSpec(
                name="hardware_shared_clock",
                accepted_for_confirmatory=True,
                description=(
                    "EEG and QRNG timestamps are produced from a shared "
                    "hardware clock."
                ),
            ),
            SyncModeSpec(
                name="software_timestamp_aligned",
                accepted_for_confirmatory=True,
                description=(
                    "EEG and QRNG streams are aligned using logged "
                    "software timestamps."
                ),
            ),
            SyncModeSpec(
                name="posthoc_clock_drift_corrected",
                accepted_for_confirmatory=True,
                description=(
                    "Clock drift is measured, logged and corrected "
                    "before analysis."
                ),
            ),
            SyncModeSpec(
                name="external_sync_marker",
                accepted_for_confirmatory=True,
                description=(
                    "External synchronization markers align EEG and "
                    "QRNG streams."
                ),
            ),
            SyncModeSpec(
                name="unknown_clock_relation",
                accepted_for_confirmatory=False,
                description="No auditable clock relation is available.",
            ),
            SyncModeSpec(
                name="manual_alignment_without_log",
                accepted_for_confirmatory=False,
                description=(
                    "Manual alignment without auditable synchronization logs."
                ),
            ),
            SyncModeSpec(
                name="random_pairing",
                accepted_for_confirmatory=False,
                description=(
                    "EEG and QRNG are paired without true temporal "
                    "co-acquisition."
                ),
            ),
            SyncModeSpec(
                name="posthoc_alignment_to_maximize_signal",
                accepted_for_confirmatory=False,
                description=(
                    "Post hoc alignment chosen to maximize signal "
                    "is not allowed."
                ),
            ),
        )
    )

    # =====================================================
    # Module toggles
    # =====================================================

    run_schema_validation: bool = True
    run_sync_validation: bool = True
    run_windowing: bool = True
    run_feature_construction: bool = True
    run_lagged_alignment: bool = True
    run_conditional_analysis: bool = True
    run_controls: bool = True
    run_metrics: bool = True
    run_gate_sensitivity: bool = True
    run_diagnostics: bool = True

    protocol_only_if_no_data: bool = True

    # =====================================================
    # EEG / QRNG state settings
    # =====================================================

    primary_eeg_channel: str = "EEG Fpz-Cz"
    secondary_eeg_channel: str = "EEG Pz-Oz"

    eeg_channels: Tuple[str, ...] = (
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
    )

    use_multichannel_eeg: bool = True

    multichannel_primary_channel: str = "EEG Fpz-Cz"
    multichannel_secondary_channel: str = "EEG Pz-Oz"

    eeg_n_bins: int = 3
    qrng_n_bins: int = 2

    info_n_bins: int = 3
    q_n_bins: int = 3
    latent_n_states: int = 3

    eeg_state_column: str = "eeg_state"
    eeg_next_state_column: str = "eeg_next_state"

    qrng_state_column: str = "qrng_state"
    qrng_next_state_column: str = "qrng_next_state"

    multichannel_state_column: str = "mc_eeg_state"
    multichannel_next_state_column: str = "mc_eeg_next_state"
    multichannel_info_bin_column: str = "mc_eeg_info_bin"

    multichannel_n_states: int = 9
    multichannel_n_bins: int = 3
    multichannel_state_max_unique_values: int = 9

    eeg_state_input_columns: Tuple[str, ...] = (
        "eeg_delta_power",
        "eeg_alpha_power",
    )

    eeg_info_input_columns: Tuple[str, ...] = (
        "eeg_entropy",
        "eeg_theta_power",
        "eeg_alpha_power",
        "eeg_beta_power",
    )

    qrng_info_input_columns: Tuple[str, ...] = (
        "qrng_bit_balance",
        "qrng_transition_rate",
        "qrng_entropy",
    )

    multichannel_state_input_columns: Tuple[str, ...] = (
        "eeg_primary_delta_power",
        "eeg_primary_alpha_power",
        "eeg_secondary_delta_power",
        "eeg_secondary_alpha_power",
        "interchannel_delta_ratio",
        "interchannel_alpha_ratio",
        "fronto_parietal_delta_shift",
        "fronto_parietal_alpha_shift",
        "cross_channel_corr",
    )

    # =====================================================
    # Multi-resolution windowing
    # =====================================================

    window_specs: Tuple[WindowSpec, ...] = field(
        default_factory=lambda: (
            WindowSpec(
                name="w05s",
                seconds=5,
                primary=False,
                min_windows_total=500,
                min_windows_per_subject=50,
                description=(
                    "Secondary short window retained to test faster "
                    "subject-specific responses."
                ),
            ),
            WindowSpec(
                name="w30s",
                seconds=30,
                primary=True,
                min_windows_total=150,
                min_windows_per_subject=15,
                description=(
                    "Primary window anchored to the positive "
                    "multichannel surrogate lead."
                ),
            ),
            WindowSpec(
                name="w60s",
                seconds=60,
                primary=True,
                min_windows_total=80,
                min_windows_per_subject=8,
                description=(
                    "Primary longer window for subtle and slower "
                    "structural-selection dynamics."
                ),
            ),
            WindowSpec(
                name="w90s",
                seconds=90,
                primary=False,
                min_windows_total=50,
                min_windows_per_subject=5,
                description=(
                    "Secondary long window for slower or delayed "
                    "subject-specific effects."
                ),
            ),
        )
    )

    primary_window_seconds: Tuple[int, ...] = (
        30,
        60,
    )

    # Mantido exatamente conforme solicitado.
    # Quando a janela também estiver em primary_window_seconds,
    # a classificação primary terá precedência.
    secondary_window_seconds: Tuple[int, ...] = (
        5,
        30,
        60,
        90,
    )

    all_window_seconds: Tuple[int, ...] = (
        5,
        30,
        60,
        90,
    )

    window_id_column: str = "window_id"
    window_size_column: str = "window_seconds"

    # =====================================================
    # Directional lags
    # =====================================================

    primary_lags_windows: Tuple[int, ...] = (
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
    )

    secondary_lags_windows: Tuple[int, ...] = ()

    all_lags_windows: Tuple[int, ...] = (
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
    )

    prevent_lag_cross_subject: bool = True
    prevent_lag_cross_session: bool = True

    # =====================================================
    # Conditional CDR models
    # =====================================================

    conditional_models: Tuple[ConditionalModelSpec, ...] = field(
        default_factory=lambda: (
            ConditionalModelSpec(
                name="E0_EEG_next_given_EEG",
                label="EEG self-transition baseline",
                family="EEG_next",
                current_components=("eeg_state",),
                target_component="eeg_next_state",
                baseline_model=None,
                primary=True,
                multichannel=False,
                description="Baseline model: P(EEG_{t+τ} | EEG_t).",
            ),
            ConditionalModelSpec(
                name="E1_EEG_next_given_EEG_QRNG",
                label="EEG transition conditioned by QRNG",
                family="EEG_next",
                current_components=(
                    "eeg_state",
                    "qrng_state",
                ),
                target_component="eeg_next_state",
                baseline_model="E0_EEG_next_given_EEG",
                primary=True,
                multichannel=False,
                description=(
                    "Augmented model: "
                    "P(EEG_{t+τ} | EEG_t, QRNG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="E2_QRNG_next_given_QRNG",
                label="QRNG self-transition baseline",
                family="QRNG_next",
                current_components=("qrng_state",),
                target_component="qrng_next_state",
                baseline_model=None,
                primary=True,
                multichannel=False,
                description="Baseline model: P(QRNG_{t+τ} | QRNG_t).",
            ),
            ConditionalModelSpec(
                name="E3_QRNG_next_given_QRNG_EEG",
                label="QRNG transition conditioned by EEG",
                family="QRNG_next",
                current_components=(
                    "qrng_state",
                    "eeg_state",
                ),
                target_component="qrng_next_state",
                baseline_model="E2_QRNG_next_given_QRNG",
                primary=True,
                multichannel=False,
                description=(
                    "Augmented model: "
                    "P(QRNG_{t+τ} | QRNG_t, EEG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="E4_MCEEG_next_given_MCEEG",
                label="Multichannel EEG self-transition baseline",
                family="MCEEG_next",
                current_components=("mc_eeg_state",),
                target_component="mc_eeg_next_state",
                baseline_model=None,
                primary=False,
                multichannel=True,
                description=(
                    "Baseline model: "
                    "P(MC-EEG_{t+τ} | MC-EEG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="E5_MCEEG_next_given_MCEEG_QRNG",
                label="Multichannel EEG transition conditioned by QRNG",
                family="MCEEG_next",
                current_components=(
                    "mc_eeg_state",
                    "qrng_state",
                ),
                target_component="mc_eeg_next_state",
                baseline_model="E4_MCEEG_next_given_MCEEG",
                primary=False,
                multichannel=True,
                description=(
                    "Augmented model: "
                    "P(MC-EEG_{t+τ} | MC-EEG_t, QRNG_t)."
                ),
            ),
            ConditionalModelSpec(
                name="E6_QRNG_next_given_QRNG_MCEEG",
                label="QRNG transition conditioned by multichannel EEG",
                family="QRNG_next",
                current_components=(
                    "qrng_state",
                    "mc_eeg_state",
                ),
                target_component="qrng_next_state",
                baseline_model="E2_QRNG_next_given_QRNG",
                primary=True,
                multichannel=True,
                description=(
                    "Primary augmented model: "
                    "P(QRNG_{t+τ} | QRNG_t, MC-EEG_t)."
                ),
            ),
        )
    )

    primary_conditional_pairs: Tuple[
        Tuple[str, str],
        ...,
    ] = (
        (
            "E0_EEG_next_given_EEG",
            "E1_EEG_next_given_EEG_QRNG",
        ),
        (
            "E2_QRNG_next_given_QRNG",
            "E3_QRNG_next_given_QRNG_EEG",
        ),
        (
            "E2_QRNG_next_given_QRNG",
            "E6_QRNG_next_given_QRNG_MCEEG",
        ),
    )

    multichannel_conditional_pairs: Tuple[
        Tuple[str, str],
        ...,
    ] = (
        (
            "E4_MCEEG_next_given_MCEEG",
            "E5_MCEEG_next_given_MCEEG_QRNG",
        ),
        (
            "E2_QRNG_next_given_QRNG",
            "E6_QRNG_next_given_QRNG_MCEEG",
        ),
    )

    # =====================================================
    # Leakage-safe split settings
    # =====================================================

    conditional_split_ref_ratio: float = 0.50
    conditional_split_calib_ratio: float = 0.20
    conditional_split_test_ratio: float = 0.30

    conditional_min_ref_rows: int = 20
    conditional_min_calib_rows: int = 10
    conditional_min_test_rows: int = 10

    conditional_min_calib_ll_gain: float = 1e-6
    conditional_min_test_ll_gain: float = 1e-6

    use_subject_chronological_split: bool = True
    use_session_chronological_split: bool = True

    # =====================================================
    # Epsilon grids and injection
    # =====================================================

    inj_eps_true: float = 0.05

    eps_grid_short: Tuple[float, ...] = tuple(
        round(x * 0.01, 2)
        for x in range(0, 51)
    )

    eps_grid_medium: Tuple[float, ...] = tuple(
        round(x * 0.01, 2)
        for x in range(0, 101)
    )

    eps_grid_full: Tuple[float, ...] = tuple(
        round(x * 0.01, 2)
        for x in range(0, 201)
    )

    eps_grid: Tuple[float, ...] = tuple(
        round(x * 0.01, 2)
        for x in range(0, 201)
    )

    injection_eps_grid: Tuple[float, ...] = (
        0.00,
        0.01,
        0.03,
        0.05,
    )

    # O teto inicial passa de 0.80 para 2.00.
    # Se o ótimo ainda estiver na borda, a grade poderá ser ampliada.
    eps_saturation_value: float = 2.00

    eps_saturation_warning_fraction: float = 0.95
    eps_moderate_upper_bound: float = 1.00

    eps_auto_expand_enabled: bool = True
    eps_auto_expand_factor: float = 1.50
    eps_auto_expand_max: float = 5.00

    # =====================================================
    # Official gates / success criteria
    # =====================================================

    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    required_control_fraction: float = 0.75

    conditional_lift_min: float = 0.03
    strong_eps_min: float = 0.07

    # 1 sujeito em 10 é suficiente nesta coorte.
    subject_effect_fraction: float = 0.10
    moderate_subject_effect_fraction: float = 0.05
    min_positive_subjects: int = 1

    # Coortes reais maiores podem utilizar limites entre 1% e 3%.
    large_cohort_subject_fraction_min: float = 0.01
    large_cohort_subject_fraction_max: float = 0.03
    large_cohort_subject_threshold: int = 100

    min_test_ll_improvement: float = 0.0

    use_bic_complexity_gate: bool = True
    use_aic_secondary: bool = True

    holdout_max_delta: float = 0.10
    sensitivity_max_delta: float = 0.12

    official_gates: Tuple[GateSpec, ...] = field(
        default_factory=lambda: (
            GateSpec(
                name="F2",
                label="Negative controls",
                required_for_candidate=True,
                description=(
                    "All required negative controls must collapse."
                ),
            ),
            GateSpec(
                name="F3",
                label="Holdout generalization",
                required_for_candidate=True,
                description=(
                    "Selected ε must improve held-out likelihood "
                    "relative to ε=0."
                ),
            ),
            GateSpec(
                name="F6",
                label="Conditional lift",
                required_for_candidate=True,
                description=(
                    "Augmented model must produce registered "
                    "conditional lift."
                ),
            ),
            GateSpec(
                name="F7",
                label="Responder-specific subject consistency",
                required_for_candidate=True,
                description=(
                    "At least one valid responder is sufficient in "
                    "the current 10-subject discovery cohort. Larger "
                    "cohorts use an adaptive threshold between 1% and 3%."
                ),
            ),
            GateSpec(
                name="F8",
                label="Complexity/BIC",
                required_for_candidate=True,
                description=(
                    "Augmented model must not be rejected by "
                    "model-complexity criteria."
                ),
            ),
            GateSpec(
                name="F9",
                label="Ablation",
                required_for_candidate=True,
                description=(
                    "Removing cross-domain information should reduce "
                    "or remove the effect."
                ),
            ),
            GateSpec(
                name="F10",
                label="Density/sparsity",
                required_for_candidate=True,
                description=(
                    "State density must remain above minimum "
                    "transition-per-state thresholds."
                ),
            ),
            GateSpec(
                name="F11",
                label="Multiple-comparison guard",
                required_for_candidate=True,
                description=(
                    "Result must survive primary/exploratory "
                    "test separation."
                ),
            ),
            GateSpec(
                name="F12",
                label="Epsilon saturation diagnostic",
                required_for_candidate=False,
                description=(
                    "Epsilon saturation is diagnostic and triggers "
                    "grid expansion. It is not an automatic veto of "
                    "a candidate signal."
                ),
            ),
        )
    )

    # =====================================================
    # Gate sensitivity diagnostics
    # =====================================================

    gate_sensitivity_specs: Tuple[
        GateSensitivitySpec,
        ...,
    ] = field(
        default_factory=lambda: (
            GateSensitivitySpec(
                name="F7_official",
                target_gate="F7",
                enabled=True,
                primary_diagnostic=True,
                description=(
                    "Responder-specific subject-effect threshold."
                ),
            ),
            GateSensitivitySpec(
                name="F7_bootstrap",
                target_gate="F7",
                enabled=True,
                primary_diagnostic=True,
                description=(
                    "Bootstrap confidence interval over "
                    "subject-level positive lift."
                ),
            ),
            GateSensitivitySpec(
                name="F7_weighted",
                target_gate="F7",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "Subject consistency weighted by sync quality "
                    "and valid window count."
                ),
            ),
            GateSensitivitySpec(
                name="F7_cluster",
                target_gate="F7",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "Stable responder-subgroup diagnostic."
                ),
            ),
            GateSensitivitySpec(
                name="F7_LOSO",
                target_gate="F7",
                enabled=True,
                primary_diagnostic=True,
                description=(
                    "Leave-one-subject-out dependence diagnostic."
                ),
            ),
            GateSensitivitySpec(
                name="F8_BIC_official",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=True,
                description="Official BIC improvement gate.",
            ),
            GateSensitivitySpec(
                name="F8_AIC_secondary",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "AIC as secondary complexity diagnostic."
                ),
            ),
            GateSensitivitySpec(
                name="F8_heldout_LL",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=True,
                description=(
                    "Held-out likelihood improvement diagnostic."
                ),
            ),
            GateSensitivitySpec(
                name="F8_cross_validated_LL",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "Cross-validated likelihood diagnostic."
                ),
            ),
            GateSensitivitySpec(
                name="F8_state_density_corrected_BIC",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "BIC interpreted relative to state density."
                ),
            ),
            GateSensitivitySpec(
                name="F8_effective_complexity",
                target_gate="F8",
                enabled=True,
                primary_diagnostic=False,
                description=(
                    "Effective complexity diagnostic for sparse "
                    "transition states."
                ),
            ),
            GateSensitivitySpec(
                name="F12_epsilon_grid_short",
                target_gate="F12",
                enabled=True,
                primary_diagnostic=True,
                description="ε stability under grid 0.00–0.50.",
            ),
            GateSensitivitySpec(
                name="F12_epsilon_grid_medium",
                target_gate="F12",
                enabled=True,
                primary_diagnostic=True,
                description="ε stability under grid 0.00–1.00.",
            ),
            GateSensitivitySpec(
                name="F12_epsilon_grid_full",
                target_gate="F12",
                enabled=True,
                primary_diagnostic=True,
                description="ε stability under grid 0.00–2.00.",
            ),
            GateSensitivitySpec(
                name="F12_saturation_fraction",
                target_gate="F12",
                enabled=True,
                primary_diagnostic=True,
                description=(
                    "Fraction of rows depending on maximum ε."
                ),
            ),
        )
    )

    f7_bootstrap_replicates: int = 1000
    f7_bootstrap_seed: int = 7321

    f7_weight_by_sync_quality: bool = True
    f7_weight_by_window_count: bool = True

    f8_density_corrected_bic_enabled: bool = True
    f8_cross_validation_folds: int = 5

    f12_run_short_grid: bool = True
    f12_run_medium_grid: bool = True
    f12_run_full_grid: bool = True

    # =====================================================
    # Multiple-comparison guard
    # =====================================================

    enable_multiple_comparison_guard: bool = True

    allow_strong_claim_from_secondary_windows: bool = False
    allow_strong_claim_from_secondary_lags: bool = False

    # O par multicanal E2 → E6 integra o recorte primário.
    allow_strong_claim_from_multichannel_only: bool = True

    primary_test_window_seconds: Tuple[int, ...] = (
        30,
        60,
    )

    primary_test_lags_windows: Tuple[int, ...] = (
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
    )

    primary_test_pairs: Tuple[
        Tuple[str, str],
        ...,
    ] = (
        (
            "E0_EEG_next_given_EEG",
            "E1_EEG_next_given_EEG_QRNG",
        ),
        (
            "E2_QRNG_next_given_QRNG",
            "E3_QRNG_next_given_QRNG_EEG",
        ),
        (
            "E2_QRNG_next_given_QRNG",
            "E6_QRNG_next_given_QRNG_MCEEG",
        ),
    )

    exploratory_test_window_seconds: Tuple[int, ...] = (
        5,
        90,
    )

    exploratory_test_lags_windows: Tuple[int, ...] = ()

    # =====================================================
    # Density / sparsity safeguards
    # =====================================================

    min_subjects_for_confirmatory: int = 1
    min_sessions_for_confirmatory: int = 1

    min_transitions_total: int = 100
    min_transitions_per_state: float = 30.0

    recommended_transitions_per_state: float = 50.0
    ideal_transitions_per_state: float = 100.0

    max_nominal_states_soft: int = 64
    enforce_state_density_warning: bool = True

    # =====================================================
    # Controls
    # =====================================================

    n_controls: int = 12
    control_seed: int = 525252

    control_types: Tuple[str, ...] = (
        "within_subject_qrng_shuffle",
        "within_subject_eeg_shuffle",
        "circular_qrng_shift",
        "subject_mismatch",
        "timestamp_jitter_control",
        "phase_randomized_eeg_control",
        "window_permutation_control",
        "conditional_target_shuffle",
        "sync_break_control",
    )

    required_control_types: Tuple[str, ...] = (
        "within_subject_qrng_shuffle",
        "within_subject_eeg_shuffle",
        "circular_qrng_shift",
        "subject_mismatch",
        "conditional_target_shuffle",
        "sync_break_control",
    )

    timestamp_jitter_ms: float = 1000.0
    sync_break_shift_windows: int = 10
    phase_randomization_preserve_power: bool = True

    # =====================================================
    # Output files — protocol/schema/sync
    # =====================================================

    protocol_status_json: Path = Path(
        "results/phase3_2e/phase3_2e_protocol_status.json"
    )

    protocol_summary_txt: Path = Path(
        "results/phase3_2e/phase3_2e_protocol_summary.txt"
    )

    schema_report_json: Path = Path(
        "results/phase3_2e/phase3_2e_schema_report.json"
    )

    schema_failures_csv: Path = Path(
        "results/phase3_2e/phase3_2e_schema_failures.csv"
    )

    sync_report_json: Path = Path(
        "results/phase3_2e/phase3_2e_sync_report.json"
    )

    sync_summary_txt: Path = Path(
        "results/phase3_2e/phase3_2e_sync_summary.txt"
    )

    sync_failures_csv: Path = Path(
        "results/phase3_2e/phase3_2e_sync_failures.csv"
    )

    # =====================================================
    # Output files — windows/features/alignment
    # =====================================================

    windows_csv: Path = Path(
        "data/interim/phase3_2e/phase3_2e_windows.csv"
    )

    features_csv: Path = Path(
        "data/interim/phase3_2e/phase3_2e_features.csv"
    )

    feature_specs_json: Path = Path(
        "data/interim/phase3_2e/phase3_2e_feature_specs.json"
    )

    alignment_inventory_csv: Path = Path(
        "data/interim/phase3_2e/"
        "phase3_2e_alignment_inventory.csv"
    )

    lagged_frames_dir: Path = Path(
        "data/interim/phase3_2e/lagged_frames"
    )

    # =====================================================
    # Output files — conditional/controls/metrics
    # =====================================================

    conditional_model_scores_csv: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_conditional_model_scores.csv"
    )

    conditional_pair_scores_csv: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_conditional_pair_scores.csv"
    )

    conditional_summary_json: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_conditional_summary.json"
    )

    conditional_summary_txt: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_conditional_summary.txt"
    )

    control_runs_csv: Path = Path(
        "results/phase3_2e/phase3_2e_control_runs.csv"
    )

    control_pair_scores_csv: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_control_pair_scores.csv"
    )

    controls_json: Path = Path(
        "results/phase3_2e/phase3_2e_controls.json"
    )

    controls_summary_txt: Path = Path(
        "results/phase3_2e/phase3_2e_controls_summary.txt"
    )

    gates_json: Path = Path(
        "results/phase3_2e/phase3_2e_gates.json"
    )

    metrics_table_csv: Path = Path(
        "results/phase3_2e/phase3_2e_metrics_table.csv"
    )

    metrics_summary_txt: Path = Path(
        "results/phase3_2e/phase3_2e_metrics_summary.txt"
    )

    multiple_comparison_guard_json: Path = Path(
        "results/phase3_2e/"
        "phase3_2e_multiple_comparison_guard.json"
    )

    # =====================================================
    # Output files — gate sensitivity
    # =====================================================

    f7_subject_sensitivity_csv: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f7_subject_sensitivity.csv"
    )

    f7_subject_sensitivity_json: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f7_subject_sensitivity.json"
    )

    f8_complexity_diagnostic_csv: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f8_complexity_diagnostic.csv"
    )

    f8_complexity_diagnostic_json: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f8_complexity_diagnostic.json"
    )

    f12_epsilon_saturation_csv: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f12_epsilon_saturation_diagnostic.csv"
    )

    f12_epsilon_saturation_json: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_f12_epsilon_saturation_diagnostic.json"
    )

    # =====================================================
    # Output files — diagnostics/runner/audit
    # =====================================================

    diagnostics_report_json: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_diagnostics_report.json"
    )

    diagnostics_summary_txt: Path = Path(
        "results/phase3_2e/diagnostics/"
        "phase3_2e_diagnostics_summary.txt"
    )

    runner_report_json: Path = Path(
        "results/phase3_2e/phase3_2e_runner_report.json"
    )

    runner_summary_txt: Path = Path(
        "results/phase3_2e/phase3_2e_runner_summary.txt"
    )

    # =====================================================
    # Result status labels
    # =====================================================

    status_protocol_ready: str = "protocol_ready"
    status_sync_invalid: str = "sync_invalid"
    status_candidate_signal: str = "candidate_signal"
    status_exploratory_lead: str = "exploratory_lead"

    status_exploratory_lead_with_saturation_warning: str = (
        "exploratory_lead_with_saturation_warning"
    )

    status_synchronized_null_result: str = (
        "synchronized_null_result"
    )

    status_pending_synchronized_data: str = (
        "pending_synchronized_data"
    )

    # =====================================================
    # Reproducibility
    # =====================================================

    random_seed: int = 20260603
    numpy_seed: int = 20260603

    save_intermediate_files: bool = True
    overwrite_existing_outputs: bool = True
    verbose: int = 1


# =========================================================
# Helper functions
# =========================================================

def load_phase3_2e_config() -> Phase32EConfig:
    """
    Return the Phase III.2E configuration and create required folders.
    """

    cfg = Phase32EConfig()

    required_dirs = [
        cfg.raw_phase3_2e_dir,
        cfg.raw_synchronized_dir,
        cfg.raw_eeg_dir,
        cfg.raw_qrng_dir,
        cfg.interim_dir,
        cfg.processed_dir,
        cfg.results_dir,
        cfg.diagnostics_dir,
        cfg.audit_bundle_dir,
        cfg.docs_dir,
        cfg.lagged_frames_dir,
    ]

    for path in required_dirs:
        Path(path).mkdir(
            parents=True,
            exist_ok=True,
        )

    return cfg


def sync_mode_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, SyncModeSpec]:
    """
    Return synchronization specifications indexed by name.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.name: spec
        for spec in cfg.sync_mode_specs
    }


def window_spec_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, WindowSpec]:
    """
    Return window specifications indexed by name.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.name: spec
        for spec in cfg.window_specs
    }


def window_seconds_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[int, WindowSpec]:
    """
    Return window specifications indexed by duration.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.seconds: spec
        for spec in cfg.window_specs
    }


def conditional_model_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, ConditionalModelSpec]:
    """
    Return conditional model specifications indexed by model name.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.name: spec
        for spec in cfg.conditional_models
    }


def gate_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, GateSpec]:
    """
    Return gate specifications indexed by gate name.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.name: spec
        for spec in cfg.official_gates
    }


def gate_sensitivity_map(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, GateSensitivitySpec]:
    """
    Return gate-sensitivity specifications indexed by name.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        spec.name: spec
        for spec in cfg.gate_sensitivity_specs
    }


def required_schema_columns(
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[str, ...]:
    """
    Return all required synchronized-input columns.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    columns: List[str] = []

    columns.extend(cfg.identity_columns)
    columns.extend(cfg.timing_columns)
    columns.extend(cfg.required_eeg_columns)
    columns.extend(cfg.required_qrng_columns)

    if cfg.use_multichannel_eeg:
        columns.extend(cfg.required_multichannel_columns)

    return tuple(
        dict.fromkeys(columns)
    )


def optional_schema_columns(
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[str, ...]:
    """
    Return recognized optional synchronized-input columns.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    columns: List[str] = []

    columns.extend(cfg.optional_eeg_columns)
    columns.extend(cfg.optional_qrng_columns)
    columns.append(cfg.sync_mode_column)

    return tuple(
        dict.fromkeys(columns)
    )


def active_conditional_pairs(
    cfg: Optional[Phase32EConfig] = None,
) -> Tuple[Tuple[str, str], ...]:
    """
    Return all active model pairs without duplicates.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    pairs: List[Tuple[str, str]] = list(
        cfg.primary_conditional_pairs
    )

    if cfg.use_multichannel_eeg:
        pairs.extend(
            list(cfg.multichannel_conditional_pairs)
        )

    return tuple(
        dict.fromkeys(pairs)
    )


def primary_test_definitions(
    cfg: Optional[Phase32EConfig] = None,
) -> List[Dict[str, object]]:
    """
    Define the primary evidence-guided replication tests.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    tests: List[Dict[str, object]] = []

    for window_seconds in cfg.primary_test_window_seconds:
        for lag_value in cfg.primary_test_lags_windows:
            for baseline, augmented in cfg.primary_test_pairs:
                tests.append(
                    {
                        "window_seconds": window_seconds,
                        "lag_windows": lag_value,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "primary": True,
                    }
                )

    return tests


def exploratory_test_definitions(
    cfg: Optional[Phase32EConfig] = None,
) -> List[Dict[str, object]]:
    """
    Define exploratory tests outside the primary window configuration.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    tests: List[Dict[str, object]] = []

    exploratory_windows = tuple(
        value
        for value in cfg.all_window_seconds
        if value not in cfg.primary_test_window_seconds
    )

    exploratory_lags = tuple(
        value
        for value in cfg.all_lags_windows
        if value not in cfg.primary_test_lags_windows
    )

    for window_seconds in exploratory_windows:
        for lag_value in cfg.all_lags_windows:
            for baseline, augmented in active_conditional_pairs(cfg):
                tests.append(
                    {
                        "window_seconds": window_seconds,
                        "lag_windows": lag_value,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "primary": False,
                        "reason": "secondary_window",
                    }
                )

    for window_seconds in cfg.primary_test_window_seconds:
        for lag_value in exploratory_lags:
            for baseline, augmented in active_conditional_pairs(cfg):
                tests.append(
                    {
                        "window_seconds": window_seconds,
                        "lag_windows": lag_value,
                        "baseline_model": baseline,
                        "augmented_model": augmented,
                        "primary": False,
                        "reason": "secondary_lag",
                    }
                )

    return tests


def describe_config(
    cfg: Optional[Phase32EConfig] = None,
) -> Dict[str, object]:
    """
    Return a compact serializable configuration description.
    """

    if cfg is None:
        cfg = load_phase3_2e_config()

    return {
        "project_name": cfg.project_name,
        "phase_name": cfg.phase_name,
        "phase_title": cfg.phase_title,
        "version": cfg.version,
        "protocol_status": cfg.protocol_status,
        "empirical_execution_status": (
            cfg.empirical_execution_status
        ),
        "synchronized_input_csv": str(
            cfg.synchronized_input_csv
        ),
        "allow_protocol_only_without_data": (
            cfg.allow_protocol_only_without_data
        ),
        "allow_synthetic_demo_data": (
            cfg.allow_synthetic_demo_data
        ),
        "require_real_synchronized_data_for_claim": (
            cfg.require_real_synchronized_data_for_claim
        ),
        "required_schema_columns": list(
            required_schema_columns(cfg)
        ),
        "optional_schema_columns": list(
            optional_schema_columns(cfg)
        ),
        "min_sync_quality": cfg.min_sync_quality,
        "max_clock_drift_ms": cfg.max_clock_drift_ms,
        "max_timestamp_gap_ms": cfg.max_timestamp_gap_ms,
        "accepted_sync_modes": list(
            cfg.accepted_sync_modes
        ),
        "rejected_sync_modes": list(
            cfg.rejected_sync_modes
        ),
        "primary_eeg_channel": cfg.primary_eeg_channel,
        "secondary_eeg_channel": cfg.secondary_eeg_channel,
        "use_multichannel_eeg": cfg.use_multichannel_eeg,
        "primary_window_seconds": list(
            cfg.primary_window_seconds
        ),
        "secondary_window_seconds": list(
            cfg.secondary_window_seconds
        ),
        "all_window_seconds": list(
            cfg.all_window_seconds
        ),
        "primary_lags_windows": list(
            cfg.primary_lags_windows
        ),
        "secondary_lags_windows": list(
            cfg.secondary_lags_windows
        ),
        "all_lags_windows": list(
            cfg.all_lags_windows
        ),
        "conditional_models": [
            spec.name
            for spec in cfg.conditional_models
        ],
        "primary_conditional_pairs": list(
            cfg.primary_conditional_pairs
        ),
        "multichannel_conditional_pairs": list(
            cfg.multichannel_conditional_pairs
        ),
        "active_conditional_pairs": list(
            active_conditional_pairs(cfg)
        ),
        "official_gates": [
            spec.name
            for spec in cfg.official_gates
        ],
        "gate_sensitivity_specs": [
            spec.name
            for spec in cfg.gate_sensitivity_specs
        ],
        "control_types": list(cfg.control_types),
        "required_control_types": list(
            cfg.required_control_types
        ),
        "n_controls": cfg.n_controls,
        "eps_grid_short": [
            min(cfg.eps_grid_short),
            max(cfg.eps_grid_short),
        ],
        "eps_grid_medium": [
            min(cfg.eps_grid_medium),
            max(cfg.eps_grid_medium),
        ],
        "eps_grid_full": [
            min(cfg.eps_grid_full),
            max(cfg.eps_grid_full),
        ],
        "eps_saturation_value": (
            cfg.eps_saturation_value
        ),
        "eps_saturation_warning_fraction": (
            cfg.eps_saturation_warning_fraction
        ),
        "eps_moderate_upper_bound": (
            cfg.eps_moderate_upper_bound
        ),
        "eps_auto_expand_enabled": (
            cfg.eps_auto_expand_enabled
        ),
        "eps_auto_expand_factor": (
            cfg.eps_auto_expand_factor
        ),
        "eps_auto_expand_max": (
            cfg.eps_auto_expand_max
        ),
        "conditional_lift_min": (
            cfg.conditional_lift_min
        ),
        "strong_eps_min": cfg.strong_eps_min,
        "subject_effect_fraction": (
            cfg.subject_effect_fraction
        ),
        "min_positive_subjects": (
            cfg.min_positive_subjects
        ),
        "large_cohort_subject_fraction_min": (
            cfg.large_cohort_subject_fraction_min
        ),
        "large_cohort_subject_fraction_max": (
            cfg.large_cohort_subject_fraction_max
        ),
        "multiple_comparison_guard": (
            cfg.enable_multiple_comparison_guard
        ),
    }


if __name__ == "__main__":
    config = load_phase3_2e_config()

    print(
        "Phase III.2E config loaded successfully."
    )

    print(
        describe_config(config)
    )