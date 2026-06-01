from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


# =========================================================
# State model specification
# =========================================================

@dataclass(frozen=True)
class StateModelSpec:
    """
    Defines a locked state-space candidate for Phase III.1.

    Each model specifies:
        - name: internal model identifier
        - label: human-readable description
        - components: feature/state columns used by the model
        - bins: number of discrete bins/states for each component
        - active: whether the runner should evaluate this model
    """

    name: str
    label: str
    components: Tuple[str, ...]
    bins: Tuple[int, ...]
    active: bool = True

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def n_states(self) -> int:
        total = 1
        for b in self.bins:
            total *= int(b)
        return int(total)


# =========================================================
# Phase III.1 configuration
# =========================================================

@dataclass
class Phase3_1Config:
    """
    Configuration for Phase III.1 — Advanced EEG + RNG Informational Coupling Redesign.

    This phase tests whether multi-subject EEG, informational features,
    latent states, and quantum-aware RNG proxies reveal stable joint structure
    that was not detected by the simpler Phase III.0 joint representation.

    Main methodological principles:
        - Do not concatenate independent RNG requests.
        - Fit latent/scaling structures on train only.
        - Preserve controls, holdout, and sensitivity gates.
        - Treat EEG + RNG as structural joint alignment, not physical synchronized coupling.
        - Compare advanced joint models against EEG-only and RNG-only baselines.
    """

    # =====================================================
    # Project / output
    # =====================================================

    project_name: str = "Phase3.1-Advanced-EEG-RNG"
    phase_name: str = "Phase III.1"
    results_dir: Path = Path("results/phase3_1")
    interim_dir: Path = Path("data/interim/phase3_1")
    processed_dir: Path = Path("data/processed/phase3_1")

    save_intermediate_features: bool = True
    save_model_comparison: bool = True
    save_subject_results: bool = True
    save_controls: bool = True
    save_injection_curve: bool = True
    save_ablation_results: bool = True

    # =====================================================
    # EEG dataset
    # =====================================================

    eeg_raw_dir: Path = Path("data/raw/eeg")

    # If empty, the loader will discover Sleep-EDF files automatically.
    # Recommended initial run: max_subjects = 5.
    subject_ids: Tuple[str, ...] = tuple()
    max_subjects: int = 10
    min_subjects_for_loso: int = 3

    # Sleep-EDF file conventions.
    psg_suffix: str = "-PSG.edf"
    hypnogram_suffix: str = "-Hypnogram.edf"

    # Preferred EEG channels. The loader should use the first available channel.
    eeg_channels_preferred: Tuple[str, ...] = (
        "EEG Fpz-Cz",
        "EEG Pz-Oz",
        "Fpz-Cz",
        "Pz-Oz",
    )

    eeg_epoch_seconds: int = 30
    eeg_resample_hz: Optional[float] = 100.0

    # Basic filtering for EEG feature extraction.
    eeg_filter_l_freq: float = 0.5
    eeg_filter_h_freq: float = 35.0

    # Optional cap for fast debugging. Keep None for final runs.
    max_epochs_per_subject: Optional[int] = None
    min_epochs_per_subject: int = 100

    # Sleep stage handling.
    # Sleep-EDF commonly uses:
    #   Sleep stage W, 1, 2, 3, 4, R, Movement time, ?
    keep_sleep_stages: Tuple[str, ...] = (
        "W",
        "N1",
        "N2",
        "N3",
        "REM",
    )
    merge_n3_n4: bool = True
    drop_unknown_stages: bool = True

    # =====================================================
    # EEG observed features
    # =====================================================

    eeg_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
        }
    )

    eeg_observed_features: Tuple[str, ...] = (
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "alpha_delta_ratio",
        "delta_alpha_balance",
    )

    # Core EEG features expected to remain important across models.
    eeg_core_features: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
    )

    # =====================================================
    # EEG informational features
    # =====================================================

    eeg_informational_features: Tuple[str, ...] = (
        "spectral_entropy",
        "permutation_entropy",
        "hjorth_mobility",
        "hjorth_complexity",
        "line_length",
        "bandpower_volatility",
        "state_instability",
    )

    permutation_entropy_order: int = 3
    permutation_entropy_delay: int = 1

    # Rolling window in EEG epochs for volatility and instability features.
    eeg_info_rolling_window_epochs: int = 5

    # =====================================================
    # RNG dataset
    # =====================================================

    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    rng_source_name: str = "ANU"
    rng_sequence_length: int = 1024

    # Mandatory for Phase III.1 unless explicitly testing raw uint8.
    rng_use_bits: bool = True

    # Preserve falsifiability: do not concatenate independent API requests.
    allow_multiple_rng_requests: bool = False

    # RNG windowing for local informational and quantum-aware proxies.
    rng_window_size: int = 64

    # The loader will create RNG local windows and resample/align them
    # deterministically to EEG epochs.
    rng_alignment_mode: str = "resample_rng_metrics_to_eeg_epochs"

    # Accepted values:
    #   "resample_rng_metrics_to_eeg_epochs"
    #   "tile_rng_metrics_declared"
    #   "truncate_to_common_length"
    rng_allowed_alignment_modes: Tuple[str, ...] = (
        "resample_rng_metrics_to_eeg_epochs",
        "tile_rng_metrics_declared",
        "truncate_to_common_length",
    )

    # =====================================================
    # RNG observed / informational / quantum-aware features
    # =====================================================

    rng_basic_features: Tuple[str, ...] = (
        "rng_bit",
        "bit_balance_local",
        "transition_rate",
        "pair_frequency_00",
        "pair_frequency_01",
        "pair_frequency_10",
        "pair_frequency_11",
    )

    rng_informational_features: Tuple[str, ...] = (
        "rng_entropy_local",
        "entropy_rate_proxy",
        "run_length_mean",
        "run_length_max",
        "run_length_std",
        "compressibility_proxy",
        "surprise_index",
        "transition_asymmetry",
        "micro_cluster_deviation",
    )

    rng_quantum_proxy_features: Tuple[str, ...] = (
        "q_entropy_rate_drift",
        "q_transition_asymmetry",
        "q_run_instability",
        "q_surprise_burst",
        "q_micro_cluster_deviation",
        "q_local_balance_deviation",
    )

    # =====================================================
    # Informational / latent / quantum-aware layers
    # =====================================================

    use_informational_layer: bool = True
    use_latent_layer: bool = True
    use_quantum_proxy_layer: bool = True

    # I_t compact components.
    eeg_info_component_name: str = "eeg_info_bin"
    rng_info_component_name: str = "rng_info_bin"

    eeg_info_source_features: Tuple[str, ...] = (
        "spectral_entropy",
        "permutation_entropy",
        "state_instability",
    )

    rng_info_source_features: Tuple[str, ...] = (
        "rng_entropy_local",
        "surprise_index",
        "compressibility_proxy",
    )

    # Z_t latent state.
    latent_component_name: str = "latent_state"
    latent_method: str = "kmeans"
    latent_k: int = 3
    latent_random_state: int = 42
    latent_fit_train_only: bool = True

    latent_features: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
        "spectral_entropy",
        "rng_entropy_local",
        "surprise_index",
    )

    # Q_t compact proxy.
    q_component_name: str = "q_rng_bin"

    q_source_features: Tuple[str, ...] = (
        "q_entropy_rate_drift",
        "q_transition_asymmetry",
        "q_run_instability",
        "q_surprise_burst",
        "q_micro_cluster_deviation",
        "q_local_balance_deviation",
    )

    # =====================================================
    # Discretization
    # =====================================================

    strategy: str = "quantile"

    # Quantiles by bin count.
    # The runner/features layer can select the correct tuple based on each model component.
    quantiles_by_bins: Dict[int, Tuple[float, ...]] = field(
        default_factory=lambda: {
            2: (0.50,),
            3: (0.33, 0.66),
            4: (0.25, 0.50, 0.75),
            5: (0.20, 0.40, 0.60, 0.80),
        }
    )

    # Alternative quantiles for F5 sensitivity.
    sensitivity_quantiles_by_bins: Dict[int, Tuple[float, ...]] = field(
        default_factory=lambda: {
            2: (0.45,),
            3: (0.30, 0.70),
            4: (0.20, 0.50, 0.80),
            5: (0.15, 0.35, 0.60, 0.80),
        }
    )

    # =====================================================
    # Candidate state models M0-M5
    # =====================================================

    state_models: Tuple[StateModelSpec, ...] = field(
        default_factory=lambda: (
            StateModelSpec(
                name="M0_observed_compact",
                label="Observed compact baseline: delta_power x rng_bit",
                components=("delta_power", "rng_bit"),
                bins=(4, 2),
                active=True,
            ),
            StateModelSpec(
                name="M1_eeg_informational",
                label="EEG informational model: delta_power x eeg_info_bin x rng_bit",
                components=("delta_power", "eeg_info_bin", "rng_bit"),
                bins=(3, 3, 2),
                active=True,
            ),
            StateModelSpec(
                name="M2_rng_quantum_proxy",
                label="RNG quantum-aware proxy model: delta_power x q_rng_bin",
                components=("delta_power", "q_rng_bin"),
                bins=(3, 3),
                active=True,
            ),
            StateModelSpec(
                name="M3_informational_joint",
                label="Informational joint model: eeg_info_bin x rng_info_bin",
                components=("eeg_info_bin", "rng_info_bin"),
                bins=(3, 3),
                active=True,
            ),
            StateModelSpec(
                name="M4_latent_joint",
                label="Latent joint model: latent_state x rng_bit",
                components=("latent_state", "rng_bit"),
                bins=(3, 2),
                active=True,
            ),
            StateModelSpec(
                name="M5_augmented_final",
                label="Augmented final model: latent_state x rng_info_bin x q_rng_bin",
                components=("latent_state", "rng_info_bin", "q_rng_bin"),
                bins=(3, 3, 3),
                active=True,
            ),
        )
    )

    final_state_model: str = "M5_augmented_final"
    baseline_state_model: str = "M0_observed_compact"

    # =====================================================
    # State-space safety rules
    # =====================================================

    max_state_space: int = 36
    min_transitions_per_state: int = 50
    min_transitions_per_observed_state: int = 30

    # If True, runner may skip models that violate density rules instead of crashing.
    skip_models_failing_density: bool = True

    # =====================================================
    # Temporal embedding / transitions
    # =====================================================

    lag: int = 1

    # =====================================================
    # Kernel parameters
    # =====================================================

    dirichlet_alpha: float = 0.01
    min_prob: float = 1e-12

    # Same grid convention used in previous phases.
    eps_grid: Tuple[float, ...] = tuple(i * 0.01 for i in range(81))

    # =====================================================
    # Holdout / validation design
    # =====================================================

    random_seed: int = 42

    # Main Phase III.1 holdout.
    holdout_mode: str = "leave_one_subject_out"

    # Optional fallback if subject count is insufficient.
    fallback_holdout_mode: str = "chronological"

    train_ratio: float = 0.75

    allowed_holdout_modes: Tuple[str, ...] = (
        "leave_one_subject_out",
        "chronological",
        "interleaved",
    )

    # Sleep-stage secondary analysis.
    enable_sleep_stage_analysis: bool = True
    sleep_stage_holdout_enabled: bool = False

    # =====================================================
    # Controls
    # =====================================================

    n_controls: int = 12
    control_tol: float = 0.05
    control_fraction: float = 0.75

    enable_control_rng_shuffle: bool = True
    enable_control_rng_circular_shift: bool = True
    enable_control_eeg_epoch_shuffle: bool = True
    enable_control_eeg_stage_stratified_shuffle: bool = True
    enable_control_joint_pairing_shuffle: bool = True
    enable_control_subject_mismatch: bool = True
    enable_control_latent_shuffle: bool = True
    enable_control_proxy_ablation: bool = True

    rng_circular_shift_min: int = 64
    eeg_block_shuffle_size_epochs: int = 10

    # =====================================================
    # Injection / injectability
    # =====================================================

    inj_eps_true: float = 0.05
    injection_eps_grid: Tuple[float, ...] = (0.01, 0.03, 0.05)

    gate_tol_abs: float = 0.05

    # Store injection curve even when final F1 uses inj_eps_true.
    compute_injection_curve: bool = True

    # =====================================================
    # Advanced gates F6-F9
    # =====================================================

    # F3
    holdout_delta: float = 0.10

    # F5
    sensitivity_delta: float = 0.12

    # F6 — Incremental Joint Lift.
    joint_lift_min: float = 0.03
    strong_joint_eps_min: float = 0.07

    # F7 — Subject Generalization.
    subject_effect_fraction: float = 0.60

    # F8 — Complexity Penalty.
    use_complexity_penalty: bool = True
    complexity_metric: str = "bic"
    min_test_ll_improvement: float = 0.0

    # F9 — Proxy Ablation Stability.
    ablation_delta_min: float = 0.02

    # =====================================================
    # Baseline comparison
    # =====================================================

    compute_eeg_only_baseline: bool = True
    compute_rng_only_baseline: bool = True
    compare_against_phase3_0_targets: bool = True

    phase3_0_eeg_eps_reference: float = 0.04
    phase3_0_rng_eps_reference: float = 0.00
    phase3_0_joint_eps_reference_low: float = 0.00
    phase3_0_joint_eps_reference_high: float = 0.03

    # =====================================================
    # Logging
    # =====================================================

    verbose: int = 1
    progress_every: int = 10

    # =====================================================
    # Path helpers
    # =====================================================

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # Validation
    # =====================================================

    def validate(self) -> None:
        # ------------------------------
        # Paths
        # ------------------------------
        if not self.eeg_raw_dir.exists():
            raise FileNotFoundError(
                f"EEG raw directory not found: {self.eeg_raw_dir}"
            )

        if not self.rng_file.exists():
            raise FileNotFoundError(
                f"RNG file not found: {self.rng_file}"
            )

        # ------------------------------
        # EEG
        # ------------------------------
        if self.max_subjects < 1:
            raise ValueError("max_subjects must be >= 1")

        if self.min_subjects_for_loso < 2:
            raise ValueError("min_subjects_for_loso must be >= 2")

        if self.eeg_epoch_seconds < 1:
            raise ValueError("eeg_epoch_seconds must be >= 1")

        if self.eeg_resample_hz is not None and self.eeg_resample_hz <= 0:
            raise ValueError("eeg_resample_hz must be > 0 or None")

        if self.eeg_filter_l_freq < 0:
            raise ValueError("eeg_filter_l_freq must be >= 0")

        if self.eeg_filter_h_freq <= self.eeg_filter_l_freq:
            raise ValueError("eeg_filter_h_freq must be > eeg_filter_l_freq")

        if len(self.eeg_channels_preferred) < 1:
            raise ValueError("eeg_channels_preferred must contain at least one channel")

        if self.min_epochs_per_subject < 10:
            raise ValueError("min_epochs_per_subject must be >= 10")

        if self.max_epochs_per_subject is not None and self.max_epochs_per_subject < 10:
            raise ValueError("max_epochs_per_subject must be >= 10 or None")

        # ------------------------------
        # RNG
        # ------------------------------
        if self.rng_sequence_length < 10:
            raise ValueError("rng_sequence_length must be >= 10")

        if self.allow_multiple_rng_requests:
            raise ValueError(
                "allow_multiple_rng_requests must remain False for the registered "
                "Phase III.1 design."
            )

        if self.rng_window_size < 8:
            raise ValueError("rng_window_size must be >= 8")

        if self.rng_alignment_mode not in self.rng_allowed_alignment_modes:
            raise ValueError(
                f"Unsupported rng_alignment_mode: {self.rng_alignment_mode}"
            )

        # ------------------------------
        # Features
        # ------------------------------
        for band_name, (lo, hi) in self.eeg_bands.items():
            if lo < 0 or hi <= lo:
                raise ValueError(f"Invalid EEG band definition for {band_name}: {(lo, hi)}")

        if self.permutation_entropy_order < 2:
            raise ValueError("permutation_entropy_order must be >= 2")

        if self.permutation_entropy_delay < 1:
            raise ValueError("permutation_entropy_delay must be >= 1")

        if self.eeg_info_rolling_window_epochs < 2:
            raise ValueError("eeg_info_rolling_window_epochs must be >= 2")

        # ------------------------------
        # Latent model
        # ------------------------------
        if self.latent_method not in {"kmeans"}:
            raise ValueError("Only latent_method='kmeans' is currently supported")

        if self.latent_k < 2:
            raise ValueError("latent_k must be >= 2")

        if not self.latent_fit_train_only:
            raise ValueError(
                "latent_fit_train_only must remain True to avoid train/test leakage"
            )

        # ------------------------------
        # Discretization
        # ------------------------------
        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        for n_bins, qs in self.quantiles_by_bins.items():
            self._validate_quantiles_for_bins(n_bins, qs, label="quantiles_by_bins")

        for n_bins, qs in self.sensitivity_quantiles_by_bins.items():
            self._validate_quantiles_for_bins(
                n_bins,
                qs,
                label="sensitivity_quantiles_by_bins",
            )

        # ------------------------------
        # State models
        # ------------------------------
        if len(self.state_models) < 1:
            raise ValueError("state_models must contain at least one model")

        model_names = set()

        for model in self.state_models:
            if model.name in model_names:
                raise ValueError(f"Duplicate state model name: {model.name}")

            model_names.add(model.name)

            if len(model.components) != len(model.bins):
                raise ValueError(
                    f"State model {model.name} has mismatched components and bins"
                )

            if model.n_components < 1:
                raise ValueError(f"State model {model.name} must have >= 1 component")

            if any(b < 2 for b in model.bins):
                raise ValueError(f"State model {model.name} has bin count < 2")

            if model.n_states > self.max_state_space:
                raise ValueError(
                    f"State model {model.name} has {model.n_states} states, "
                    f"exceeding max_state_space={self.max_state_space}"
                )

            for b in model.bins:
                if b not in self.quantiles_by_bins:
                    raise ValueError(
                        f"State model {model.name} uses {b} bins, but no quantiles "
                        f"are defined for this bin count"
                    )

                if b not in self.sensitivity_quantiles_by_bins:
                    raise ValueError(
                        f"State model {model.name} uses {b} bins, but no sensitivity "
                        f"quantiles are defined for this bin count"
                    )

        if self.final_state_model not in model_names:
            raise ValueError(
                f"final_state_model not found in state_models: {self.final_state_model}"
            )

        if self.baseline_state_model not in model_names:
            raise ValueError(
                f"baseline_state_model not found in state_models: {self.baseline_state_model}"
            )

        if not any(m.active for m in self.state_models):
            raise ValueError("At least one state model must be active")

        # ------------------------------
        # State-space safety
        # ------------------------------
        if self.max_state_space < 2:
            raise ValueError("max_state_space must be >= 2")

        if self.min_transitions_per_state < 1:
            raise ValueError("min_transitions_per_state must be >= 1")

        if self.min_transitions_per_observed_state < 1:
            raise ValueError("min_transitions_per_observed_state must be >= 1")

        # ------------------------------
        # Kernel / epsilon grid
        # ------------------------------
        if self.lag < 1:
            raise ValueError("lag must be >= 1")

        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        if len(self.eps_grid) < 2:
            raise ValueError("eps_grid must contain at least two values")

        if any(e < 0 for e in self.eps_grid):
            raise ValueError("eps_grid cannot contain negative values")

        if tuple(sorted(self.eps_grid)) != self.eps_grid:
            raise ValueError("eps_grid must be sorted ascending")

        # ------------------------------
        # Holdout
        # ------------------------------
        if self.holdout_mode not in self.allowed_holdout_modes:
            raise ValueError(f"Unsupported holdout_mode: {self.holdout_mode}")

        if self.fallback_holdout_mode not in self.allowed_holdout_modes:
            raise ValueError(
                f"Unsupported fallback_holdout_mode: {self.fallback_holdout_mode}"
            )

        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        # ------------------------------
        # Controls
        # ------------------------------
        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")

        if self.control_tol <= 0:
            raise ValueError("control_tol must be > 0")

        if not (0.0 < self.control_fraction <= 1.0):
            raise ValueError("control_fraction must be in (0, 1]")

        if self.rng_circular_shift_min < 1:
            raise ValueError("rng_circular_shift_min must be >= 1")

        if self.eeg_block_shuffle_size_epochs < 2:
            raise ValueError("eeg_block_shuffle_size_epochs must be >= 2")

        # ------------------------------
        # Injection / gates
        # ------------------------------
        if self.inj_eps_true <= 0:
            raise ValueError("inj_eps_true must be > 0")

        if self.gate_tol_abs <= 0:
            raise ValueError("gate_tol_abs must be > 0")

        if len(self.injection_eps_grid) < 1:
            raise ValueError("injection_eps_grid must contain at least one value")

        if any(e <= 0 for e in self.injection_eps_grid):
            raise ValueError("injection_eps_grid must contain positive values only")

        if self.inj_eps_true not in self.injection_eps_grid:
            raise ValueError(
                "inj_eps_true should be included in injection_eps_grid for Phase III.1"
            )

        if self.holdout_delta <= 0:
            raise ValueError("holdout_delta must be > 0")

        if self.sensitivity_delta <= 0:
            raise ValueError("sensitivity_delta must be > 0")

        if self.joint_lift_min <= 0:
            raise ValueError("joint_lift_min must be > 0")

        if self.strong_joint_eps_min <= 0:
            raise ValueError("strong_joint_eps_min must be > 0")

        if not (0.0 < self.subject_effect_fraction <= 1.0):
            raise ValueError("subject_effect_fraction must be in (0, 1]")

        if self.complexity_metric not in {"bic", "mdl", "test_ll"}:
            raise ValueError("complexity_metric must be one of: 'bic', 'mdl', 'test_ll'")

        if self.ablation_delta_min <= 0:
            raise ValueError("ablation_delta_min must be > 0")

        # ------------------------------
        # Logging
        # ------------------------------
        if self.verbose < 0:
            raise ValueError("verbose must be >= 0")

        if self.progress_every < 1:
            raise ValueError("progress_every must be >= 1")

    @staticmethod
    def _validate_quantiles_for_bins(
        n_bins: int,
        quantiles: Tuple[float, ...],
        label: str,
    ) -> None:
        if n_bins < 2:
            raise ValueError(f"{label}: n_bins must be >= 2")

        expected_len = n_bins - 1

        if len(quantiles) != expected_len:
            raise ValueError(
                f"{label}: for n_bins={n_bins}, expected {expected_len} quantiles, "
                f"got {len(quantiles)}"
            )

        if any(q <= 0 or q >= 1 for q in quantiles):
            raise ValueError(f"{label}: quantiles must be between 0 and 1")

        if tuple(sorted(quantiles)) != quantiles:
            raise ValueError(f"{label}: quantiles must be strictly ascending")


# =========================================================
# Convenience loader
# =========================================================

def load_phase3_1_config() -> Phase3_1Config:
    cfg = Phase3_1Config()
    cfg.ensure_paths()
    cfg.validate()
    return cfg