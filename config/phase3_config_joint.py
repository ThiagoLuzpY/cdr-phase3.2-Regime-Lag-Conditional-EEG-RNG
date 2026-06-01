from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Phase3JointConfig:
    """
    Configuration for Phase III joint validation (EEG + RNG).
    """

    # =========================
    # EEG dataset
    # =========================

    eeg_dataset_root: Path = Path("data/raw/eeg")
    eeg_psg_file: str = "SC4001E0-PSG.edf"
    eeg_hypnogram_file: str = "SC4001EC-Hypnogram.edf"

    # =========================
    # RNG dataset
    # =========================

    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    rng_sequence_length: int = 1024
    rng_use_bits: bool = True
    rng_state_window: int = 2

    # =========================
    # EEG preprocessing
    # =========================

    eeg_channel_name: str = "EEG Fpz-Cz"
    eeg_epoch_seconds: int = 30
    eeg_max_epochs: Optional[int] = None

    eeg_allowed_stages: Tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    )

    # =========================
    # Joint state construction
    # =========================

    # Reduced joint state:
    # EEG side -> only delta_power
    eeg_state_columns: Tuple[str, ...] = (
        "delta_power",
    )

    # RNG side -> only x0
    rng_state_columns: Tuple[str, ...] = (
        "x0",
    )

    # Keep disabled for first compact joint test
    include_stage_code_in_joint_state: bool = False

    # =========================
    # EEG spectral feature extraction
    # =========================

    eeg_delta_band: Tuple[float, float] = (0.5, 4.0)
    eeg_theta_band: Tuple[float, float] = (4.0, 8.0)
    eeg_alpha_band: Tuple[float, float] = (8.0, 12.0)
    eeg_beta_band: Tuple[float, float] = (12.0, 30.0)

    eeg_log_bandpower: bool = True

    # =========================
    # Discretization
    # =========================

    # EEG remains discretized in 4 bins
    eeg_n_bins: int = 5
    eeg_quantiles: Tuple[float, ...] = (0.25, 0.50, 0.75)
    eeg_sensitivity_quantiles: Tuple[float, ...] = (0.30, 0.70)

    # RNG bits remain binary
    rng_n_bins: int = 2

    strategy: str = "quantile"

    # =========================
    # Temporal alignment
    # =========================

    alignment_mode: str = "truncate_to_shortest"

    lag: int = 1

    # =========================
    # Kernel parameters
    # =========================

    dirichlet_alpha: float = 0.01
    min_prob: float = 1e-12
    eps_grid: Tuple[float, ...] = tuple(i * 0.01 for i in range(81))

    # =========================
    # Validation controls
    # =========================

    random_seed: int = 42
    train_ratio: float = 0.75

    f3_holdout_mode: str = "block_interleaved"
    f3_block_size: int = 20

    # Joint controls
    n_controls: int = 12
    joint_control_block_size: int = 12
    joint_control_shuffle_rng_only: bool = True
    joint_control_shuffle_eeg_only: bool = True

    # Gates / thresholds
    inj_eps_true: float = 0.05
    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    control_fraction: float = 0.75
    sensitivity_delta: float = 0.12
    holdout_delta: float = 0.10

    # =========================
    # Output
    # =========================

    results_dir: Path = Path("results/phase3_joint")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        # -------------------------
        # EEG files
        # -------------------------
        if not self.eeg_dataset_root.exists():
            raise FileNotFoundError(f"EEG dataset root not found: {self.eeg_dataset_root}")

        eeg_psg_path = self.eeg_dataset_root / self.eeg_psg_file
        eeg_hyp_path = self.eeg_dataset_root / self.eeg_hypnogram_file

        if not eeg_psg_path.exists():
            raise FileNotFoundError(f"EEG PSG file not found: {eeg_psg_path}")

        if not eeg_hyp_path.exists():
            raise FileNotFoundError(f"EEG hypnogram file not found: {eeg_hyp_path}")

        # -------------------------
        # RNG file
        # -------------------------
        if not self.rng_file.exists():
            raise FileNotFoundError(f"RNG file not found: {self.rng_file}")

        # -------------------------
        # EEG settings
        # -------------------------
        if self.eeg_epoch_seconds < 1:
            raise ValueError("eeg_epoch_seconds must be >= 1")

        if self.eeg_max_epochs is not None and self.eeg_max_epochs < 2:
            raise ValueError("eeg_max_epochs must be >= 2 or None")

        if len(self.eeg_allowed_stages) < 1:
            raise ValueError("eeg_allowed_stages must contain at least one stage")

        # -------------------------
        # RNG settings
        # -------------------------
        if self.rng_sequence_length < 10:
            raise ValueError("rng_sequence_length must be >= 10")

        if self.rng_state_window < 2:
            raise ValueError("rng_state_window must be >= 2")

        # -------------------------
        # State columns
        # -------------------------
        if len(self.eeg_state_columns) < 1:
            raise ValueError("eeg_state_columns must contain at least 1 variable")

        if len(self.rng_state_columns) < 1:
            raise ValueError("rng_state_columns must contain at least 1 variable")

        allowed_eeg_state_columns = {
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "stage_code",
        }

        for col in self.eeg_state_columns:
            if col not in allowed_eeg_state_columns:
                raise ValueError(f"Unsupported EEG joint state column: {col}")

        for col in self.rng_state_columns:
            if not col.startswith("x"):
                raise ValueError(f"Unsupported RNG joint state column: {col}")

        # -------------------------
        # Discretization
        # -------------------------
        if self.eeg_n_bins < 2:
            raise ValueError("eeg_n_bins must be >= 2")

        if self.rng_n_bins != 2:
            raise ValueError("rng_n_bins must be 2 for binary RNG bits")

        if not isinstance(self.eeg_quantiles, tuple):
            raise ValueError("eeg_quantiles must be a tuple")

        if any(q <= 0 or q >= 1 for q in self.eeg_quantiles):
            raise ValueError("eeg_quantiles must be between 0 and 1")

        if self.eeg_n_bins == 2:
            if len(self.eeg_quantiles) != 1:
                raise ValueError("for eeg_n_bins=2, eeg_quantiles must have exactly 1 value")
        elif self.eeg_n_bins == 3:
            if len(self.eeg_quantiles) != 2:
                raise ValueError("for eeg_n_bins=3, eeg_quantiles must have exactly 2 values")
            q_lo, q_hi = self.eeg_quantiles
            if not (0.0 < q_lo < q_hi < 1.0):
                raise ValueError("eeg_quantiles must satisfy 0 < q_lo < q_hi < 1")
        else:
            if len(self.eeg_quantiles) < 1:
                raise ValueError("eeg_quantiles must contain at least 1 value")

        if self.eeg_n_bins == 3:
            if len(self.eeg_sensitivity_quantiles) != 2:
                raise ValueError("for eeg_n_bins=3, eeg_sensitivity_quantiles must have exactly 2 values")
            q_lo, q_hi = self.eeg_sensitivity_quantiles
            if not (0.0 < q_lo < q_hi < 1.0):
                raise ValueError("eeg_sensitivity_quantiles must satisfy 0 < q_lo < q_hi < 1")

        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        if self.alignment_mode not in {"truncate_to_shortest"}:
            raise ValueError("alignment_mode must currently be 'truncate_to_shortest'")

        if self.lag < 1:
            raise ValueError("lag must be >= 1")

        # -------------------------
        # Split / controls
        # -------------------------
        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        if self.f3_holdout_mode not in {"block_interleaved", "interleaved", "chronological"}:
            raise ValueError(
                "f3_holdout_mode must be 'block_interleaved', 'interleaved', or 'chronological'"
            )

        if self.f3_block_size < 2:
            raise ValueError("f3_block_size must be >= 2")

        if self.joint_control_block_size < 2:
            raise ValueError("joint_control_block_size must be >= 2")

        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")

        # -------------------------
        # Kernel
        # -------------------------
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        # -------------------------
        # Bands
        # -------------------------
        for band_name, band in {
            "eeg_delta_band": self.eeg_delta_band,
            "eeg_theta_band": self.eeg_theta_band,
            "eeg_alpha_band": self.eeg_alpha_band,
            "eeg_beta_band": self.eeg_beta_band,
        }.items():
            if len(band) != 2:
                raise ValueError(f"{band_name} must contain exactly 2 values")
            lo, hi = band
            if lo < 0 or hi <= lo:
                raise ValueError(f"{band_name} must satisfy 0 <= low < high")


def load_phase3_joint_config() -> Phase3JointConfig:
    cfg = Phase3JointConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg