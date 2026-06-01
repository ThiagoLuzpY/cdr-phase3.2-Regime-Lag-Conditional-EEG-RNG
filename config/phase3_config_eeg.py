from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Phase3EEGConfig:
    """
    Configuration for Phase III EEG-only validation.
    """

    # =========================
    # Dataset
    # =========================

    dataset_root: Path = Path("data/raw/eeg")

    psg_file: str = "SC4001E0-PSG.edf"
    hypnogram_file: str = "SC4001EC-Hypnogram.edf"

    # =========================
    # EEG preprocessing
    # =========================

    channel_name: str = "EEG Fpz-Cz"

    epoch_seconds: int = 30
    max_epochs: Optional[int] = None

    # Sleep stages to include
    allowed_stages: Tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    )

    # =========================
    # Variables / State space
    # =========================

    state_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
    )

    # Keep stage information available in the loader/results,
    # but do not force it into the core state for now.
    include_stage_code_in_state: bool = False

    # =========================
    # Spectral feature extraction
    # =========================

    delta_band: Tuple[float, float] = (0.5, 4.0)
    theta_band: Tuple[float, float] = (4.0, 8.0)
    alpha_band: Tuple[float, float] = (8.0, 12.0)
    beta_band: Tuple[float, float] = (12.0, 30.0)

    # Optional log scaling of bandpowers
    log_bandpower: bool = True

    # =========================
    # Discretization
    # =========================

    n_bins: int = 3
    quantiles: Tuple[float, ...] = (0.33, 0.66)
    strategy: str = "quantile"

    # Alternative quantiles for F5
    sensitivity_quantiles: Tuple[float, ...] = (0.30, 0.70)

    # =========================
    # Temporal embedding
    # =========================

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

    # F3 holdout strategy
    f3_holdout_mode: str = "block_interleaved"
    f3_block_size: int = 20

    # EEG control settings
    n_controls: int = 12
    eeg_control_block_size: int = 12
    eeg_control_stage_shuffle: bool = True

    # Gates / thresholds
    inj_eps_true: float = 0.05
    gate_tol_abs: float = 0.08
    control_tol: float = 0.05
    control_fraction: float = 0.75
    sensitivity_delta: float = 0.12
    holdout_delta: float = 0.10

    # =========================
    # Output
    # =========================

    results_dir: Path = Path("results/phase3_eeg")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        psg_path = self.dataset_root / self.psg_file
        hyp_path = self.dataset_root / self.hypnogram_file

        if not psg_path.exists():
            raise FileNotFoundError(f"PSG file not found: {psg_path}")

        if not hyp_path.exists():
            raise FileNotFoundError(f"Hypnogram file not found: {hyp_path}")

        if self.epoch_seconds < 1:
            raise ValueError("epoch_seconds must be >= 1")

        if self.max_epochs is not None and self.max_epochs < 2:
            raise ValueError("max_epochs must be >= 2 or None")

        if len(self.allowed_stages) < 1:
            raise ValueError("allowed_stages must contain at least one stage")

        if len(self.state_columns) < 2:
            raise ValueError("state_columns must contain at least 2 variables")

        allowed_state_columns = {
            "delta_power",
            "theta_power",
            "alpha_power",
            "beta_power",
            "stage_code",
        }

        for col in self.state_columns:
            if col not in allowed_state_columns:
                raise ValueError(f"Unsupported EEG state column: {col}")

        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")

        if not isinstance(self.quantiles, tuple):
            raise ValueError("quantiles must be a tuple")

        if any(q <= 0 or q >= 1 for q in self.quantiles):
            raise ValueError("quantiles must be between 0 and 1")

        if self.n_bins == 2:
            if len(self.quantiles) != 1:
                raise ValueError("for n_bins=2, quantiles must have exactly 1 value, e.g. (0.5,)")
        elif self.n_bins == 3:
            if len(self.quantiles) != 2:
                raise ValueError("for n_bins=3, quantiles must have exactly 2 values, e.g. (0.33, 0.66)")
            q_lo, q_hi = self.quantiles
            if not (0.0 < q_lo < q_hi < 1.0):
                raise ValueError("quantiles must satisfy 0 < q_lo < q_hi < 1")
        else:
            if len(self.quantiles) < 1:
                raise ValueError("quantiles must contain at least 1 value")

        if self.n_bins == 3:
            if len(self.sensitivity_quantiles) != 2:
                raise ValueError("for n_bins=3, sensitivity_quantiles must have exactly 2 values")
            q_lo, q_hi = self.sensitivity_quantiles
            if not (0.0 < q_lo < q_hi < 1.0):
                raise ValueError("sensitivity_quantiles must satisfy 0 < q_lo < q_hi < 1")

        if self.lag < 1:
            raise ValueError("lag must be >= 1")

        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        if self.f3_holdout_mode not in {"interleaved", "chronological", "block_interleaved"}:
            raise ValueError(
                "f3_holdout_mode must be 'interleaved', 'chronological', or 'block_interleaved'"
            )

        if self.f3_block_size < 2:
            raise ValueError("f3_block_size must be >= 2")

        if self.eeg_control_block_size < 2:
            raise ValueError("eeg_control_block_size must be >= 2")

        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")

        for band_name, band in {
            "delta_band": self.delta_band,
            "theta_band": self.theta_band,
            "alpha_band": self.alpha_band,
            "beta_band": self.beta_band,
        }.items():
            if len(band) != 2:
                raise ValueError(f"{band_name} must contain exactly 2 values")
            lo, hi = band
            if lo < 0 or hi <= lo:
                raise ValueError(f"{band_name} must satisfy 0 <= low < high")


def load_phase3_eeg_config() -> Phase3EEGConfig:
    cfg = Phase3EEGConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg