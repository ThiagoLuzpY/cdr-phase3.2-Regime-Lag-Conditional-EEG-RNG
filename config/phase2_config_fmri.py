from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Phase2FMRIConfig:
    """
    Configuration for Phase II.1B (fMRI validation)
    """

    # =========================
    # Dataset
    # =========================

    dataset_root: Path = Path("data/raw/fmri/ds002938")
    subject: str = "sub-01"
    task: str = "effort"

    # =========================
    # Atlas configuration
    # =========================

    atlas_name: str = "harvard_oxford"
    atlas_cache_dir: Optional[Path] = None

    # =========================
    # fMRI preprocessing
    # =========================

    default_tr: float = 2.0
    standardize: bool = True
    detrend: bool = True
    smoothing_fwhm: Optional[float] = None
    low_pass: Optional[float] = None
    high_pass: Optional[float] = None

    # =========================
    # Discretization
    # =========================

    n_bins: int = 2
    quantiles: Tuple[float, float] = (0.5,)
    strategy: str = "quantile"

    # =========================
    # Temporal embedding
    # =========================

    lag: int = 1

    # =========================
    # Kernel parameters
    # =========================

    dirichlet_alpha: float = 0.1  # Smoothing for empirical kernel P0
    min_prob: float = 1e-12  # Minimum probability for log-likelihood
    eps_grid: Tuple[float, ...] = tuple([i * 0.01 for i in range(0, 81)])  # 0.00 to 0.80

    # =========================
    # Validation controls
    # =========================

    random_seed: int = 42
    train_ratio: float = 0.75

    # Gates / thresholds
    inj_eps_true: float = 0.05
    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    control_fraction: float = 2 / 3
    n_controls: int = 20
    sensitivity_delta: float = 0.12
    holdout_delta: float = 0.10

    # =========================
    # Output
    # =========================

    results_dir: Path = Path("results/phase2_fmri")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

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

        if self.lag < 1:
            raise ValueError("lag must be >= 1")

        if self.strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be 'quantile' or 'uniform'")

        if not (0.0 < self.train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")


def load_phase2_fmri_config() -> Phase2FMRIConfig:
    cfg = Phase2FMRIConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg