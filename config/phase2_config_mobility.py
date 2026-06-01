from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Phase2MobilityConfig:
    """
    Configuration for Phase II.2 (human mobility validation)
    """

    # =========================
    # Dataset
    # =========================

    dataset_root: Path = Path(
        "data/raw/geolife/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data"
    )

    max_users: Optional[int] = 3
    min_points_per_traj: int = 50

    # Optional resampling / thinning
    sampling_seconds: Optional[int] = None

    # =========================
    # Variables / State space
    # =========================

    state_columns: Tuple[str, ...] = ("speed", "accel", "turn", "stop")

    # =========================
    # Discretization
    # =========================

    n_bins: int = 3
    quantiles: Tuple[float, ...] = (0.33, 0.66)
    strategy: str = "quantile"

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

    # Gates / thresholds
    inj_eps_true: float = 0.30
    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    control_fraction: float = 2 / 3
    n_controls: int = 10
    sensitivity_delta: float = 0.12
    holdout_delta: float = 0.10

    # =========================
    # Output
    # =========================

    results_dir: Path = Path("results/phase2_mobility")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        if self.max_users is not None and self.max_users < 1:
            raise ValueError("max_users must be >= 1 or None")

        if self.min_points_per_traj < 2:
            raise ValueError("min_points_per_traj must be >= 2")

        if self.sampling_seconds is not None and self.sampling_seconds < 1:
            raise ValueError("sampling_seconds must be >= 1 or None")

        if len(self.state_columns) < 2:
            raise ValueError("state_columns must contain at least 2 variables")

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

        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")


def load_phase2_mobility_config() -> Phase2MobilityConfig:
    cfg = Phase2MobilityConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg