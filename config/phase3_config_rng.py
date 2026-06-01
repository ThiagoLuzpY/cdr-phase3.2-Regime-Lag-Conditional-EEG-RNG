from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class Phase3RNGConfig:
    """
    Configuration for Phase III RNG-only baseline validation.
    """

    # =========================
    # Dataset
    # =========================

    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    sequence_length: int = 1024

    # =========================
    # RNG representation / state construction
    # =========================

    # Convert uint8 values into bits (0/1) before building windows.
    use_bits: bool = True

    # Use sliding windows over the RNG sequence:
    # state_t   = (x_t,   x_t+1)
    # state_t+1 = (x_t+1, x_t+2)
    state_window: int = 2

    # =========================
    # Discretization
    # =========================

    # For bits, 2 bins is the natural choice.
    n_bins: int = 2
    quantiles: Tuple[float, ...] = (0.5,)
    strategy: str = "quantile"

    # Alternative quantiles for F5
    sensitivity_quantiles: Tuple[float, ...] = (0.45,)

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
    f3_holdout_mode: str = "chronological"

    # RNG control settings
    n_controls: int = 12
    rng_control_block_size: int = 64

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

    results_dir: Path = Path("results/phase3_rng")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.rng_file.exists():
            raise FileNotFoundError(f"RNG file not found: {self.rng_file}")

        if self.sequence_length < 10:
            raise ValueError("sequence_length must be >= 10")

        if self.state_window < 2:
            raise ValueError("state_window must be >= 2")

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

        if self.n_bins == 2:
            if len(self.sensitivity_quantiles) != 1:
                raise ValueError("for n_bins=2, sensitivity_quantiles must have exactly 1 value")
            q = self.sensitivity_quantiles[0]
            if not (0.0 < q < 1.0):
                raise ValueError("sensitivity_quantiles must satisfy 0 < q < 1")
        elif self.n_bins == 3:
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

        if self.f3_holdout_mode not in {"chronological", "interleaved"}:
            raise ValueError("f3_holdout_mode must be 'chronological' or 'interleaved'")

        if self.rng_control_block_size < 2:
            raise ValueError("rng_control_block_size must be >= 2")

        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")


def load_phase3_rng_config() -> Phase3RNGConfig:
    cfg = Phase3RNGConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg