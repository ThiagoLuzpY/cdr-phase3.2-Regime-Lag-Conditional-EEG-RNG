from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Phase2ProteinConfig:
    """
    Configuration for Phase II.4 (protein dynamics validation)
    """

    # =========================
    # Dataset
    # =========================

    dataset_root: Path = Path("data/raw/protein")

    pdb_file: str = "alanine-dipeptide-nowater.pdb"

    xtc_files: Tuple[str, ...] = (
        "alanine-dipeptide-0-250ns-nowater.xtc",
        "alanine-dipeptide-1-250ns-nowater.xtc",
        "alanine-dipeptide-2-250ns-nowater.xtc",
    )

    # Optional loading / thinning
    frame_stride: int = 10
    max_frames_per_traj: Optional[int] = None

    # =========================
    # Variables / State space
    # =========================

    state_columns: Tuple[str, ...] = ("phi", "psi")

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

    # F3 holdout strategy
    f3_holdout_mode: str = "leave_one_trajectory_out"
    f3_holdout_traj_index: int = -1

    # Gates / thresholds
    inj_eps_true: float = 0.25
    gate_tol_abs: float = 0.05
    control_tol: float = 0.05
    control_fraction: float = 2 / 3
    n_controls: int = 10
    sensitivity_delta: float = 0.12
    holdout_delta: float = 0.10

    # =========================
    # Output
    # =========================

    results_dir: Path = Path("results/phase2_protein")

    # =========================
    # Logging
    # =========================

    verbose: int = 1

    def ensure_paths(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        pdb_path = self.dataset_root / self.pdb_file
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        if len(self.xtc_files) < 1:
            raise ValueError("xtc_files must contain at least one trajectory file")

        for fname in self.xtc_files:
            xtc_path = self.dataset_root / fname
            if not xtc_path.exists():
                raise FileNotFoundError(f"XTC file not found: {xtc_path}")

        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

        if self.max_frames_per_traj is not None and self.max_frames_per_traj < 2:
            raise ValueError("max_frames_per_traj must be >= 2 or None")

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

        if self.f3_holdout_mode not in {"leave_one_trajectory_out"}:
            raise ValueError("f3_holdout_mode must be 'leave_one_trajectory_out'")

        if not isinstance(self.f3_holdout_traj_index, int):
            raise ValueError("f3_holdout_traj_index must be an integer")

        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")

        if self.min_prob <= 0:
            raise ValueError("min_prob must be > 0")

        if self.n_controls < 1:
            raise ValueError("n_controls must be >= 1")


def load_phase2_protein_config() -> Phase2ProteinConfig:
    cfg = Phase2ProteinConfig()
    cfg.ensure_paths()
    cfg.validate()
    return cfg