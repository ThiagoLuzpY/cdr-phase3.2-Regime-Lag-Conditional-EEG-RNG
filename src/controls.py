from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.estimators import estimate_epsilon_mle_grid, EpsilonGridFit
from src.ising_kernel import IsingKernel


@dataclass(frozen=True)
class ControlEpsilonResult:
    control_name: str
    seed: int
    eps_hat: float
    fit: Optional[EpsilonGridFit]
    notes: str = ""


def time_shuffle_trajectory(trajectory: np.ndarray, seed: int) -> np.ndarray:
    """Destroy temporal order while preserving the multiset of states."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(trajectory.shape[0])
    return trajectory[idx].copy()


def marginal_preserving_surrogate(trajectory: np.ndarray, seed: int) -> np.ndarray:
    """Preserve per-component marginals exactly, destroy cross-component coupling."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
    rng = np.random.default_rng(seed)
    out = trajectory.copy()
    rng.shuffle(out[:, 0])  # permute component 1 over time
    rng.shuffle(out[:, 1])  # permute component 2 over time
    return out


def autocorr_preserving_surrogate_circular_shift(trajectory: np.ndarray, seed: int) -> np.ndarray:
    """Preserve each component's temporal pattern up to circular shift; disrupt cross-component alignment."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
    rng = np.random.default_rng(seed)
    T = trajectory.shape[0]
    # independent nonzero shifts (when possible)
    shift1 = int(rng.integers(0, T))
    shift2 = int(rng.integers(0, T))
    out = np.empty_like(trajectory)
    out[:, 0] = np.roll(trajectory[:, 0], shift1)
    out[:, 1] = np.roll(trajectory[:, 1], shift2)
    return out


def estimate_epsilon_on_control(
    trajectory: np.ndarray,
    J: float,
    h: float,
    eps_grid: Sequence[float],
    control_name: str,
    seed: int,
    kernel: Optional[IsingKernel] = None,
) -> ControlEpsilonResult:
    """Run epsilon grid-MLE on a control trajectory."""
    fit = estimate_epsilon_mle_grid(trajectory, J=J, h=h, eps_grid=eps_grid, kernel=kernel)
    return ControlEpsilonResult(
        control_name=control_name,
        seed=seed,
        eps_hat=fit.eps_hat_mle,
        fit=fit,
    )


def run_control_suite(
    trajectory: np.ndarray,
    J: float,
    h: float,
    eps_grid: Sequence[float],
    controls_seed: int,
    kernel: Optional[IsingKernel] = None,
) -> List[ControlEpsilonResult]:
    """Apply canonical Phase I controls and estimate epsilon on each."""
    results: List[ControlEpsilonResult] = []

    # Time-shuffle
    traj_ts = time_shuffle_trajectory(trajectory, seed=controls_seed)
    results.append(
        estimate_epsilon_on_control(
            traj_ts, J=J, h=h, eps_grid=eps_grid,
            control_name="time_shuffle", seed=controls_seed, kernel=kernel
        )
    )

    # Marginal-preserving (independent permutations)
    traj_mp = marginal_preserving_surrogate(trajectory, seed=controls_seed + 1)
    results.append(
        estimate_epsilon_on_control(
            traj_mp, J=J, h=h, eps_grid=eps_grid,
            control_name="marginal_preserving", seed=controls_seed + 1, kernel=kernel
        )
    )

    # Autocorr-preserving via circular shifts
    traj_cs = autocorr_preserving_surrogate_circular_shift(trajectory, seed=controls_seed + 2)
    results.append(
        estimate_epsilon_on_control(
            traj_cs, J=J, h=h, eps_grid=eps_grid,
            control_name="circular_shift", seed=controls_seed + 2, kernel=kernel
        )
    )

    return results