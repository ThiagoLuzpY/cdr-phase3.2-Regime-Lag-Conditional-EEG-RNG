"""Global configuration for CDR Phase I validation (toy-model).

This file centralizes pre-registered defaults for reproducible runs.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Phase1Config:
    """Pre-registered defaults for Phase I toy-model runs."""

    # Toy-model parameters (can be overridden in experiments)
    J_default: float = 0.5
    h_default: float = 0.1
    epsilon_h1_default: float = 0.3

    # Trajectory settings
    n_steps_default: int = 1000

    # Replication settings
    n_reps_default: int = 20
    h0_seed_start: int = 42
    h1_seed_start: int = 100
    controls_seed: int = 99999

    # State space (binary 2-component)
    n_components: int = 2
    binary_states: Tuple[int, int] = (0, 1)