"""Conditional Ising-like kernel for CDR Phase I toy-model validation.

This module provides an exact (enumerated) 2-component binary conditional
transition kernel P0(I' | I), the CDR kernel-local statistic Δχ(I'; I), and
the reweighted kernel Pε(I' | I).

External state representation:
    I = (i1, i2), with ik in {0, 1}

Internal spin representation (for energy-like scoring):
    s = (s1, s2), with sk in {-1, +1}, via sk = 2*ik - 1

Design choice:
    We use a *conditional coupled* kernel so that Δχ is generally nonzero.
    This is essential for CDR Phase I identifiability and detectability tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


State = Tuple[int, int]
SpinState = Tuple[int, int]


@dataclass(frozen=True)
class IsingKernel:
    """Exact 2-component conditional Ising-like kernel for CDR Phase I.

    The baseline kernel P0(I'|I) is defined by an energy-like score over next
    state I' conditioned on current state I:

        score(I', I) = J * s'_1 s'_2 + h * (s'_1 s_1 + s'_2 s_2)

    where s and s' are spin encodings of I and I' in {-1,+1}^2.

    Notes:
        - `J` controls cross-component coupling in the *next* state.
        - `h` acts as a temporal alignment/persistence field (toy-model choice).
        - This produces a non-factorized conditional kernel in general,
          enabling nontrivial Δχ(I'; I).

    Attributes:
        states: Fixed enumeration of binary states in {0,1}^2.
    """

    states: Tuple[State, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

    # ---------------------------------------------------------------------
    # Basic utilities
    # ---------------------------------------------------------------------
    def validate_state(self, state: Sequence[int]) -> State:
        """Validate and normalize a state into a canonical 2-tuple of ints.

        Args:
            state: Sequence of length 2 with binary entries.

        Returns:
            Canonical state tuple (i1, i2).

        Raises:
            ValueError: If state is invalid.
        """
        if len(state) != 2:
            raise ValueError(f"State must have length 2, got length={len(state)}")
        try:
            s0 = int(state[0])
            s1 = int(state[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"State entries must be integers: {state}") from exc
        if s0 not in (0, 1) or s1 not in (0, 1):
            raise ValueError(f"State entries must be in {{0,1}}, got {state}")
        return (s0, s1)

    def state_to_spin(self, state: Sequence[int]) -> SpinState:
        """Convert binary state {0,1}^2 to spin state {-1,+1}^2.

        Args:
            state: Binary state.

        Returns:
            Spin state tuple.
        """
        i1, i2 = self.validate_state(state)
        return (2 * i1 - 1, 2 * i2 - 1)

    def spin_to_state(self, spin_state: Sequence[int]) -> State:
        """Convert spin state {-1,+1}^2 to binary state {0,1}^2.

        Args:
            spin_state: Spin state.

        Returns:
            Binary state tuple.

        Raises:
            ValueError: If input is not a valid spin state.
        """
        if len(spin_state) != 2:
            raise ValueError("Spin state must have length 2")
        a, b = int(spin_state[0]), int(spin_state[1])
        if a not in (-1, 1) or b not in (-1, 1):
            raise ValueError(f"Spin entries must be in {{-1,+1}}, got {spin_state}")
        return ((a + 1) // 2, (b + 1) // 2)

    def state_index(self, state: Sequence[int]) -> int:
        """Return the enumeration index of a binary state."""
        st = self.validate_state(state)
        try:
            return self.states.index(st)
        except ValueError as exc:
            raise ValueError(f"State {st} not found in enumeration {self.states}") from exc

    def index_state(self, idx: int) -> State:
        """Return state by enumeration index."""
        if idx < 0 or idx >= len(self.states):
            raise ValueError(f"Index out of range: {idx}")
        return self.states[idx]

    # ---------------------------------------------------------------------
    # Baseline kernel P0
    # ---------------------------------------------------------------------
    def _baseline_score(self, state_next: Sequence[int], state_curr: Sequence[int], J: float, h: float) -> float:
        """Energy-like score used to define the baseline conditional kernel P0.

        Args:
            state_next: Next state I'.
            state_curr: Current state I.
            J: Cross-component coupling parameter.
            h: Temporal alignment/persistence field parameter.

        Returns:
            Scalar score.
        """
        s_next = self.state_to_spin(state_next)
        s_curr = self.state_to_spin(state_curr)

        # score = J * s'_1 s'_2 + h * (s'_1 s_1 + s'_2 s_2)
        return float(
            J * s_next[0] * s_next[1]
            + h * (s_next[0] * s_curr[0] + s_next[1] * s_curr[1])
        )

    def p0_distribution(self, state_curr: Sequence[int], J: float, h: float) -> np.ndarray:
        """Compute the exact baseline conditional distribution P0(. | I).

        Args:
            state_curr: Current state I.
            J: Coupling parameter.
            h: Temporal alignment field.

        Returns:
            Array of shape (4,) with probabilities in the order `self.states`.
        """
        curr = self.validate_state(state_curr)
        scores = np.array(
            [self._baseline_score(s_next, curr, J, h) for s_next in self.states],
            dtype=float,
        )

        # Stable softmax
        max_score = float(np.max(scores))
        weights = np.exp(scores - max_score)
        probs = weights / np.sum(weights)

        return probs

    def compute_P0(self, I_prime: Sequence[int], I: Sequence[int], J: float, h: float) -> float:
        """Compute baseline conditional probability P0(I' | I).

        Args:
            I_prime: Next state I'.
            I: Current state I.
            J: Coupling parameter.
            h: Temporal alignment field.

        Returns:
            Probability P0(I' | I).
        """
        probs = self.p0_distribution(I, J, h)
        idx = self.state_index(I_prime)
        return float(probs[idx])

    def p0_marginals_next_components(self, state_curr: Sequence[int], J: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute conditional marginals of next-state components under P0.

        Returns:
            Tuple `(marg1, marg2)` where:
                marg1[a] = P0(I'_1 = a | I), a in {0,1}
                marg2[b] = P0(I'_2 = b | I), b in {0,1}
        """
        probs = self.p0_distribution(state_curr, J, h)

        marg1 = np.zeros(2, dtype=float)
        marg2 = np.zeros(2, dtype=float)

        for idx, s_next in enumerate(self.states):
            i1p, i2p = s_next
            p = probs[idx]
            marg1[i1p] += p
            marg2[i2p] += p

        return marg1, marg2

    # ---------------------------------------------------------------------
    # CDR kernel-local statistic Δχ
    # ---------------------------------------------------------------------
    def compute_delta_chi(self, I_prime: Sequence[int], I: Sequence[int], J: float, h: float) -> float:
        """Compute the CDR kernel-local integration gain Δχ(I'; I).

        For n=2 components, the canonical statistic is:

            Δχ(I'; I) = log [ P0(I'|I) / (P0(I'_1|I) P0(I'_2|I)) ]

        Args:
            I_prime: Next state I'.
            I: Current state I.
            J: Coupling parameter of the baseline kernel.
            h: Temporal alignment field of the baseline kernel.

        Returns:
            Scalar Δχ(I'; I).

        Raises:
            FloatingPointError: If numerical underflow produces nonpositive terms.
        """
        I_prime_valid = self.validate_state(I_prime)
        I_valid = self.validate_state(I)

        p_joint = self.compute_P0(I_prime_valid, I_valid, J, h)
        marg1, marg2 = self.p0_marginals_next_components(I_valid, J, h)
        p_prod = float(marg1[I_prime_valid[0]] * marg2[I_prime_valid[1]])

        if p_joint <= 0.0 or p_prod <= 0.0:
            raise FloatingPointError(
                "Encountered nonpositive probability in Δχ computation. "
                f"p_joint={p_joint}, p_prod={p_prod}"
            )

        return float(np.log(p_joint / p_prod))

    def delta_chi_table(self, I: Sequence[int], J: float, h: float) -> Dict[State, float]:
        """Compute Δχ(s'; s) for all possible next states given current state s."""
        curr = self.validate_state(I)
        return {s_next: self.compute_delta_chi(s_next, curr, J, h) for s_next in self.states}

    # ---------------------------------------------------------------------
    # Reweighted kernel Pε
    # ---------------------------------------------------------------------
    def peps_distribution(self, state_curr: Sequence[int], J: float, h: float, epsilon: float) -> np.ndarray:
        """Compute the exact reweighted conditional distribution Pε(. | I).

        CDR reweighting:
            Pε(I'|I) ∝ P0(I'|I) * exp(epsilon * Δχ(I';I))

        Args:
            state_curr: Current state I.
            J: Baseline coupling parameter.
            h: Baseline temporal alignment field.
            epsilon: Selection-bias parameter.

        Returns:
            Array of shape (4,) with probabilities in the order `self.states`.
        """
        curr = self.validate_state(state_curr)

        p0 = self.p0_distribution(curr, J, h)
        delta_vals = np.array(
            [self.compute_delta_chi(s_next, curr, J, h) for s_next in self.states],
            dtype=float,
        )

        # Stable normalization in log-space style
        logw = np.log(p0) + float(epsilon) * delta_vals
        max_logw = float(np.max(logw))
        weights = np.exp(logw - max_logw)
        probs = weights / np.sum(weights)

        return probs

    def compute_Peps(self, I_prime: Sequence[int], I: Sequence[int], J: float, h: float, epsilon: float) -> float:
        """Compute reweighted probability Pε(I' | I)."""
        probs = self.peps_distribution(I, J, h, epsilon)
        idx = self.state_index(I_prime)
        return float(probs[idx])

    def partition_function(self, I: Sequence[int], J: float, h: float, epsilon: float) -> float:
        """Compute the local partition function Z(I) for the reweighted kernel.

        Z(I) = sum_{I'} P0(I'|I) * exp(epsilon * Δχ(I';I))
        """
        curr = self.validate_state(I)
        p0 = self.p0_distribution(curr, J, h)
        delta_vals = np.array(
            [self.compute_delta_chi(s_next, curr, J, h) for s_next in self.states],
            dtype=float,
        )
        return float(np.sum(p0 * np.exp(float(epsilon) * delta_vals)))

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------
    def sample_next_state(
        self,
        state_curr: Sequence[int],
        J: float,
        h: float,
        epsilon: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> State:
        """Sample one next state from Pε(. | I).

        Args:
            state_curr: Current state I.
            J: Baseline coupling parameter.
            h: Baseline temporal alignment field.
            epsilon: Reweighting parameter.
            rng: Optional NumPy Generator. If None, a fresh generator is used.

        Returns:
            Sampled next state I'.
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.peps_distribution(state_curr, J, h, epsilon)
        idx = int(rng.choice(len(self.states), p=probs))
        return self.states[idx]

    def sample_trajectory(
        self,
        J: float,
        h: float,
        epsilon: float,
        n_steps: int,
        seed: int,
        initial_state: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Sample a Markov trajectory under Pε with deterministic seeding.

        Args:
            J: Baseline coupling parameter.
            h: Baseline temporal alignment field.
            epsilon: Reweighting parameter.
            n_steps: Number of transitions to simulate.
            seed: RNG seed (critical for replicability).
            initial_state: Optional starting state. If None, sampled uniformly.

        Returns:
            Trajectory array of shape (n_steps + 1, 2), dtype=int.

        Notes:
            - `trajectory[t]` is I_t
            - transitions are from `trajectory[t] -> trajectory[t+1]`
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        rng = np.random.default_rng(seed)
        traj = np.empty((n_steps + 1, 2), dtype=np.int64)

        if initial_state is None:
            # Uniform initial state over {0,1}^2
            start_idx = int(rng.integers(0, len(self.states)))
            traj[0] = self.states[start_idx]
        else:
            traj[0] = self.validate_state(initial_state)

        for t in range(n_steps):
            curr = (int(traj[t, 0]), int(traj[t, 1]))
            nxt = self.sample_next_state(curr, J, h, epsilon=epsilon, rng=rng)
            traj[t + 1] = nxt

        return traj

    # ---------------------------------------------------------------------
    # Trajectory-derived diagnostics (useful for E.6 artifacts)
    # ---------------------------------------------------------------------
    def trajectory_delta_chi_series(self, trajectory: np.ndarray, J: float, h: float) -> np.ndarray:
        """Compute per-step Δχ(I_{t+1}; I_t) along a trajectory.

        Args:
            trajectory: Array of shape (T+1, 2).
            J: Baseline coupling parameter.
            h: Baseline temporal alignment field.

        Returns:
            Array of shape (T,) with per-transition Δχ values.
        """
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                "Trajectory must have shape (T+1, 2). "
                f"Got shape={trajectory.shape}"
            )
        if trajectory.shape[0] < 2:
            raise ValueError("Trajectory must contain at least two states.")

        deltas = np.empty(trajectory.shape[0] - 1, dtype=float)
        for t in range(trajectory.shape[0] - 1):
            I = (int(trajectory[t, 0]), int(trajectory[t, 1]))
            I_prime = (int(trajectory[t + 1, 0]), int(trajectory[t + 1, 1]))
            deltas[t] = self.compute_delta_chi(I_prime, I, J, h)

        return deltas

    def empirical_state_occupancy(self, trajectory: np.ndarray) -> Dict[State, float]:
        """Compute empirical state occupancy \\hat{rho}(s) from a trajectory.

        Args:
            trajectory: Array of shape (T+1, 2).

        Returns:
            Dictionary mapping each state to empirical frequency.
        """
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                "Trajectory must have shape (T+1, 2). "
                f"Got shape={trajectory.shape}"
            )

        counts = {s: 0 for s in self.states}
        for row in trajectory:
            s = self.validate_state((int(row[0]), int(row[1])))
            counts[s] += 1

        total = float(trajectory.shape[0])
        return {s: counts[s] / total for s in self.states}

    def e6_artifact_bundle(
        self,
        J: float,
        h: float,
        epsilon: float,
        n_steps: int,
        seed: int,
        initial_state: Optional[Sequence[int]] = None,
    ) -> Dict[str, object]:
        """Generate a minimal E.6-compliant artifact bundle in-memory.

        This supports the Appendix E.6 requirement to emit:
            - trajectory
            - config snapshot
            - RNG seed manifest
            - per-step Δχ
            - baseline state occupancy \\hat{rho}(s)

        Returns:
            Dictionary with numpy arrays and metadata suitable for serialization.
        """
        traj = self.sample_trajectory(
            J=J,
            h=h,
            epsilon=epsilon,
            n_steps=n_steps,
            seed=seed,
            initial_state=initial_state,
        )
        delta_series = self.trajectory_delta_chi_series(traj, J=J, h=h)
        occupancy = self.empirical_state_occupancy(traj)

        return {
            "trajectory": traj,
            "config_snapshot": {
                "J": float(J),
                "h": float(h),
                "epsilon": float(epsilon),
                "n_steps": int(n_steps),
                "initial_state": None if initial_state is None else tuple(map(int, initial_state)),
            },
            "rng_seed_manifest": {"trajectory_seed": int(seed)},
            "delta_chi_per_step": delta_series,
            "rho_hat": {str(k): float(v) for k, v in occupancy.items()},
        }