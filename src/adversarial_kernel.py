from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from src.ising_kernel import IsingKernel, State


@dataclass(frozen=True)
class AdversarialIsingKernel(IsingKernel):
    """Adversarial baseline kernel P0' with bounded extra flexibility.

    Adds a cross-lag coupling term:
        g * (s'_1 s_2 + s'_2 s_1)

    Purpose:
        Stress-test degeneracy: can a modestly richer baseline mimic the effect
        attributed to epsilon under CDR?

    This kernel is used only for Phase I+ Gate G6.
    """

    def _baseline_score_adv(
        self,
        state_next: Sequence[int],
        state_curr: Sequence[int],
        J: float,
        h: float,
        g: float,
    ) -> float:
        s_next = self.state_to_spin(state_next)
        s_curr = self.state_to_spin(state_curr)
        # base: J*s1'*s2' + h*(s1'*s1 + s2'*s2)
        base = J * s_next[0] * s_next[1] + h * (s_next[0] * s_curr[0] + s_next[1] * s_curr[1])
        # adversarial enrichment: cross-lag coupling
        enrich = g * (s_next[0] * s_curr[1] + s_next[1] * s_curr[0])
        return float(base + enrich)

    def p0_distribution_adv(self, state_curr: Sequence[int], J: float, h: float, g: float) -> np.ndarray:
        curr = self.validate_state(state_curr)
        scores = np.array(
            [self._baseline_score_adv(s_next, curr, J, h, g) for s_next in self.states],
            dtype=float,
        )
        m = float(np.max(scores))
        w = np.exp(scores - m)
        p = w / np.sum(w)
        return p

    def compute_P0_adv(self, I_prime: Sequence[int], I: Sequence[int], J: float, h: float, g: float) -> float:
        p = self.p0_distribution_adv(I, J=J, h=h, g=g)
        idx = self.state_index(I_prime)
        return float(p[idx])

    def loglik_h0_adv(self, trajectory: np.ndarray, J: float, h: float, g: float) -> float:
        """Log-likelihood under adversarial baseline null: epsilon fixed to 0, baseline has g."""
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(f"trajectory must have shape (T+1,2). got {trajectory.shape}")
        if trajectory.shape[0] < 2:
            raise ValueError("trajectory must contain at least two states.")

        ll = 0.0
        for t in range(trajectory.shape[0] - 1):
            I = (int(trajectory[t, 0]), int(trajectory[t, 1]))
            Ip = (int(trajectory[t + 1, 0]), int(trajectory[t + 1, 1]))
            p = self.compute_P0_adv(Ip, I, J=J, h=h, g=g)
            ll += np.log(max(p, 1e-300))
        return float(ll)