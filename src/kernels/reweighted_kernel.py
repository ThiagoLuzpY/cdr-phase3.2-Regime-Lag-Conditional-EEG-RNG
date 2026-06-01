from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.kernels.empirical_kernel import EmpiricalKernel


@dataclass
class ReweightedKernel:
    base: EmpiricalKernel
    epsilon: float
    min_prob: float = 1e-12

    def row_probs(self, curr_state: int) -> np.ndarray:
        """
        Compute Pε(j|i) ∝ P0(j|i) * exp(eps * Δχ(j;i))
        """

        i = int(np.asarray(curr_state).item())

        p0 = self.base.P[i, :].astype(float)

        deltas = np.array(
            [
                self.base.delta_chi(j, i, min_prob=self.min_prob)
                for j in range(self.base.n_states)
            ],
            dtype=float,
        )

        w = p0 * np.exp(self.epsilon * deltas)

        s = float(np.sum(w))

        if not np.isfinite(s) or s <= 0:
            w = p0.copy()
            s = float(np.sum(w))

        w = w / s

        w = np.clip(w, self.min_prob, 1.0)

        w = w / float(np.sum(w))

        return w

    def p_joint(self, nxt_state: int, curr_state: int) -> float:
        """
        Joint transition probability P(j|i)
        """

        i = int(np.asarray(curr_state).item())
        j = int(np.asarray(nxt_state).item())

        row = self.row_probs(i)

        return float(row[j])

    def loglik(self, curr: np.ndarray, nxt: np.ndarray) -> float:
        """
        Log-likelihood of sequence of transitions.
        """

        curr = np.asarray(curr).reshape(-1)
        nxt = np.asarray(nxt).reshape(-1)

        ll = 0.0

        for i, j in zip(curr, nxt):

            i = int(np.asarray(i).item())
            j = int(np.asarray(j).item())

            p = self.p_joint(j, i)

            ll += float(np.log(max(p, self.min_prob)))

        return float(ll)

    def sample_next(self, curr_state: int, rng: np.random.Generator) -> int:
        """
        Sample next state from reweighted transition distribution.
        """

        i = int(np.asarray(curr_state).item())

        probs = self.row_probs(i)

        return int(rng.choice(len(probs), p=probs))