from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.build_states import StateEncoding, decode_state


@dataclass
class EmpiricalKernel:
    n_states: int
    enc: StateEncoding
    P: np.ndarray  # shape (n_states, n_states)
    marginals: List[np.ndarray]  # list of (n_states, n_bins) for each component

    @staticmethod
    def from_transitions(
        curr: np.ndarray,
        nxt: np.ndarray,
        n_states: int,
        enc: StateEncoding,
        alpha: float = 1e-2,
    ) -> "EmpiricalKernel":
        counts = np.full((n_states, n_states), float(alpha), dtype=float)
        for i, j in zip(curr, nxt):
            counts[int(i), int(j)] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        P = counts / row_sums

        # precompute marginals over next components
        n_bins = enc.n_bins
        n_components = enc.n_components

        # map next_state -> component values
        comp_vals = np.zeros((n_states, n_components), dtype=int)
        for s in range(n_states):
            comp_vals[s, :] = decode_state(s, enc)

        marginals: List[np.ndarray] = []
        for k in range(n_components):
            M = np.zeros((n_states, n_bins), dtype=float)
            # for each current state, sum prob over next states with given comp value
            for s in range(n_states):
                # vectorized grouping
                for v in range(n_bins):
                    mask = (comp_vals[:, k] == v)
                    M[s, v] = float(np.sum(P[s, mask]))
            marginals.append(M)

        return EmpiricalKernel(n_states=n_states, enc=enc, P=P, marginals=marginals)

    def p_joint(self, nxt_state: int, curr_state: int) -> float:
        return float(self.P[int(curr_state), int(nxt_state)])

    def p_comp(self, comp_idx: int, comp_val: int, curr_state: int) -> float:
        return float(self.marginals[comp_idx][int(curr_state), int(comp_val)])

    def delta_chi(self, nxt_state: int, curr_state: int, min_prob: float = 1e-12) -> float:
        # Δχ = log( P0(next|curr) / Πk P0(next_k|curr) )
        joint = max(self.p_joint(nxt_state, curr_state), min_prob)

        nxt_vec = decode_state(int(nxt_state), self.enc)
        prod = 1.0
        for k in range(self.enc.n_components):
            prod *= max(self.p_comp(k, int(nxt_vec[k]), curr_state), min_prob)

        prod = max(prod, min_prob)
        return float(np.log(joint / prod))