"""
controls_phase2_ecology.py
--------------------------

Adversarial controls for Phase II.3 — Ecological dynamics.

FINAL VERSION (aligned with loader V3):

- Only 2 variables (hare, lynx)
- No temporal proxy (year_norm removed)
- No exogenous variables (SOI removed)

Design goals:
- destroy ecological causality
- avoid artificial structure
- maintain statistical sanity
"""

import numpy as np


# ---------------------------------------------------------
# CONTROL 1 — SHUFFLE TIME (FULL DESTRUCTION)
# ---------------------------------------------------------

def shuffle_time(X, rng):
    """
    Completely destroys temporal structure.
    """

    idx = rng.permutation(len(X))
    return X[idx]


# ---------------------------------------------------------
# CONTROL 2 — BLOCK SHUFFLE (LOCAL STRUCTURE ONLY)
# ---------------------------------------------------------

def block_shuffle(X, rng, block_size=5):
    """
    Preserves short-term local patterns but destroys global dynamics.
    """

    n = len(X)

    if block_size >= n:
        return shuffle_time(X, rng)

    blocks = [
        X[i:i + block_size]
        for i in range(0, n, block_size)
    ]

    rng.shuffle(blocks)

    return np.vstack(blocks)


# ---------------------------------------------------------
# CONTROL 3 — SPECIES SWAP
# ---------------------------------------------------------

def species_swap(X, rng):
    """
    Breaks predator-prey causality.

    Assumes:
    col 0 = hare_log_return
    col 1 = lynx_log_return
    """

    if X.shape[1] < 2:
        return X

    X_swapped = X.copy()

    X_swapped[:, 0], X_swapped[:, 1] = X[:, 1], X[:, 0]

    return X_swapped


# ---------------------------------------------------------
# CONTROL 4 — TRANSITION RANDOMIZATION
# ---------------------------------------------------------

def transition_randomization(X, rng):
    """
    Preserves marginal distribution but destroys temporal dependence.
    """

    n = len(X)
    idx = rng.integers(0, n, size=n)

    return X[idx]


# ---------------------------------------------------------
# CONTROL REGISTRY
# ---------------------------------------------------------

CONTROL_REGISTRY = {
    "shuffle_time": shuffle_time,
    "block_shuffle": block_shuffle,
    "species_swap": species_swap,
    "transition_randomization": transition_randomization,
}