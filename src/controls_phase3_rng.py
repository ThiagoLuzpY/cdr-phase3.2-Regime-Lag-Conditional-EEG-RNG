from __future__ import annotations

import numpy as np

from src.kernels.reweighted_kernel import ReweightedKernel


def _estimate_epsilon_grid_local(
    curr: np.ndarray,
    nxt: np.ndarray,
    base,
    eps_grid: np.ndarray,
    min_prob: float,
    *,
    label: str = "",
    progress_every: int = 20,
):
    """
    Local copy of epsilon-grid estimation to avoid circular imports.
    Compatible with the EmpiricalKernel/ReweightedKernel used elsewhere.
    """
    ll = np.zeros_like(eps_grid, dtype=float)
    n_grid = int(len(eps_grid))

    if label:
        print(f"[Phase3-RNG] Estimating epsilon on {label} (grid={n_grid})...")

    for i, eps in enumerate(eps_grid):
        k = ReweightedKernel(base=base, epsilon=float(eps), min_prob=min_prob)
        ll[i] = k.loglik(curr, nxt)

        if (i == 0) or ((i + 1) % progress_every == 0) or (i + 1 == n_grid):
            pct = int(round(100 * (i + 1) / n_grid))
            if label:
                print(f"  - {label}: {pct}% ({i + 1}/{n_grid})")
            else:
                print(f"  - {pct}% ({i + 1}/{n_grid})")

    j = int(np.argmax(ll))
    eps_hat = float(eps_grid[j])

    if label:
        print(f"[Phase3-RNG] Done epsilon({label}): eps_hat={eps_hat:.4f}")

    return eps_hat, ll


def _shuffle_next_states(curr: np.ndarray, nxt: np.ndarray, seed: int):
    """
    Strong null control:
    destroys temporal pairing between consecutive RNG states.
    """
    rng = np.random.default_rng(seed)

    curr_ctrl = curr.copy()
    nxt_ctrl = nxt.copy()

    rng.shuffle(nxt_ctrl)

    return curr_ctrl, nxt_ctrl


def _circular_shift_next_states(curr: np.ndarray, nxt: np.ndarray, seed: int):
    """
    Preserves the marginal distribution of next states while breaking
    the correct temporal alignment between current and next RNG states.
    """
    rng = np.random.default_rng(seed)

    curr_ctrl = curr.copy()
    nxt_ctrl = nxt.copy()

    if len(nxt_ctrl) <= 1:
        return curr_ctrl, nxt_ctrl

    shift = int(rng.integers(1, len(nxt_ctrl)))
    nxt_ctrl = np.roll(nxt_ctrl, shift)

    return curr_ctrl, nxt_ctrl


def _block_shuffle_pairs(curr: np.ndarray, nxt: np.ndarray, seed: int, block_size: int):
    """
    Shuffle short temporal blocks of RNG state transitions.
    Preserves local short-range structure while destroying larger-scale ordering.
    """
    rng = np.random.default_rng(seed)

    n = len(curr)

    if n <= block_size:
        idx = rng.permutation(n)
        return curr[idx].copy(), nxt[idx].copy()

    starts = list(range(0, n, block_size))
    blocks = []

    for s in starts:
        e = min(s + block_size, n)
        blocks.append((curr[s:e].copy(), nxt[s:e].copy()))

    rng.shuffle(blocks)

    curr_ctrl = np.concatenate([b[0] for b in blocks], axis=0)
    nxt_ctrl = np.concatenate([b[1] for b in blocks], axis=0)

    return curr_ctrl, nxt_ctrl


def _reverse_next_states(curr: np.ndarray, nxt: np.ndarray):
    """
    Aggressive null control:
    reverses the order of next states, preserving value distribution but
    strongly breaking temporal directionality.
    """
    curr_ctrl = curr.copy()
    nxt_ctrl = nxt[::-1].copy()

    return curr_ctrl, nxt_ctrl


def run_rng_controls(
    curr: np.ndarray,
    nxt: np.ndarray,
    kernel,
    cfg,
):
    """
    Run RNG-specific negative controls and estimate epsilon on each.

    Parameters
    ----------
    curr, nxt
        Arrays of current and next encoded RNG states.
    kernel
        EmpiricalKernel baseline P0 fitted on train transitions.
    cfg
        Phase3RNGConfig.

    Returns
    -------
    list[float]
        Estimated epsilons for controls.
    """
    eps_grid = np.array(cfg.eps_grid, dtype=float)

    methods = [
        "shuffle_next",
        "circular_shift",
        "block_shuffle",
        "reverse_next",
    ]

    eps_controls = []

    for i in range(cfg.n_controls):
        method = methods[i % len(methods)]
        seed = cfg.random_seed + i

        if method == "shuffle_next":
            c_ctrl, n_ctrl = _shuffle_next_states(curr, nxt, seed=seed)

        elif method == "circular_shift":
            c_ctrl, n_ctrl = _circular_shift_next_states(curr, nxt, seed=seed)

        elif method == "block_shuffle":
            c_ctrl, n_ctrl = _block_shuffle_pairs(
                curr,
                nxt,
                seed=seed,
                block_size=cfg.rng_control_block_size,
            )

        elif method == "reverse_next":
            c_ctrl, n_ctrl = _reverse_next_states(curr, nxt)

        else:
            raise ValueError(f"Unknown control method: {method}")

        eps_c, _ = _estimate_epsilon_grid_local(
            c_ctrl,
            n_ctrl,
            kernel,
            eps_grid,
            cfg.min_prob,
            label=f"control_{i + 1}_{method}",
            progress_every=20,
        )

        eps_controls.append(float(eps_c))
        print(f"  Control {i + 1}/{cfg.n_controls} [{method}]: eps={eps_c:.4f}")

    return eps_controls