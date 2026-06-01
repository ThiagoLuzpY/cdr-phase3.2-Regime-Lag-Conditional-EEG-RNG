from __future__ import annotations

import json
import numpy as np

from config.phase3_config_eeg import load_phase3_eeg_config

from src.eeg_loader import load_eeg
from src.discretize import fit_and_discretize
from src.build_states import (
    make_encoding,
    encode_states,
    build_transitions,
)
from src.kernels.empirical_kernel import EmpiricalKernel
from src.validators_phase2 import (
    gate_F1_injection_recovery,
    gate_F2_controls_collapse,
    gate_F3_holdout_generalization,
    gate_F5_sensitivity,
    summarize,
)
from src.controls_phase3_eeg import run_eeg_controls
from src.phase2_runner import _estimate_epsilon_grid, _simulate_trajectory


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _make_interleaved_split(n: int):
    """
    Interleaved split for temporally structured EEG epochs.
    Even indices -> train
    Odd indices  -> test
    """
    idx = np.arange(n, dtype=int)

    idx_train = idx[idx % 2 == 0]
    idx_test = idx[idx % 2 == 1]

    if len(idx_train) < 2 or len(idx_test) < 2:
        raise RuntimeError(
            f"Interleaved split too small: train={len(idx_train)}, test={len(idx_test)}"
        )

    return idx_train, idx_test


def _make_chronological_split(n: int, train_ratio: float):
    """
    Standard chronological split.
    """
    split = int(n * train_ratio)

    if split < 2 or split >= n - 1:
        raise RuntimeError(
            f"Invalid chronological split: split={split}, n={n}, train_ratio={train_ratio}"
        )

    idx_train = np.arange(0, split, dtype=int)
    idx_test = np.arange(split, n, dtype=int)

    return idx_train, idx_test


def _make_block_interleaved_split(n: int, block_size: int):
    """
    Split EEG epochs into contiguous blocks and alternate train/test blocks.

    Example with block_size=20:
    block 0 -> train
    block 1 -> test
    block 2 -> train
    block 3 -> test
    ...
    """
    if block_size < 2:
        raise RuntimeError(f"block_size must be >= 2, got {block_size}")

    idx_train_parts = []
    idx_test_parts = []

    starts = list(range(0, n, block_size))

    for b, s in enumerate(starts):
        e = min(s + block_size, n)
        idx_block = np.arange(s, e, dtype=int)

        if b % 2 == 0:
            idx_train_parts.append(idx_block)
        else:
            idx_test_parts.append(idx_block)

    if len(idx_train_parts) == 0 or len(idx_test_parts) == 0:
        raise RuntimeError(
            f"Block interleaved split too small: n={n}, block_size={block_size}"
        )

    idx_train = np.concatenate(idx_train_parts, axis=0)
    idx_test = np.concatenate(idx_test_parts, axis=0)

    if len(idx_train) < 2 or len(idx_test) < 2:
        raise RuntimeError(
            f"Block interleaved split too small: train={len(idx_train)}, test={len(idx_test)}"
        )

    return idx_train, idx_test


def _make_epoch_split(cfg, n: int):
    """
    Select epoch-level split according to cfg.f3_holdout_mode.
    This split is used both for the main train/test partition and for F3.
    """
    if cfg.f3_holdout_mode == "interleaved":
        return _make_interleaved_split(n)
    elif cfg.f3_holdout_mode == "chronological":
        return _make_chronological_split(n, cfg.train_ratio)
    elif cfg.f3_holdout_mode == "block_interleaved":
        return _make_block_interleaved_split(n, cfg.f3_block_size)
    else:
        raise ValueError(f"Unsupported F3 holdout mode: {cfg.f3_holdout_mode}")


def _transition_indices_from_epoch_split(
    n_epochs: int,
    idx_train_epochs: np.ndarray,
    idx_test_epochs: np.ndarray,
):
    """
    Convert an epoch split into transition indices without leakage.

    Transition k corresponds to epoch pair (k -> k+1).
    A transition belongs to TRAIN only if both k and k+1 are in train epochs.
    A transition belongs to TEST  only if both k and k+1 are in test epochs.
    """
    if n_epochs < 2:
        raise RuntimeError(f"Need at least 2 epochs, got {n_epochs}")

    train_mask = np.zeros(n_epochs, dtype=bool)
    test_mask = np.zeros(n_epochs, dtype=bool)

    train_mask[idx_train_epochs] = True
    test_mask[idx_test_epochs] = True

    train_t = []
    test_t = []

    for k in range(n_epochs - 1):
        if train_mask[k] and train_mask[k + 1]:
            train_t.append(k)
        elif test_mask[k] and test_mask[k + 1]:
            test_t.append(k)

    idx_train_trans = np.array(train_t, dtype=int)
    idx_test_trans = np.array(test_t, dtype=int)

    if len(idx_train_trans) < 2 or len(idx_test_trans) < 2:
        raise RuntimeError(
            f"Transition split too small after leakage-safe alignment: "
            f"train={len(idx_train_trans)}, test={len(idx_test_trans)}"
        )

    return idx_train_trans, idx_test_trans


# ---------------------------------------------------------
# Phase III EEG Runner
# ---------------------------------------------------------

def run_phase3_eeg():
    cfg = load_phase3_eeg_config()
    cfg.ensure_paths()

    print("\n==============================")
    print("CDR Phase III — EEG Domain")
    print("==============================\n")

    print("[Phase3-EEG] Configuration loaded")
    print("Dataset root:", cfg.dataset_root)
    print("PSG file:", cfg.psg_file)
    print("Hypnogram file:", cfg.hypnogram_file)
    print("Channel:", cfg.channel_name)
    print("State columns:", cfg.state_columns)
    print("Epoch seconds:", cfg.epoch_seconds)
    print("F3 holdout mode:", cfg.f3_holdout_mode)
    print("F3 block size:", cfg.f3_block_size)
    print("Log bandpower:", cfg.log_bandpower)

    # -----------------------------------------------------
    # Load EEG data
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Loading EEG dataset...")

    df_full = load_eeg(
        dataset_root=cfg.dataset_root,
        psg_file=cfg.psg_file,
        hypnogram_file=cfg.hypnogram_file,
        channel_name=cfg.channel_name,
        epoch_seconds=cfg.epoch_seconds,
        max_epochs=cfg.max_epochs,
        allowed_stages=cfg.allowed_stages,
        delta_band=cfg.delta_band,
        theta_band=cfg.theta_band,
        alpha_band=cfg.alpha_band,
        beta_band=cfg.beta_band,
        log_bandpower=cfg.log_bandpower,
        verbose=cfg.verbose,
    )

    state_cols = list(cfg.state_columns)
    if cfg.include_stage_code_in_state and "stage_code" not in state_cols:
        state_cols.append("stage_code")

    df = df_full[state_cols]

    print(f"[Phase3-EEG] Observations: {len(df)}")
    print(f"[Phase3-EEG] Columns: {list(df.columns)}")

    if len(df) < 10:
        raise RuntimeError("Too few EEG epochs loaded for Phase III EEG")

    # -----------------------------------------------------
    # Epoch split (used consistently across the domain)
    # -----------------------------------------------------

    idx_train_epochs, idx_test_epochs = _make_epoch_split(cfg, len(df))

    print(
        f"[Phase3-EEG] Epoch split: train={len(idx_train_epochs)} | test={len(idx_test_epochs)}"
    )

    # -----------------------------------------------------
    # Discretization
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Discretizing state variables...")

    # Important: fit bins on TRAIN epochs only
    df_disc, specs = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=cfg.quantiles,
        fit_on_index=idx_train_epochs,
    )

    print("[Phase3-EEG] Discretization completed")

    comps = df_disc.to_numpy(dtype=int)
    n_components = comps.shape[1]

    enc = make_encoding(
        n_components=n_components,
        n_bins=cfg.n_bins,
    )

    n_states = cfg.n_bins ** n_components

    print(
        f"[Phase3-EEG] State space: "
        f"{n_components} variables x {cfg.n_bins} bins = {n_states} states"
    )

    ids = encode_states(comps, enc)

    # -----------------------------------------------------
    # Transitions
    # -----------------------------------------------------

    curr_all, nxt_all = build_transitions(ids)

    if len(curr_all) < 10:
        raise RuntimeError("Too few transitions after EEG state encoding")

    print(f"[Phase3-EEG] Total transitions: {len(curr_all)}")

    idx_train_trans, idx_test_trans = _transition_indices_from_epoch_split(
        n_epochs=len(df),
        idx_train_epochs=idx_train_epochs,
        idx_test_epochs=idx_test_epochs,
    )

    curr_train = curr_all[idx_train_trans]
    nxt_train = nxt_all[idx_train_trans]

    curr_test = curr_all[idx_test_trans]
    nxt_test = nxt_all[idx_test_trans]

    print(
        f"[Phase3-EEG] Leakage-safe transitions: "
        f"train={len(curr_train)} | test={len(curr_test)}"
    )

    # -----------------------------------------------------
    # Empirical Kernel
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Building empirical baseline kernel P0...")

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    print("[Phase3-EEG] P0 built.")

    eps_grid = np.array(cfg.eps_grid, dtype=float)

    # -----------------------------------------------------
    # Estimate epsilon (train)
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Estimating epsilon (train)...")

    eps_hat_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        cfg.min_prob,
        label="eeg_train",
        progress_every=10,
    )

    print(f"[Phase3-EEG] eps_hat_train = {eps_hat_train:.4f}")

    # -----------------------------------------------------
    # Estimate epsilon (test)
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Estimating epsilon (test)...")

    eps_hat_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        cfg.min_prob,
        label="eeg_test",
        progress_every=10,
    )

    print(f"[Phase3-EEG] eps_hat_test = {eps_hat_test:.4f}")

    # -----------------------------------------------------
    # F1 — Injection recovery
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Running Gate F1 (injection recovery)...")

    sim_traj = _simulate_trajectory(
        P0,
        eps=cfg.inj_eps_true,
        n_steps=len(curr_train),
        seed=cfg.random_seed,
    )

    sim_curr, sim_nxt = build_transitions(sim_traj)

    eps_hat_inj, ll_inj = _estimate_epsilon_grid(
        sim_curr,
        sim_nxt,
        P0,
        eps_grid,
        cfg.min_prob,
        label="eeg_injection",
        progress_every=10,
    )

    gate1 = gate_F1_injection_recovery(
        eps_hat=eps_hat_inj,
        eps_true=cfg.inj_eps_true,
        tol_abs=cfg.gate_tol_abs,
    )

    # -----------------------------------------------------
    # F2 — Controls collapse
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Running Gate F2 (controls collapse)...")

    eps_controls = run_eeg_controls(
        curr=curr_train,
        nxt=nxt_train,
        kernel=P0,
        cfg=cfg,
    )

    gate2 = gate_F2_controls_collapse(
        eps_controls=eps_controls,
        tol=cfg.control_tol,
        required_fraction=cfg.control_fraction,
    )

    # -----------------------------------------------------
    # F3 — Holdout generalization
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Running Gate F3 (holdout generalization)...")

    # F3 reuses the same leakage-safe split, but recomputes discretization
    # strictly on TRAIN epochs to preserve formal consistency.
    df_disc_f3, _ = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=cfg.quantiles,
        fit_on_index=idx_train_epochs,
    )

    comps_f3 = df_disc_f3.to_numpy(dtype=int)

    if comps_f3.shape[1] != enc.n_components:
        raise RuntimeError("F3 discretization changed number of EEG components")

    ids_f3 = encode_states(comps_f3, enc)

    curr_f3_all, nxt_f3_all = build_transitions(ids_f3)

    idx_train_trans_f3, idx_test_trans_f3 = _transition_indices_from_epoch_split(
        n_epochs=len(df),
        idx_train_epochs=idx_train_epochs,
        idx_test_epochs=idx_test_epochs,
    )

    curr_train_f3 = curr_f3_all[idx_train_trans_f3]
    nxt_train_f3 = nxt_f3_all[idx_train_trans_f3]

    curr_test_f3 = curr_f3_all[idx_test_trans_f3]
    nxt_test_f3 = nxt_f3_all[idx_test_trans_f3]

    print(
        f"[Phase3-EEG] F3 transitions: "
        f"train={len(curr_train_f3)} | test={len(curr_test_f3)}"
    )

    P0_f3 = EmpiricalKernel.from_transitions(
        curr_train_f3,
        nxt_train_f3,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    eps_hat_train_f3, _ = _estimate_epsilon_grid(
        curr_train_f3,
        nxt_train_f3,
        P0_f3,
        eps_grid,
        cfg.min_prob,
        label="eeg_f3_train",
        progress_every=10,
    )

    eps_hat_test_f3, _ = _estimate_epsilon_grid(
        curr_test_f3,
        nxt_test_f3,
        P0_f3,
        eps_grid,
        cfg.min_prob,
        label="eeg_f3_test",
        progress_every=10,
    )

    print(
        f"[Phase3-EEG] F3 eps_train = {eps_hat_train_f3:.4f} | "
        f"eps_test = {eps_hat_test_f3:.4f}"
    )

    gate3 = gate_F3_holdout_generalization(
        eps_train=eps_hat_train_f3,
        eps_test=eps_hat_test_f3,
        max_delta=cfg.holdout_delta,
    )

    # -----------------------------------------------------
    # F5 — Sensitivity
    # -----------------------------------------------------

    print("\n[Phase3-EEG] Running Gate F5 (sensitivity)...")

    alt_quantiles = cfg.sensitivity_quantiles

    df_disc_alt, _ = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=alt_quantiles,
        fit_on_index=idx_train_epochs,
    )

    comps_alt = df_disc_alt.to_numpy(dtype=int)

    if comps_alt.shape[1] != enc.n_components:
        raise RuntimeError("Sensitivity discretization changed number of EEG components")

    ids_alt = encode_states(comps_alt, enc)

    curr_alt_all, nxt_alt_all = build_transitions(ids_alt)

    curr_alt_train = curr_alt_all[idx_train_trans]
    nxt_alt_train = nxt_alt_all[idx_train_trans]

    P0_alt = EmpiricalKernel.from_transitions(
        curr_alt_train,
        nxt_alt_train,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    eps_alt, ll_alt = _estimate_epsilon_grid(
        curr_alt_train,
        nxt_alt_train,
        P0_alt,
        eps_grid,
        cfg.min_prob,
        label="eeg_bins_alt",
        progress_every=10,
    )

    gate5 = gate_F5_sensitivity(
        eps_binsA=eps_hat_train,
        eps_binsB=eps_alt,
        max_delta=cfg.sensitivity_delta,
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    gates = [gate1, gate2, gate3, gate5]
    summary = summarize(gates)

    print("\n==============================")
    print("CDR Phase III (EEG) — Gates")
    print("==============================")

    for g in gates:
        print(f"{g.name}: {'PASS' if g.passed else 'FAIL'} | metrics={g.metrics}")

    print("==============================")
    print(f"FINAL: {'PASS ✅' if summary['passed_all'] else 'FAIL ❌'}")

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results = {
        "eps_hat_train": float(eps_hat_train),
        "eps_hat_test": float(eps_hat_test),
        "eps_hat_injection": float(eps_hat_inj),
        "eps_controls": [float(x) for x in eps_controls],
        "eps_hat_bins_alt": float(eps_alt),
        "gates": summary,
        "metadata": {
            "state_columns": state_cols,
            "n_bins": int(cfg.n_bins),
            "train_ratio": float(cfg.train_ratio),
            "channel_name": cfg.channel_name,
            "epoch_seconds": int(cfg.epoch_seconds),
            "psg_file": cfg.psg_file,
            "hypnogram_file": cfg.hypnogram_file,
            "f3_holdout_mode": cfg.f3_holdout_mode,
            "f3_block_size": int(cfg.f3_block_size),
            "log_bandpower": bool(cfg.log_bandpower),
            "delta_band": list(cfg.delta_band),
            "theta_band": list(cfg.theta_band),
            "alpha_band": list(cfg.alpha_band),
            "beta_band": list(cfg.beta_band),
            "f3_eps_hat_train": float(eps_hat_train_f3),
            "f3_eps_hat_test": float(eps_hat_test_f3),
        },
    }

    out_file = cfg.results_dir / "phase3_eeg_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {out_file}")


# ---------------------------------------------------------

if __name__ == "__main__":
    run_phase3_eeg()