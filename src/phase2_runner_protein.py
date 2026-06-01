from __future__ import annotations

import json
import numpy as np

from config.phase2_config_protein import load_phase2_protein_config

from src.protein_loader import load_protein
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

from src.controls_phase2_protein import run_protein_controls
from src.phase2_runner import _estimate_epsilon_grid, _simulate_trajectory


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _build_transitions_by_group(
    state_ids: np.ndarray,
    groups: np.ndarray,
):
    """
    Build transitions only within each group, never across group boundaries.

    This is critical for protein trajectories because each .xtc file is an
    independent simulation and we must not create artificial transitions
    between the end of one trajectory and the start of another.
    """
    state_ids = np.asarray(state_ids).reshape(-1)
    groups = np.asarray(groups).reshape(-1)

    if len(state_ids) != len(groups):
        raise ValueError("state_ids and groups must have the same length")

    curr_parts = []
    nxt_parts = []

    unique_groups = list(dict.fromkeys(groups.tolist()))

    for g in unique_groups:
        idx = np.where(groups == g)[0]

        if len(idx) < 2:
            continue

        ids_g = state_ids[idx]
        c_g, n_g = build_transitions(ids_g)

        curr_parts.append(c_g)
        nxt_parts.append(n_g)

    if len(curr_parts) == 0:
        raise RuntimeError("No valid grouped transitions could be built")

    curr = np.concatenate(curr_parts, axis=0)
    nxt = np.concatenate(nxt_parts, axis=0)

    return curr, nxt


# ---------------------------------------------------------
# Phase II.4 Runner
# ---------------------------------------------------------

def run_phase2_protein():
    cfg = load_phase2_protein_config()
    cfg.ensure_paths()

    print("\n==============================")
    print("CDR Phase II.4 — Protein Domain")
    print("==============================\n")

    print("[Phase2-Protein] Configuration loaded")
    print("Dataset root:", cfg.dataset_root)
    print("PDB file:", cfg.pdb_file)
    print("XTC files:", cfg.xtc_files)
    print("State columns:", cfg.state_columns)
    print("Frame stride:", cfg.frame_stride)

    # -----------------------------------------------------
    # Load protein data
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Loading protein dataset...")

    df_full = load_protein(
        dataset_root=cfg.dataset_root,
        pdb_file=cfg.pdb_file,
        xtc_files=cfg.xtc_files,
        frame_stride=cfg.frame_stride,
        max_frames_per_traj=cfg.max_frames_per_traj,
        verbose=cfg.verbose,
    )

    # Keep full dataframe because traj_file is needed for F3
    df = df_full[list(cfg.state_columns)]

    print(f"[Phase2-Protein] Observations: {len(df)}")
    print(f"[Phase2-Protein] Columns: {list(df.columns)}")

    if len(df) < 10:
        raise RuntimeError("Too few observations loaded for protein phase")

    # -----------------------------------------------------
    # Discretization
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Discretizing state variables...")

    fit_idx = np.arange(len(df), dtype=int)

    df_disc, specs = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=cfg.quantiles,
        fit_on_index=fit_idx,
    )

    print("[Phase2-Protein] Discretization completed")

    comps = df_disc.to_numpy(dtype=int)
    n_components = comps.shape[1]

    enc = make_encoding(
        n_components=n_components,
        n_bins=cfg.n_bins,
    )

    n_states = cfg.n_bins ** n_components

    print(
        f"[Phase2-Protein] State space: "
        f"{n_components} variables x {cfg.n_bins} bins = {n_states} states"
    )

    ids = encode_states(comps, enc)

    # -----------------------------------------------------
    # Transitions (original sequential logic preserved
    # for F1, F2 and F5 exactly as before)
    # -----------------------------------------------------

    curr_all, nxt_all = build_transitions(ids)

    if len(curr_all) < 10:
        raise RuntimeError("Too few transitions after state encoding")

    split = int(len(curr_all) * cfg.train_ratio)

    if split < 2 or split >= len(curr_all):
        raise RuntimeError(
            f"Invalid train/test split for protein domain: split={split}, n={len(curr_all)}"
        )

    curr_train = curr_all[:split]
    nxt_train = nxt_all[:split]

    curr_test = curr_all[split:]
    nxt_test = nxt_all[split:]

    print(
        f"[Phase2-Protein] Transitions: "
        f"train={len(curr_train)} | test={len(curr_test)}"
    )

    # -----------------------------------------------------
    # Empirical Kernel
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Building empirical baseline kernel P0...")

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    print("[Phase2-Protein] P0 built.")

    eps_grid = np.array(cfg.eps_grid, dtype=float)

    # -----------------------------------------------------
    # Estimate epsilon (train)
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Estimating epsilon (train)...")

    eps_hat_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        cfg.min_prob,
        label="protein_train",
        progress_every=10,
    )

    print(f"[Phase2-Protein] eps_hat_train = {eps_hat_train:.4f}")

    # -----------------------------------------------------
    # Estimate epsilon (test)
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Estimating epsilon (test)...")

    eps_hat_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        cfg.min_prob,
        label="protein_test",
        progress_every=10,
    )

    print(f"[Phase2-Protein] eps_hat_test = {eps_hat_test:.4f}")

    # -----------------------------------------------------
    # F1 — Injection recovery
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Running Gate F1 (injection recovery)...")

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
        label="protein_injection",
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

    print("\n[Phase2-Protein] Running Gate F2 (controls collapse)...")

    eps_controls = run_protein_controls(
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
    # ONLY THIS PART CHANGED
    # -----------------------------------------------------

    print("\n[Phase2-Protein] Running Gate F3 (holdout generalization)...")

    traj_labels = df_full["traj_file"].to_numpy(dtype=str)
    unique_trajs = sorted(np.unique(traj_labels))

    if len(unique_trajs) < 2:
        raise RuntimeError("F3 requires at least 2 independent trajectories")

    # Deterministic trajectory-level holdout:
    # train on all trajectories except the last one, test on the last one.
    if cfg.f3_holdout_mode == "leave_one_trajectory_out":
        heldout_traj = unique_trajs[cfg.f3_holdout_traj_index]
    else:
        raise ValueError(f"Unsupported F3 holdout mode: {cfg.f3_holdout_mode}")

    train_mask = traj_labels != heldout_traj
    test_mask = traj_labels == heldout_traj

    train_idx_f3 = np.where(train_mask)[0]

    if train_idx_f3.size < 2 or np.sum(test_mask) < 2:
        raise RuntimeError("Invalid trajectory holdout split for F3")

    # Refit discretization on trajectory-level TRAIN only
    df_disc_f3, _ = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=cfg.quantiles,
        fit_on_index=train_idx_f3,
    )

    comps_f3 = df_disc_f3.to_numpy(dtype=int)

    if comps_f3.shape[1] != enc.n_components:
        raise RuntimeError("F3 discretization changed number of components")

    ids_f3 = encode_states(comps_f3, enc)

    # Build transitions ONLY within each trajectory
    curr_train_f3, nxt_train_f3 = _build_transitions_by_group(
        ids_f3[train_mask],
        traj_labels[train_mask],
    )

    curr_test_f3, nxt_test_f3 = _build_transitions_by_group(
        ids_f3[test_mask],
        traj_labels[test_mask],
    )

    print(
        f"[Phase2-Protein] F3 holdout trajectory: {heldout_traj} | "
        f"train_transitions={len(curr_train_f3)} | test_transitions={len(curr_test_f3)}"
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
        label="protein_f3_train",
        progress_every=10,
    )

    eps_hat_test_f3, _ = _estimate_epsilon_grid(
        curr_test_f3,
        nxt_test_f3,
        P0_f3,
        eps_grid,
        cfg.min_prob,
        label="protein_f3_test",
        progress_every=10,
    )

    print(
        f"[Phase2-Protein] F3 eps_train = {eps_hat_train_f3:.4f} | "
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

    print("\n[Phase2-Protein] Running Gate F5 (sensitivity)...")

    if cfg.n_bins == 2:
        alt_quantiles = (0.45,)
    elif cfg.n_bins == 3:
        alt_quantiles = (0.30, 0.70)
    else:
        alt_quantiles = cfg.quantiles

    df_disc_alt, _ = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=alt_quantiles,
        fit_on_index=fit_idx,
    )

    comps_alt = df_disc_alt.to_numpy(dtype=int)

    if comps_alt.shape[1] != enc.n_components:
        raise RuntimeError("Sensitivity discretization changed number of components")

    ids_alt = encode_states(comps_alt, enc)

    curr_alt_all, nxt_alt_all = build_transitions(ids_alt)

    curr_alt_train = curr_alt_all[:split]
    nxt_alt_train = nxt_alt_all[:split]

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
        label="protein_bins_alt",
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
    print("CDR Phase II.4 (Protein) — Gates")
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
            "state_columns": list(cfg.state_columns),
            "n_bins": int(cfg.n_bins),
            "train_ratio": float(cfg.train_ratio),
            "frame_stride": int(cfg.frame_stride),
            "pdb_file": cfg.pdb_file,
            "xtc_files": list(cfg.xtc_files),
            "f3_holdout_mode": "leave_one_trajectory_out_last_as_test",
            "f3_heldout_traj": heldout_traj,
            "f3_eps_hat_train": float(eps_hat_train_f3),
            "f3_eps_hat_test": float(eps_hat_test_f3),
        },
    }

    out_file = cfg.results_dir / "phase2_protein_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {out_file}")


# ---------------------------------------------------------

if __name__ == "__main__":
    run_phase2_protein()