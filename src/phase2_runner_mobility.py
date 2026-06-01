from __future__ import annotations

import json
import numpy as np

from config.phase2_config_mobility import load_phase2_mobility_config

from src.geolife_loader import load_geolife
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

from src.controls_phase2_mobility import run_mobility_controls
from src.phase2_runner import _estimate_epsilon_grid, _simulate_trajectory


# ---------------------------------------------------------
# Phase II.2 Runner
# ---------------------------------------------------------

def run_phase2_mobility():
    cfg = load_phase2_mobility_config()
    cfg.ensure_paths()

    print("\n==============================")
    print("CDR Phase II.2 — Mobility Domain")
    print("==============================\n")

    print("[Phase2-Mobility] Configuration loaded")
    print("Dataset root:", cfg.dataset_root)
    print("Max users:", cfg.max_users)
    print("State columns:", cfg.state_columns)

    # -----------------------------------------------------
    # Load mobility data
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Loading GeoLife dataset...")

    df = load_geolife(max_users=cfg.max_users)
    df = df[list(cfg.state_columns)]

    print(f"[Phase2-Mobility] Observations: {len(df)}")
    print(f"[Phase2-Mobility] Columns: {list(df.columns)}")

    # -----------------------------------------------------
    # Discretization
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Discretizing state variables...")

    df_disc, specs = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=cfg.quantiles,
        fit_on_index=np.arange(len(df), dtype=int),
    )

    print("[Phase2-Mobility] Discretization completed")

    comps = df_disc.to_numpy(dtype=int)
    n_components = comps.shape[1]

    enc = make_encoding(
        n_components=n_components,
        n_bins=cfg.n_bins,
    )

    n_states = cfg.n_bins ** n_components

    print(
        f"[Phase2-Mobility] State space: "
        f"{n_components} variables x {cfg.n_bins} bins = {n_states} states"
    )

    ids = encode_states(comps, enc)

    # -----------------------------------------------------
    # Transitions
    # -----------------------------------------------------

    curr_all, nxt_all = build_transitions(ids)

    split = int(len(curr_all) * cfg.train_ratio)

    curr_train = curr_all[:split]
    nxt_train = nxt_all[:split]

    curr_test = curr_all[split:]
    nxt_test = nxt_all[split:]

    print(
        f"[Phase2-Mobility] Transitions: "
        f"train={len(curr_train)} | test={len(curr_test)}"
    )

    # -----------------------------------------------------
    # Empirical Kernel
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Building empirical baseline kernel P0...")

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    print("[Phase2-Mobility] P0 built.")

    eps_grid = np.array(cfg.eps_grid, dtype=float)

    # -----------------------------------------------------
    # Estimate epsilon (train)
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Estimating epsilon (train)...")

    eps_hat_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        cfg.min_prob,
        label="mobility_train",
        progress_every=10,
    )

    print(f"[Phase2-Mobility] eps_hat_train = {eps_hat_train:.4f}")

    # -----------------------------------------------------
    # Estimate epsilon (test)
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Estimating epsilon (test)...")

    eps_hat_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        cfg.min_prob,
        label="mobility_test",
        progress_every=10,
    )

    print(f"[Phase2-Mobility] eps_hat_test = {eps_hat_test:.4f}")

    # -----------------------------------------------------
    # F1 — Injection recovery
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Running Gate F1 (injection recovery)...")

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
        label="injection",
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

    print("\n[Phase2-Mobility] Running Gate F2 (controls collapse)...")

    eps_controls = run_mobility_controls(
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

    print("\n[Phase2-Mobility] Running Gate F3 (holdout generalization)...")

    gate3 = gate_F3_holdout_generalization(
        eps_train=eps_hat_train,
        eps_test=eps_hat_test,
        max_delta=cfg.holdout_delta,
    )

    # -----------------------------------------------------
    # F5 — Sensitivity
    # -----------------------------------------------------

    print("\n[Phase2-Mobility] Running Gate F5 (sensitivity)...")

    df_disc_alt, _ = fit_and_discretize(
        df,
        n_bins=cfg.n_bins,
        quantiles=(0.30, 0.70),
        fit_on_index=np.arange(len(df), dtype=int),
    )

    comps_alt = df_disc_alt.to_numpy(dtype=int)
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
        label="mobility_bins_alt",
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
    print("CDR Phase II.2 (Mobility) — Gates")
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
    }

    out_file = cfg.results_dir / "phase2_mobility_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {out_file}")


# ---------------------------------------------------------

if __name__ == "__main__":
    run_phase2_mobility()