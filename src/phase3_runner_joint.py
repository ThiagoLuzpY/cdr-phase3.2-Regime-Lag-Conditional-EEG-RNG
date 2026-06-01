from __future__ import annotations

import json
import numpy as np
import pandas as pd

from config.phase3_config_joint import load_phase3_joint_config

from src.joint_loader import load_joint
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
from src.controls_phase3_joint import run_joint_controls
from src.phase2_runner import _estimate_epsilon_grid, _simulate_trajectory


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _make_chronological_split(n: int, train_ratio: float):
    split = int(n * train_ratio)

    if split < 2 or split >= n - 1:
        raise RuntimeError(
            f"Invalid chronological split: split={split}, n={n}, train_ratio={train_ratio}"
        )

    idx_train = np.arange(0, split, dtype=int)
    idx_test = np.arange(split, n, dtype=int)

    return idx_train, idx_test


def _make_interleaved_split(n: int):
    idx = np.arange(n, dtype=int)

    idx_train = idx[idx % 2 == 0]
    idx_test = idx[idx % 2 == 1]

    if len(idx_train) < 2 or len(idx_test) < 2:
        raise RuntimeError(
            f"Interleaved split too small: train={len(idx_train)}, test={len(idx_test)}"
        )

    return idx_train, idx_test


def _make_block_interleaved_split(n: int, block_size: int):
    if block_size < 2:
        raise ValueError("block_size must be >= 2")

    train_idx = []
    test_idx = []

    start = 0
    block_id = 0

    while start < n:
        end = min(start + block_size, n)
        block = np.arange(start, end, dtype=int)

        if block_id % 2 == 0:
            train_idx.append(block)
        else:
            test_idx.append(block)

        start = end
        block_id += 1

    idx_train = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    idx_test = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)

    if len(idx_train) < 2 or len(idx_test) < 2:
        raise RuntimeError(
            f"Block-interleaved split too small: train={len(idx_train)}, test={len(idx_test)}"
        )

    return idx_train, idx_test


def _make_epoch_split(cfg, n: int):
    if cfg.f3_holdout_mode == "chronological":
        return _make_chronological_split(n, cfg.train_ratio)
    elif cfg.f3_holdout_mode == "interleaved":
        return _make_interleaved_split(n)
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
    Convert an epoch split into leakage-safe transition indices.

    Transition k corresponds to epoch pair (k -> k+1).
    A transition belongs to TRAIN only if both k and k+1 are in train epochs.
    A transition belongs to TEST  only if both k and k+1 are in test epochs.
    """
    if n_epochs < 2:
        raise RuntimeError(f"Need at least 2 aligned samples, got {n_epochs}")

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


def _build_joint_dataframe(
    df_joint: pd.DataFrame,
    cfg,
    idx_train_epochs: np.ndarray,
    *,
    use_sensitivity_quantiles: bool = False,
) -> pd.DataFrame:
    """
    Build the final joint component dataframe used for state encoding.

    EEG columns are discretized.
    RNG columns are already binary and are used directly.
    """
    eeg_cols = [f"eeg_{c}" for c in cfg.eeg_state_columns]
    rng_cols = [f"rng_{c}" for c in cfg.rng_state_columns]

    if cfg.include_stage_code_in_joint_state:
        eeg_cols = eeg_cols + ["eeg_stage_code"]

    missing = [c for c in (eeg_cols + rng_cols) if c not in df_joint.columns]
    if missing:
        raise RuntimeError(f"Joint dataframe missing columns: {missing}")

    # EEG branch: discretize
    df_eeg = df_joint[eeg_cols].copy()

    if cfg.include_stage_code_in_joint_state:
        eeg_feature_cols = [c for c in eeg_cols if c != "eeg_stage_code"]
        df_eeg_features = df_eeg[eeg_feature_cols].copy()
    else:
        eeg_feature_cols = eeg_cols
        df_eeg_features = df_eeg.copy()

    quantiles = (
        cfg.eeg_sensitivity_quantiles if use_sensitivity_quantiles else cfg.eeg_quantiles
    )

    df_eeg_disc, _ = fit_and_discretize(
        df_eeg_features,
        n_bins=cfg.eeg_n_bins,
        quantiles=quantiles,
        fit_on_index=idx_train_epochs,
    )

    # If stage_code is included, append it already as discrete component
    if cfg.include_stage_code_in_joint_state:
        stage_vals = df_eeg["eeg_stage_code"].to_numpy(dtype=int)
        df_eeg_disc["eeg_stage_code"] = stage_vals

    # RNG branch: already binary
    df_rng = df_joint[rng_cols].copy()

    rng_vals = df_rng.to_numpy(dtype=int)
    if np.any((rng_vals < 0) | (rng_vals >= cfg.rng_n_bins)):
        raise RuntimeError("RNG joint components out of expected binary range [0, 1]")

    # Final joint component ordering:
    # EEG first, RNG last
    df_final = pd.DataFrame(index=df_joint.index)

    for col in df_eeg_disc.columns:
        df_final[col] = df_eeg_disc[col].to_numpy(dtype=int)

    for col in rng_cols:
        df_final[col] = df_rng[col].to_numpy(dtype=int)

    return df_final


# ---------------------------------------------------------
# Phase III Joint Runner
# ---------------------------------------------------------

def run_phase3_joint():
    cfg = load_phase3_joint_config()
    cfg.ensure_paths()

    print("\n==============================")
    print("CDR Phase III — JOINT Domain (EEG + RNG)")
    print("==============================\n")

    print("[Phase3-Joint] Configuration loaded")
    print("EEG root:", cfg.eeg_dataset_root)
    print("EEG PSG file:", cfg.eeg_psg_file)
    print("EEG hypnogram file:", cfg.eeg_hypnogram_file)
    print("RNG file:", cfg.rng_file)
    print("EEG state columns:", cfg.eeg_state_columns)
    print("RNG state columns:", cfg.rng_state_columns)
    print("Include stage in joint state:", cfg.include_stage_code_in_joint_state)
    print("Alignment mode:", cfg.alignment_mode)
    print("F3 holdout mode:", cfg.f3_holdout_mode)

    # -----------------------------------------------------
    # Load joint data
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Loading JOINT dataset...")

    df_joint_raw = load_joint(
        eeg_dataset_root=cfg.eeg_dataset_root,
        eeg_psg_file=cfg.eeg_psg_file,
        eeg_hypnogram_file=cfg.eeg_hypnogram_file,
        eeg_channel_name=cfg.eeg_channel_name,
        eeg_epoch_seconds=cfg.eeg_epoch_seconds,
        eeg_max_epochs=cfg.eeg_max_epochs,
        eeg_allowed_stages=cfg.eeg_allowed_stages,
        eeg_state_columns=cfg.eeg_state_columns,
        include_stage_code_in_joint_state=cfg.include_stage_code_in_joint_state,
        rng_file=cfg.rng_file,
        rng_sequence_length=cfg.rng_sequence_length,
        rng_use_bits=cfg.rng_use_bits,
        rng_state_window=cfg.rng_state_window,
        rng_state_columns=cfg.rng_state_columns,
        alignment_mode=cfg.alignment_mode,
        verbose=cfg.verbose,
    )

    print(f"[Phase3-Joint] Aligned observations: {len(df_joint_raw)}")
    print(f"[Phase3-Joint] Raw columns: {list(df_joint_raw.columns)}")

    if len(df_joint_raw) < 10:
        raise RuntimeError("Too few aligned observations for joint phase")

    # -----------------------------------------------------
    # Epoch split
    # -----------------------------------------------------

    idx_train_epochs, idx_test_epochs = _make_epoch_split(cfg, len(df_joint_raw))

    print(
        f"[Phase3-Joint] Epoch split: train={len(idx_train_epochs)} | test={len(idx_test_epochs)}"
    )

    # -----------------------------------------------------
    # Build joint state components
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Building joint state components...")

    df_joint_comp = _build_joint_dataframe(
        df_joint_raw,
        cfg,
        idx_train_epochs,
        use_sensitivity_quantiles=False,
    )

    print(f"[Phase3-Joint] Joint component columns: {list(df_joint_comp.columns)}")

    comps = df_joint_comp.to_numpy(dtype=int)
    n_components = comps.shape[1]

    # Effective base must accommodate all component alphabets.
    # Since EEG bins are 3 and RNG bins are 2, we encode the full joint state
    # using base = max(eeg_n_bins, rng_n_bins), while validating component values.
    effective_n_bins = max(cfg.eeg_n_bins, cfg.rng_n_bins)

    if np.any((comps < 0) | (comps >= effective_n_bins)):
        raise RuntimeError("Joint components out of encoding range")

    enc = make_encoding(
        n_components=n_components,
        n_bins=effective_n_bins,
    )

    n_states = effective_n_bins ** n_components

    print(
        f"[Phase3-Joint] State space: "
        f"{n_components} variables x {effective_n_bins} bins = {n_states} states"
    )

    ids = encode_states(comps, enc)

    # -----------------------------------------------------
    # Transitions
    # -----------------------------------------------------

    curr_all, nxt_all = build_transitions(ids)

    if len(curr_all) < 10:
        raise RuntimeError("Too few transitions after joint state encoding")

    print(f"[Phase3-Joint] Total transitions: {len(curr_all)}")

    idx_train_trans, idx_test_trans = _transition_indices_from_epoch_split(
        n_epochs=len(df_joint_raw),
        idx_train_epochs=idx_train_epochs,
        idx_test_epochs=idx_test_epochs,
    )

    curr_train = curr_all[idx_train_trans]
    nxt_train = nxt_all[idx_train_trans]

    curr_test = curr_all[idx_test_trans]
    nxt_test = nxt_all[idx_test_trans]

    print(
        f"[Phase3-Joint] Leakage-safe transitions: "
        f"train={len(curr_train)} | test={len(curr_test)}"
    )

    # -----------------------------------------------------
    # Empirical Kernel
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Building empirical baseline kernel P0...")

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=cfg.dirichlet_alpha,
    )

    print("[Phase3-Joint] P0 built.")

    eps_grid = np.array(cfg.eps_grid, dtype=float)

    # -----------------------------------------------------
    # Estimate epsilon (train)
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Estimating epsilon (train)...")

    eps_hat_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        cfg.min_prob,
        label="joint_train",
        progress_every=10,
    )

    print(f"[Phase3-Joint] eps_hat_train = {eps_hat_train:.4f}")

    # -----------------------------------------------------
    # Estimate epsilon (test)
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Estimating epsilon (test)...")

    eps_hat_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        cfg.min_prob,
        label="joint_test",
        progress_every=10,
    )

    print(f"[Phase3-Joint] eps_hat_test = {eps_hat_test:.4f}")

    # -----------------------------------------------------
    # F1 — Injection recovery
    # -----------------------------------------------------

    print("\n[Phase3-Joint] Running Gate F1 (injection recovery)...")

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
        label="joint_injection",
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

    print("\n[Phase3-Joint] Running Gate F2 (controls collapse)...")

    eps_controls = run_joint_controls(
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

    print("\n[Phase3-Joint] Running Gate F3 (holdout generalization)...")

    df_joint_comp_f3 = _build_joint_dataframe(
        df_joint_raw,
        cfg,
        idx_train_epochs,
        use_sensitivity_quantiles=False,
    )

    comps_f3 = df_joint_comp_f3.to_numpy(dtype=int)

    if np.any((comps_f3 < 0) | (comps_f3 >= effective_n_bins)):
        raise RuntimeError("F3 joint components out of encoding range")

    ids_f3 = encode_states(comps_f3, enc)

    curr_f3_all, nxt_f3_all = build_transitions(ids_f3)

    idx_train_trans_f3, idx_test_trans_f3 = _transition_indices_from_epoch_split(
        n_epochs=len(df_joint_raw),
        idx_train_epochs=idx_train_epochs,
        idx_test_epochs=idx_test_epochs,
    )

    curr_train_f3 = curr_f3_all[idx_train_trans_f3]
    nxt_train_f3 = nxt_f3_all[idx_train_trans_f3]

    curr_test_f3 = curr_f3_all[idx_test_trans_f3]
    nxt_test_f3 = nxt_f3_all[idx_test_trans_f3]

    print(
        f"[Phase3-Joint] F3 transitions: "
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
        label="joint_f3_train",
        progress_every=10,
    )

    eps_hat_test_f3, _ = _estimate_epsilon_grid(
        curr_test_f3,
        nxt_test_f3,
        P0_f3,
        eps_grid,
        cfg.min_prob,
        label="joint_f3_test",
        progress_every=10,
    )

    print(
        f"[Phase3-Joint] F3 eps_train = {eps_hat_train_f3:.4f} | "
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

    print("\n[Phase3-Joint] Running Gate F5 (sensitivity)...")

    df_joint_comp_alt = _build_joint_dataframe(
        df_joint_raw,
        cfg,
        idx_train_epochs,
        use_sensitivity_quantiles=True,
    )

    comps_alt = df_joint_comp_alt.to_numpy(dtype=int)

    if np.any((comps_alt < 0) | (comps_alt >= effective_n_bins)):
        raise RuntimeError("Sensitivity joint components out of encoding range")

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
        label="joint_bins_alt",
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
    print("CDR Phase III (JOINT EEG + RNG) — Gates")
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
            "eeg_state_columns": list(cfg.eeg_state_columns),
            "rng_state_columns": list(cfg.rng_state_columns),
            "include_stage_code_in_joint_state": bool(cfg.include_stage_code_in_joint_state),
            "eeg_n_bins": int(cfg.eeg_n_bins),
            "rng_n_bins": int(cfg.rng_n_bins),
            "effective_n_bins": int(effective_n_bins),
            "alignment_mode": cfg.alignment_mode,
            "train_ratio": float(cfg.train_ratio),
            "rng_use_bits": bool(cfg.rng_use_bits),
            "rng_sequence_length": int(cfg.rng_sequence_length),
            "f3_holdout_mode": cfg.f3_holdout_mode,
            "f3_eps_hat_train": float(eps_hat_train_f3),
            "f3_eps_hat_test": float(eps_hat_test_f3),
        },
    }

    out_file = cfg.results_dir / "phase3_joint_results.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {out_file}")


# ---------------------------------------------------------

if __name__ == "__main__":
    run_phase3_joint()