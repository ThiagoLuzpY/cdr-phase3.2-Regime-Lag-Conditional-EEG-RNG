from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np

from config.phase2_config import CFG
from src.artifacts import ensure_dir, write_json, write_text, plot_curve
from src.build_states import (
    make_encoding,
    build_components_matrix,
    encode_states,
    build_transitions,
)
from src.discretize import fit_and_discretize, apply_specs
from src.controls_phase2 import (
    shuffle_rows_global_df,
    shuffle_columns_independently_df,
    shuffle_week_blocks_df,
    shuffle_within_hour_weektype_df,
)
from src.kernels.empirical_kernel import EmpiricalKernel
from src.kernels.reweighted_kernel import ReweightedKernel
from src.opsp_loader import resolve_columns, load_timeseries, quick_report
from src.validators_phase2 import (
    GateResult,
    gate_F1_injection_recovery,
    gate_F2_controls_collapse,
    gate_F3_holdout_generalization,
    gate_F5_sensitivity,
    summarize,
)


def _make_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _make_serializable(asdict(obj))
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def _estimate_epsilon_grid(
    curr: np.ndarray,
    nxt: np.ndarray,
    base: EmpiricalKernel,
    eps_grid: np.ndarray,
    min_prob: float,
    *,
    label: str = "",
    progress_every: int = 10,
) -> Tuple[float, np.ndarray]:

    ll = np.zeros_like(eps_grid, dtype=float)
    n_grid = int(len(eps_grid))

    if label:
        print(f"[Phase2] Estimating epsilon on {label} (grid={n_grid})...")

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
        print(f"[Phase2] Done epsilon({label}): eps_hat={eps_hat:.4f}")

    return eps_hat, ll


def _simulate_trajectory(
    base: EmpiricalKernel,
    eps: float,
    n_steps: int,
    seed: int,
) -> np.ndarray:

    rng = np.random.default_rng(seed)
    k = ReweightedKernel(base=base, epsilon=float(eps), min_prob=1e-12)

    s = int(rng.integers(0, base.n_states))
    traj = [s]

    for _ in range(n_steps):
        s = k.sample_next(s, rng)
        traj.append(s)

    return np.array(traj, dtype=int)


def run_phase2() -> None:

    root = Path(__file__).resolve().parents[1]
    out_dir = root / CFG.results_dir

    ensure_dir(out_dir)

    print("[Phase2] Starting CDR Phase II runner")

    csv_path = root / CFG.data.csv_path

    sel = resolve_columns(
        csv_path=csv_path,
        country=CFG.data.country,
        patterns=CFG.data.patterns,
        explicit=CFG.data.explicit_columns,
    )

    df = load_timeseries(csv_path, sel, start=CFG.data.start, end=CFG.data.end)

    rep = quick_report(df)

    if CFG.disc.missing_policy == "drop":
        df = df.dropna()
    else:
        raise ValueError("Only drop supported")

    write_json(out_dir / "data_report.json", _make_serializable(rep))
    write_json(out_dir / "selection.json", _make_serializable({"country": sel.country, "columns": sel.columns}))
    write_json(out_dir / "phase2_config.json", _make_serializable({"CFG": CFG}))

    n = len(df)
    split = int(0.75 * n)

    idx_train = np.arange(0, split, dtype=int)
    idx_test = np.arange(split, n, dtype=int)

    df_bins, specs = fit_and_discretize(
        df,
        n_bins=CFG.disc.n_bins,
        quantiles=CFG.disc.quantiles,
        fit_on_index=idx_train,
    )

    comp_order = ["load", "wind", "solar"] + (["price"] if "price" in df_bins.columns else [])

    components = build_components_matrix(df_bins, comp_order)

    enc = make_encoding(n_components=components.shape[1], n_bins=CFG.disc.n_bins)

    n_states = CFG.disc.n_bins ** enc.n_components

    state_ids = encode_states(components, enc)

    curr_all, nxt_all = build_transitions(state_ids)

    curr_train = curr_all[: split - 1]
    nxt_train = nxt_all[: split - 1]

    curr_test = curr_all[split - 1 :]
    nxt_test = nxt_all[split - 1 :]

    P0 = EmpiricalKernel.from_transitions(
        curr_train,
        nxt_train,
        n_states=n_states,
        enc=enc,
        alpha=CFG.kernel.dirichlet_alpha,
    )

    eps_grid = np.array(list(CFG.kernel.eps_grid), dtype=float)

    eps_hat_train, ll_train = _estimate_epsilon_grid(
        curr_train,
        nxt_train,
        P0,
        eps_grid,
        CFG.kernel.min_prob,
        label="train",
    )

    eps_hat_test, ll_test = _estimate_epsilon_grid(
        curr_test,
        nxt_test,
        P0,
        eps_grid,
        CFG.kernel.min_prob,
        label="test",
    )

    sim_traj = _simulate_trajectory(P0, eps=CFG.gates.inj_eps_true, n_steps=len(curr_train), seed=12345)

    sim_curr, sim_nxt = build_transitions(sim_traj)

    eps_hat_inj, ll_inj = _estimate_epsilon_grid(
        sim_curr,
        sim_nxt,
        P0,
        eps_grid,
        CFG.kernel.min_prob,
        label="injection",
    )

    print("[Phase2] Generating control datasets")

    control_builders = {
        "rows_shuffle": lambda d, s: shuffle_rows_global_df(d, s),
        "columns_shuffle": lambda d, s: shuffle_columns_independently_df(d, s),
        "weekly_blocks": lambda d, s: shuffle_week_blocks_df(d, 168, s),
        "seasonal_strata": lambda d, s: shuffle_within_hour_weektype_df(d, s),
    }

    eps_controls = {}

    for name, builder in control_builders.items():

        seed = CFG.control_seeds[0]

        print(f"[Phase2] Running control: {name}")

        df_ctrl = builder(df, seed)

        df_ctrl_bins = apply_specs(df_ctrl, specs)

        comps_ctrl = build_components_matrix(df_ctrl_bins, comp_order)

        ids_ctrl = encode_states(comps_ctrl, enc)

        c_ctrl, n_ctrl = build_transitions(ids_ctrl)

        eps_hat_ctrl, _ = _estimate_epsilon_grid(
            c_ctrl[: split - 1],
            n_ctrl[: split - 1],
            P0,
            eps_grid,
            CFG.kernel.min_prob,
            label=f"control_{name}",
        )

        eps_controls[name] = eps_hat_ctrl

    write_json(out_dir / "checkpoint_controls.json", _make_serializable(eps_controls))

    eps_controls_collapse = [
        eps_controls[c] for c in CFG.collapse_controls if c in eps_controls
    ]

    bins4 = CFG.gates.sensitivity_bins4

    df_bins4, _ = fit_and_discretize(
        df,
        n_bins=bins4,
        quantiles=CFG.disc.quantiles,
        fit_on_index=idx_train,
    )

    comps4 = build_components_matrix(df_bins4, comp_order)

    enc4 = make_encoding(n_components=comps4.shape[1], n_bins=bins4)

    ids4 = encode_states(comps4, enc4)

    c4, n4 = build_transitions(ids4)

    c4_train = c4[: split - 1]
    n4_train = n4[: split - 1]

    P0_4 = EmpiricalKernel.from_transitions(
        c4_train,
        n4_train,
        n_states=bins4 ** enc4.n_components,
        enc=enc4,
        alpha=CFG.kernel.dirichlet_alpha,
    )

    eps_hat_bins4, _ = _estimate_epsilon_grid(
        c4_train,
        n4_train,
        P0_4,
        eps_grid,
        CFG.kernel.min_prob,
        label="bins4_train",
    )

    gates: List[GateResult] = []

    gates.append(
        gate_F1_injection_recovery(
            eps_hat_inj,
            CFG.gates.inj_eps_true,
            CFG.gates.inj_tol_abs,
        )
    )

    gates.append(
        gate_F2_controls_collapse(
            eps_controls_collapse,
            CFG.gates.controls_tol,
            CFG.gates.controls_required_fraction,
        )
    )

    gates.append(
        gate_F3_holdout_generalization(
            eps_hat_train,
            eps_hat_test,
            CFG.gates.holdout_max_delta,
        )
    )

    gates.append(
        gate_F5_sensitivity(
            eps_hat_train,
            eps_hat_bins4,
            CFG.gates.sensitivity_max_delta,
        )
    )

    summary = summarize(gates)

    write_json(
        out_dir / "phase2_results.json",
        _make_serializable(
            {
                "eps_hat_train": eps_hat_train,
                "eps_hat_test": eps_hat_test,
                "eps_hat_injection": eps_hat_inj,
                "eps_controls": eps_controls,
                "eps_hat_bins4": eps_hat_bins4,
                "gates": summary,
            }
        ),
    )

    plot_curve(out_dir, eps_grid, ll_train, "Train", "ll_train.png")
    plot_curve(out_dir, eps_grid, ll_test, "Test", "ll_test.png")
    plot_curve(out_dir, eps_grid, ll_inj, "Injection", "ll_injection.png")

    lines = []

    lines.append("CDR Phase II — Gates")
    lines.append("────────────────────────────────")

    for g in gates:
        lines.append(f"{g.name}: {'PASS' if g.passed else 'FAIL'}")

    lines.append("────────────────────────────────")
    lines.append(f"FINAL: {'PASS' if summary['passed_all'] else 'FAIL'}")

    report = "\n".join(lines)

    write_text(out_dir / "report.txt", report)

    print(report)

    print("[Phase2] Done")


if __name__ == "__main__":
    run_phase2()