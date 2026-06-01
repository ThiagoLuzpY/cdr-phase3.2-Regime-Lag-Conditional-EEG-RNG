import numpy as np

from src.phase1_plus_runner import Phase1PlusConfig, Phase1PlusValidator


def test_phase1_plus_runs_fast_smoke():
    # Smaller config for tests (fast)
    cfg = Phase1PlusConfig(
        n_steps=200,
        stability_steps_alt=160,
        n_reps=3,
        eps_grid=tuple(np.linspace(0.0, 0.6, 31)),
        g_grid=tuple(np.linspace(0.0, 0.20, 11)),
    )
    v = Phase1PlusValidator(cfg)
    summary = v.run_phase1_plus(out_dir="results/_test_phase1_plus")

    assert "phase1plus_pass" in summary
    assert "gates" in summary
    assert set(summary["gates"].keys()) >= {
        "G1_H0_recovery",
        "G2_H1_recovery",
        "G3_controls_collapse",
        "G4_identifiability",
        "G5_stability",
        "G6_adversarial",
        "G7_out_of_sample",
    }