from src.phase1_plus_runner import Phase1PlusConfig, Phase1PlusValidator

def main() -> None:
    cfg = Phase1PlusConfig()  # defaults: n_steps=1000, n_reps=20
    runner = Phase1PlusValidator(cfg)
    summary = runner.run_phase1_plus(out_dir="results/phase1_plus_full")

    print("PASS:", summary["phase1plus_pass"])
    print("Saved to:", "results/phase1_plus_full")

if __name__ == "__main__":
    main()