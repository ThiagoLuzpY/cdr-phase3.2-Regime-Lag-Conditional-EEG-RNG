from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    clean = {k: _to_jsonable(v) for k, v in data.items()}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_seed_manifest(path: Path, seeds: Dict[str, Any]) -> None:
    write_json(path, {"seeds": seeds})


def plot_histograms(
    out_dir: Path,
    eps_h0: np.ndarray,
    eps_h1: np.ndarray,
    eps_controls: Optional[np.ndarray] = None,
) -> None:
    ensure_dir(out_dir)

    plt.figure()
    plt.hist(eps_h0, bins=15)
    plt.title("Epsilon-hat (H0)")
    plt.xlabel("epsilon_hat")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "eps_hat_h0_hist.png")
    plt.close()

    plt.figure()
    plt.hist(eps_h1, bins=15)
    plt.title("Epsilon-hat (H1)")
    plt.xlabel("epsilon_hat")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "eps_hat_h1_hist.png")
    plt.close()

    if eps_controls is not None:
        plt.figure()
        plt.hist(eps_controls, bins=10)
        plt.title("Epsilon-hat (Controls)")
        plt.xlabel("epsilon_hat")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "eps_hat_controls_hist.png")
        plt.close()


def plot_curve(out_dir: Path, x: np.ndarray, y: np.ndarray, title: str, fname: str, xlabel: str = "x", ylabel: str = "y") -> None:
    ensure_dir(out_dir)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / fname)
    plt.close()