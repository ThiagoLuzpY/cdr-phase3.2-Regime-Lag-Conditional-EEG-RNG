from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import mdtraj as md
except ImportError as e:
    raise ImportError(
        "mdtraj is required for protein_loader.py. Install it with: pip install mdtraj"
    ) from e


# ---------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------

@dataclass
class ProteinConfig:
    dataset_root: Path = Path("data/raw/protein")

    pdb_file: str = "alanine-dipeptide-nowater.pdb"

    xtc_files: tuple[str, ...] = (
        "alanine-dipeptide-0-250ns-nowater.xtc",
        "alanine-dipeptide-1-250ns-nowater.xtc",
        "alanine-dipeptide-2-250ns-nowater.xtc",
    )

    frame_stride: int = 10
    max_frames_per_traj: Optional[int] = None
    verbose: int = 1


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

class ProteinLoader:

    def __init__(self, cfg: ProteinConfig):
        self.cfg = cfg

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg)

    def _resolve_topology_path(self) -> Path:
        path = self.cfg.dataset_root / self.cfg.pdb_file
        if not path.exists():
            raise FileNotFoundError(f"PDB file not found: {path}")
        return path

    def _resolve_trajectory_paths(self) -> List[Path]:
        files: List[Path] = []

        for fname in self.cfg.xtc_files:
            path = self.cfg.dataset_root / fname
            if not path.exists():
                raise FileNotFoundError(f"XTC file not found: {path}")
            files.append(path)

        if len(files) == 0:
            raise RuntimeError("No XTC files found")

        return files

    def _load_single_trajectory(self, xtc_path: Path, top_path: Path) -> pd.DataFrame:
        self._log(f"[ProteinLoader] Loading trajectory: {xtc_path.name}")

        traj = md.load_xtc(str(xtc_path), top=str(top_path), stride=self.cfg.frame_stride)

        if self.cfg.max_frames_per_traj is not None:
            traj = traj[: self.cfg.max_frames_per_traj]

        if traj.n_frames < 3:
            raise RuntimeError(
                f"Trajectory {xtc_path.name} has too few frames after stride/max_frames filtering"
            )

        # MDTraj returns radians for phi/psi
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)

        if phi.shape[1] < 1 or psi.shape[1] < 1:
            raise RuntimeError(
                f"Could not compute valid phi/psi angles for trajectory {xtc_path.name}"
            )

        # Alanine dipeptide should expose a single relevant phi and psi trace
        phi_main = phi[:, 0]
        psi_main = psi[:, 0]

        # Keep angles in radians to preserve physical continuity and sign structure
        df = pd.DataFrame(
            {
                "phi": phi_main.astype(float),
                "psi": psi_main.astype(float),
            }
        )

        # Optional derived quantities kept for diagnostics only if needed later
        df["traj_file"] = xtc_path.name
        df["frame_idx"] = np.arange(len(df), dtype=int)

        return df

    def load(self) -> pd.DataFrame:
        self._log("[ProteinLoader] Loading protein trajectories...")

        top_path = self._resolve_topology_path()
        xtc_paths = self._resolve_trajectory_paths()

        segments: List[pd.DataFrame] = []

        for xtc_path in xtc_paths:
            try:
                seg = self._load_single_trajectory(xtc_path, top_path)
                segments.append(seg)
            except Exception as e:
                self._log(f"[ProteinLoader] Skipping {xtc_path.name}: {e}")

        if len(segments) == 0:
            raise RuntimeError("No protein trajectories loaded")

        df = pd.concat(segments, ignore_index=True)

        # Preserve only the state variables for the main pipeline
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["phi", "psi"]).reset_index(drop=True)

        self._log(f"[ProteinLoader] Loaded {len(df)} observations")
        self._log(f"[ProteinLoader] Columns: {list(df.columns)}")

        return df


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_protein(
    dataset_root: Path = Path("data/raw/protein"),
    pdb_file: str = "alanine-dipeptide-nowater.pdb",
    xtc_files: tuple[str, ...] = (
        "alanine-dipeptide-0-250ns-nowater.xtc",
        "alanine-dipeptide-1-250ns-nowater.xtc",
        "alanine-dipeptide-2-250ns-nowater.xtc",
    ),
    frame_stride: int = 10,
    max_frames_per_traj: Optional[int] = None,
    verbose: int = 1,
) -> pd.DataFrame:
    cfg = ProteinConfig(
        dataset_root=dataset_root,
        pdb_file=pdb_file,
        xtc_files=xtc_files,
        frame_stride=frame_stride,
        max_frames_per_traj=max_frames_per_traj,
        verbose=verbose,
    )

    loader = ProteinLoader(cfg)

    df = loader.load()

    return df