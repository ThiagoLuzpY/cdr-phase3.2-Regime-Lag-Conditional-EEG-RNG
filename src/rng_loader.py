from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------

@dataclass
class RNGConfig:
    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    sequence_length: int = 1024
    state_window: int = 2
    use_bits: bool = True
    verbose: int = 1


# ---------------------------------------------------------
# Data container
# ---------------------------------------------------------

@dataclass(frozen=True)
class RNGData:
    """
    Container for RNG sequence data.
    """
    sequence: np.ndarray
    n_numbers: int
    source: str
    representation: str  # "uint8" or "bits"


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

class RNGLoader:

    def __init__(self, cfg: RNGConfig):
        self.cfg = cfg

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg)

    def _resolve_path(self) -> Path:
        if not self.cfg.rng_file.exists():
            raise FileNotFoundError(f"RNG file not found: {self.cfg.rng_file}")
        return self.cfg.rng_file

    def _convert_to_bits(self, sequence_uint8: np.ndarray) -> np.ndarray:
        """
        Convert uint8 sequence (0..255) into a flat binary sequence of bits (0/1).

        Example:
            [125, 235] -> [0,1,1,1,1,1,0,1, 1,1,1,0,1,0,1,1]
        """
        bits = np.unpackbits(sequence_uint8)
        return bits.astype(np.uint8)

    def _load_json(self, path: Path) -> RNGData:
        self._log(f"[RNGLoader] Loading RNG JSON: {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not raw.get("success", False):
            raise ValueError(f"RNG JSON indicates failure: {raw}")

        if "data" not in raw:
            raise ValueError("RNG JSON missing 'data' field")

        seq_uint8 = np.asarray(raw["data"], dtype=np.uint8).reshape(-1)

        if self.cfg.sequence_length is not None:
            seq_uint8 = seq_uint8[: self.cfg.sequence_length]

        if len(seq_uint8) < self.cfg.state_window + 2:
            raise RuntimeError(
                f"RNG sequence too short after truncation: "
                f"len={len(seq_uint8)}, state_window={self.cfg.state_window}"
            )

        if self.cfg.use_bits:
            seq = self._convert_to_bits(seq_uint8)
            representation = "bits"
        else:
            seq = seq_uint8.astype(np.uint8)
            representation = "uint8"

        if len(seq) < self.cfg.state_window + 2:
            raise RuntimeError(
                f"RNG sequence too short after representation conversion: "
                f"len={len(seq)}, state_window={self.cfg.state_window}, "
                f"representation={representation}"
            )

        source = "ANU"

        return RNGData(
            sequence=seq,
            n_numbers=len(seq),
            source=source,
            representation=representation,
        )

    def _construct_rng_dataframe(self, sequence: np.ndarray) -> pd.DataFrame:
        """
        Build a dataframe of overlapping sliding-window state components.

        For state_window = 2:
            x0_t = sequence[t]
            x1_t = sequence[t+1]

        This produces a component matrix that can later be discretized and encoded
        into state ids using the standard CDR pipeline.
        """
        n = len(sequence)
        w = self.cfg.state_window

        if w < 2:
            raise ValueError("state_window must be >= 2")

        rows = []

        for i in range(n - w + 1):
            row = {}

            for j in range(w):
                row[f"x{j}"] = int(sequence[i + j])

            row["window_start"] = i
            rows.append(row)

        if len(rows) < 3:
            raise RuntimeError("Too few RNG windows constructed")

        df = pd.DataFrame(rows)

        return df

    def load(self) -> pd.DataFrame:
        path = self._resolve_path()
        data = self._load_json(path)

        self._log(
            f"[RNGLoader] Loaded {data.n_numbers} values from {data.source} "
            f"using representation={data.representation}"
        )

        df = self._construct_rng_dataframe(data.sequence)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)

        self._log(f"[RNGLoader] Constructed {len(df)} RNG windows")
        self._log(f"[RNGLoader] Columns: {list(df.columns)}")

        return df


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_rng(
    rng_file: Path = Path("data/raw/rng/anu_sample.json"),
    sequence_length: int = 1024,
    state_window: int = 2,
    use_bits: bool = True,
    verbose: int = 1,
) -> pd.DataFrame:
    cfg = RNGConfig(
        rng_file=rng_file,
        sequence_length=sequence_length,
        state_window=state_window,
        use_bits=use_bits,
        verbose=verbose,
    )

    loader = RNGLoader(cfg)

    df = loader.load()

    return df