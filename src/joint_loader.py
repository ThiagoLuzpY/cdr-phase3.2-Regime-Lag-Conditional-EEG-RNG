from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.eeg_loader import load_eeg
from src.rng_loader import load_rng


# ---------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------

@dataclass
class JointConfig:
    # EEG
    eeg_dataset_root: Path = Path("data/raw/eeg")
    eeg_psg_file: str = "SC4001E0-PSG.edf"
    eeg_hypnogram_file: str = "SC4001EC-Hypnogram.edf"
    eeg_channel_name: str = "EEG Fpz-Cz"
    eeg_epoch_seconds: int = 30
    eeg_max_epochs: Optional[int] = None
    eeg_allowed_stages: Tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    )
    eeg_state_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
    )
    include_stage_code_in_joint_state: bool = False

    # RNG
    rng_file: Path = Path("data/raw/rng/anu_sample.json")
    rng_sequence_length: int = 1024
    rng_use_bits: bool = True
    rng_state_window: int = 2
    rng_state_columns: Tuple[str, ...] = (
        "x0",
        "x1",
    )

    # Alignment
    alignment_mode: str = "truncate_to_shortest"

    # Logging
    verbose: int = 1


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

class JointLoader:

    def __init__(self, cfg: JointConfig):
        self.cfg = cfg

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg)

    def _load_eeg(self) -> pd.DataFrame:
        self._log("[JointLoader] Loading EEG branch...")

        df_eeg = load_eeg(
            dataset_root=self.cfg.eeg_dataset_root,
            psg_file=self.cfg.eeg_psg_file,
            hypnogram_file=self.cfg.eeg_hypnogram_file,
            channel_name=self.cfg.eeg_channel_name,
            epoch_seconds=self.cfg.eeg_epoch_seconds,
            max_epochs=self.cfg.eeg_max_epochs,
            allowed_stages=self.cfg.eeg_allowed_stages,
            verbose=self.cfg.verbose,
        )

        required_cols = list(self.cfg.eeg_state_columns)
        if self.cfg.include_stage_code_in_joint_state:
            required_cols.append("stage_code")

        missing = [c for c in required_cols if c not in df_eeg.columns]
        if missing:
            raise RuntimeError(f"EEG joint loader missing columns: {missing}")

        out = df_eeg.copy()

        # Prefix EEG columns to avoid any ambiguity in the joint dataframe
        rename_map = {}
        for col in self.cfg.eeg_state_columns:
            rename_map[col] = f"eeg_{col}"

        if self.cfg.include_stage_code_in_joint_state:
            rename_map["stage_code"] = "eeg_stage_code"

        # Keep useful metadata too
        for meta_col in ["stage_label", "epoch_idx", "onset_sec"]:
            if meta_col in out.columns:
                rename_map[meta_col] = f"eeg_{meta_col}"

        out = out.rename(columns=rename_map)

        keep_cols = list(rename_map.values())
        out = out[keep_cols].reset_index(drop=True)

        self._log(f"[JointLoader] EEG rows loaded: {len(out)}")
        self._log(f"[JointLoader] EEG columns: {list(out.columns)}")

        return out

    def _load_rng(self) -> pd.DataFrame:
        self._log("[JointLoader] Loading RNG branch...")

        df_rng = load_rng(
            rng_file=self.cfg.rng_file,
            sequence_length=self.cfg.rng_sequence_length,
            state_window=self.cfg.rng_state_window,
            use_bits=self.cfg.rng_use_bits,
            verbose=self.cfg.verbose,
        )

        required_cols = list(self.cfg.rng_state_columns)
        missing = [c for c in required_cols if c not in df_rng.columns]
        if missing:
            raise RuntimeError(f"RNG joint loader missing columns: {missing}")

        out = df_rng.copy()

        # Prefix RNG columns to avoid ambiguity
        rename_map = {}
        for col in self.cfg.rng_state_columns:
            rename_map[col] = f"rng_{col}"

        if "window_start" in out.columns:
            rename_map["window_start"] = "rng_window_start"

        out = out.rename(columns=rename_map)

        keep_cols = list(rename_map.values())
        out = out[keep_cols].reset_index(drop=True)

        self._log(f"[JointLoader] RNG rows loaded: {len(out)}")
        self._log(f"[JointLoader] RNG columns: {list(out.columns)}")

        return out

    def _align_truncate_to_shortest(
        self,
        df_eeg: pd.DataFrame,
        df_rng: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simple, falsifiable alignment:
        truncate both streams to the same length using the shortest one.

        We do NOT fabricate timing synchronization here.
        We only build a paired observational stream of equal length.
        """
        n = min(len(df_eeg), len(df_rng))

        if n < 10:
            raise RuntimeError(
                f"Too few aligned samples for joint domain: EEG={len(df_eeg)}, RNG={len(df_rng)}, aligned={n}"
            )

        eeg_trim = df_eeg.iloc[:n].reset_index(drop=True)
        rng_trim = df_rng.iloc[:n].reset_index(drop=True)

        joint = pd.concat([eeg_trim, rng_trim], axis=1)

        joint["joint_index"] = np.arange(n, dtype=int)

        return joint

    def load(self) -> pd.DataFrame:
        if self.cfg.alignment_mode != "truncate_to_shortest":
            raise ValueError(
                f"Unsupported alignment_mode: {self.cfg.alignment_mode}"
            )

        df_eeg = self._load_eeg()
        df_rng = self._load_rng()

        self._log("[JointLoader] Aligning EEG and RNG streams...")

        df_joint = self._align_truncate_to_shortest(df_eeg, df_rng)

        df_joint = df_joint.replace([np.inf, -np.inf], np.nan)
        df_joint = df_joint.dropna().reset_index(drop=True)

        self._log(f"[JointLoader] Joint rows: {len(df_joint)}")
        self._log(f"[JointLoader] Joint columns: {list(df_joint.columns)}")

        return df_joint


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_joint(
    eeg_dataset_root: Path = Path("data/raw/eeg"),
    eeg_psg_file: str = "SC4001E0-PSG.edf",
    eeg_hypnogram_file: str = "SC4001EC-Hypnogram.edf",
    eeg_channel_name: str = "EEG Fpz-Cz",
    eeg_epoch_seconds: int = 30,
    eeg_max_epochs: Optional[int] = None,
    eeg_allowed_stages: Tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    ),
    eeg_state_columns: Tuple[str, ...] = (
        "delta_power",
        "alpha_power",
    ),
    include_stage_code_in_joint_state: bool = False,
    rng_file: Path = Path("data/raw/rng/anu_sample.json"),
    rng_sequence_length: int = 1024,
    rng_use_bits: bool = True,
    rng_state_window: int = 2,
    rng_state_columns: Tuple[str, ...] = (
        "x0",
        "x1",
    ),
    alignment_mode: str = "truncate_to_shortest",
    verbose: int = 1,
) -> pd.DataFrame:
    cfg = JointConfig(
        eeg_dataset_root=eeg_dataset_root,
        eeg_psg_file=eeg_psg_file,
        eeg_hypnogram_file=eeg_hypnogram_file,
        eeg_channel_name=eeg_channel_name,
        eeg_epoch_seconds=eeg_epoch_seconds,
        eeg_max_epochs=eeg_max_epochs,
        eeg_allowed_stages=eeg_allowed_stages,
        eeg_state_columns=eeg_state_columns,
        include_stage_code_in_joint_state=include_stage_code_in_joint_state,
        rng_file=rng_file,
        rng_sequence_length=rng_sequence_length,
        rng_use_bits=rng_use_bits,
        rng_state_window=rng_state_window,
        rng_state_columns=rng_state_columns,
        alignment_mode=alignment_mode,
        verbose=verbose,
    )

    loader = JointLoader(cfg)

    df = loader.load()

    return df