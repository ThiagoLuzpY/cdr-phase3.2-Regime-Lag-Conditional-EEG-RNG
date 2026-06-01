from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import mne
except ImportError as e:
    raise ImportError(
        "mne is required for eeg_loader.py. Install it with: pip install mne"
    ) from e


# ---------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------

@dataclass
class EEGConfig:
    dataset_root: Path = Path("data/raw/eeg")

    psg_file: str = "SC4001E0-PSG.edf"
    hypnogram_file: str = "SC4001EC-Hypnogram.edf"

    channel_name: str = "EEG Fpz-Cz"

    epoch_seconds: int = 30
    max_epochs: Optional[int] = None

    allowed_stages: tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    )

    delta_band: Tuple[float, float] = (0.5, 4.0)
    theta_band: Tuple[float, float] = (4.0, 8.0)
    alpha_band: Tuple[float, float] = (8.0, 12.0)
    beta_band: Tuple[float, float] = (12.0, 30.0)

    log_bandpower: bool = True

    verbose: int = 1


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

class EEGLoader:

    def __init__(self, cfg: EEGConfig):
        self.cfg = cfg

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg)

    def _resolve_paths(self) -> tuple[Path, Path]:
        psg_path = self.cfg.dataset_root / self.cfg.psg_file
        hyp_path = self.cfg.dataset_root / self.cfg.hypnogram_file

        if not psg_path.exists():
            raise FileNotFoundError(f"PSG file not found: {psg_path}")

        if not hyp_path.exists():
            raise FileNotFoundError(f"Hypnogram file not found: {hyp_path}")

        return psg_path, hyp_path

    def _stage_map(self) -> Dict[str, int]:
        return {
            "Sleep stage W": 0,
            "Sleep stage 1": 1,
            "Sleep stage 2": 2,
            "Sleep stage 3": 3,
            "Sleep stage 4": 3,  # merge N3/N4 into same coarse deep-sleep code
            "Sleep stage R": 4,
        }

    def _bandpower(self, x: np.ndarray, sfreq: float, fmin: float, fmax: float) -> float:
        freqs = np.fft.rfftfreq(len(x), d=1.0 / sfreq)
        psd = np.abs(np.fft.rfft(x)) ** 2

        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            return 0.0

        p = float(np.sum(psd[mask]))

        if self.cfg.log_bandpower:
            return float(np.log1p(max(p, 0.0)))

        return p

    def _extract_epoch_features(
        self,
        raw: mne.io.BaseRaw,
        ann: mne.Annotations,
    ) -> pd.DataFrame:
        sfreq = float(raw.info["sfreq"])

        if self.cfg.channel_name not in raw.ch_names:
            raise ValueError(
                f"Channel '{self.cfg.channel_name}' not found. Available channels: {raw.ch_names}"
            )

        stage_map = self._stage_map()

        rows = []
        n_kept = 0

        for k, desc in enumerate(ann.description):
            if desc not in self.cfg.allowed_stages:
                continue

            onset_sec = float(ann.onset[k])
            duration_sec = float(ann.duration[k])

            if duration_sec < self.cfg.epoch_seconds:
                continue

            start_sample = int(round(onset_sec * sfreq))
            stop_sample = start_sample + int(round(self.cfg.epoch_seconds * sfreq))

            data = raw.get_data(
                picks=[self.cfg.channel_name],
                start=start_sample,
                stop=stop_sample,
            )

            if data.shape[1] == 0:
                continue

            x = data[0].astype(float)

            if len(x) < 4:
                continue

            # Remove DC component
            x = x - np.mean(x)

            delta_power = self._bandpower(
                x, sfreq, self.cfg.delta_band[0], self.cfg.delta_band[1]
            )
            theta_power = self._bandpower(
                x, sfreq, self.cfg.theta_band[0], self.cfg.theta_band[1]
            )
            alpha_power = self._bandpower(
                x, sfreq, self.cfg.alpha_band[0], self.cfg.alpha_band[1]
            )
            beta_power = self._bandpower(
                x, sfreq, self.cfg.beta_band[0], self.cfg.beta_band[1]
            )

            rows.append(
                {
                    "delta_power": delta_power,
                    "theta_power": theta_power,
                    "alpha_power": alpha_power,
                    "beta_power": beta_power,
                    "stage_code": stage_map[desc],
                    "stage_label": desc,
                    "epoch_idx": n_kept,
                    "onset_sec": onset_sec,
                }
            )

            n_kept += 1

            if self.cfg.max_epochs is not None and n_kept >= self.cfg.max_epochs:
                break

        if len(rows) == 0:
            raise RuntimeError("No EEG epochs extracted from EDF annotations")

        df = pd.DataFrame(rows)

        return df

    def load(self) -> pd.DataFrame:
        self._log("[EEGLoader] Loading Sleep-EDF files...")

        psg_path, hyp_path = self._resolve_paths()

        self._log(f"[EEGLoader] PSG: {psg_path.name}")
        self._log(f"[EEGLoader] Hypnogram: {hyp_path.name}")

        raw = mne.io.read_raw_edf(
            str(psg_path),
            preload=True,
            verbose="ERROR",
        )

        ann = mne.read_annotations(str(hyp_path))

        df = self._extract_epoch_features(raw, ann)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(
            subset=[
                "delta_power",
                "theta_power",
                "alpha_power",
                "beta_power",
                "stage_code",
            ]
        ).reset_index(drop=True)

        self._log(f"[EEGLoader] Loaded {len(df)} epochs")
        self._log(f"[EEGLoader] Columns: {list(df.columns)}")

        return df


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_eeg(
    dataset_root: Path = Path("data/raw/eeg"),
    psg_file: str = "SC4001E0-PSG.edf",
    hypnogram_file: str = "SC4001EC-Hypnogram.edf",
    channel_name: str = "EEG Fpz-Cz",
    epoch_seconds: int = 30,
    max_epochs: Optional[int] = None,
    allowed_stages: tuple[str, ...] = (
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    ),
    delta_band: Tuple[float, float] = (0.5, 4.0),
    theta_band: Tuple[float, float] = (4.0, 8.0),
    alpha_band: Tuple[float, float] = (8.0, 12.0),
    beta_band: Tuple[float, float] = (12.0, 30.0),
    log_bandpower: bool = True,
    verbose: int = 1,
) -> pd.DataFrame:
    cfg = EEGConfig(
        dataset_root=dataset_root,
        psg_file=psg_file,
        hypnogram_file=hypnogram_file,
        channel_name=channel_name,
        epoch_seconds=epoch_seconds,
        max_epochs=max_epochs,
        allowed_stages=allowed_stages,
        delta_band=delta_band,
        theta_band=theta_band,
        alpha_band=alpha_band,
        beta_band=beta_band,
        log_bandpower=log_bandpower,
        verbose=verbose,
    )

    loader = EEGLoader(cfg)

    df = loader.load()

    return df