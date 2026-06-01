from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

EARTH_RADIUS_M = 6371000


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Compute distance between two GPS coordinates in meters.
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS_M * c


def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute direction of travel in degrees.
    """
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def angle_difference(a, b):
    """
    Smallest difference between two angles.
    """
    diff = abs(a - b)
    return min(diff, 360 - diff)


# ---------------------------------------------------------
# Loader configuration
# ---------------------------------------------------------

@dataclass
class GeoLifeConfig:

    dataset_root: Path = Path("data/raw/geolife/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data")

    max_users: Optional[int] = None

    min_points_per_traj: int = 50

    sampling_seconds: Optional[int] = None

    verbose: int = 1


# ---------------------------------------------------------
# Loader
# ---------------------------------------------------------

class GeoLifeLoader:

    def __init__(self, cfg: GeoLifeConfig):
        self.cfg = cfg

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg)

    def _list_user_dirs(self) -> List[Path]:

        users = sorted(self.cfg.dataset_root.glob("*"))

        if self.cfg.max_users is not None:
            users = users[: self.cfg.max_users]

        return users

    def _load_trajectory(self, file: Path) -> pd.DataFrame:

        df = pd.read_csv(
            file,
            skiprows=6,
            header=None,
            names=[
                "lat",
                "lon",
                "unused",
                "altitude",
                "date_days",
                "date",
                "time",
            ],
        )

        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

        df = df[["lat", "lon", "datetime"]]

        df = df.sort_values("datetime")

        return df

    def _process_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:

        lat = df["lat"].values
        lon = df["lon"].values
        time = df["datetime"].values.astype("datetime64[s]").astype("int64")

        n = len(df)

        dist = np.zeros(n)
        speed = np.zeros(n)
        accel = np.zeros(n)
        bearing = np.zeros(n)
        turn = np.zeros(n)

        for i in range(1, n):

            dist[i] = haversine_distance(
                lat[i - 1],
                lon[i - 1],
                lat[i],
                lon[i],
            )

            dt = time[i] - time[i - 1]

            if dt > 0:
                speed[i] = dist[i] / dt

            bearing[i] = compute_bearing(
                lat[i - 1],
                lon[i - 1],
                lat[i],
                lon[i],
            )

        for i in range(2, n):

            dt = time[i] - time[i - 1]

            if dt > 0:
                accel[i] = (speed[i] - speed[i - 1]) / dt

            turn[i] = angle_difference(
                bearing[i - 1],
                bearing[i],
            )

        stop = (speed < 0.5).astype(int)

        out = pd.DataFrame(
            {
                "speed": speed,
                "accel": accel,
                "turn": turn,
                "stop": stop,
            }
        )

        return out

    def load(self) -> pd.DataFrame:

        self._log("Loading GeoLife trajectories...")

        users = self._list_user_dirs()

        all_segments = []

        for user in users:

            traj_dir = user / "Trajectory"

            if not traj_dir.exists():
                continue

            files = sorted(traj_dir.glob("*.plt"))

            for f in files:

                try:

                    df = self._load_trajectory(f)

                    if len(df) < self.cfg.min_points_per_traj:
                        continue

                    seg = self._process_trajectory(df)

                    all_segments.append(seg)

                except Exception as e:

                    self._log(f"Skipping {f}: {e}")

        if len(all_segments) == 0:
            raise RuntimeError("No trajectories loaded")

        df = pd.concat(all_segments, ignore_index=True)

        df = df.replace([np.inf, -np.inf], np.nan)

        df = df.dropna()

        self._log(f"Loaded {len(df)} observations")

        return df


# ---------------------------------------------------------
# Convenience function
# ---------------------------------------------------------

def load_geolife(max_users: Optional[int] = None):

    cfg = GeoLifeConfig(max_users=max_users)

    loader = GeoLifeLoader(cfg)

    df = loader.load()

    return df