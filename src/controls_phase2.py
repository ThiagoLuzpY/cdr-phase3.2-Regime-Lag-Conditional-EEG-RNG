from __future__ import annotations

import numpy as np
import pandas as pd


def shuffle_state_ids_global(state_ids: np.ndarray, seed: int) -> np.ndarray:
    """
    Destroy all temporal structure by permuting state ids globally.
    Strongest possible null control on already-encoded states.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(state_ids))
    return state_ids[idx].copy()


def shuffle_rows_global_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle rows of the dataframe globally.

    Effect:
    - destroys temporal transitions completely
    - preserves instantaneous coupling between variables within each row

    Note:
    - for OPSD this behaves more like a stress/adversarial perturbation
      than a realistic collapse control
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    out = df.iloc[idx].copy()
    return out


def shuffle_columns_independently_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Shuffle each column independently.

    Effect:
    - destroys cross-variable coupling
    - destroys temporal structure
    - preserves each variable's marginal distribution exactly

    Note:
    - for OPSD this is usually too aggressive for F2
      and should be treated as stress/adversarial control
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    for col in out.columns:
        values = out[col].to_numpy(copy=True)
        rng.shuffle(values)
        out[col] = values

    return out


def shuffle_week_blocks_df(
    df: pd.DataFrame,
    block_size: int = 168,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Shuffle weekly blocks (default: 168 hours).

    Effect:
    - preserves intra-week local dynamics
    - destroys longer-range temporal ordering between weeks

    This is a good OPSD collapse control because it preserves
    realistic local patterns while breaking long-range sequencing.
    """
    rng = np.random.default_rng(seed)

    n = len(df)
    n_blocks = n // block_size

    blocks = [
        df.iloc[i * block_size:(i + 1) * block_size].copy()
        for i in range(n_blocks)
    ]

    rng.shuffle(blocks)

    out = pd.concat(blocks, axis=0)

    # append remainder if exists
    remainder = df.iloc[n_blocks * block_size:].copy()
    if len(remainder) > 0:
        out = pd.concat([out, remainder], axis=0)

    return out


def shuffle_within_hour_weektype_df(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Shuffle rows within seasonal strata:
    (hour of day, weekday/weekend).

    Effect:
    - preserves coarse seasonality
    - destroys finer temporal organization

    This is a good OPSD collapse control because it respects
    obvious daily/weekly structure.
    """
    rng = np.random.default_rng(seed)

    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for seasonal stratified shuffle")

    hours = out.index.hour
    weektype = (out.index.weekday < 5).astype(int)

    strata: dict[tuple[int, int], list[int]] = {}

    for i in range(len(out)):
        key = (int(hours[i]), int(weektype[i]))
        strata.setdefault(key, []).append(i)

    new_df = out.copy()

    for _, idxs in strata.items():
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        new_df.iloc[idxs] = out.iloc[shuffled].to_numpy()

    return new_df


def shuffle_within_month_hour_weektype_df(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Shuffle rows within finer seasonal strata:
    (month, hour of day, weekday/weekend).

    Effect:
    - preserves monthly seasonality
    - preserves hour-of-day structure
    - preserves weekday/weekend regime
    - destroys finer-grained temporal organization

    This is a stronger and more realistic OPSD collapse control than
    plain hour/weektype shuffle for highly seasonal energy data.
    """
    rng = np.random.default_rng(seed)

    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex for monthly/hour/weektype shuffle")

    months = out.index.month
    hours = out.index.hour
    weektype = (out.index.weekday < 5).astype(int)

    strata: dict[tuple[int, int, int], list[int]] = {}

    for i in range(len(out)):
        key = (int(months[i]), int(hours[i]), int(weektype[i]))
        strata.setdefault(key, []).append(i)

    new_df = out.copy()

    for _, idxs in strata.items():
        if len(idxs) <= 1:
            continue
        shuffled = idxs.copy()
        rng.shuffle(shuffled)
        new_df.iloc[idxs] = out.iloc[shuffled].to_numpy()

    return new_df