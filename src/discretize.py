from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BinSpec:
    edges: np.ndarray  # len = n_bins+1, includes -inf/+inf style edges
    n_bins: int


def fit_quantile_bins(series: pd.Series, n_bins: int, quantiles: Tuple[float, ...]) -> BinSpec:
    """
    Fit bin edges from quantiles on the training data.
    Supports n_bins = 2, 3, or >=4.
    """

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    # --- 2 bins (median split) ---
    if n_bins == 2:
        if len(quantiles) != 1:
            raise ValueError("n_bins=2 requires quantiles like (0.5,)")

        q = float(series.quantile(quantiles[0]))
        edges = np.array([-np.inf, q, np.inf], dtype=float)

        return BinSpec(edges=edges, n_bins=2)

    # --- 3 bins (two quantiles) ---
    if n_bins == 3:
        if len(quantiles) != 2:
            raise ValueError("n_bins=3 requires quantiles like (0.33, 0.66)")

        q_lo = float(series.quantile(quantiles[0]))
        q_hi = float(series.quantile(quantiles[1]))

        edges = np.array([-np.inf, q_lo, q_hi, np.inf], dtype=float)

        return BinSpec(edges=edges, n_bins=3)

    # --- n_bins >= 4 ---
    # ignore provided quantiles and compute evenly spaced quantiles
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]

    cuts = [float(series.quantile(q)) for q in qs]

    edges = np.array([-np.inf] + cuts + [np.inf], dtype=float)

    return BinSpec(edges=edges, n_bins=n_bins)


def apply_bins(series: pd.Series, spec: BinSpec) -> np.ndarray:
    """
    Apply pre-fitted bins to a series.
    """

    x = series.to_numpy(dtype=float)

    # remove -inf and +inf
    bins_internal = spec.edges[1:-1]

    # digitize
    b = np.digitize(x, bins_internal, right=False)

    return b.astype(int)


def fit_and_discretize(
    df: pd.DataFrame,
    n_bins: int,
    quantiles: Tuple[float, ...],
    fit_on_index: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, BinSpec]]:
    """
    Fit bin specifications on training subset and discretize the full dataframe.
    """

    specs: Dict[str, BinSpec] = {}

    out = pd.DataFrame(index=df.index)

    for col in df.columns:

        s_fit = df[col].iloc[fit_on_index]

        spec = fit_quantile_bins(
            s_fit,
            n_bins=n_bins,
            quantiles=quantiles,
        )

        specs[col] = spec

        out[col] = apply_bins(df[col], spec)

    return out, specs


def apply_specs(
    df: pd.DataFrame,
    specs: Dict[str, BinSpec],
) -> pd.DataFrame:
    """
    Apply previously fitted BinSpec objects to a new dataframe.

    This is critical for controls:
    - bins MUST be identical to those fitted on real data
    - prevents controls from adapting their own discretization
    """

    out = pd.DataFrame(index=df.index)

    for col, spec in specs.items():

        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in dataframe for discretization")

        out[col] = apply_bins(df[col], spec)

    return out