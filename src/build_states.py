from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StateEncoding:
    n_components: int
    n_bins: int
    powers: np.ndarray  # base powers for encoding


def make_encoding(n_components: int, n_bins: int) -> StateEncoding:
    powers = np.array([n_bins ** i for i in range(n_components)], dtype=int)
    return StateEncoding(n_components=n_components, n_bins=n_bins, powers=powers)


def encode_states(components: np.ndarray, enc: StateEncoding) -> np.ndarray:
    # components: shape (T, n_components) values in [0..n_bins-1]
    if components.ndim != 2 or components.shape[1] != enc.n_components:
        raise ValueError("components must be (T, n_components)")
    return (components @ enc.powers).astype(int)


def decode_state(state_id: int, enc: StateEncoding) -> np.ndarray:
    out = np.zeros(enc.n_components, dtype=int)
    x = int(state_id)
    for i in range(enc.n_components):
        out[i] = x % enc.n_bins
        x //= enc.n_bins
    return out


def build_components_matrix(df_bins: pd.DataFrame, order: List[str]) -> np.ndarray:
    # order define a ordem dos componentes (ex: ["load","wind","solar","price"])
    return df_bins[order].to_numpy(dtype=int)


def build_transitions(state_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if state_ids.ndim != 1 or len(state_ids) < 2:
        raise ValueError("state_ids must be 1D with len>=2")
    curr = state_ids[:-1].astype(int)
    nxt = state_ids[1:].astype(int)
    return curr, nxt