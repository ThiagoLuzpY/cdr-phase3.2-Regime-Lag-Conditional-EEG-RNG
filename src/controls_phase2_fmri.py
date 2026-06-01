from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft


def phase_randomize_bold(data: np.ndarray, seed: int) -> np.ndarray:
    """
    Phase randomized surrogate for BOLD signals.

    Preserves:
    - power spectrum
    - autocorrelation

    Destroys:
    - temporal phase structure

    Standard surrogate used in neuroimaging.
    """

    rng = np.random.default_rng(seed)

    out = data.copy()

    n_scans, n_rois = out.shape

    for i in range(n_rois):

        x = out[:, i]

        fft_x = fft(x)

        amplitudes = np.abs(fft_x)

        phases = rng.uniform(0, 2 * np.pi, len(x))

        fft_surr = amplitudes * np.exp(1j * phases)

        surr = np.real(ifft(fft_surr))

        out[:, i] = surr

    return out