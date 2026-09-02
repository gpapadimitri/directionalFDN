"""Block-processing adapter over :func:`pyFDN.process_fdn`.

The hand-written loops this replaces read the delay lines and then injected
feedback computed in the *previous* block, giving a round trip of
``m + frame_size`` samples rather than ``m``.  ``pyFDN.process_fdn`` reads and
writes the same buffer positions inside one block, so the round trip is
exactly ``m`` and no compensation term is needed anywhere.

This module only converts orientation: pyFDN is samples-first
``(n_samples, n_channels)``, this package is channels-first
``(n_channels, n_samples)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyFDN import process_fdn


class ScalarGains:
    """Frequency-independent per-delay-line gains as a process_fdn filter hook.

    Parameters
    ----------
    gains : NDArray, shape (N,)
        Linear gain applied to each delay-line output.
    """

    __slots__ = ("gains",)

    def __init__(self, gains: NDArray) -> None:
        self.gains = np.asarray(gains, dtype=float).ravel()

    def filter(self, block: NDArray) -> NDArray:
        """Scale a ``(block_size, N)`` block channel-wise."""
        return np.asarray(block, dtype=float) * self.gains[np.newaxis, :]


def render_fdn(
    input_data: NDArray,
    delays: NDArray,
    A: NDArray,
    b: NDArray,
    c: NDArray,
    d: NDArray,
    absorption: object | None = None,
) -> NDArray:
    """Run an FDN recursion and return the output channels-first.

    Parameters
    ----------
    input_data : NDArray, shape (n_in, n_samples)
        Input signal, channels-first.
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.  All must be positive.
    A : NDArray, shape (N, N) or (N, N, order)
        Feedback matrix.  A 3-D matrix is treated as an FIR polynomial matrix.
    b : NDArray, shape (N, n_in)
        Input mixing.
    c : NDArray, shape (n_out, N)
        Output mixing, applied to the absorbed delay-line output.
    d : NDArray, shape (n_out, n_in)
        Direct-path gain.
    absorption : object, optional
        Anything with ``filter(block: (F, N)) -> (F, N)``, e.g. a
        :class:`ScalarGains` or a :class:`pyFDN.SOSFilterBank`.  Applied to the
        delay-line output before both the output tap and the feedback path,
        which is where the previous implementation applied it.

    Returns
    -------
    NDArray, shape (n_out, n_samples)
    """
    x = np.asarray(input_data, dtype=float)
    if x.ndim != 2:
        raise ValueError("input_data must be 2-D (n_in, n_samples)")

    y = process_fdn(
        x.T,
        np.asarray(delays, dtype=int).ravel(),
        A,
        b,
        c,
        d,
        post_delay=absorption,
    )
    y = np.asarray(y, dtype=float)
    # process_fdn squeezes its output; restore the channel axis.
    if y.ndim == 1:
        y = y[:, np.newaxis]
    return y.T
