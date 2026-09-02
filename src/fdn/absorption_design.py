"""Absorption filter design via graphic EQ (Schlecht and Habets, DAFx 2017).

Wraps :func:`pyFDN.design_geq` so that each delay line can carry its own
T60(f) curve, which :func:`pyFDN.absorption_geq` does not support.  For a
single shared curve the output is identical to ``pyFDN.absorption_geq``.

The GEQ design grid is fixed by pyFDN at DC, 63, 125, 250, 500, 1k, 2k, 4k,
8k Hz and Nyquist; a user-supplied T60(f) curve is interpolated onto it in
log frequency and clamped at the edges.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyFDN import design_geq, rt_to_slope

#: Interior points of ``pyFDN.design_geq``'s target grid.  The DC point (1 Hz)
#: and Nyquist are appended at design time because Nyquist depends on ``fs``.
_GEQ_INTERIOR_HZ = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def t60_on_geq_grid(t60_s: NDArray, freqs: NDArray, fs: float) -> NDArray:
    """Interpolate a T60(f) curve onto the 10-point GEQ design grid.

    Parameters
    ----------
    t60_s : NDArray, shape (n_bands,)
        Reverberation time in **seconds** at *freqs*.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz, ascending.
    fs : float
        Sample rate in Hz.

    Returns
    -------
    NDArray, shape (10,)
        T60 in seconds at [1, 63, 125, 250, 500, 1k, 2k, 4k, 8k, fs] Hz.
        Values outside the supplied range are held flat at the edge value.
    """
    t60_s = np.asarray(t60_s, dtype=float).ravel()
    freqs = np.asarray(freqs, dtype=float).ravel()
    if t60_s.size != freqs.size:
        raise ValueError("t60_s and freqs must have the same length")
    if np.any(freqs <= 0.0):
        raise ValueError("freqs must be positive")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("freqs must be strictly ascending")

    grid = np.concatenate(([1.0], _GEQ_INTERIOR_HZ, [float(fs)]))
    # np.interp holds the edge values outside the data range, which is the
    # same flat extension the previous spline fitter applied at its cutoffs.
    return np.interp(np.log(grid), np.log(freqs), t60_s)


def design_absorption_sos(
    t60_s: NDArray,
    freqs: NDArray,
    delays: NDArray,
    fs: float,
) -> NDArray:
    """Design one GEQ absorption cascade per delay line.

    Parameters
    ----------
    t60_s : NDArray
        Shape ``(n_bands,)`` for one curve shared by all delay lines, or
        ``(N, n_bands)`` for one curve per delay line.  Seconds.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz at which *t60_s* is specified.
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.
    fs : float
        Sample rate in Hz.

    Returns
    -------
    NDArray, shape (11, 6, N)
        Per-channel SOS bank in pyFDN's canonical layout, ready for
        :class:`pyFDN.SOSFilterBank`.  All sections are normalised to a0 = 1.
    """
    delays = np.asarray(delays, dtype=float).ravel()
    num_delays = delays.size
    if num_delays == 0:
        raise ValueError("delays must not be empty")

    curves = np.atleast_2d(np.asarray(t60_s, dtype=float))
    if curves.shape[0] == 1:
        curves = np.repeat(curves, num_delays, axis=0)
    if curves.shape[0] != num_delays:
        raise ValueError(
            f"t60_s must supply one T60 curve per delay line: expected "
            f"{num_delays} rows, got {curves.shape[0]}"
        )

    sos_out: NDArray | None = None
    for i in range(num_delays):
        target_db = rt_to_slope(t60_on_geq_grid(curves[i], freqs, fs), fs) * delays[i]
        sos, _ = design_geq(target_db, fs=fs)
        sos = sos / sos[:, 3:4]  # normalise a0 = 1
        if sos_out is None:
            sos_out = np.zeros((sos.shape[0], 6, num_delays))
        sos_out[:, :, i] = sos
    assert sos_out is not None
    return sos_out
