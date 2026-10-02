"""Directional decay, level and correlation analysis for grid and Ambisonic RIRs.

Octave bands are pyfar's, as in the hybrid-rir ODEON notebook; decay fits are
pyrato's ISO 3382 regression. The intercept of the fit gives the initial
power, as in the EDR analysis of Gallien et al. (DAFx26, Sec. 3.2).
"""

from __future__ import annotations

import numpy as np
import pyfar as pf
from numpy.typing import NDArray
from pyrato.edc import schroeder_integration
from pyrato.parameters import reverberation_time_linear_regression

#pyrato's ISO 3382 intervals, needed here for the truncation guard
_DECAY_INTERVALS_DB = {"EDT": (-0.1, -10.1), "T20": (-5.0, -25.0), "T30": (-5.0, -35.0)}


def octave_bands(x: NDArray, fs: float, bands: NDArray) -> NDArray:
    """Octave-band split along the last axis, band axis first: (B, C, n)."""
    bands = np.asarray(bands, dtype=float)
    y = pf.dsp.filter.fractional_octave_bands(
        pf.Signal(np.atleast_2d(x), fs), 1, frequency_range=(bands[0], bands[-1])
    ).time
    if y.shape[0] != bands.size:
        raise ValueError(
            f"bands must be consecutive octave centres; pyfar gives {y.shape[0]} "
            f"bands between {bands[0]:g} and {bands[-1]:g} Hz, not {bands.size}."
        )
    return y


def band_edc_db(x: NDArray, fs: float, bands: NDArray) -> NDArray:
    """Octave-band Schroeder EDC in dB, 0 dB at t = 0, shape (B, C, n)."""
    edc = schroeder_integration(pf.Signal(octave_bands(x, fs, bands), fs)).time
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(edc / edc[..., :1])


def band_energy(x: NDArray, fs: float, bands: NDArray, start_s: float) -> NDArray:
    """Energy per octave band from *start_s* to the end, shape (B, C).

    The window is applied before the filters, so energy before *start_s*
    cannot ring into the window through them.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
    x[:, : int(round(start_s * fs))] = 0.0
    return np.sum(octave_bands(x, fs, bands) ** 2, axis=-1)


def decay_fit(
    x: NDArray, fs: float, bands: NDArray, parameter: str = "T20",
) -> tuple[NDArray, NDArray]:
    """Per-channel, per-band reverberation time and initial power.

    Parameters
    ----------
    x : NDArray, shape (C, n) or (n,)
        Channel signals (grid directions, beams, or SH channels).
    fs : float
        Sample rate in Hz.
    bands : NDArray
        Consecutive octave centres in Hz.
    parameter : {"EDT", "T20", "T30"}
        ISO 3382 regression interval.

    Returns
    -------
    t60_s : NDArray, shape (C, B)
        Reverberation time in seconds, extrapolated to 60 dB.
    p0_db : NDArray, shape (C, B)
        Initial power from the regression intercept as a one-sided PSD
        (band power per Hz, dB). Biased high below 250 Hz by the onset of
        pyfar's filters; compare it relatively there. NaN where the fit
        window is too close to the end of the signal.
    """
    if parameter not in _DECAY_INTERVALS_DB:
        raise ValueError(
            f"parameter must be one of {sorted(_DECAY_INTERVALS_DB)}, got {parameter!r}."
        )
    upper, lower = _DECAY_INTERVALS_DB[parameter]
    bands = np.asarray(bands, dtype=float)
    edc = schroeder_integration(pf.Signal(octave_bands(x, fs, bands), fs)).time
    n = edc.shape[-1]
    edc0 = edc[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = edc / edc0[..., np.newaxis]
        edc_db = 10.0 * np.log10(norm)
        i0 = np.argmin(np.abs(upper - edc_db), axis=-1)
        i1 = np.argmin(np.abs(lower - edc_db), axis=-1)
        t, intercept = reverberation_time_linear_regression(
            pf.TimeData(norm, np.arange(n) / fs), T=parameter, return_intercept=True
        )
        tau = t / (6.0 * np.log(10.0))
        bandwidth = bands[:, np.newaxis] / np.sqrt(2.0)
        p0 = 10.0 * np.log10(intercept * edc0 / (tau * fs * bandwidth))
    #a truncated tail bends the backward integral down and pyrato does not
    #check, so require at least as much signal after the window as in it
    valid = (i1 - i0 >= 2) & (n - i1 >= i1 - i0) & (t > 0)
    return np.where(valid, t, np.nan).T, np.where(valid, p0, np.nan).T


def band_correlation(
    x: NDArray, fs: float, bands: NDArray, t_range_s: tuple[float, float] = (0.05, 1.0),
) -> NDArray:
    """Normalised zero-lag inter-channel correlation per band, shape (B, C, C)."""
    i0, i1 = int(t_range_s[0] * fs), int(t_range_s[1] * fs)
    y = octave_bands(x, fs, bands)[..., i0:i1]
    cov = y @ np.swapaxes(y, -1, -2)
    d = np.sqrt(np.einsum("bii->bi", cov))
    return cov / (d[:, :, np.newaxis] * d[:, np.newaxis, :])


def mean_offdiag_abs(r: NDArray) -> NDArray:
    """Mean absolute off-diagonal entry of each matrix in a (B, C, C) stack."""
    mask = ~np.eye(r.shape[-1], dtype=bool)
    return np.abs(r[:, mask]).mean(axis=-1)
