"""IIR filter design via Prony's method and frequency-warped Prony.

Adapted from Jatin Chowdhury's *audio_dspy* toolbox and J. Abel's
all-pass warping implementation.

Reference
---------
Chowdhury, J. (2022). audio_dspy — https://github.com/jatinchowdhury18/audio_dspy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq, ifft, irfft, rfftfreq
from scipy.interpolate import splev, splrep
from scipy.linalg import toeplitz
from scipy.signal import hilbert, tf2zpk, zpk2tf

from ..utils import db, db2lin

# ---------------------------------------------------------------------------
# Spectrum helpers
# ---------------------------------------------------------------------------

def _one_to_two_sided(one_sided: NDArray, *, is_even: bool = True) -> NDArray:
    """Convert a single-sided spectrum to conjugate-symmetric two-sided form."""
    two_sided: NDArray
    if is_even:
        one_sided = one_sided.copy()
        one_sided[-1] = np.real(one_sided[-1])
        two_sided = np.concatenate((one_sided, np.conj(one_sided[-2:0:-1])))
    else:
        two_sided = np.concatenate((one_sided, np.conj(one_sided[-1:0:-1])))
    two_sided[0] = np.real(two_sided[0])
    return two_sided


def interpolate_magnitude_spectrum(
    mag_spec: NDArray,
    freqs: NDArray,
    fs: float,
    n_fft: int,
    *,
    cutoff: tuple[float, float] = (63.0, 16_000.0),
    rolloff_dc_db: float | None = None,
    rolloff_nyq_db: float | None = None,
    return_one_sided: bool = True,
    bands_per_octave: int = 1,
) -> tuple[NDArray, NDArray]:
    """Spline-interpolate a sparse magnitude spectrum onto a dense FFT grid.

    Parameters
    ----------
    mag_spec : NDArray
        Sparse magnitude values (linear).
    freqs : NDArray
        Corresponding frequency points in Hz.
    fs : float
        Sample rate.
    n_fft : int
        FFT length for the dense grid.
    cutoff : (float, float)
        Low and high frequency limits; magnitude is gently rolled off outside.
    rolloff_dc_db, rolloff_nyq_db : float, optional
        Roll-off slopes (dB/octave) below/above *cutoff*.  Estimated from
        neighbours if not given.
    return_one_sided : bool
        If True return only the positive-frequency half.

    Returns
    -------
    mag_interp : NDArray
        Interpolated magnitude spectrum.
    new_freqs : NDArray
        Corresponding frequency axis.
    """
    if rolloff_nyq_db is None:
        rolloff_nyq_db = db(mag_spec[-1] - mag_spec[-2]) * bands_per_octave
    if rolloff_dc_db is None:
        rolloff_dc_db = -db(np.abs(mag_spec[1] - mag_spec[0])) * bands_per_octave

    new_freqs = rfftfreq(n_fft, d=1.0 / fs)
    n_bins = new_freqs.size
    end_at = int(np.searchsorted(new_freqs, cutoff[1]))
    start_from = int(np.searchsorted(new_freqs, cutoff[0]))

    spline = splrep(freqs, mag_spec, k=2)
    mag_interp = splev(new_freqs, spline)

    # Low-frequency roll-off: hold edge value constant (flat extension)
    mag_interp[:start_from] = mag_interp[start_from]
    # High-frequency roll-off: hold edge value constant
    mag_interp[end_at:] = mag_interp[end_at]

    if not return_one_sided:
        mag_interp = _one_to_two_sided(mag_interp, is_even=(n_fft % 2 == 0))
        new_freqs = fftfreq(n_fft, d=1.0 / fs)
    return mag_interp, new_freqs


def tf2minphase(
    tf_spec: NDArray,
    *,
    is_even_fft: bool = True,
    return_ir: bool = False,
) -> NDArray:
    """Convert a single-sided magnitude spectrum to minimum phase.

    Parameters
    ----------
    tf_spec : NDArray
        Single-sided magnitude spectrum.
    is_even_fft : bool
        Whether the original two-sided FFT had an even number of bins.
    return_ir : bool
        If True, return the minimum-phase impulse response instead of the
        spectral coefficients.
    """
    n_bins = len(tf_spec)
    two_sided = _one_to_two_sided(tf_spec, is_even=is_even_fft)
    mag = np.abs(two_sided)
    phu = np.imag(hilbert(-np.log(mag + np.finfo(np.float64).eps)))
    tf_mp = mag * np.exp(1j * phu)
    tf_mp = tf_mp[:n_bins]
    if return_ir:
        return irfft(tf_mp)
    return tf_mp


# ---------------------------------------------------------------------------
# Prony's method
# ---------------------------------------------------------------------------

def prony(h: NDArray, nb: int, na: int) -> tuple[NDArray, NDArray]:
    """IIR filter design via Prony's method.

    Parameters
    ----------
    h : NDArray
        Target minimum-phase impulse response.
    nb, na : int
        Number of numerator / denominator coefficients.

    Returns
    -------
    b, a : NDArray
        Filter coefficients.
    """
    k = len(h) - 1
    H = toeplitz(h, np.r_[1.0, np.zeros(k)])
    H = H[:, : na + 1]
    H1 = H[: nb + 1, :]
    h1 = H[nb + 1 : k + 1, 0]
    H2 = H[nb + 1 : k + 1, 1 : na + 1]
    a_tail, *_ = np.linalg.lstsq(H2, -h1, rcond=None)
    a = np.r_[1.0, a_tail]
    b = (H1 @ a)[: nb + 1]
    return b, a


# ---------------------------------------------------------------------------
# All-pass warping (J. Abel implementation)
# ---------------------------------------------------------------------------

def _allpass_warp_ir(ir: NDArray, rho: float) -> NDArray:
    """Apply all-pass frequency warping to an impulse response."""
    nsamp = len(ir)
    nbinsmax = 65536
    stretch = (1 + abs(rho)) / (1 - abs(rho))
    nbins = min(nbinsmax, int(2 ** np.ceil(np.log2(nsamp * stretch))))
    w = np.pi * np.arange(nbins) / nbins
    z = np.exp(1j * w)
    zeta = (z - rho) / (1 - rho * z)
    ww = np.angle(zeta)

    from scipy.interpolate import interp1d
    tf_full = fft(ir, 2 * nbins)
    tf_half = tf_full[:nbins]
    interp_fn = interp1d(w, tf_half, kind="cubic", fill_value="extrapolate")
    tfw = np.r_[interp_fn(ww), np.conj(np.flip(interp_fn(ww)[1:nbins]))]
    return np.real(ifft(tfw, 2 * nbins))[:nsamp]


def _allpass_warp_roots(
    rho: float, b: NDArray, a: NDArray
) -> tuple[NDArray, NDArray]:
    """Apply all-pass warping to filter pole/zero positions."""
    z, p, k = tf2zpk(b, a)
    zw = (z + rho) / (1 + rho * z)
    pw = (p + rho) / (1 + rho * p)
    npz = len(p) - len(z)
    if npz > 0:
        zw = np.r_[zw, np.full(npz, rho)]
    elif npz < 0:
        pw = np.r_[pw, np.full(-npz, rho)]
    kw = k * np.prod(1 + z * rho) / np.prod(1 + p * rho)
    bw, aw = zpk2tf(zw, pw, kw)
    return np.real(bw), np.real(aw)


def prony_warped(
    h: NDArray, nb: int, na: int, rho: float
) -> tuple[NDArray, NDArray]:
    """Frequency-warped Prony's method for IIR filter design.

    Parameters
    ----------
    h : NDArray
        Target impulse response.
    nb, na : int
        Numerator / denominator orders.
    rho : float
        Warping coefficient (−1, 1). Positive expands low-frequency
        resolution; negative expands high-frequency resolution.

    Returns
    -------
    b, a : NDArray
        Filter coefficients.
    """
    h_warp = _allpass_warp_ir(h, rho)
    b_w, a_w = prony(h_warp, nb, na)
    return _allpass_warp_roots(-rho, b_w, a_w)
