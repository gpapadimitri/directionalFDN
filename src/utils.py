from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyfar as pf
import pyFDN
from numpy.typing import NDArray
from pyrato.edc import energy_decay_curve_chu_lundeby
from pyrato.parameters import reverberation_time_linear_regression
from scipy.signal import butter, sosfilt

#: Linear magnitude to decibels, and back.  Re-exported from pyFDN rather than
#: reimplemented; both clamp instead of biasing, so they are exact inverses.
lin2db = pyFDN.lin_to_db
db2lin = pyFDN.db_to_lin

#: Milliseconds to samples.  pyFDN rounds where the previous local version
#: truncated -- a sub-sample difference on delay-range bounds.
ms_to_samps = pyFDN.ms_to_smp


# ---------------------------------------------------------------------------
# T30 reverberation time estimation & plotting
# ---------------------------------------------------------------------------

OCTAVE_BANDS = np.array([125, 250, 500, 1000, 2000, 4000, 8000])



def estimate_t30(ir: NDArray, fs: float, bands: NDArray = OCTAVE_BANDS) -> NDArray:
    """Estimate T30 (extrapolated to T60) per octave band.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    bands : NDArray
        Octave-band center frequencies in Hz.

    Returns
    -------
    NDArray, shape ``(len(bands),)``
        T60 values in seconds (NaN where estimation fails).
    """
    t60 = np.full(len(bands), np.nan)
    for i, fc in enumerate(bands):
        f_lo = fc / np.sqrt(2)
        f_hi = fc * np.sqrt(2)
        if f_hi >= fs / 2:
            continue
        sos = butter(4, [f_lo, f_hi], btype="band", fs=fs, output="sos")
        filtered = sosfilt(sos, ir)
        peak = np.max(np.abs(filtered))
        if peak > 0:
            noise_floor = peak * 1e-6  # −120 dB below peak
            rng = np.random.default_rng(42)
            dithered = filtered + rng.standard_normal(len(filtered)) * noise_floor
        else:
            dithered = filtered
        sig = pf.Signal(dithered[np.newaxis, :], fs)
        edc = energy_decay_curve_chu_lundeby(sig, freq="broadband")
        t30 = reverberation_time_linear_regression(edc, T="T30")
        t60[i] = float(t30[0])
    return t60


def plot_t30(ir: NDArray, fs: float) -> None:
    """Plot octave-band reverberation time from a *synthesised* impulse response.

    Uses :func:`estimate_t60_bands`, i.e. pyFDN's Schroeder fit over its own
    octave filterbank.  For a *measured* response prefer :func:`estimate_t30`,
    whose Chu-Lundeby estimator subtracts the measurement noise floor.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """
    t60, f_centre = estimate_t60_bands(ir, fs)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{fc:.0f}" if fc < 1000 else f"{fc / 1000:.0f}k" for fc in f_centre]
    ax.plot(np.arange(len(f_centre)), t60, color="red", marker="o")
    ax.set_ylim(bottom=0, top=np.nanmax(t60) * 1.1)
    ax.set_xticks(np.arange(len(f_centre)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("T60 (s)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_time_domain(ir: NDArray, fs: float) -> None:
    """Plot an impulse response in the time domain.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(ir)) / fs
    ax.plot(t, ir, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Impulse Response")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_edc(ir: NDArray, fs: float) -> None:
    """Plot broadband and octave-band energy decay curves in dB.

    Built on pyFDN's :func:`~pyFDN.edc`, :func:`~pyFDN.octave_bands` and
    :func:`~pyFDN.octave_band_filterbank`, so the band edges and the decay
    curve match what :func:`estimate_t60_bands` fits its slopes to.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """
    ir = np.asarray(ir, dtype=float).ravel()
    t = np.arange(len(ir)) / fs

    def _db(curve: NDArray) -> NDArray:
        return 10.0 * np.log10(curve / (curve[0] + 1e-30) + 1e-30)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, _db(pyFDN.edc(ir)), color="black", linewidth=1.5, label="Broadband")

    bands, f_centre = pyFDN.octave_bands(fc=1000.0, start=-4, n=8, fs=fs)
    sos_bank = pyFDN.octave_band_filterbank(bands, fs)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(f_centre)))
    for i, (fc, sos) in enumerate(zip(f_centre, sos_bank, strict=True)):
        band_edc = pyFDN.edc(sosfilt(sos, ir))
        label = f"{fc:.0f} Hz" if fc < 1000 else f"{fc / 1000:.0f} kHz"
        ax.plot(t, _db(band_edc), color=cmap[i], linewidth=0.9, alpha=0.8, label=label)
    ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    ax.set_ylim(bottom=-80, top=0)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_frequency_response(x, fs, smooth=3):
    """
    Plot the frequency response of a signal using pyfar.plot.freq.

    Parameters
    ----------
    x : array_like
        The impulse response signal.
    fs : int or float
        The sampling frequency of the signal in Hz.
    smooth : int or None, optional
        Apply octave smoothing: 1 for octave, 3 for 1/3 octave, None for no smoothing.
    """
    sig = pf.Signal(np.asarray(x, dtype=np.float64), int(fs))
    fig, ax = plt.subplots(figsize=(10, 6))

    if smooth is not None:
        # Plot raw with transparency
        pf.plot.freq(sig, ax=ax, alpha=0.3, color='gray', label='Raw')
        # Apply smoothing (newer pyfar returns (smoothed, mask) tuple)
        result = pf.dsp.smooth_fractional_octave(sig, smooth)
        smoothed_sig = result[0] if isinstance(result, tuple) else result
        pf.plot.freq(smoothed_sig, ax=ax, label=f'1/{smooth} Octave Smoothed')
        ax.legend(loc='best')
    else:
        pf.plot.freq(sig, ax=ax)

    ax.set_xlim([20, 20000])
    ax.relim()
    ax.autoscale(axis='y')
    ax.set_title("Frequency Response")
    plt.tight_layout()
    plt.show()


def estimate_t60_bands(
    ir: NDArray, fs: float
) -> tuple[NDArray, NDArray]:
    """Octave-band T60 by Schroeder-curve fit, without noise compensation.

    Preferred over :func:`estimate_t30` for *synthesised* impulse responses,
    which have no noise floor: the linear fit runs from -5 dB over 30 dB and is
    extrapolated to 60 dB.  For *measured* responses keep :func:`estimate_t30`,
    whose Chu-Lundeby estimator subtracts the measurement noise floor.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.

    Returns
    -------
    rt : NDArray, shape (8,)
        T60 in seconds at 62.5 Hz to 8 kHz.
    f_centre : NDArray, shape (8,)
        Band centre frequencies in Hz.
    """
    return pyFDN.estimate_rt_bands(np.asarray(ir).ravel(), fs, n=8, start=-4)


def mixing_time_ms(ir: NDArray, fs: float, pre_delay: int = 0) -> float:
    """Mixing time in milliseconds (Abel and Huang 2006 echo density).

    The time at which the normalised echo density first reaches 1, i.e. when
    the response becomes indistinguishable from Gaussian noise.  This is the
    quantitative answer to whether a DFDN tail is actually diffuse, which is
    what the ``scattering_amount`` and ``lateral_mix`` parameters are for.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response, at least ~100 ms long.
    fs : float
        Sample rate in Hz.
    pre_delay : int
        Onset offset in samples to measure from.

    Returns
    -------
    float
        Mixing time in ms, or ``nan`` if the response never reaches the
        threshold.  ``pyFDN.echo_density`` reports that case as 0.0, which
        would read as "instantly mixed"; nan is the honest value for
        "undefined" and keeps comparisons from silently inverting.
    """
    import warnings

    # Coupled to pyFDN.echo_density's exact wording: this is the only way to
    # distinguish "never reached threshold" from an ordinary 0.0 result, since
    # echo_density reports both the same way otherwise. Any other warning it
    # raises is re-emitted below rather than silently swallowed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t_abel, _ = pyFDN.echo_density(
            np.asarray(ir).ravel(), n=1024, fs=fs, pre_delay=pre_delay
        )
    not_found = False
    for w in caught:
        if "Mixing time not found" in str(w.message):
            not_found = True
        else:
            warnings.warn_explicit(
                w.message, w.category, w.filename, w.lineno
            )
    if not_found:
        return float("nan")
    return float(t_abel)



def pole_bounds(
    delays: NDArray,
    sos: NDArray,
    feedback_matrix: NDArray,
    fs: float,
    nfft: int = 2**12,
) -> tuple[NDArray, NDArray, NDArray]:
    """Bounds on the FDN loop's pole magnitudes against frequency.

    A stable, correctly-absorbing FDN keeps the upper curve below 1 everywhere.
    With per-direction absorption filters that is a real design check, since
    the single-channel intuition does not carry over.

    This reimplements :func:`pyFDN.pole_boundaries` from the SOS cascade rather
    than calling it.  That function needs a direct-form ``(b, a)`` pair per
    channel, and flattening an 11-section cascade with ``scipy.signal.sos2tf``
    is numerically indefensible at order 22: the flattened response reports a
    peak magnitude of 5.69 where the cascade's true peak is 0.92, and picks up
    a pole at 1.13 that the cascade does not have.  The maths below is the
    same; only the route to the filter response differs.

    Parameters
    ----------
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.
    sos : NDArray, shape (n_sections, 6, N)
        Per-channel absorption SOS bank, as returned by
        :func:`src.fdn.absorption_design.design_absorption_sos`.
    feedback_matrix : NDArray, shape (N, N) or (N, N, order)
        Scalar or FIR-polynomial feedback matrix.
    fs : float
        Sample rate in Hz.
    nfft : int
        Number of frequency points from DC to Nyquist.

    Returns
    -------
    min_curve, max_curve : NDArray, shape (nfft,)
        Lower and upper bounds on pole magnitude.
    freqs : NDArray, shape (nfft,)
        Frequencies in Hz.
    """
    from scipy.signal import sosfreqz

    delays_arr = np.asarray(delays, dtype=float).ravel()
    sos_arr = np.asarray(sos, dtype=float)
    if sos_arr.ndim != 3 or sos_arr.shape[1] != 6:
        raise ValueError("sos must have shape (n_sections, 6, N)")
    num = delays_arr.size
    if sos_arr.shape[2] != num:
        raise ValueError("sos must carry one cascade per delay line")

    omega = np.linspace(0.0, np.pi, nfft)

    # Feedback-matrix singular values per bin.  A scalar matrix is frequency
    # independent, so its curves are flat.
    A = np.asarray(feedback_matrix, dtype=float)
    if A.ndim == 2:
        A = A[:, :, np.newaxis]
    A_w = np.fft.fft(A, n=2 * nfft, axis=2)[:, :, :nfft]
    sv_min = np.empty(nfft)
    sv_max = np.empty(nfft)
    for k in range(nfft):
        s = np.linalg.svd(A_w[:, :, k], compute_uv=False)
        sv_min[k] = np.min(np.abs(s)) ** (1.0 / delays_arr.min())
        sv_max[k] = np.max(np.abs(s)) ** (1.0 / delays_arr.max())

    # Per-channel absorption, as a gain per sample of the loop it sits in.
    # The group delay is differentiated from the cascade's own unwrapped phase
    # rather than taken from a flattened transfer function.
    per_sample = np.empty((nfft, num))
    for i in range(num):
        _, h = sosfreqz(sos_arr[:, :, i], worN=omega)
        group_delay = -np.gradient(np.unwrap(np.angle(h)), omega)
        loop = np.maximum(delays_arr[i] + group_delay, 1.0)
        per_sample[:, i] = np.abs(h) ** (1.0 / loop)

    return (
        per_sample.min(axis=1) * sv_min,
        per_sample.max(axis=1) * sv_max,
        omega / np.pi * fs / 2.0,
    )
