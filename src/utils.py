from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt
import pyfar as pf
from pyrato.edc import energy_decay_curve_chu_lundeby, schroeder_integration
from pyrato.parameters import reverberation_time_linear_regression
import pyFDN
from scipy.signal import sosfreqz


def db(x: NDArray | float, is_squared: bool = False) -> NDArray:
    """Convert a linear magnitude to decibels.

    Parameters
    ----------
    x : array_like
        Linear magnitude(s).
    is_squared : bool
        If *True*, treat *x* as a power quantity (10·log10);
        otherwise as an amplitude quantity (20·log10).

    Returns
    -------
    NDArray
        Value(s) in dB, clipped to −200 dB minimum.
    """
    x = np.asarray(x, dtype=float)
    factor = 10.0 if is_squared else 20.0
    eps = np.finfo(np.float64).eps
    return np.clip(factor * np.log10(np.abs(x) + eps), a_min=-200, a_max=None)


def db2lin(dB: NDArray | float) -> NDArray | float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (np.asarray(dB, dtype=float) / 20.0)


def lin2db(linear: NDArray | float) -> NDArray | float:
    """Convert linear amplitude to decibels."""
    return 20.0 * np.log10(np.abs(np.asarray(linear, dtype=float)) + 1e-32)


def ms_to_samps(ms: NDArray | float, fs: float) -> int | NDArray:
    """Convert milliseconds to an integer sample count.

    Parameters
    ----------
    ms : float or array_like
        Time in milliseconds.
    fs : float
        Sample rate in Hz.
    """
    samp = np.asarray(ms, dtype=float) * 1e-3 * fs
    if samp.ndim == 0:
        return int(samp)
    return samp.astype(np.int64)


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
    """Plot frequency-dependent reverberation time from an impulse response.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """

    bands = OCTAVE_BANDS
    t60 = estimate_t30(ir, fs, bands)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{int(f)}" if f < 1000 else f"{f:.0f}" for f in bands]
    ax.plot(np.arange(len(bands)), t60, color="red", marker="o")
    ax.set_ylim(bottom=0, top=np.nanmax(t60) * 1.1)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("T30 (s)")
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
    """Plot the energy decay curve (EDC) in dB, optionally per octave band.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """
    bands = OCTAVE_BANDS
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(ir)) / fs

    # Broadband EDC via pyrato Schroeder integration
    sig = pf.Signal(ir[np.newaxis, :], fs)
    edc_pf = schroeder_integration(sig)
    edc_data = edc_pf.time[0]
    edc_db = 10.0 * np.log10(edc_data / (edc_data[0] + 1e-30) + 1e-30)
    ax.plot(t, edc_db, color="black", linewidth=1.5, label="Broadband")

    # Per-band EDCs via pyrato Schroeder integration
    if bands is not None:
        cmap = plt.cm.coolwarm(np.linspace(0, 1, len(bands)))
        for i, fc in enumerate(bands):
            f_lo = fc / np.sqrt(2)
            f_hi = fc * np.sqrt(2)
            if f_hi >= fs / 2:
                continue
            sos = butter(4, [f_lo, f_hi], btype="band", fs=fs, output="sos")
            filtered = sosfilt(sos, ir)
            band_sig = pf.Signal(filtered[np.newaxis, :], fs)
            band_edc_pf = schroeder_integration(band_sig)
            band_edc_data = band_edc_pf.time[0]
            band_edc_db = 10.0 * np.log10(
                band_edc_data / (band_edc_data[0] + 1e-30) + 1e-30
            )
            label = f"{int(fc)} Hz" if fc < 1000 else f"{fc/1000:.0f} kHz"
            ax.plot(t, band_edc_db, color=cmap[i], linewidth=0.9,
                    alpha=0.8, label=label)
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
        Mixing time in ms, or 0.0 if the threshold is never crossed.
    """
    t_abel, _ = pyFDN.echo_density(
        np.asarray(ir).ravel(), n=1024, fs=fs, pre_delay=pre_delay
    )
    return float(t_abel)


def pole_bounds(
    delays: NDArray, sos: NDArray, feedback_matrix: NDArray, fs: float,
    nfft: int = 2**12,
) -> tuple[NDArray, NDArray, NDArray]:
    """Upper and lower bounds on the FDN loop pole magnitudes vs frequency.

    A stable, correctly-absorbing FDN keeps the upper curve below 1 at every
    frequency.  Useful as a design check on the directional absorption, where
    each direction has its own filter and the usual single-channel intuition
    does not apply.

    Parameters
    ----------
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.
    sos : NDArray, shape (n_sections, 6, N)
        Per-channel absorption SOS bank, as returned by
        :func:`src.fdn.absorption_design.design_absorption_sos`.
    feedback_matrix : NDArray, shape (N, N)
        Scalar feedback matrix.
    fs : float
        Sample rate in Hz.
    nfft : int
        Number of frequency points.

    Returns
    -------
    min_curve, max_curve : NDArray, shape (nfft,)
        Lower and upper pole-magnitude bounds.
    freqs : NDArray, shape (nfft,)
        Frequencies in Hz.
    """
    delays = np.asarray(delays, dtype=float).ravel()
    num = delays.size

    # pole_boundaries wants an object exposing per-channel (b, a) of shape
    # (N, 1, len); convert each SOS cascade to a single transfer function.
    from scipy.signal import sos2tf

    b_list, a_list = [], []
    for i in range(num):
        b_i, a_i = sos2tf(np.asarray(sos)[:, :, i])
        b_list.append(b_i)
        a_list.append(a_i)
    order = max(max(len(v) for v in b_list), max(len(v) for v in a_list))
    b_arr = np.zeros((num, 1, order))
    a_arr = np.zeros((num, 1, order))
    for i in range(num):
        b_arr[i, 0, : len(b_list[i])] = b_list[i]
        a_arr[i, 0, : len(a_list[i])] = a_list[i]

    class _Absorption:
        b = b_arr
        a = a_arr

    A3 = np.asarray(feedback_matrix, dtype=float)
    if A3.ndim == 2:
        A3 = A3[:, :, np.newaxis]
    return pyFDN.pole_boundaries(delays, _Absorption(), A3, fs, nfft=nfft)

