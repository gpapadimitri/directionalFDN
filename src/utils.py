from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt
import pyfar as pf
from pyrato.edc import energy_decay_curve_chu_lundeby, schroeder_integration
from pyrato.parameters import reverberation_time_linear_regression


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

