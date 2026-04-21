from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt
import pyfar as pf
from pyrato.edc import energy_decay_curve_chu_lundeby
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


def _t30_from_edc(edc_db: NDArray, fs: float) -> float:
    """Estimate T30 from a dB-scale energy decay curve.

    Uses pyrato's ISO 3382–compliant linear regression via
    Chu–Lundeby EDC estimation.

    Parameters
    ----------
    edc_db : NDArray
        dB-scale energy decay curve (unused — kept for API compat).
        The actual EDC is recomputed by pyrato from the filtered signal
        passed via :func:`estimate_t30`.
    fs : float
        Sample rate.

    Returns
    -------
    float
        T60 in seconds (extrapolated from T30), or NaN if fit fails.
    """
    # This function is now only called from estimate_t30 which passes
    # a filtered signal as a pyfar.Signal directly through the new path.
    # Kept for backward compatibility if anyone calls it directly.
    from scipy.stats import linregress
    start_db, end_db = -5.0, -35.0
    idx = np.where((edc_db <= start_db) & (edc_db >= end_db))[0]
    if len(idx) < 10:
        return np.nan
    t = idx / fs
    slope, intercept, r, _, _ = linregress(t, edc_db[idx])
    if slope >= 0 or r**2 < 0.8:
        return np.nan
    return -60.0 / slope


def _schroeder_t30(filtered: NDArray, fs: float) -> float:
    """Fallback T30 via Schroeder backward integration.

    Used when pyrato's Chu–Lundeby method fails (e.g. synthetic RIRs
    with zero noise floor).
    """
    from scipy.stats import linregress
    edc = np.cumsum(filtered[::-1] ** 2)[::-1]
    edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
    idx = np.where((edc_db <= -5.0) & (edc_db >= -35.0))[0]
    if len(idx) < 10:
        return np.nan
    t = idx / fs
    slope, intercept, r, _, _ = linregress(t, edc_db[idx])
    if slope >= 0 or r ** 2 < 0.8:
        return np.nan
    return -60.0 / slope


def estimate_t30(ir: NDArray, fs: float, bands: NDArray = OCTAVE_BANDS) -> NDArray:
    """Estimate T30 (extrapolated to T60) per octave band.

    Uses pyrato's Chu–Lundeby EDC estimation and ISO 3382–compliant
    linear regression for robust reverberation time estimation.

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
        # Add dither noise floor for synthetic RIRs whose tail is
        # exactly zero — pyrato's Lundeby algorithm needs a finite
        # noise floor to estimate the intersection time.
        peak = np.max(np.abs(filtered))
        if peak > 0:
            noise_floor = peak * 1e-6  # −120 dB below peak
            rng = np.random.default_rng(42)
            dithered = filtered + rng.standard_normal(len(filtered)) * noise_floor
        else:
            dithered = filtered
        try:
            sig = pf.Signal(dithered[np.newaxis, :], fs)
            edc = energy_decay_curve_chu_lundeby(sig, freq="broadband")
            t30 = reverberation_time_linear_regression(edc, T="T30")
            val = float(t30[0])
            if not np.isnan(val):
                t60[i] = val
                continue
        except Exception:
            pass
        # Fallback: simple Schroeder backward integration
        t60[i] = _schroeder_t30(filtered, fs)
    return t60


def plot_t30(ir: NDArray, fs: float, *, bands: NDArray = OCTAVE_BANDS,
             title: str = "T30", ax=None) -> None:
    """Plot frequency-dependent reverberation time from an impulse response.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    bands : NDArray
        Octave-band center frequencies.
    title : str
        Plot title.
    ax : matplotlib Axes, optional
        Axes to draw on; creates a new figure if *None*.
    """

    t60 = estimate_t30(ir, fs, bands)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    labels = [f"{int(f)}" if f < 1000 else f"{f:.0f}" for f in bands]
    valid = ~np.isnan(t60)
    ax.plot(np.arange(len(bands))[valid], t60[valid], color="red", marker="o")
    ax.set_ylim(bottom=0, top=np.nanmax(t60) * 1.1)
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("T30 (s)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_time_domain(ir: NDArray, fs: float, title: str = "Impulse Response") -> None:
    """Plot an impulse response in the time domain.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(ir)) / fs
    ax.plot(t, ir, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_edc(ir: NDArray, fs: float, title: str = "Energy Decay Curve",
             bands: NDArray | None = None) -> None:
    """Plot the energy decay curve (EDC) in dB, optionally per octave band.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    title : str
        Plot title.
    bands : NDArray, optional
        Octave-band centre frequencies.  If given, per-band EDCs are drawn
        alongside the broadband curve.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(len(ir)) / fs

    # Broadband EDC
    edc = np.cumsum(ir[::-1] ** 2)[::-1]
    edc_db = 10.0 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
    ax.plot(t, edc_db, color="black", linewidth=1.5, label="Broadband")

    # Per-band EDCs
    if bands is not None:
        cmap = plt.cm.coolwarm(np.linspace(0, 1, len(bands)))
        for i, fc in enumerate(bands):
            f_lo = fc / np.sqrt(2)
            f_hi = fc * np.sqrt(2)
            if f_hi >= fs / 2:
                continue
            sos = butter(4, [f_lo, f_hi], btype="band", fs=fs, output="sos")
            filtered = sosfilt(sos, ir)
            band_edc = np.cumsum(filtered[::-1] ** 2)[::-1]
            band_edc_db = 10.0 * np.log10(
                band_edc / (edc[0] + 1e-30) + 1e-30
            )
            label = f"{int(fc)} Hz" if fc < 1000 else f"{fc/1000:.0f} kHz"
            ax.plot(t, band_edc_db, color=cmap[i], linewidth=0.9,
                    alpha=0.8, label=label)
        ax.legend(fontsize=7, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (dB)")
    ax.set_ylim(bottom=-80, top=0)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_frequency_response(ir: NDArray, fs: float, title: str = "Frequency Response", smooth: float = 1/3) -> None:
    """Plot the frequency response of an impulse response.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    title : str
        Plot title.
    smooth : float
        Fractional-octave smoothing width (e.g. 1/3 for third-octave).
        Set to 0 to disable smoothing.
    """
    freqs = rfftfreq(len(ir), 1 / fs)
    mag = np.abs(rfft(ir))

    if smooth > 0 and len(freqs) > 1:
        mag_smooth = np.copy(mag)
        half = smooth / 2.0
        for i in range(1, len(freqs)):
            fc = freqs[i]
            f_lo = fc * 2.0 ** (-half)
            f_hi = fc * 2.0 ** (half)
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if np.any(mask):
                mag_smooth[i] = np.mean(mag[mask])
        mag = mag_smooth

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs[1:], db(mag[1:]), color="blue")
    ax.set_xscale("log")
    ax.set_xlim(left=20, right=fs / 2)
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.show()

# def plot_frequency_response(x, fs, title="Frequency Response"):
#     """
#     Plot the frequency response and time-domain impulse response of a signal.

#     Parameters
#     ----------
#     x : array_like
#         The impulse response signal.
#     fs : int or float
#         The sampling frequency of the signal in Hz.
#     title : str, optional
#         The title for the frequency response plot (default is "Frequency Response").

#     Returns
#     -------
#     None
#         This function displays the plots and does not return any value.
#     """

#     Nfft = 2 ** int(np.ceil(np.log2(len(x))))
#     IR_fft = np.fft.rfft(x, n=Nfft)
#     freqs = np.fft.rfftfreq(Nfft, 1/fs)

#     mag_db = 20 * np.log10(np.abs(IR_fft) / np.max(np.abs(IR_fft)) + 1e-12)

#     plt.figure(figsize=(10, 4))
#     plt.semilogx(freqs, mag_db, linewidth=0.3)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude (dB)')
#     plt.title(title)
    
#     # Set x-axis ticks and labels
#     xtick_positions = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
#     xtick_labels = ['50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
#     plt.xticks(xtick_positions, xtick_labels)
#     plt.xlim([50, 20000])
    
#     plt.grid(True, which="both", ls="--", alpha=0.7)
#     plt.show()