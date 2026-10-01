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

#: Octave-band centres for *analysis* -- the band T60 estimate and the plots.
#:
#: 16 kHz is in and 62.5 Hz is out.  Air absorption does nearly all its work in
#: the top band, while 62.5 Hz sits below what a delay range bracketing the
#: mean free path resolves, and its Schroeder fit is dominated by the modal
#: sparsity of the network rather than by the absorption design.
#:
#: These are the top eight points of the graphic EQ's design grid (see
#: :func:`~src.fdn.fdn_components.geq_band_frequencies`), so a measurement
#: lines up with a ten-value target index for index as ``target[2:]``.
OCTAVE_BANDS = np.array(
    [125, 250, 500, 1000, 2000, 4000, 8000, 16000], dtype=float
)

#: Reference centre for the octave grid; pyFDN counts bands in octaves from it.
_BAND_REF_HZ = 1000.0



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

    Uses :func:`pyFDN.estimate_rt_bands`, i.e. pyFDN's Schroeder fit over its own
    octave filterbank.  For a *measured* response prefer :func:`estimate_t30`,
    whose Chu-Lundeby estimator subtracts the measurement noise floor.

    Parameters
    ----------
    ir : NDArray
        Single-channel impulse response.
    fs : float
        Sample rate in Hz.
    """
    t60, f_centre = pyFDN.estimate_rt_bands(
        ir, fs, start=_band_start(OCTAVE_BANDS), n=len(OCTAVE_BANDS)
    )

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
    curve match what :func:`pyFDN.estimate_rt_bands` fits its slopes to.

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

    bands, f_centre = pyFDN.octave_bands(
        fc=_BAND_REF_HZ,
        start=_band_start(OCTAVE_BANDS),
        n=len(OCTAVE_BANDS),
        fs=fs,
    )
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


def _band_start(bands: NDArray) -> float:
    """Octave offset of *bands*[0] from :data:`_BAND_REF_HZ`, as pyFDN wants it."""
    return float(np.log2(np.asarray(bands, dtype=float)[0] / _BAND_REF_HZ))


def energy_decay_deviation(directional_irs: NDArray) -> NDArray:
    """Energy decay deviation per direction, in dB against time.

    Alary et al., J. Acoust. Soc. Am. 149(5), 2021, Eqs. 4-7: each
    direction's energy decay curve in dB, minus the mean taken across
    directions at the same instant.  Zero everywhere is an isotropic decay;
    a spread that holds open is an anisotropic one; a spread that closes as
    the tail develops is anisotropy relaxing into diffusion, which a single
    T60 per direction cannot show -- the fitting window averages straight
    across it and reports one number or the other.

    Parameters
    ----------
    directional_irs : NDArray, shape (num_directions, num_samples)
        One impulse response per direction, already decomposed.  Beamforming
        is the caller's, since the beam pattern sets how much of the true
        deviation survives the analysis (Alary et al. 2021, Sec. II A) --
        pass :meth:`~src.fdn.planewave_dfdn.PlaneWaveDFDN.render`'s
        ``grid_output`` to leave it out of the measurement entirely.

    Returns
    -------
    NDArray, shape (num_directions, num_samples)
        Deviation in dB.  Sums to zero across directions at every instant.
    """
    irs = np.asarray(directional_irs, dtype=float)
    if irs.ndim != 2:
        raise ValueError(
            f"Expected (num_directions, num_samples), got shape {irs.shape}."
        )
    if irs.shape[0] < 2:
        raise ValueError(
            "The deviation needs at least two directions to have a mean to "
            "deviate from; one direction is its own mean."
        )
    edc_db = 10.0 * np.log10(
        np.stack([pyFDN.edc(row) for row in irs]) + 1e-30
    )
    return edc_db - edc_db.mean(axis=0)


#: Bounds on the FDN loop's pole magnitudes against frequency.  Takes the
#: absorption as the ``(n_sections, 6, N)`` bank of
#: :func:`src.fdn.fdn_components.design_absorption_sos` and works from the
#: cascade, not a flattened transfer function.
pole_bounds = pyFDN.pole_boundaries
