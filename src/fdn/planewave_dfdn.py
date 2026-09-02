"""Frequency-Dependent Directional FDN (Plane-Wave Domain).

Implementation of Alary & Politis (2020) *"Frequency-Dependent Directional
Feedback Delay Network"*, ICASSP 2020.

Architecture
------------
Unlike the JAES 2019 DFDN (which operates in the SH domain with DWM
transforms), this formulation works in the **plane-wave domain** where
each channel of a delay-line group corresponds directly to an incident
direction on a spatial grid of Q points.

Key differences from :class:`DirectionalFDN`:

1. **Per-direction absorption filters** — each of the Q channels in each
   delay-line group gets its own IIR filter parameterised by
   T60(ω, φ, θ) (Eqs. 6–10).
2. **Independent delay lengths** within a group — each of the Q channels
   has a randomly-chosen delay length, improving echo density (Fig. 4).
3. **No DWM transform** — directional shaping is handled entirely by
   the per-channel filters.
4. **Early reflection module** — VBAP-encoded reflections injected at
   tap-in points along the delay lines (Eqs. 3–5).
5. **Output encoding matrix E** — converts grid signals to Ambisonics,
   loudspeaker, or binaural format (Eq. 13).

Usage
-----
>>> from directional_fdn.fdn.planewave_dfdn import PlaneWaveDFDN
>>> pw = PlaneWaveDFDN(
...     sample_rate=48000,
...     num_delay_line_groups=8,
...     sh_order=2,  # Q ≥ (L+1)² grid points
...     frame_size=64,
... )
>>> pw.setup(
...     delay_range_ms=(10, 50),
...     t60_per_direction_ms=t60_map,  # shape (Q, num_bands) or (Q,)
...     freqs=np.array([125, 250, 500, 1000, 2000, 4000, 8000]),
... )
>>> rir = pw.render(source_azimuth=0.0, source_elevation=0.0,
...                  length_seconds=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, tf2zpk, zpk2tf

from scipy.linalg import hadamard

from pyFDN import SOSFilterBank

from . import spatial
from .absorption_design import design_absorption_sos
from .delay_line import generate_delay_lengths
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .filter_fitting import (
    interpolate_magnitude_spectrum,
    prony_warped,
    tf2minphase,
)
from .render import ScalarGains, render_fdn
from ..utils import db2lin


# Speed of sound constant (m/s) — used for air absorption and delay range
_SPEED_OF_SOUND = 343.0


def air_absorption_db_per_meter(
    freqs: NDArray,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> NDArray:
    """Atmospheric absorption coefficient (dB/m) per ISO 9613-1:1993.

    Implements the full O\u2082 / N\u2082 vibrational-relaxation model, valid
    roughly for 50\u202fHz \u2013 10\u202fMHz, 0\u2013100\u202f% relative humidity, and a wide
    temperature range. Returns positive values (attenuation).

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz.
    temperature_c : float
        Air temperature in degrees Celsius.
    humidity_percent : float
        Relative humidity in percent (0\u2013100).
    pressure_kpa : float
        Ambient atmospheric pressure in kPa (reference 101.325).

    Returns
    -------
    NDArray
        Absorption coefficient \u03b1 in dB/m.

    References
    ----------
    ISO 9613-1:1993, "Acoustics \u2014 Attenuation of sound during propagation
    outdoors \u2014 Part 1: Calculation of the absorption of sound by the
    atmosphere."
    """
    f = np.asarray(freqs, dtype=float)
    T = temperature_c + 273.15           # K
    T0 = 293.15                          # reference temperature (20 \u00b0C)
    T01 = 273.16                         # triple point of water
    pa = pressure_kpa
    pr = 101.325                         # reference pressure
    pa_pr = pa / pr

    # Saturation vapour pressure ratio psat/pr (ISO 9613-1, Annex B)
    V = 10.79586 * (1.0 - T01 / T) \
        - 5.02808 * np.log10(T / T01) \
        + 1.50474e-4 * (1.0 - 10.0 ** (-8.29692 * (T / T01 - 1.0))) \
        + 0.42873e-3 * (-1.0 + 10.0 ** (4.76955 * (1.0 - T01 / T))) \
        - 2.2195983
    psat_pr = 10.0 ** V

    # Molar concentration of water vapour, %
    h = humidity_percent * psat_pr / pa_pr

    # Relaxation frequencies (Hz) for O2 and N2
    frO = pa_pr * (
        24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    )
    frN = pa_pr * (T / T0) ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / T0) ** (-1.0 / 3.0) - 1.0))
    )

    # Absorption coefficient \u03b1 in nepers/m, then convert to dB/m
    f2 = f * f
    alpha = (
        8.686 * f2 * (
            1.84e-11 * (1.0 / pa_pr) * np.sqrt(T / T0)
            + (T / T0) ** -2.5 * (
                0.01275 * np.exp(-2239.1 / T) / (frO + f2 / frO)
                + 0.1068 * np.exp(-3352.0 / T) / (frN + f2 / frN)
            )
        )
    )
    return alpha


def air_absorption_split(
    t60_s: NDArray,
    freqs: NDArray,
    delays: NDArray,
    fs: float,
    *,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> tuple[NDArray, NDArray]:
    """Attribute the required round-trip attenuation to medium and boundary.

    Air absorption is part of the T60 budget, not an extra loss on top of it:
    the medium takes its share of the attenuation each round trip must lose,
    and the boundary supplies the rest.  The realised T60 is therefore
    unchanged by this function -- it only says how the loss divides.

    A non-positive boundary share means air absorption alone already decays
    faster than the target, i.e. no real room could sustain that T60 at these
    atmospheric conditions; this raises a ``RuntimeWarning`` naming the worst
    band.

    Parameters
    ----------
    t60_s : NDArray, shape (N, n_bands)
        Target T60 in seconds, one row per delay line.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz.
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.  The air share is computed over the
        physical propagation path only.
    fs : float
        Sample rate in Hz.
    temperature_c, humidity_percent, pressure_kpa : float
        Atmospheric conditions, passed to :func:`air_absorption_db_per_meter`.

    Returns
    -------
    air_db, boundary_db : NDArray, shape (N, n_bands)
        Attenuation in dB per round trip attributed to each.
    """
    import warnings

    t60_s = np.atleast_2d(np.asarray(t60_s, dtype=float))
    delays = np.asarray(delays, dtype=float).ravel()
    alpha_db_per_m = air_absorption_db_per_meter(
        np.asarray(freqs, dtype=float),
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        pressure_kpa=pressure_kpa,
    )

    required_db = 60.0 * delays[:, np.newaxis] / (fs * np.maximum(t60_s, 1e-3))
    path_m = delays[:, np.newaxis] / fs * _SPEED_OF_SOUND
    air_db = alpha_db_per_m[np.newaxis, :] * path_m
    boundary_db = required_db - air_db

    worst = float(np.min(boundary_db))
    if worst < 0.0:
        band = float(np.asarray(freqs)[int(np.argmin(np.min(boundary_db, axis=0)))])
        warnings.warn(
            f"Target T60 exceeds what air absorption alone permits at "
            f"{band:.0f} Hz (boundary share {worst:.3f} dB per round trip, "
            f"i.e. negative). The requested T60 is still realised, but no real "
            f"room could sustain it at these atmospheric conditions -- shorten "
            f"the target T60 in that band, or lower the humidity/temperature.",
            RuntimeWarning,
            stacklevel=2,
        )
    return air_db, boundary_db


def delay_range_from_room(
    room: tuple[float, float, float],
    speed_of_sound: float = _SPEED_OF_SOUND,
    spread: tuple[float, float] = (0.4, 2.0),
) -> tuple[float, float]:
    """Compute FDN delay range (ms) from room dimensions.

    The range brackets the **mean free path** τ = 4V/(Sc) (Sabine), which
    is the standard FDN heuristic (Jot).  The room diagonal is *not* a
    physically meaningful upper bound for inter-reflection delay times.

    Parameters
    ----------
    room : (Lx, Ly, Lz)
        Room dimensions in metres.
    speed_of_sound : float
        Speed of sound in m/s.
    spread : (float, float)
        Multiplicative spread around the mean free path. Default
        (0.4, 2.0) — wide enough to provide ~688 distinct integer delay
        lengths in typical rooms (NQ for sh_order=3, N=8).

    Returns
    -------
    (min_ms, max_ms) : (float, float)
        Delay range in milliseconds.
    """
    Lx, Ly, Lz = room
    V = Lx * Ly * Lz
    S = 2.0 * (Lx * Ly + Ly * Lz + Lx * Lz)
    mean_free_path = 4.0 * V / S
    tau_ms = 1000.0 * mean_free_path / speed_of_sound
    return (spread[0] * tau_ms, spread[1] * tau_ms)



# ---------------------------------------------------------------------------
# Early Reflection Module (Eqs. 3–5)
# ---------------------------------------------------------------------------

@dataclass
class Reflection:
    """One early reflection with its gain, direction, and delay."""
    gain: float
    azimuth: float      # radians
    elevation: float    # radians
    delay_samples: int  # l_r — tap-in offset into the delay lines


def shoebox_reflections(
    room: tuple[float, float, float],
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    sample_rate: float,
    t60_broadband_s: float,
    speed_of_sound: float = 343.0,
    alpha_per_wall: tuple[float, float, float, float, float, float] | None = None,
    model: str = "eyring",
) -> list[Reflection]:
    """Compute first-order image-source reflections for a shoebox room.

    Returns 6 :class:`Reflection` objects (one per wall), sorted by
    arrival time, suitable for :meth:`PlaneWaveDFDN.set_early_reflections`.

    The pressure reflection coefficient is derived from the broadband T60
    via Sabine's formula, so the ER gains are physically consistent with
    the late-reverb absorption:

    .. math::
        \\alpha = \\frac{0.161\\,V}{S\\,T_{60}}, \\qquad
        r_{\\text{wall}} = \\sqrt{1 - \\alpha}

    where *S* is the total wall area and *V* the room volume.

    Parameters
    ----------
    room : (Lx, Ly, Lz)
        Room dimensions in metres.
    source : (sx, sy, sz)
        Source position in metres.
    receiver : (rx, ry, rz)
        Receiver position in metres.
    sample_rate : float
        Sample rate (Hz) — used to convert distances to delay_samples.
    t60_broadband_s : float
        Broadband reverberation time in seconds.  Used to derive the
        wall pressure reflection coefficient via Sabine's formula.
    speed_of_sound : float
        Speed of sound in m/s.

    Returns
    -------
    list[Reflection]
        Six first-order reflections sorted by delay (earliest first).
    """
    import warnings

    Lx, Ly, Lz = room
    sx, sy, sz = source
    rx, ry, rz = receiver

    # Derive wall reflectivity from T60.  Default: Eyring inversion
    # (α = 1 − exp(−0.161 V/(S T60))), which stays physical for high
    # absorption.  Sabine (α = 0.161 V/(S T60)) is available for legacy
    # parity but warns / clips when α ≥ 1.
    if alpha_per_wall is None:
        V = Lx * Ly * Lz
        S = 2.0 * (Lx * Ly + Ly * Lz + Lx * Lz)
        if model == "eyring":
            alpha = 1.0 - np.exp(-0.161 * V / (S * t60_broadband_s))
        elif model == "sabine":
            alpha = 0.161 * V / (S * t60_broadband_s)
        else:
            raise ValueError(f"Unknown model {model!r}; use 'eyring' or 'sabine'.")
        if alpha >= 1.0:
            warnings.warn(
                f"Derived α={alpha:.3f} ≥ 1 (T60={t60_broadband_s:.2f}s is "
                f"too long for room volume V={V:.1f} m³, S={S:.1f} m²). "
                "Clipping to 0.999.",
                RuntimeWarning,
                stacklevel=2,
            )
            alpha = 0.999
        alpha_per_wall = (alpha,) * 6
    if len(alpha_per_wall) != 6:
        raise ValueError("alpha_per_wall must have 6 entries (one per wall).")
    refl = tuple(float(np.sqrt(max(0.0, 1.0 - a))) for a in alpha_per_wall)

    # Image sources: mirror the source across each wall
    image_sources = [
        ((-sx, sy, sz),       refl[0]),   # x = 0
        ((2*Lx - sx, sy, sz), refl[1]),   # x = Lx
        ((sx, -sy, sz),       refl[2]),   # y = 0
        ((sx, 2*Ly - sy, sz), refl[3]),   # y = Ly
        ((sx, sy, -sz),       refl[4]),   # z = 0 (floor)
        ((sx, sy, 2*Lz - sz), refl[5]),   # z = Lz (ceiling)
    ]

    # Direct-sound distance (for gain normalisation)
    d_direct = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)
    d_direct = max(d_direct, 1e-6)

    reflections: list[Reflection] = []
    for (ix, iy, iz), r in image_sources:
        dx = ix - rx
        dy = iy - ry
        dz = iz - rz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = max(dist, 1e-6)

        # Gain: wall reflectivity × absolute inverse-distance law (1/r).
        # Using absolute amplitude (not normalised to d_direct) keeps ref.gain
        # on the same scale as direct_gain = 1/d_direct, so that the ER/direct
        # amplitude ratio is r_wall × d_direct/dist — the physically correct
        # image-source value.
        gain = r / dist

        # Direction from receiver to image source
        azimuth = float(np.arctan2(dy, dx))
        elevation = float(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))

        # Delay relative to direct sound
        delay_s = (dist - d_direct) / speed_of_sound
        delay_samples = max(1, int(np.round(delay_s * sample_rate)))

        reflections.append(Reflection(
            gain=gain,
            azimuth=azimuth,
            elevation=elevation,
            delay_samples=delay_samples,
        ))

    # Sort by arrival time
    reflections.sort(key=lambda r: r.delay_samples)
    return reflections


class EarlyReflectionModule:
    """Early reflection generator per ICASSP 2020 Eqs. 3–5.

    Each reflection is SH-encoded directly into the Ambisonics domain,
    bypassing the plane-wave grid.  This avoids directional quantisation
    artefacts from nearest-neighbour VBAP on coarse grids.

    Parameters
    ----------
    reflections : list[Reflection]
        Reflection definitions.
    sh_order : int
        Ambisonics order for SH encoding.
    """

    def __init__(
        self,
        reflections: list[Reflection],
        sh_order: int,
    ) -> None:
        self.reflections = reflections
        self.sh_order = sh_order
        # Pre-compute per-reflection SH encoding vectors (gain-weighted)
        self.sh_vectors: list[NDArray] = []
        for ref in reflections:
            y = spatial.encode_direction(ref.azimuth, ref.elevation, sh_order)
            self.sh_vectors.append(y * ref.gain)


# ---------------------------------------------------------------------------
# Per-Direction Absorption Filters (Eqs. 6–10)
# ---------------------------------------------------------------------------

class DirectionalAbsorptionFilters:
    """Per-channel IIR absorption for the plane-wave DFDN.

    Each of the total_delay_lines channels gets its own IIR filter fitted
    to the target T60(ω, φ_q, θ_q) and delay length m.

    Parameters
    ----------
    total_delay_lines : int
        N × Q — total number of delay lines.
    delay_lengths : NDArray, shape (N*Q,)
        Per-channel delay lengths in samples.
    sample_rate : float
    frame_size : int
    t60_per_channel_ms : NDArray, shape (N*Q, n_bands) or (N*Q,)
        Target T60 in ms for each delay line at each frequency band.
        If 1D, broadband (scalar per channel).
    freqs : NDArray, optional
        Frequency points (Hz) for frequency-dependent T60.
    filter_order : int
        IIR filter order per channel.
    warp_factor : float
        All-pass warping coefficient.
    air_absorption : bool
        Account for atmospheric absorption *within* the target T60 budget
        (ISO 9613-1).  The realised T60 is unchanged — enabling this does
        not shorten the decay.  What it does is split the per-round-trip
        attenuation into a medium share and a boundary share, exposed as
        :attr:`air_attenuation_db` and :attr:`boundary_attenuation_db`,
        and warn when the requested T60 is longer than air absorption
        alone permits.  Requires frequency-dependent *t60_per_channel_ms*.
    """

    def __init__(
        self,
        total_delay_lines: int,
        delay_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        t60_per_channel_ms: NDArray,
        freqs: NDArray | None = None,
        filter_order: int = 6,
        warp_factor: float = 0.0,
        air_absorption: bool = False,
    ) -> None:
        self.N_total = total_delay_lines
        self.delay_lengths = np.asarray(delay_lengths, dtype=np.int64)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_order = filter_order
        self.rho = warp_factor
        self.air_absorption = air_absorption

        t60 = np.asarray(t60_per_channel_ms, dtype=float)
        if t60.ndim == 1:
            # Shape (N_total,) — broadband scalar per channel
            if air_absorption:
                import warnings

                warnings.warn(
                    "air_absorption is ignored for a broadband T60: the "
                    "absorption coefficient is frequency-dependent, so the "
                    "medium/boundary split cannot be evaluated. Pass a "
                    "frequency-dependent T60 with freqs to use it.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._use_filters = False
            t60_s = np.maximum(t60, 1.0) * 1e-3
            #round trip is m+frame_size, not m, since block processing
            #reinjects feedback one block late. without it the realised
            #t60 runs long by ~(1+frame_size/m)
            self._gains = db2lin(
                -60.0
                * (self.delay_lengths + self.frame_size)
                / (self.sample_rate * t60_s)
            )
        elif t60.ndim == 2 and t60.shape[0] == total_delay_lines:
            # Shape (N_total, n_bands) — frequency-dependent per channel
            self._use_filters = True
            if freqs is None:
                raise ValueError("freqs required for frequency-dependent T60.")
            self.freqs = np.asarray(freqs, dtype=float)
            self._num_coeffs = np.zeros((total_delay_lines, filter_order + 1))
            self._den_coeffs = np.zeros_like(self._num_coeffs)
            self._zi = np.zeros((total_delay_lines, filter_order))
            self._fit_filters(t60)
        else:
            raise ValueError(
                f"t60_per_channel_ms must be shape ({total_delay_lines},) or "
                f"({total_delay_lines}, n_bands), got {t60.shape}."
            )

    def _fit_filters(self, t60_ms: NDArray, n_fft: int = 4096) -> None:
        """Fit one IIR filter per channel from T60(ω) and delay length."""
        import warnings

        # Pre-compute air absorption if enabled
        air_atten_db = None
        if self.air_absorption:
            air_atten_db = air_absorption_db_per_meter(self.freqs)
            self.boundary_attenuation_db = np.zeros(
                (self.N_total, len(self.freqs))
            )
            self.air_attenuation_db = np.zeros_like(self.boundary_attenuation_db)
            worst_deficit = 0.0
            worst_band = None

        for i in range(self.N_total):
            # Eq. 6-8: target gain spectrum for this channel
            t60_s = np.maximum(t60_ms[i], 1.0) * 1e-3
            #frame_size for the one-block feedback latency, ceil(order/2)
            #for the filter group delay, so the round trip is sized right
            effective_len = (
                self.delay_lengths[i]
                + self.frame_size
                + int(np.ceil(self.filter_order / 2))
            )
            # Total attenuation the recirculation must lose per round trip
            # to realise the target T60.
            required_db = 60.0 * effective_len / (self.sample_rate * t60_s)
            if air_atten_db is not None:
                # Air absorption is part of this budget, not an extra loss on
                # top of it: the medium takes its share of ``required_db`` and
                # the boundary supplies the rest, so the realised T60 stays
                # equal to the target.  (The filter is the only loss element
                # in the loop, so it still implements the full ``required_db``;
                # the split below is what the model attributes to each.)
                # The air share is computed over the physical path only —
                # frame_size and filter group delay are block-processing
                # latency, not propagation.
                path_m = self.delay_lengths[i] / self.sample_rate * _SPEED_OF_SOUND
                air_db = air_atten_db * path_m
                boundary_db = required_db - air_db
                self.air_attenuation_db[i] = air_db
                self.boundary_attenuation_db[i] = boundary_db
                # A non-positive boundary share means air absorption alone
                # already decays faster than the target: no real room could
                # sustain this T60 at these atmospheric conditions.
                deficit = float(np.min(boundary_db))
                if deficit < worst_deficit:
                    worst_deficit = deficit
                    worst_band = float(self.freqs[int(np.argmin(boundary_db))])
            target_gains = db2lin(-required_db)
            interp, _ = interpolate_magnitude_spectrum(
                target_gains,
                self.freqs,
                self.sample_rate,
                n_fft=n_fft,
                cutoff=(100.0, 16_000.0),
                rolloff_dc_db=-60.0,
                return_one_sided=True,
            )
            ir = tf2minphase(interp, is_even_fft=True, return_ir=True)
            b, a = prony_warped(
                ir, self.filter_order, self.filter_order, self.rho
            )
            # Stability check: ensure all poles are inside the unit circle
            _, p, _ = tf2zpk(b, a)
            max_pole = np.max(np.abs(p))
            if max_pole >= 1.0:
                z, p, k = tf2zpk(b, a)
                p = p * (0.999 / max_pole)
                b, a = zpk2tf(z, p, k)
            self._num_coeffs[i], self._den_coeffs[i] = b, a

        if air_atten_db is not None and worst_band is not None:
            warnings.warn(
                f"Target T60 exceeds what air absorption alone permits at "
                f"{worst_band:.0f} Hz (boundary share {worst_deficit:.3f} dB "
                f"per round trip, i.e. negative). The requested T60 is still "
                f"realised, but no real room could sustain it at these "
                f"atmospheric conditions — shorten the target T60 in that "
                f"band, or lower the humidity/temperature.",
                RuntimeWarning,
                stacklevel=4,
            )

    def process(self, x: NDArray) -> NDArray:
        """Apply per-channel absorption to a (N_total, frame_size) block."""
        if not self._use_filters:
            return x * self._gains[:, np.newaxis]
        out = np.empty_like(x)
        for i in range(self.N_total):
            out[i], self._zi[i] = lfilter(
                self._num_coeffs[i], self._den_coeffs[i], x[i], zi=self._zi[i]
            )
        return out

    def reset(self) -> None:
        if self._use_filters:
            self._zi[:] = 0.0


# ---------------------------------------------------------------------------
# Plane-Wave DFDN
# ---------------------------------------------------------------------------

class PlaneWaveDFDN:
    """Frequency-Dependent Directional FDN (ICASSP 2020).

    Operates in the plane-wave domain: each channel of a delay-line group
    corresponds to a direction on a Q-point spatial grid.  Per-channel
    absorption filters provide both frequency- and direction-dependent
    T60 control.

    Parameters
    ----------
    sample_rate : float
        Audio sample rate in Hz.
    num_delay_line_groups : int
        *N* — number of delay-line groups.
    sh_order : int
        SH order *L* controlling grid density: Q ≥ (L+1)².
    output_sh_order : int, optional
        SH order of the rendered Ambisonics RIR.  Defaults to *sh_order*.
        Setting this lower than *sh_order* lets you keep a dense plane-wave
        grid (better directional resolution inside the FDN) while emitting
        a lower-order Ambisonics output (e.g. ``sh_order=3``,
        ``output_sh_order=1`` to render a FOA RIR).
        Must satisfy ``output_sh_order <= sh_order``.
    frame_size : int
        Processing block size in samples.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        num_delay_line_groups: int = 8,
        sh_order: int = 2,
        frame_size: int = 64,
        seed: int | None = None,
        output_sh_order: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_delay_line_groups = num_delay_line_groups  # N
        self.sh_order = sh_order
        if output_sh_order is None:
            output_sh_order = sh_order
        if output_sh_order > sh_order:
            raise ValueError(
                f"output_sh_order ({output_sh_order}) must not exceed "
                f"sh_order ({sh_order})."
            )
        self.output_sh_order = output_sh_order
        self.Q = spatial.num_channels(sh_order)  # grid points
        self.frame_size = frame_size
        self.seed = seed

        self.total_delay_lines = num_delay_line_groups * self.Q  # N × Q
        self.num_outputs = (output_sh_order + 1) ** 2  # Ambisonics channels

        # State
        self.delay_length_samps: NDArray | None = None
        self.absorption: ScalarGains | SOSFilterBank | None = None
        self.absorption_sos: NDArray | None = None
        self.air_attenuation_db: NDArray | None = None
        self.boundary_attenuation_db: NDArray | None = None
        self._kronecker_matrix: NDArray | None = None
        self._base_feedback: FeedbackMatrix | None = None
        self._encoding_matrix: NDArray | None = None  # E (Eq. 13)
        self._early_reflections: EarlyReflectionModule | None = None

        # Grid directions
        self.grid_azimuths: NDArray | None = None
        self.grid_elevations: NDArray | None = None

    # ==================================================================
    # Grid generation
    # ==================================================================

    def _init_grid(self) -> None:
        """Generate the Q-point spatial grid.

        Uses proper spherical quadrature grids (t-designs) for orders
        1–3, which satisfy the discrete orthogonality condition for SH
        products.  Falls back to Fibonacci spiral for unsupported orders.
        Note: for order ≥ 2, Q may be larger than (L+1)² to satisfy
        the quadrature condition.
        """
        try:
            self.grid_azimuths, self.grid_elevations = spatial.quadrature_grid(
                self.sh_order
            )
            # Update Q to match actual grid size (may differ from (L+1)²)
            actual_Q = len(self.grid_azimuths)
            if actual_Q != self.Q:
                self.Q = actual_Q
                self.total_delay_lines = self.num_delay_line_groups * self.Q
        except ValueError:
            self.grid_azimuths, self.grid_elevations = spatial.uniform_directions(
                self.Q
            )

    # ==================================================================
    # High-level setup
    # ==================================================================

    def setup(
        self,
        delay_range_ms: tuple[float, float] = (10.0, 50.0),
        t60_per_direction_ms: NDArray | float = 800.0,
        *,
        freqs: NDArray | None = None,
        filter_order: int = 6,
        warp_factor: float = 0.0,
        feedback_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
        air_absorption: bool = False,
        lateral_mix: float = 0.0,
        lambda_group: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise all components of the plane-wave DFDN.

        Parameters
        ----------
        delay_range_ms : (float, float)
            Range for randomly-chosen delay lengths (per channel).
        t60_per_direction_ms : float or NDArray
            Controls the target reverberation time.  Accepted formats:

            * **scalar** — same T60 for all directions and frequencies
              (isotropic, broadband).
            * **shape (n_bands,)** — frequency-dependent T60, same for all
              directions (isotropic, freq-dependent).  Requires *freqs*.
            * **shape (Q,)** — direction-dependent T60, broadband.
            * **shape (Q, n_bands)** — direction- *and* frequency-dependent.
              Row *q* gives the T60 for grid direction *q*; columns match
              *freqs*.  Requires *freqs*.
        freqs : NDArray, optional
            Frequency bands (Hz) — required when T60 is frequency-dependent.
        filter_order : int
            IIR filter order for absorption.
        warp_factor : float
            All-pass warping coefficient for Prony fitting.
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        air_absorption : bool, optional
            Account for atmospheric absorption (ISO 9613-1) within the
            target T60 budget rather than as an extra loss on top of it.
            The realised T60 still equals *t60_per_direction_ms*; what
            changes is that the required attenuation is split into a
            medium share and a boundary share, and a warning is raised if
            the requested T60 is longer than air absorption alone permits.
            Requires *freqs*.  Default ``False``.
        lateral_mix : float, optional
            Inter-direction mixing strength in [0, 1].  ``0`` (default,
            faithful to ICASSP 2020) uses ``A ⊗ I_Q`` and preserves
            per-direction T60.  ``1`` uses ``A ⊗ H_Q`` (full Hadamard
            mixing across directions per feedback iteration), which
            improves late-reverb isotropy at the cost of directional T60
            accuracy.  Intermediate values interpolate via the matrix
            exponential ``M_Q = exp(lateral_mix · log H_Q)``.
        lambda_group : int, optional
            Index of the delay-line group that receives the early-reflection
            re-injection ``s'(n)`` (Eq. 12).  Default ``0``.
        rng : np.random.Generator, optional
        """
        # 0. Spatial grid (may update self.Q for quadrature grids)
        self._init_grid()
        if not (0 <= lambda_group < self.num_delay_line_groups):
            raise ValueError(
                f"lambda_group must be in [0, {self.num_delay_line_groups}), "
                f"got {lambda_group}."
            )
        self._lambda_group = int(lambda_group)

        N, F = self.num_delay_line_groups, self.frame_size
        Q = self.Q  # read AFTER grid init, since Q may have changed

        # 1. Delay-line lengths — independent per channel (ICASSP Sec. 3)
        if rng is None and self.seed is not None:
            rng = np.random.default_rng(self.seed)
        all_lengths = generate_delay_lengths(
            delay_range_ms, N * Q, self.sample_rate, rng=rng
        )
        self.delay_length_samps = all_lengths

        # 2. Feedback matrix A ⊗ M_Q (M_Q = I_Q by default per ICASSP 2020)
        self._init_kronecker_feedback(feedback_type, lateral_mix=lateral_mix)

        # 3. Per-direction absorption
        t60 = np.asarray(t60_per_direction_ms, dtype=float)
        n_bands = len(freqs) if freqs is not None else 0

        if t60.ndim == 0:
            # Scalar → same broadband T60 for all channels
            t60_expanded = np.full(N * Q, float(t60))

        elif t60.ndim == 1 and t60.shape[0] == Q and freqs is None:
            # (Q,) → direction-dependent broadband
            t60_expanded = np.tile(t60, N)

        elif t60.ndim == 1 and freqs is not None and t60.shape[0] == n_bands:
            # (n_bands,) → frequency-dependent, same for all directions
            # Tile to (N*Q, n_bands)
            t60_expanded = np.tile(t60[np.newaxis, :], (N * Q, 1))

        elif t60.ndim == 1 and t60.shape[0] == Q and freqs is not None:
            # (Q,) with freqs given → direction-dependent broadband
            # (treat as broadband even though freqs is passed)
            t60_expanded = np.tile(t60, N)

        elif t60.ndim == 2 and t60.shape[0] == Q:
            # (Q, n_bands) → direction- and frequency-dependent
            t60_expanded = np.tile(t60, (N, 1))

        else:
            raise ValueError(
                f"t60_per_direction_ms must be scalar, (n_bands,), (Q={Q},), "
                f"or (Q={Q}, n_bands), got shape {t60.shape}."
            )
        if freqs is None:
            # Broadband: one scalar gain per channel, round trip = m exactly.
            t60_s = np.maximum(t60_expanded, 1.0) * 1e-3
            gains = db2lin(-60.0 * all_lengths / (self.sample_rate * t60_s))
            self.absorption = ScalarGains(gains)
            self.absorption_sos = None
            if air_absorption:
                import warnings

                warnings.warn(
                    "air_absorption is ignored for a broadband T60: the "
                    "absorption coefficient is frequency-dependent, so the "
                    "medium/boundary split cannot be evaluated. Pass a "
                    "frequency-dependent T60 with freqs to use it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            t60_2d = np.atleast_2d(t60_expanded) * 1e-3
            if t60_2d.shape[0] == 1:
                t60_2d = np.repeat(t60_2d, N * Q, axis=0)
            self.absorption_sos = design_absorption_sos(
                t60_2d, freqs, all_lengths, self.sample_rate
            )
            self.absorption = SOSFilterBank(self.absorption_sos, N * Q)
            if air_absorption:
                self.air_attenuation_db, self.boundary_attenuation_db = (
                    air_absorption_split(t60_2d, freqs, all_lengths, self.sample_rate)
                )

        # 5. Output encoding matrix E (Eq. 13) — default: Ambisonics
        self._init_ambisonics_encoding()

    def set_early_reflections(
        self, reflections: list[Reflection]
    ) -> None:
        """Configure the early reflection module (Eqs. 3–5).

        Must be called after :meth:`setup` so that the grid is available.
        """
        if self.grid_azimuths is None:
            raise RuntimeError("Call setup() before set_early_reflections().")
        self._early_reflections = EarlyReflectionModule(
            reflections, self.output_sh_order,
        )

    # ==================================================================
    # Output encoding (Eq. 13)
    # ==================================================================

    def _init_ambisonics_encoding(self) -> None:
        """Build Ambisonics encoding matrix E.

        Shape: ((L_out+1)², Q) — maps Q grid signals to Ambisonics.

        With a t-design grid of degree 2L and SN3D normalisation,
        ``(4π/Q) Yᵀ Y = diag(1/(2l+1))`` (NOT identity).  We use
        ``E = Yᵀ · √(4π/Q)``, which is the SN3D plane-wave-decomposition
        encoder: it preserves total energy under the spatially-white
        (isotropic) assumption that matches the late reverb's grid
        statistics.  IMPORTANT: this encoder does NOT reproduce a
        unit-amplitude single plane wave from a single grid point—that
        requires the ``(2l+1)`` per-degree compensation applied in the
        ER path (see :meth:`_build_er_grid_signal`).  Do not 'fix' the
        SN3D scaling here without also revisiting that path.

        No SH-order truncation compensation is applied: when
        ``output_sh_order < sh_order`` the higher-order spatial detail
        is discarded; boosting only the reverb path would break the
        balance with the direct and early-reflection paths.
        """
        # Y matrix: (Q, (L_out+1)²) in SN3D — uses output_sh_order
        Y = spatial.encode_directions(
            self.grid_azimuths, self.grid_elevations, self.output_sh_order
        )
        self._encoding_matrix = Y.T * np.sqrt(4.0 * np.pi / self.Q)  # ((L_out+1)², Q)

    # ==================================================================
    # Kronecker feedback
    # ==================================================================

    def _init_kronecker_feedback(
        self, matrix_type: FeedbackMatrixType, lateral_mix: float = 0.0,
    ) -> None:
        """Build A ⊗ M_Q feedback.

        ``M_Q = exp(lateral_mix · log H_Q)`` interpolates unitarily
        between the identity (``lateral_mix=0``, paper-faithful) and the
        Q×Q normalised Hadamard / random-orthogonal matrix
        (``lateral_mix=1``).  ``lateral_mix=0`` preserves per-direction
        T60 exactly (no inter-direction coupling); larger values diffuse
        directional energy at the cost of directional T60 accuracy.
        """
        self._base_feedback = FeedbackMatrix(
            self.num_delay_line_groups,
            matrix_type,
            self.frame_size,
            seed=self.seed,
        )
        Q = self.Q
        if lateral_mix <= 0.0:
            M_Q = np.eye(Q)
        else:
            if Q > 0 and (Q & (Q - 1)) == 0:
                H_Q = hadamard(Q).astype(float) / np.sqrt(Q)
            else:
                rng = np.random.default_rng(
                    self.seed + 7777 if self.seed is not None else None
                )
                M = rng.standard_normal((Q, Q))
                H_Q, _ = np.linalg.qr(M)
            if lateral_mix >= 1.0:
                M_Q = H_Q
            else:
                # Unitary interpolation via matrix log/exp
                from scipy.linalg import logm, expm
                M_Q = np.real(expm(float(lateral_mix) * logm(H_Q)))
        self._kronecker_matrix = np.kron(self._base_feedback.matrix, M_Q)
        # Verify unitarity
        NQ = self.total_delay_lines
        err = np.linalg.norm(
            self._kronecker_matrix.T @ self._kronecker_matrix - np.eye(NQ),
            "fro",
        )
        if err > 1e-8:
            raise RuntimeError(
                f"Kronecker feedback matrix unitarity error: {err:.2e}"
            )

    # ==================================================================
    # Input encoding
    # ==================================================================

    def _build_input_gains(self, source_azimuth: float, source_elevation: float) -> NDArray:
        """Build input vector b_i — omnidirectional per ICASSP 2020 Sec. 3.

        Each of the N×Q delay-line channels receives equal gain
        ``1 / √(N·Q)`` so that the total input energy is unity.
        """
        N, Q = self.num_delay_line_groups, self.Q
        b = np.full((N * Q, 1), 1.0 / np.sqrt(N * Q))
        return b

    def _build_direct_gain(
        self, source_azimuth: float, source_elevation: float,
    ) -> NDArray:
        """Build direct-path gain vector in Ambisonics domain.

        Returns the unit-amplitude SH encoding of the source direction.
        Direct-to-reverberant ratio is controlled by the user-supplied
        ``direct_gain`` and ``reverb_gain`` knobs at render time, not by
        an implicit ``1/√N`` factor (which would silently couple DRR to
        the number of delay-line groups).

        Returns shape ((L+1)², 1).
        """
        y_src = spatial.encode_direction(
            source_azimuth, source_elevation, self.output_sh_order,
        )  # ((L_out+1)²,)
        return y_src.reshape(-1, 1)

    # ==================================================================
    # Output encoding
    # ==================================================================

    def _build_output_gains(self) -> NDArray:
        """Build output matrix c_i for all delay-line groups.

        Per Eq. 11, C_i = diag(c_i) weights each grid direction.
        With unit weights, the sum across N groups gives the correct
        grid-domain signal; spatial normalisation is handled by
        the encoding matrix E (Eq. 13).
        Shape: (Q, N*Q).
        """
        N, Q = self.num_delay_line_groups, self.Q
        c = np.zeros((Q, N * Q))
        for g in range(N):
            s = g * Q
            for q in range(Q):
                c[q, s + q] = 1.0
        return c

    # ==================================================================
    # Rendering
    # ==================================================================

    def render(
        self,
        source_azimuth: float = 0.0,
        source_elevation: float = 0.0,
        length_seconds: float = 1.0,
        direct_gain: float = 1.0,
        er_gain: float = 1.0,
        reverb_gain: float = 1.0,
    ) -> NDArray:
        """Generate an Ambisonics RIR.

        Parameters
        ----------
        source_azimuth, source_elevation : float
            Source direction in radians.
        length_seconds : float
            Duration of the impulse response.
        direct_gain : float
            Linear gain applied to the direct-path impulse.  Set to
            ``1 / distance`` (in metres) to match a 1/r free-field law.
            Defaults to 1.0.
        er_gain : float
            Additional linear scale factor applied to all early reflections.
            Use to adjust the direct-to-reverb balance independently of
            ``wall_reflectivity``.  Defaults to 1.0 (no change).
        reverb_gain : float
            Linear scale factor applied to the FDN input vector ``b``,
            globally attenuating (or boosting) the late-reverb buildup
            from the omnidirectional source-to-FDN path.  Use to lower
            the overall direct-to-reverberant ratio.  Defaults to 1.0.

        Returns
        -------
        NDArray, shape ((L_out+1)², num_samples)
            Ambisonics RIR in ACN/SN3D format.
        """
        num_samples = int(length_seconds * self.sample_rate)
        F = self.frame_size
        num_samples = (num_samples // F) * F

        impulse = np.zeros((1, num_samples))
        impulse[0, 0] = 1.0

        return self.process(
            impulse, source_azimuth, source_elevation,
            direct_gain=direct_gain, er_gain=er_gain,
            reverb_gain=reverb_gain,
        )

    def process(
        self,
        input_data: NDArray,
        source_azimuth: float = 0.0,
        source_elevation: float = 0.0,
        direct_gain: float = 1.0,
        er_gain: float = 1.0,
        reverb_gain: float = 1.0,
    ) -> NDArray:
        """Process mono input through the plane-wave DFDN.

        Signal flow (Fig. 1, Eqs. 11–12 of ICASSP 2020):

        For non-ER groups (i ≠ λ):
            s_i = Z_i [b_i x + Σ A_ij G_j s_j]      (Eq. 12)
        For ER group (i = λ):
            s_λ = s'(n) + Z_λ [Σ A_λj G_j s_j]       (Eq. 12)
        Output:
            y' = Σ C_i G_i s_i + d x                  (Eq. 11)
            y  = E y'                                  (Eq. 13)

        Parameters
        ----------
        input_data : NDArray, shape (1, num_samples)
            Mono input.
        source_azimuth, source_elevation : float
            Source direction in radians.

        Returns
        -------
        NDArray, shape ((L+1)², num_samples)
        """
        self._check_ready()
        if input_data.shape[0] != 1:
            raise ValueError("Input must be mono (1 channel).")

        N, Q = self.num_delay_line_groups, self.Q
        NQ = N * Q
        ir_len = input_data.shape[-1]

        b = self._build_input_gains(source_azimuth, source_elevation) * float(
            reverb_gain
        )                                            # (NQ, 1)
        c = self._build_output_gains()               # (Q, NQ)
        d = self._build_direct_gain(source_azimuth, source_elevation) * float(
            direct_gain
        )                                            # (n_out, 1)
        E = self._encoding_matrix                    # (n_out, Q)

        # Early reflections are a precomputed, time-varying grid signal.
        # process_fdn only takes static B and D, so they ride in as Q extra
        # input channels: routed into delay-line group lambda by B, and tapped
        # straight to the output through the encoder by D.  This reproduces
        # dl_in = b x + er_grid and y = E (c s + er_grid) + d x exactly.
        if self._early_reflections is not None:
            er_grid = self._build_er_grid_signal(input_data, ir_len, er_gain)
            X = np.vstack([input_data, er_grid])                      # (1+Q, len)
            B = np.zeros((NQ, 1 + Q))
            B[:, 0:1] = b
            lam_s = self._lambda_group * Q
            B[lam_s : lam_s + Q, 1:] = np.eye(Q)
            D = np.zeros((self.num_outputs, 1 + Q))
            D[:, 0:1] = d
            D[:, 1:] = E
        else:
            X, B, D = input_data, b, d

        return render_fdn(
            X,
            self.delay_length_samps,
            self._kronecker_matrix,
            B,
            E @ c,
            D,
            absorption=self.absorption,
        )

    # ==================================================================
    # Early reflections buffer
    # ==================================================================

    def _build_er_grid_signal(
        self, input_data: NDArray, ir_len: int, er_gain: float = 1.0,
    ) -> NDArray:
        """Pre-compute the early-reflection signal on the plane-wave grid.

        For each reflection, decomposes a plane wave from its direction
        onto the Q-point grid using
        ``w[q] = (4π/Q) · Y_grid(q) · Y(az,el)``,
        the dual of the encoder ``E = Y_grid^T · √(4π/Q)``.  When this
        grid signal later passes through ``E``, it reconstructs a true
        plane wave in the SH output with amplitude equal to ``ref.gain``
        (wall reflectivity × 1/r), matching the direct-path convention
        ``d = Y(az_src, el_src) · direct_gain``.

        Returns shape (Q, ir_len) — added to ``grid_output`` inside the
        render loop.
        """
        er_buf = np.zeros((self.Q, ir_len))

        if self._early_reflections is None:
            return er_buf

        # SH basis evaluated on the grid at output_sh_order (same order
        # as E, since anything above L_out is annihilated by E anyway).
        Y_grid = spatial.encode_directions(
            self.grid_azimuths, self.grid_elevations, self.output_sh_order,
        )  # (Q, (L_out+1)²)
        # Calibrate the plane-wave decomposition so that (E @ w) == y_dir.
        # With (4π/Q) Yᵀ Y = diag(1/(2l+1)) and E = Yᵀ √(4π/Q):
        #     E @ w = √(Q/(4π)) · diag(1/(2l+1)) · (comp · y_dir)
        # so we need ``comp[lm] = (2l+1) · √(4π/Q)``.
        comp_factor = np.sqrt(4.0 * np.pi / self.Q)
        comp = np.zeros((Y_grid.shape[1],))
        idx_sh = 0
        for l in range(self.output_sh_order + 1):
            n = 2 * l + 1
            comp[idx_sh:idx_sh + n] = (2 * l + 1) * comp_factor
            idx_sh += n

        for ref in self._early_reflections.reflections:
            l_r = ref.delay_samples
            if l_r >= ir_len:
                continue
            end = ir_len - l_r
            y_dir = spatial.encode_direction(
                ref.azimuth, ref.elevation, self.output_sh_order,
            )  # ((L_out+1)²,)
            weights = Y_grid @ (comp * y_dir)  # (Q,)
            er_buf[:, l_r:l_r + end] += (
                ref.gain * er_gain * weights[:, np.newaxis]
                * input_data[0:1, :end]
            )

        return er_buf

    # ==================================================================
    # State management
    # ==================================================================

    def reset(self) -> None:
        """Clear all internal state."""
        if self.absorption_sos is not None:
            self.absorption = SOSFilterBank(self.absorption_sos, self.total_delay_lines)

    def _check_ready(self) -> None:
        if self.delay_length_samps is None:
            raise RuntimeError("Call setup() before processing.")
        if self._kronecker_matrix is None:
            raise RuntimeError("Kronecker feedback matrix not initialised.")
        if self.absorption is None:
            raise RuntimeError("Absorption not initialised.")
        if self._encoding_matrix is None:
            raise RuntimeError("Encoding matrix not initialised.")
