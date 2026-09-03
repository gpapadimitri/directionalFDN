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
from pyFDN import SOSFilterBank
from scipy.linalg import hadamard

from ..utils import db2lin
from . import spatial
from .absorption_design import design_absorption_sos
from .delay_line import generate_delay_lengths
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .render import ScalarGains, render_fdn

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


def air_absorption_t60(
    freqs: NDArray,
    *,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> NDArray:
    """Reverberation time the medium alone would impose, in seconds.

    Sound travelling ``c`` metres per second through air losing
    ``alpha`` dB per metre falls 60 dB in ``60 / (alpha * c)`` seconds,
    independently of any room.  This is the ceiling on T60 at each frequency:
    no boundary treatment can make a room ring longer than its air allows.

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz.
    temperature_c, humidity_percent, pressure_kpa : float
        Atmospheric conditions, passed to :func:`air_absorption_db_per_meter`.
    speed_of_sound : float
        Propagation speed in m/s.

    Returns
    -------
    NDArray
        T60 in seconds, one per frequency.  Falls steeply above a few kHz.
    """
    alpha_db_per_m = air_absorption_db_per_meter(
        np.asarray(freqs, dtype=float),
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        pressure_kpa=pressure_kpa,
    )
    return 60.0 / (alpha_db_per_m * speed_of_sound)


def air_limited_t60(
    t60_boundary_s: NDArray, freqs: NDArray, **atmosphere: float
) -> NDArray:
    """Combine boundary and medium decay into a realisable T60 target.

    Decay rates add, so

    .. math::
        \frac{1}{T_{60}} = \frac{1}{T_\text{boundary}} + \frac{1}{T_\text{air}}

    which is algebraically the ``4mV`` air term of the standard Sabine and
    Eyring formulae, ``T60 = 0.161 V / (S alpha + 4 m V)``, rewritten without
    reference to a specific room.

    This is where air absorption belongs in an FDN.  Every reference
    implementation -- fdnToolbox's ``absorptionGEQ``, ``onePoleAbsorption`` and
    ``firstOrderAbsorption``; the ``pyFDN`` translations of them; the DAFx24
    tutorial's ``absorption_filter`` -- takes reverberation time and delay
    lengths and nothing else.  None models a propagation medium, so the medium
    has to reach the network through the target it is asked to realise.

    Parameters
    ----------
    t60_boundary_s : NDArray
        Target T60 in seconds from boundary absorption alone.  Shape
        ``(n_bands,)`` or ``(N, n_bands)``.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz at which the target is specified.
    **atmosphere
        Forwarded to :func:`air_absorption_t60`.

    Returns
    -------
    NDArray
        Realisable T60 in seconds, never longer than *t60_boundary_s* and
        appreciably shorter above a few kHz.
    """
    t60_boundary_s = np.asarray(t60_boundary_s, dtype=float)
    t_air = air_absorption_t60(freqs, **atmosphere)
    return 1.0 / (1.0 / t60_boundary_s + 1.0 / t_air)


def air_absorption_split(
    t60_boundary_s: NDArray,
    freqs: NDArray,
    delays: NDArray,
    fs: float,
    **atmosphere: float,
) -> tuple[NDArray, NDArray]:
    """Split the round-trip attenuation into medium and boundary shares.

    A diagnostic companion to :func:`air_limited_t60`, saying how much of each
    delay line's per-round-trip loss the air supplies and how much the
    boundary does.  Because decay rates add, so do the attenuations, and the
    two shares sum exactly to the loss required by the combined target.

    Both shares are positive by construction.  An earlier formulation
    subtracted the air share from a fixed budget and could report a negative
    boundary share, meaning the requested T60 was longer than air alone
    permits; under the combined model that situation cannot arise, because the
    air share shortens the target instead of competing with it.

    Parameters
    ----------
    t60_boundary_s : NDArray, shape (N, n_bands) or (n_bands,)
        Boundary-only target T60 in seconds.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz.
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.  The air share is computed over the
        physical propagation path they represent.
    fs : float
        Sample rate in Hz.
    **atmosphere
        Forwarded to :func:`air_absorption_t60`.

    Returns
    -------
    air_db, boundary_db : NDArray, shape (N, n_bands)
        Attenuation in dB per round trip attributed to each.
    """
    t60_boundary_s = np.atleast_2d(np.asarray(t60_boundary_s, dtype=float))
    delays = np.asarray(delays, dtype=float).ravel()[:, np.newaxis]
    t_air = air_absorption_t60(freqs, **atmosphere)[np.newaxis, :]

    air_db = 60.0 * delays / (fs * t_air)
    boundary_db = 60.0 * delays / (fs * np.maximum(t60_boundary_s, 1e-6))
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
    """One early reflection: where it comes from, when, and how loud.

    ``gain`` is the broadband amplitude, always populated.  ``band_gains``, when
    present, carries the same quantity resolved per frequency -- wall loss, air
    loss and the 1/r spread combined -- and supersedes ``gain`` at render time,
    where the reflection becomes a short filtered arrival rather than a single
    sample.
    """

    gain: float
    azimuth: float      # radians
    elevation: float    # radians
    delay_samples: int  # l_r -- tap-in offset into the delay lines
    order: int = 1      # number of wall bounces
    band_gains: NDArray | None = None   # (n_bands,) linear amplitude
    freqs: NDArray | None = None        # (n_bands,) Hz, matching band_gains


def _wall_reflectivity(
    room: tuple[float, float, float],
    t60_s: NDArray,
    model: str = "eyring",
) -> NDArray:
    """Pressure reflection coefficient implied by a reverberation time.

    Inverts Sabine or Eyring for the average absorption coefficient, so the
    early reflections lose energy at the same rate the late field does.
    Accepts a scalar or a per-band array and returns the matching shape.
    """
    import warnings

    lx, ly, lz = room
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + ly * lz + lx * lz)
    t60_s = np.asarray(t60_s, dtype=float)

    if model == "eyring":
        alpha = 1.0 - np.exp(-0.161 * volume / (surface * t60_s))
    elif model == "sabine":
        alpha = 0.161 * volume / (surface * t60_s)
    else:
        raise ValueError(f"Unknown model {model!r}; use 'eyring' or 'sabine'.")

    if np.any(alpha >= 1.0):
        warnings.warn(
            f"Derived absorption reaches {float(np.max(alpha)):.3f} >= 1 for "
            f"V={volume:.1f} m3, S={surface:.1f} m2; the requested T60 is too "
            "long for this room. Clipping to 0.999.",
            RuntimeWarning,
            stacklevel=3,
        )
        alpha = np.minimum(alpha, 0.999)
    return np.sqrt(np.maximum(0.0, 1.0 - alpha))


def _image_sources(
    room: tuple[float, float, float],
    source: tuple[float, float, float],
    order: int,
) -> list[tuple[tuple[float, float, float], int]]:
    """Enumerate shoebox image sources up to *order* reflections.

    Uses the Allen and Berkley indexing: along each axis an image sits at
    ``2*m*L + (1-2q)*s`` for integer ``m`` and parity ``q``, having bounced
    ``|m - q|`` times off the wall at 0 and ``|m|`` times off the wall at L.
    The reflection order is the sum of those counts over the three axes, which
    yields the expected ``4n^2 + 2`` images at order ``n``.

    Returns ``(position, order)`` pairs, excluding the direct path.
    """
    from itertools import product

    span = order + 1
    images: list[tuple[tuple[float, float, float], int]] = []
    for mx, my, mz in product(range(-span, span + 1), repeat=3):
        for qx, qy, qz in product((0, 1), repeat=3):
            position = []
            total = 0
            for m, q, s, length in (
                (mx, qx, source[0], room[0]),
                (my, qy, source[1], room[1]),
                (mz, qz, source[2], room[2]),
            ):
                position.append(2 * m * length + (1 - 2 * q) * s)
                total += abs(m - q) + abs(m)
            if 0 < total <= order:
                images.append((tuple(position), total))
    return images


def shoebox_reflections(
    room: tuple[float, float, float],
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    sample_rate: float,
    t60_broadband_s: float,
    speed_of_sound: float = _SPEED_OF_SOUND,
    model: str = "eyring",
    *,
    order: int = 1,
    freqs: NDArray | None = None,
    t60_per_band_s: NDArray | None = None,
    air_absorption: bool = True,
    temperature_c: float = 20.0,
    humidity_percent: float = 50.0,
    pressure_kpa: float = 101.325,
) -> list[Reflection]:
    """Image-source early reflections for a shoebox room.

    Returns :class:`Reflection` objects sorted by arrival time, ready for
    :meth:`PlaneWaveDFDN.set_early_reflections`.  A shoebox has ``4n^2 + 2``
    images at reflection order *n*, so the returned count is 6, 24, 62 or 128
    for ``order`` 1 to 4.

    Wall loss comes from inverting Eyring (or Sabine) on the supplied
    reverberation time, so the early field decays at the same rate as the late
    one.  Supplying *freqs* and *t60_per_band_s* resolves that per band and
    populates :attr:`Reflection.band_gains`, at which point each reflection
    renders as a short filtered arrival rather than a bare impulse.

    Parameters
    ----------
    room : (Lx, Ly, Lz)
        Room dimensions in metres.
    source, receiver : (x, y, z)
        Positions in metres.
    sample_rate : float
        Sample rate in Hz, used to convert arrival times to samples.
    t60_broadband_s : float
        Broadband reverberation time, used for the broadband ``gain`` and as
        the fallback when *t60_per_band_s* is not given.
    speed_of_sound : float
        Propagation speed in m/s.
    model : str
        ``"eyring"`` (default) or ``"sabine"`` for the absorption inversion.
    order : int
        Maximum reflection order.  First order alone leaves the first tens of
        milliseconds nearly empty, which is what makes individual arrivals
        stand out; order 2 to 3 fills it.
    freqs : NDArray, optional
        Frequencies in Hz for the per-band treatment.  Requires
        *t60_per_band_s*.
    t60_per_band_s : NDArray, optional
        Reverberation time per band, in seconds, giving frequency-dependent
        wall loss.
    air_absorption : bool
        Apply ISO 9613-1 atmospheric loss over each reflection's own path
        length.  Only meaningful per band, so it is ignored without *freqs*.
    temperature_c, humidity_percent, pressure_kpa : float
        Atmospheric conditions.

    Returns
    -------
    list[Reflection]
        Sorted by arrival time, earliest first.
    """
    receiver_arr = np.asarray(receiver, dtype=float)
    direct_distance = max(
        float(np.linalg.norm(np.asarray(source, dtype=float) - receiver_arr)), 1e-6
    )

    per_band = freqs is not None and t60_per_band_s is not None
    if per_band:
        freqs = np.asarray(freqs, dtype=float)
        r_wall_band = _wall_reflectivity(room, t60_per_band_s, model)
        alpha_air = (
            air_absorption_db_per_meter(
                freqs,
                temperature_c=temperature_c,
                humidity_percent=humidity_percent,
                pressure_kpa=pressure_kpa,
            )
            if air_absorption
            else np.zeros_like(freqs)
        )
    r_wall = float(_wall_reflectivity(room, t60_broadband_s, model))

    reflections: list[Reflection] = []
    for position, reflection_order in _image_sources(room, source, order):
        offset = np.asarray(position, dtype=float) - receiver_arr
        distance = max(float(np.linalg.norm(offset)), 1e-6)

        band_gains = None
        if per_band:
            # Wall loss compounds once per bounce; air loss accrues over the
            # whole path the reflection actually travelled.
            band_gains = (
                r_wall_band**reflection_order
                * db2lin(-alpha_air * distance)
                / distance
            )

        reflections.append(
            Reflection(
                gain=r_wall**reflection_order / distance,
                azimuth=float(np.arctan2(offset[1], offset[0])),
                elevation=float(
                    np.arctan2(offset[2], np.hypot(offset[0], offset[1]))
                ),
                delay_samples=max(
                    1,
                    int(
                        round(
                            (distance - direct_distance)
                            / speed_of_sound
                            * sample_rate
                        )
                    ),
                ),
                order=reflection_order,
                band_gains=band_gains,
                freqs=freqs if per_band else None,
            )
        )

    reflections.sort(key=lambda r: r.delay_samples)
    return reflections


def _reflection_fir(
    reflection: Reflection, sample_rate: float, num_taps: int = 63
) -> NDArray:
    """Linear-phase FIR realising a reflection's per-band magnitude.

    ``firwin2`` needs the gain specified from DC to Nyquist, so the band edges
    are held flat outside the supplied range.  The result is symmetric, with a
    group delay of ``(num_taps - 1) / 2`` samples that the caller removes so
    the reflection still arrives when the geometry says it should.
    """
    from scipy.signal import firwin2

    freqs = np.asarray(reflection.freqs, dtype=float)
    gains = np.asarray(reflection.band_gains, dtype=float)
    nyquist = sample_rate / 2.0

    inside = freqs < nyquist
    freqs, gains = freqs[inside], gains[inside]
    grid = np.concatenate(([0.0], freqs, [nyquist]))
    values = np.concatenate(([gains[0]], gains, [gains[-1]]))
    return firwin2(num_taps, grid / nyquist, values)


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
# Plane-Wave DFDN
# ---------------------------------------------------------------------------

class _EarlyReflectionInjection:
    """Adds the early reflections to delay-line group lambda after its delay.

    Alary and Politis (ICASSP 2020) Eq. 12::

        s_i = s'(z) + Z_i(z)[sum_j A_ij G_j s_j],   i = lambda

    The reflections join the group's *post-delay* state, so the ``l_r`` offsets
    already carried by ``s'`` set their arrival times, and they then pass
    through the absorption filter ``G_lambda`` and the output matrix
    ``C_lambda`` exactly like the recirculating signals -- and recirculate
    through ``A`` as well.  One injection point serves both purposes, where an
    input-side injection would delay them by a whole delay line and a separate
    output tap would let them bypass the absorption.

    ``process_fdn`` applies its ``post_delay`` hook to the delay-line output,
    which is precisely that point, and calls it once per block in order, so the
    running position below stays aligned with the render.
    """

    def __init__(
        self,
        absorption: object,
        er_grid: NDArray,
        lambda_group: int,
        num_channels: int,
    ) -> None:
        self._absorption = absorption
        self._er = er_grid
        self._slice = slice(lambda_group * num_channels,
                            (lambda_group + 1) * num_channels)
        self._position = 0

    def filter(self, block: NDArray) -> NDArray:
        block = np.array(block, dtype=float, copy=True)
        n = block.shape[0]
        segment = self._er[:, self._position : self._position + n]
        block[: segment.shape[1], self._slice] += segment.T
        self._position += n
        return self._absorption.filter(block)


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

    Attributes
    ----------
    realised_t60_s : NDArray or None
        The T60 actually designed for, shape ``(N*Q, n_bands)``.  Equals the
        requested target unless ``air_absorption`` shortened it.
    air_attenuation_db, boundary_attenuation_db : NDArray or None
        Per-round-trip attenuation in dB attributed to the medium and to the
        boundary, shape ``(N*Q, n_bands)``.  Set only when ``air_absorption``
        is enabled; they sum to the loss the realised target requires.
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
        self.realised_t60_s: NDArray | None = None
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
            Ignored. Retained for API compatibility; absorption is now
            designed as a graphic EQ (see :mod:`src.fdn.absorption_design`),
            which takes no filter order or warping coefficient.
        warp_factor : float
            Ignored. Retained for API compatibility; absorption is now
            designed as a graphic EQ (see :mod:`src.fdn.absorption_design`),
            which takes no filter order or warping coefficient.
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        air_absorption : bool, optional
            Treat *t60_per_direction_ms* as the **boundary-only** target and
            add atmospheric absorption (ISO 9613-1) on top of it, so the
            realised decay is ``1/T = 1/T_boundary + 1/T_air`` -- the ``4mV``
            term of Sabine and Eyring.  The realised T60 is therefore shorter
            than requested, appreciably so above a few kHz, and is recorded in
            :attr:`realised_t60_s`.  The medium and boundary shares of each
            round trip are exposed as :attr:`air_attenuation_db` and
            :attr:`boundary_attenuation_db`.  Requires *freqs*; ignored with a
            warning for a broadband T60, whose single value cannot carry a
            frequency-dependent medium.  Default ``False``.
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

        N = self.num_delay_line_groups
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
        # Branch on the shape actually computed above, not on whether freqs
        # was supplied: a (Q,) T60 is broadband-per-channel even when freqs
        # is passed (see the branch above), so it must take the ScalarGains
        # path regardless.
        self.air_attenuation_db = None
        self.boundary_attenuation_db = None
        self.realised_t60_s = None
        if t60_expanded.ndim == 1:
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
            t60_2d = t60_expanded * 1e-3
            if air_absorption:
                # The medium enters through the target, which is the only place
                # an FDN has for it: every reference implementation designs
                # absorption from reverberation time and delay lengths alone.
                # Summing decay rates is the 4mV term of Sabine and Eyring.
                self.air_attenuation_db, self.boundary_attenuation_db = (
                    air_absorption_split(t60_2d, freqs, all_lengths, self.sample_rate)
                )
                t60_2d = air_limited_t60(t60_2d, freqs)
            self.realised_t60_s = np.broadcast_to(
                t60_2d, (N * Q, np.size(freqs))
            ).copy()
            self.absorption_sos = design_absorption_sos(
                t60_2d, freqs, all_lengths, self.sample_rate
            )
            self.absorption = SOSFilterBank(self.absorption_sos, N * Q)

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
        """Build the Ambisonics encoding matrix E of Eq. 13.

        ``E = Y(L, U_grid)``: the real spherical harmonics of the grid
        directions, with no quadrature scaling.  Under this convention a grid
        channel of amplitude ``s`` encodes to ``s * y(theta_q)`` exactly, so the
        direct sound, the early reflections and the recirculating signals --
        all of which live on the grid -- share one level convention and their
        balance does not shift with the grid size.

        Shape ``((L_out+1)^2, Q)``.  No compensation is applied when
        ``output_sh_order < sh_order``: the higher-order spatial detail is
        discarded, and boosting only the reverb path would break that balance.
        """
        Y = spatial.encode_directions(
            self.grid_azimuths, self.grid_elevations, self.output_sh_order
        )
        self._encoding_matrix = Y.T  # ((L_out+1)^2, Q)

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
                from scipy.linalg import expm, logm
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
        """Direct path panned onto the grid, per Eq. 3.

        ``d = vbap(U_grid, (phi, theta))``.  The direct sound joins the grid
        like every other signal and reaches the output through ``E``, so it
        needs no separate spherical-harmonic convention of its own.

        Returns shape ``(Q, 1)``.  Absolute level stays with the caller's
        ``direct_gain``; nothing here divides by the number of delay lines.
        """
        gains = spatial.vbap_gains(
            self.grid_azimuths, self.grid_elevations,
            source_azimuth, source_elevation,
        )
        return gains.reshape(-1, 1)

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

        # Eq. 11: y' = sum_i C_i G_i s_i + d x, then y = E y'.  Everything
        # lives on the grid, so E is the only encoder in the signal path.
        absorption = (
            SOSFilterBank(self.absorption_sos, NQ)
            if self.absorption_sos is not None
            else self.absorption
        )
        if self._early_reflections is not None:
            # Eq. 12: the reflections replace this group's input injection and
            # join its state after the delay, so they are wrapped around the
            # absorption hook rather than routed through B and D.
            absorption = _EarlyReflectionInjection(
                absorption,
                self._build_er_grid_signal(input_data, ir_len, er_gain),
                self._lambda_group,
                Q,
            )

        return render_fdn(
            input_data,
            self.delay_length_samps,
            self._kronecker_matrix,
            b,
            E @ c,
            E @ d,
            absorption=absorption,
        )

    # ==================================================================
    # Early reflections buffer
    # ==================================================================

    def _build_er_grid_signal(
        self, input_data: NDArray, ir_len: int, er_gain: float = 1.0,
    ) -> NDArray:
        """Early-reflection signal on the plane-wave grid, per Eqs. 4-5.

        Each reflection is panned onto the grid with VBAP and delayed by its
        own ``l_r``::

            lambda_r = lambda_r * vbap(U_grid, (phi_r, theta_r))
            s'_r(n + l_r) = lambda_r x(n)
            s'(n) = sum_r s'_r(n)

        No spherical-harmonic compensation is involved: the reflections live on
        the grid alongside the recirculating signals and are encoded by the
        same ``E``.  A reflection carrying per-band gains contributes a short
        filtered arrival instead of a bare impulse, its group delay removed so
        the geometry still sets the arrival time.

        Returns shape ``(Q, ir_len)``.
        """
        er_buf = np.zeros((self.Q, ir_len))
        if self._early_reflections is None:
            return er_buf

        for ref in self._early_reflections.reflections:
            l_r = ref.delay_samples
            if l_r >= ir_len:
                continue
            weights = spatial.vbap_gains(
                self.grid_azimuths, self.grid_elevations,
                ref.azimuth, ref.elevation,
            )
            if ref.band_gains is not None:
                taps = _reflection_fir(ref, self.sample_rate)
                start = max(0, l_r - (taps.size - 1) // 2)
                signal = np.convolve(input_data[0], taps)[: ir_len - start]
            else:
                start = l_r
                signal = ref.gain * input_data[0, : ir_len - start]

            er_buf[:, start : start + signal.size] += (
                er_gain * weights[:, np.newaxis] * signal[np.newaxis, :]
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
