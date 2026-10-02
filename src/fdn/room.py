"""Room-acoustics helpers shared by both directional networks.

* Air absorption (ISO 9613-1) and its place in a T60 target.
* The delay-line range a room's mean free path implies.
* Image-source early reflections for a shoebox room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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

    Implements the full O2 / N2 vibrational-relaxation model, valid roughly
    for 50 Hz - 10 MHz, 0-100 % relative humidity, and a wide temperature
    range. Returns positive values (attenuation).

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz.
    temperature_c : float
        Air temperature in degrees Celsius.
    humidity_percent : float
        Relative humidity in **percent** (0\u2013100).  Note the convention:
        :func:`pyfar.constants.air_attenuation`, the same standard, takes a
        0\u20131 fraction instead.  Passing ``0.5`` here means 0.5\u202f% relative
        humidity, which is legal and very dry, so no validation catches
        that confusion -- only reading this line can.
    pressure_kpa : float
        Ambient atmospheric pressure in kPa (reference 101.325).

    Returns
    -------
    NDArray
        Absorption coefficient \u03b1 in dB/m.

    Raises
    ------
    ValueError
        If the atmosphere is unphysical or the frequencies are negative.
        \u03b1 goes as ``f\u00b2`` and is smooth in humidity, so out-of-range inputs
        otherwise return plausible numbers rather than failing.

    References
    ----------
    ISO 9613-1:1993, "Acoustics \u2014 Attenuation of sound during propagation
    outdoors \u2014 Part 1: Calculation of the absorption of sound by the
    atmosphere."
    """
    f = np.asarray(freqs, dtype=float)
    if np.any(f < 0.0):
        raise ValueError(
            "Frequencies must be non-negative; alpha goes as f^2, so a sign "
            "error would otherwise pass silently."
        )
    if not 0.0 <= humidity_percent <= 100.0:
        raise ValueError(
            f"Relative humidity must be a percentage in [0, 100], got "
            f"{humidity_percent}. pyfar's air_attenuation takes a 0-1 "
            "fraction; this takes percent."
        )
    if temperature_c <= -273.15:
        raise ValueError(
            f"Temperature must be above absolute zero, got {temperature_c} C."
        )
    if pressure_kpa <= 0.0:
        raise ValueError(
            f"Atmospheric pressure must be positive, got {pressure_kpa} kPa."
        )
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
        Infinite where alpha is zero -- DC loses nothing to the medium, so it
        caps no boundary target.
    """
    alpha_db_per_m = air_absorption_db_per_meter(
        np.asarray(freqs, dtype=float),
        temperature_c=temperature_c,
        humidity_percent=humidity_percent,
        pressure_kpa=pressure_kpa,
    )
    rate = alpha_db_per_m * speed_of_sound
    return np.divide(
        60.0, rate, out=np.full_like(rate, np.inf), where=rate > 0.0
    )


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
# Image-source early reflections (shoebox)
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
    delay_samples: int  # l_r -- arrival offset, Eq. 4
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
    :meth:`DirectionalFDN.set_early_reflections`.  A shoebox has ``4n^2 + 2``
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
