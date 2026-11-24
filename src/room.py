"""Room-acoustics helpers shared by both directional networks.

* The delay-line range a room's mean free path implies.
* Image-source early reflections for a shoebox room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .air import SPEED_OF_SOUND as _SPEED_OF_SOUND
from .air import Atmosphere


def delay_range_from_room(
    room: tuple[float, float, float],
    speed_of_sound: float = _SPEED_OF_SOUND,
    spread: tuple[float, float] = (0.4, 2.0),
) -> tuple[float, float]:
    """Compute FDN delay range (ms) from room dimensions.

    The range brackets the **mean free path** tau = 4V/(Sc) (Sabine), which
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
        (0.4, 2.0) -- wide enough to give every delay line of a typical
        network (N*Q = 208 for sh_order=3, N=8) a distinct integer length.

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
    air_absorption: bool | Atmosphere = False,
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
    air_absorption : bool or Atmosphere
        Apply ISO 9613-1 atmospheric loss over each reflection's own path
        length: ``True`` for the default :class:`~src.air.Atmosphere`, or an
        instance for other conditions. Only meaningful per band, so it is
        ignored without *freqs*. Default ``False``.

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
        atmosphere = (
            air_absorption
            if isinstance(air_absorption, Atmosphere)
            else Atmosphere() if air_absorption else None
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
            band_gains = r_wall_band**reflection_order / distance
            if atmosphere is not None:
                band_gains = band_gains * atmosphere.path_gain(freqs, distance)

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
