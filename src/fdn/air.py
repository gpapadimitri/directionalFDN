"""Atmospheric absorption (ISO 9613-1) and its place in a T60 target.

One :class:`Atmosphere` holds the conditions and answers every question the
package asks of the medium: the absorption coefficient, the T60 the air alone
would impose, a boundary target limited by it, the per-round-trip attenuation
split between air and boundary, and the gain over one propagation path.

The coefficient is :func:`pyfar.constants.air_attenuation`, the same standard,
which is the package's reference implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyfar
from numpy.typing import NDArray

from ..utils import db2lin

SPEED_OF_SOUND = 343.0

#: pyfar's lower frequency limit (the standard's frequency-to-pressure ratio);
#: below it alpha is continued as f^2, its low-frequency law.
_PYFAR_MIN_HZ = 50.0


@dataclass(frozen=True)
class AirBudget:
    """Round-trip attenuation of each delay line, split into air and boundary.

    The two shares sum exactly to the loss the realised target needs, and both
    are positive by construction: the air share shortens the target instead of
    competing with it for a fixed budget.

    Attributes
    ----------
    air_db, boundary_db : NDArray, shape (N, n_bands)
        Attenuation in dB per round trip attributed to the medium and to the
        boundary.
    realised_t60_s : NDArray, shape (N, n_bands)
        ``1/T = 1/T_boundary + 1/T_air``: never longer than the boundary target.
    """

    air_db: NDArray
    boundary_db: NDArray
    realised_t60_s: NDArray


@dataclass(frozen=True)
class Atmosphere:
    """Conditions of the propagation medium.

    Parameters
    ----------
    temperature_c : float
        Air temperature in degrees Celsius.
    humidity_percent : float
        Relative humidity in **percent** (0-100). :func:`pyfar.constants.
        air_attenuation` takes a 0-1 fraction; passing ``0.5`` here means 0.5 %
        relative humidity, which is legal and very dry, so no validation catches
        the confusion.
    pressure_kpa : float
        Ambient pressure in kPa (pyfar takes pascal).
    speed_of_sound : float
        Propagation speed in m/s, which turns dB per metre into a decay time.
    """

    temperature_c: float = 20.0
    humidity_percent: float = 50.0
    pressure_kpa: float = 101.325
    speed_of_sound: float = SPEED_OF_SOUND

    def __post_init__(self) -> None:
        if not 0.0 <= self.humidity_percent <= 100.0:
            raise ValueError(
                "Relative humidity must be in 0-100 percent, got "
                f"{self.humidity_percent}. pyfar's air_attenuation takes a 0-1 "
                "fraction; this takes percent."
            )
        if self.temperature_c <= -273.15:
            raise ValueError(
                f"Temperature must be above absolute zero, got {self.temperature_c} C."
            )
        if self.pressure_kpa <= 0.0:
            raise ValueError(
                f"Atmospheric pressure must be positive, got {self.pressure_kpa} kPa."
            )

    def alpha_db_per_m(self, freqs: NDArray) -> NDArray:
        """Absorption coefficient in dB/m, positive, at *freqs* in Hz.

        Below 50 Hz, where pyfar's implementation stops, alpha is continued as
        ``f^2`` from its 50 Hz value. That is the low-frequency law but not
        exact (about 30 % high at 31 Hz for 20 C / 50 %), which is of no
        consequence at 1e-4 dB/m; it keeps the graphic EQ's 31.25 Hz command
        frequency and DC usable.

        Raises
        ------
        ValueError
            For negative frequencies, or an atmosphere pyfar rejects (below
            -73 C).
        """
        f = np.asarray(freqs, dtype=float)
        if np.any(f < 0.0):
            raise ValueError(f"Frequencies must be non-negative, got min {f.min()} Hz.")

        def coefficient(hz: NDArray) -> NDArray:
            alpha, _, _ = pyfar.constants.air_attenuation(
                self.temperature_c,
                hz.ravel(),
                self.humidity_percent / 100.0,
                self.pressure_kpa * 1e3,
            )
            return np.asarray(alpha, dtype=float).reshape(hz.shape)

        #pyfar wants one strictly increasing grid
        grid, inverse = np.unique(f, return_inverse=True)
        out = np.empty_like(grid)
        high = grid >= _PYFAR_MIN_HZ
        if high.any():
            out[high] = coefficient(grid[high])
        if (~high).any():
            edge = coefficient(np.array([_PYFAR_MIN_HZ]))[0]
            out[~high] = edge * (grid[~high] / _PYFAR_MIN_HZ) ** 2
        return out[inverse].reshape(f.shape)

    def t60(self, freqs: NDArray) -> NDArray:
        """Reverberation time the medium alone would impose, in seconds.

        Sound travelling ``c`` metres per second through air losing ``alpha`` dB
        per metre falls 60 dB in ``60 / (alpha c)`` seconds, independently of any
        room: the ceiling on T60 at each frequency. Infinite where alpha is zero
        (DC), so it caps no boundary target.
        """
        rate = self.alpha_db_per_m(freqs) * self.speed_of_sound
        return np.divide(60.0, rate, out=np.full_like(rate, np.inf), where=rate > 0.0)

    def limit(self, t60_boundary_s: NDArray, freqs: NDArray) -> NDArray:
        """Boundary T60 combined with the medium's, in seconds.

        Decay rates add, ``1/T = 1/T_boundary + 1/T_air``, which is the ``4mV``
        air term of the Sabine and Eyring formulae without reference to a room.
        This is where the medium belongs in an FDN: every reference
        implementation designs absorption from reverberation time and delay
        lengths alone, so the medium has to reach the network through the target.

        Parameters
        ----------
        t60_boundary_s : NDArray, shape (n_bands,) or (N, n_bands)
        freqs : NDArray, shape (n_bands,)
        """
        t60_boundary_s = np.asarray(t60_boundary_s, dtype=float)
        return 1.0 / (1.0 / t60_boundary_s + 1.0 / self.t60(freqs))

    def budget(
        self, t60_boundary_s: NDArray, freqs: NDArray, delays: NDArray, fs: float,
    ) -> AirBudget:
        """Split each delay line's round-trip loss between medium and boundary.

        Parameters
        ----------
        t60_boundary_s : NDArray, shape (N, n_bands) or (n_bands,)
            Boundary-only target T60 in seconds.
        freqs : NDArray, shape (n_bands,)
        delays : NDArray, shape (N,)
            Delay-line lengths in samples; the air share is evaluated over the
            propagation path they represent.
        fs : float
        """
        t60_boundary_s = np.atleast_2d(np.asarray(t60_boundary_s, dtype=float))
        delays = np.asarray(delays, dtype=float).ravel()[:, np.newaxis]
        t_air = self.t60(freqs)[np.newaxis, :]
        return AirBudget(
            air_db=60.0 * delays / (fs * t_air),
            boundary_db=60.0 * delays / (fs * np.maximum(t60_boundary_s, 1e-6)),
            realised_t60_s=self.limit(t60_boundary_s, freqs),
        )

    def path_gain(self, freqs: NDArray, distance_m: float) -> NDArray:
        """Linear amplitude gain over one path of *distance_m* metres, per frequency."""
        return db2lin(-self.alpha_db_per_m(freqs) * float(distance_m))
