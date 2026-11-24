"""Atmospheric absorption after ISO 9613-1, through pyfar."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyfar as pf
from numpy.typing import NDArray

#: pyfar's lowest frequency for ISO 9613-1
_MIN_HZ = 50.0


@dataclass(frozen=True)
class Atmosphere:
    """Propagation medium, in the units of :func:`pyfar.constants.air_attenuation`.

    Parameters
    ----------
    temperature : float
        Degrees Celsius.
    relative_humidity : float
        Fraction from 0 to 1.
    atmospheric_pressure : float
        Pascal.
    """

    temperature: float = pf.constants.reference_air_temperature_celsius
    relative_humidity: float = 0.5
    atmospheric_pressure: float = pf.constants.reference_atmospheric_pressure

    def __post_init__(self) -> None:
        #pyfar validates the conditions
        self.alpha_db_per_m(1000.0)

    def alpha_db_per_m(self, freqs: NDArray | float) -> NDArray:
        """Absorption coefficient in dB/m at *freqs* in Hz.

        Below 50 Hz, where pyfar stops, alpha follows its low-frequency law
        ``f^2`` from the 50 Hz value: about 30 % high at 31.25 Hz, at 1e-4 dB/m.
        """
        f = np.asarray(freqs, dtype=float)
        if np.any(f < 0.0):
            raise ValueError(f"Frequencies must be non-negative, got min {f.min()} Hz.")
        #pyfar wants one strictly increasing grid from 50 Hz
        grid, inverse = np.unique(np.maximum(f, _MIN_HZ), return_inverse=True)
        alpha, _, _ = pf.constants.air_attenuation(
            self.temperature, grid, self.relative_humidity, self.atmospheric_pressure
        )
        alpha = np.asarray(alpha, dtype=float).ravel()[inverse].reshape(f.shape)
        return alpha * np.minimum(f / _MIN_HZ, 1.0) ** 2
