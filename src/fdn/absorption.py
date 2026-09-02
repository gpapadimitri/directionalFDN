"""Absorption filters / gains for the FDN feedback loop.

Three implementations are provided:

* :class:`AbsorptionGains` — frequency-independent scalar gains per delay line.
* :class:`AbsorptionFilters` — frequency-dependent IIR filters (warped Prony).
* :class:`AbsorptionSOS` — pre-computed cascaded second-order sections.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, sosfilt, tf2zpk
from pyFDN import SOSFilterBank

from ..utils import db2lin
from .absorption_design import design_absorption_sos
from .filter_fitting import (
    interpolate_magnitude_spectrum,
    prony_warped,
    tf2minphase,
)
from .planewave_dfdn import _SPEED_OF_SOUND, air_absorption_db_per_meter


class Absorption(ABC):
    """Base class for absorption in the FDN feedback loop.

    Parameters
    ----------
    num_delay_lines : int
    delay_line_lengths : NDArray of int
    sample_rate : float
    frame_size : int
    des_t60_ms : float or NDArray
        Desired T60 in **milliseconds**.  A scalar gives homogeneous decay;
        an array gives frequency-dependent decay.
    """

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        des_t60_ms: float | NDArray,
    ) -> None:
        self.num_delay_lines = num_delay_lines
        self.delay_line_lengths = np.asarray(delay_line_lengths).copy()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.des_t60_ms = des_t60_ms

    @abstractmethod
    def process(self, x: NDArray) -> NDArray:
        """Apply absorption to a (num_delay_lines, frame_size) block."""
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset any internal filter state (no-op by default)."""


# ---------------------------------------------------------------------------
# Homogeneous (scalar) absorption
# ---------------------------------------------------------------------------

class AbsorptionGains(Absorption):
    """Per-delay-line scalar gains giving a flat (frequency-independent) T60.

    The gain for delay-line *i* with length *m_i* is::

        g_i = 10^( −60 · m_i / (f_s · T60_seconds · 20) )
    """

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        des_t60_ms: float,
    ) -> None:
        super().__init__(
            num_delay_lines, delay_line_lengths, sample_rate, frame_size, des_t60_ms
        )
        t60_s = float(des_t60_ms) * 1e-3
        self._gains: NDArray = db2lin(
            -60.0 * self.delay_line_lengths / (self.sample_rate * t60_s)
        )

    def process(self, x: NDArray) -> NDArray:
        return x * self._gains[:, np.newaxis]


# ---------------------------------------------------------------------------
# Frequency-dependent absorption (warped Prony IIR)
# ---------------------------------------------------------------------------

class AbsorptionFilters(Absorption):
    """Per-delay-line IIR filters for frequency-dependent T60.

    Uses interpolated magnitude → minimum-phase → warped Prony fitting.

    Parameters
    ----------
    freqs : NDArray
        Frequencies (Hz) at which *des_t60_ms* is specified.
    filter_order : int
        IIR filter order per delay line.
    warp_factor : float
        All-pass warping coefficient (−1, 1).
    """

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        des_t60_ms: NDArray,
        freqs: NDArray,
        filter_order: int,
        warp_factor: float,
        air_absorption: bool = False,
    ) -> None:
        super().__init__(
            num_delay_lines, delay_line_lengths, sample_rate, frame_size, des_t60_ms
        )
        des_t60_ms = np.asarray(des_t60_ms)
        freqs = np.asarray(freqs)
        if len(des_t60_ms) != len(freqs):
            raise ValueError("des_t60_ms and freqs must have the same length.")

        self.freqs = freqs
        self.filter_order = filter_order
        self.rho = warp_factor
        self.air_absorption = air_absorption

        self._num_coeffs = np.zeros((num_delay_lines, filter_order + 1))
        self._den_coeffs = np.zeros_like(self._num_coeffs)
        self._fit_filters()
        self._zi = np.zeros((num_delay_lines, filter_order))

    # -- Internal -----------------------------------------------------------

    def _fit_filters(self, n_fft: int = 4096) -> None:
        """Fit IIR absorption filters for every delay line."""
        air_atten_db = None
        if self.air_absorption:
            air_atten_db = air_absorption_db_per_meter(self.freqs)

        for i in range(self.num_delay_lines):
            #target gain spectrum per delay line. frame_size accounts for
            #the one-block feedback latency, ceil(order/2) for the filter
            #group delay, so the round trip is sized correctly
            target_gains = db2lin(
                -60.0
                * (
                    self.delay_line_lengths[i]
                    + self.frame_size
                    + int(np.ceil(self.filter_order / 2))
                )
                / (self.sample_rate * np.asarray(self.des_t60_ms) * 1e-3)
            )
            if air_atten_db is not None:
                path_m = self.delay_line_lengths[i] / self.sample_rate * _SPEED_OF_SOUND
                target_gains *= db2lin(-air_atten_db * path_m)
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
                # Nudge unstable poles inside the unit circle
                from scipy.signal import zpk2tf
                z, p, k = tf2zpk(b, a)
                p = p * (0.999 / max_pole)
                b, a = zpk2tf(z, p, k)
            self._num_coeffs[i], self._den_coeffs[i] = b, a

    # -- Processing ---------------------------------------------------------

    def process(self, x: NDArray) -> NDArray:
        out = np.empty_like(x)
        for i in range(self.num_delay_lines):
            out[i], self._zi[i] = lfilter(
                self._num_coeffs[i], self._den_coeffs[i], x[i], zi=self._zi[i]
            )
        return out

    def reset(self) -> None:
        self._zi[:] = 0.0


# ---------------------------------------------------------------------------
# Pre-computed SOS absorption
# ---------------------------------------------------------------------------

class AbsorptionSOS(Absorption):
    """Cascaded second-order section (biquad) absorption filters.

    Parameters
    ----------
    sos_coeffs : NDArray
        Shape ``(N, sections, 6)`` — one biquad cascade per delay line.
    """

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        sos_coeffs: NDArray,
        des_t60_ms: float | NDArray | None = None,
    ) -> None:
        super().__init__(
            num_delay_lines, delay_line_lengths, sample_rate, frame_size, des_t60_ms
        )
        sos_coeffs = np.asarray(sos_coeffs, dtype=float)

        # Accept (N, 1, sections, 6) → squeeze channel dim
        if sos_coeffs.ndim == 4 and sos_coeffs.shape[1] == 1:
            sos_coeffs = sos_coeffs[:, 0, :, :]
        if sos_coeffs.ndim != 3 or sos_coeffs.shape[-1] != 6:
            raise ValueError("sos_coeffs must have shape (N, sections, 6).")
        if sos_coeffs.shape[0] != num_delay_lines:
            raise ValueError(
                f"Expected {num_delay_lines} SOS cascades, got {sos_coeffs.shape[0]}."
            )

        self.sos_coeffs = sos_coeffs
        self._zi = np.zeros(
            (num_delay_lines, sos_coeffs.shape[1], 2), dtype=float
        )

    def process(self, x: NDArray) -> NDArray:
        out = np.empty_like(x)
        for i in range(self.num_delay_lines):
            out[i], self._zi[i] = sosfilt(
                self.sos_coeffs[i], x[i], zi=self._zi[i]
            )
        return out

    def reset(self) -> None:
        self._zi[:] = 0.0


# ---------------------------------------------------------------------------
# Graphic-EQ absorption (Schlecht and Habets, DAFx 2017)
# ---------------------------------------------------------------------------

class AbsorptionGEQ(Absorption):
    """Per-delay-line graphic-EQ absorption for frequency-dependent T60.

    Replaces the warped-Prony IIR fit of :class:`AbsorptionFilters` with an
    11-section SOS cascade designed by constrained least squares.  Being SOS,
    it is stable by construction, so no pole nudging is needed.

    The round trip is taken as ``m_i`` samples exactly.  It carries no
    ``frame_size`` correction because it is built for the corrected recursion
    in :mod:`src.fdn.render`, where the feedback re-enters the delay line in
    the same block it was read from.

    Parameters
    ----------
    des_t60_ms : NDArray
        Shape ``(n_bands,)`` shared across delay lines, or ``(N, n_bands)``
        per delay line.  Milliseconds.
    freqs : NDArray, shape (n_bands,)
        Frequencies in Hz at which *des_t60_ms* is specified.
    """

    def __init__(
        self,
        num_delay_lines: int,
        delay_line_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        des_t60_ms: NDArray,
        freqs: NDArray,
    ) -> None:
        super().__init__(
            num_delay_lines, delay_line_lengths, sample_rate, frame_size, des_t60_ms
        )
        self.freqs = np.asarray(freqs, dtype=float)
        self.sos = design_absorption_sos(
            np.asarray(des_t60_ms, dtype=float) * 1e-3,
            self.freqs,
            self.delay_line_lengths,
            sample_rate,
        )
        self._bank = SOSFilterBank(self.sos, num_delay_lines)

    def process(self, x: NDArray) -> NDArray:
        # SOSFilterBank is samples-first; this ABC is channels-first.
        return self._bank.filter(np.asarray(x).T).T

    def reset(self) -> None:
        self._bank = SOSFilterBank(self.sos, self.num_delay_lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_absorption(
    num_delay_lines: int,
    delay_line_lengths: NDArray,
    sample_rate: float,
    frame_size: int,
    des_t60_ms: float | NDArray,
    *,
    freqs: NDArray | None = None,
    filter_order: int | None = None,
    warp_factor: float | None = None,
    method: str = "geq",
) -> Absorption:
    """Convenience factory that picks the right absorption type.

    * Scalar *des_t60_ms* -> :class:`AbsorptionGains`
    * Array *des_t60_ms* with *freqs* -> :class:`AbsorptionGEQ` (default)
    * ``method="prony"`` -> legacy :class:`AbsorptionFilters`, which also
      needs *filter_order* and *warp_factor*

    ``filter_order`` and ``warp_factor`` are accepted and ignored by the GEQ
    path so that existing call sites keep working unchanged.
    """
    if isinstance(des_t60_ms, np.ndarray) and des_t60_ms.ndim > 0:
        if freqs is None:
            raise ValueError("freqs is required for frequency-dependent T60.")
        if method == "geq":
            return AbsorptionGEQ(
                num_delay_lines,
                delay_line_lengths,
                sample_rate,
                frame_size,
                des_t60_ms,
                freqs,
            )
        if method == "prony":
            if filter_order is None or warp_factor is None:
                raise ValueError(
                    "filter_order and warp_factor are required for method='prony'."
                )
            return AbsorptionFilters(
                num_delay_lines,
                delay_line_lengths,
                sample_rate,
                frame_size,
                des_t60_ms,
                freqs,
                filter_order,
                warp_factor,
            )
        raise ValueError(f"Unknown absorption method {method!r}; use 'geq' or 'prony'.")
    return AbsorptionGains(
        num_delay_lines, delay_line_lengths, sample_rate, frame_size, des_t60_ms
    )
