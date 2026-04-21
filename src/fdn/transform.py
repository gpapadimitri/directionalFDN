"""Directional Weighting Matrix (DWM) transform T_i for anisotropic decay.

Implements Section 3 of Alary et al. (2019) "Directional Feedback Delay Network".

The DWM modifies the direction-dependent energy distribution of ambisonic
signals in the recirculation path.  Given a target direction-dependent
reverberation time T60(phi, theta), the transform creates per-group Q x Q
matrices that recursively shape the spatial energy decay.

Construction follows the truncated discrete SHT form (Eqs. 35-36)::

    T_i = pinv(Y_L) @ diag(g_i) @ Y_L

where Y_L is the K x Q SH matrix at K uniform directions, and g_i is the
direction-dependent gain for group *i* computed from Eqs. 24-26.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import spatial


class DirectionalWeighting:
    """Directional Weighting Matrix (DWM) per Alary et al. 2019, Section 3.

    Computes per-group Q x Q transform matrices that encode direction-
    dependent absorption into the SH domain.  These are applied in the
    DFDN recirculation path *after* the base absorption g_i (Eq. 23).

    Parameters
    ----------
    num_groups : int
        Number of delay-line groups (*N*).
    ambisonics_order : int
        Ambisonics order *L*.
    group_delay_lengths : NDArray
        Per-group delay-line lengths in samples, shape ``(N,)``.
    sample_rate : float
        Audio sample rate in Hz.
    t60_at_directions_ms : NDArray
        T60 in milliseconds at each sampling direction, shape ``(K,)``.
    azimuths : NDArray
        Azimuth of each sampling direction in radians, shape ``(K,)``.
    elevations : NDArray
        Elevation of each sampling direction in radians, shape ``(K,)``.
    """

    def __init__(
        self,
        num_groups: int,
        ambisonics_order: int,
        group_delay_lengths: NDArray,
        sample_rate: float,
        t60_at_directions_ms: NDArray,
        azimuths: NDArray,
        elevations: NDArray,
    ) -> None:
        self.N = num_groups
        self.L = ambisonics_order
        self.Q = spatial.num_channels(ambisonics_order)
        self.group_delay_lengths = np.asarray(group_delay_lengths, dtype=np.int64)
        self.sample_rate = sample_rate

        t60_s = np.asarray(t60_at_directions_ms, dtype=float) * 1e-3
        azimuths = np.asarray(azimuths, dtype=float)
        elevations = np.asarray(elevations, dtype=float)
        K = len(t60_s)

        if len(azimuths) != K or len(elevations) != K:
            raise ValueError("t60, azimuths, elevations must have the same length.")
        if K < self.Q:
            raise ValueError(
                f"Need at least Q={self.Q} sampling directions, got {K}."
            )

        # T60_max (Eq. 6)
        self.t60_max_s = float(np.max(t60_s))
        self.t60_max_ms = self.t60_max_s * 1e3

        # SH matrix at sampling directions: K x Q
        Y = spatial.encode_directions(azimuths, elevations, order=self.L)

        # Pseudo-inverse for robust SHT: Q x K
        Y_pinv = np.linalg.pinv(Y)

        # Null-space complement: ensures SH channels unresolvable by the
        # sampling geometry (e.g. Z when all elevations are 0) pass through
        # unchanged instead of being zeroed out.
        P = Y_pinv @ Y  # projector onto the row space of Y
        null_complement = np.eye(self.Q) - P

        # Build per-group DWM T_i (Eqs. 24-26, truncated Eq. 35-36)
        self.transform_matrices: list[NDArray] = []
        for i in range(self.N):
            g_sampled = self._directional_gain(
                t60_s,
                self.t60_max_s,
                int(self.group_delay_lengths[i]),
                self.sample_rate,
            )
            # Truncated DWM (Eq. 36): T_i = pinv(Y) @ diag(g_i) @ Y
            # + null-space identity for unresolvable channels
            T_i = Y_pinv @ (g_sampled[:, np.newaxis] * Y) + null_complement
            self.transform_matrices.append(T_i)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _directional_gain(
        t60_s: NDArray,
        t60_max_s: float,
        delay_length: int,
        sample_rate: float,
    ) -> NDArray:
        """Compute direction-dependent gain g_i(phi, theta) per Eqs. 24-26.

        Parameters
        ----------
        t60_s : NDArray, shape (K,)
            Direction-dependent T60 in seconds.
        t60_max_s : float
            Maximum T60 in seconds (used for base absorption).
        delay_length : int
            Delay-line length m_i in samples.
        sample_rate : float
            Audio sample rate in Hz.

        Returns
        -------
        NDArray, shape (K,)
            Per-direction gain for this delay-line group.
        """
        # delta_T60 = T60_max - T60  (Eq. 6)
        delta_t60 = t60_max_s - t60_s

        # Avoid division by zero
        safe_t60 = np.maximum(t60_s, 1e-6)

        # Eq. 24: per-sample gain in dB
        #   g_dB = -60/fs * (1/T60 - 1/T60_max)
        #        = -60/fs * delta_T60 / (T60 * T60_max)
        g_dB_per_sample = (
            -60.0 / sample_rate * delta_t60 / (safe_t60 * t60_max_s)
        )

        # Eq. 25: per-sample linear gain
        g_lin_per_sample = 10.0 ** (g_dB_per_sample / 20.0)

        # Eq. 26: per-delay-line gain
        return g_lin_per_sample ** delay_length
