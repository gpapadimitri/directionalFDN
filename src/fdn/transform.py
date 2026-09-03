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

import warnings

import numpy as np
from numpy.typing import NDArray

from . import spatial


def weighting_effective_order(
    gains: NDArray,
    azimuths: NDArray,
    elevations: NDArray,
    energy_fraction: float = 0.95,
    max_order: int = 8,
) -> int:
    """Effective spherical-harmonic order L' of a directional weighting.

    Alary et al. (JAES 2019) Sec. 3.2: the weighting function is expanded in
    spherical harmonics and truncated at the lowest order holding
    *energy_fraction* of the coefficient energy.  L' matters because the DWM
    multiplies the weighting by the sound scene, and a product of orders L and
    L' has order L + L' -- so the quadrature grid must resolve that sum
    (Eq. 35), not merely L.

    In practice L' is often 0: a per-direction decay gain is a smooth positive
    function dominated by its mean, so most of its energy sits in the zeroth
    harmonic.  It rises only for sharply lobed anisotropy.

    Parameters
    ----------
    gains : NDArray, shape (K,)
        Directional weighting sampled at K directions.
    azimuths, elevations : NDArray, shape (K,)
        Those directions, in radians.
    energy_fraction : float
        Fraction of coefficient energy the retained orders must hold.
    max_order : int
        Highest order to test against.

    Returns
    -------
    int
        The effective order L'.
    """
    gains = np.asarray(gains, dtype=float).ravel()
    order = min(max_order, max(1, int(np.sqrt(len(gains))) - 1))
    basis = spatial.encode_directions(azimuths, elevations, order)

    # A sampling that does not span the basis -- a ring of constant elevation,
    # say -- has no well-defined expansion, and a pseudo-inverse would smear a
    # smooth function across every order.  Cap the expansion at what the
    # geometry can actually resolve.
    rank = int(np.linalg.matrix_rank(basis))
    resolvable = max(0, int(np.sqrt(rank)) - 1)
    if resolvable < order:
        order = resolvable
        if order < 1:
            return 0
        basis = spatial.encode_directions(azimuths, elevations, order)

    coefficients = np.linalg.pinv(basis) @ gains

    per_order = np.array(
        [
            np.sum(coefficients[degree**2 : (degree + 1) ** 2] ** 2)
            for degree in range(order + 1)
        ]
    )
    total = per_order.sum()
    if total <= 0.0:
        return 0
    return int(np.argmax(np.cumsum(per_order) / total >= energy_fraction))


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

        # Eq. 35 evaluates the weighting on a t-design of degree >= 2(L+L'),
        # since the DWM multiplies an order-L scene by an order-L' weighting.
        # A grid that cannot resolve the product aliases it back onto the
        # retained orders.
        probe = self._directional_gain(
            t60_s, self.t60_max_s, int(self.group_delay_lengths[0]), sample_rate
        )
        self.weighting_order = weighting_effective_order(probe, azimuths, elevations)
        rank = int(np.linalg.matrix_rank(Y))
        if rank < self.Q:
            # Sec. 3.1 samples T60 over a sphere; a degenerate set such as a
            # ring of constant elevation spans only part of the basis, and the
            # channels it misses are the ones null_complement passes through
            # untouched.  Those directions get no weighting at all.
            warnings.warn(
                f"The {K} sampling directions span only {rank} of the {self.Q} "
                f"spherical harmonics at order {self.L}, so "
                f"{self.Q - rank} channel(s) receive no directional weighting. "
                "Sample over a sphere (see spatial.uniform_directions or a "
                "t-design) rather than a ring.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif K < (2 * (self.L + self.weighting_order) + 1) ** 2:
            required = 2 * (self.L + self.weighting_order)
            warnings.warn(
                f"The {K}-point grid cannot resolve the order-"
                f"{self.L + self.weighting_order} product of an order-{self.L} "
                f"scene with an order-{self.weighting_order} weighting: Eq. 35 "
                f"wants a t-design of degree >= {required}, i.e. at least "
                f"{(required + 1) ** 2} points. The directional decay will be "
                "smoothed beyond the truncation the method already implies.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            # Sec. 2.4: the transform reshapes the directional distribution but
            # must not inject energy, or the recursion grows without bound.
            spectral_radius = float(np.max(np.abs(np.linalg.eigvals(T_i))))
            if spectral_radius > 1.0 + 1e-9:
                raise RuntimeError(
                    f"Directional weighting matrix for group {i} has spectral "
                    f"radius {spectral_radius:.6f} > 1 and would add energy on "
                    "recirculation. Check that every t60_at_directions_ms value "
                    "is at or below the maximum."
                )
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
