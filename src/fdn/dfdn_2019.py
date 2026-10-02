"""Directional Feedback Delay Network in the spherical-harmonic domain (archival).

Implementation of Alary, B., Politis, A., Schlecht, S. J., and Valimaki, V.,
*"Directional Feedback Delay Network"*, J. Audio Eng. Soc., 67(10), 752-762,
2019. Kept for comparison only: the package's working reverberator is
:class:`~src.fdn.directional_fdn.DirectionalFDN` (ICASSP 2020), which replaces
the spherical-harmonic state and its directional weighting matrix with one
channel per incident direction on a spatial grid.

Architecture
------------
1. **Input** ``x(n)`` is encoded to Ambisonics via input gains ``b_i`` (Eq. 17).
2. **Delay-line groups** ``s_i(n)`` — *N* groups of *Q* Ambisonics channels each.
3. **Feedback matrix** ``A ⊗ I_Q`` — Kronecker product of an *N×N* unitary
   matrix with the *Q×Q* identity (Eq. 20).
4. **Transform** ``T_i`` — optional directional weighting for anisotropic
   decay (Section 3).
5. **Output** ``y(n) = Σ g_i (c_i · s_i(n)) + d · x(n)`` (Eq. 22).

Usage
-----
>>> from src.fdn.dfdn_2019 import DFDN2019
>>> dfdn = DFDN2019(
...     sample_rate=48000,
...     num_delay_line_groups=4,
...     ambisonics_order=1,
... )
>>> dfdn.setup(delay_range_ms=(10, 30), des_t60_ms=800)
>>> rir = dfdn.render(source_azimuth=0.0, source_elevation=0.0,
...                    length_seconds=1.0)
"""

from __future__ import annotations

import numpy as np
import pyFDN
from numpy.typing import NDArray
from scipy.linalg import block_diag

from . import spatial
from .fdn_components import (
    FeedbackMatrix,
    FeedbackMatrixType,
    design_absorption_sos,
    generate_delay_lengths,
)
from .transform import DirectionalWeighting


class DFDN2019:
    """Directional Feedback Delay Network, spherical-harmonic domain (Alary et al. 2019).

    Archival; see :class:`~src.fdn.directional_fdn.DirectionalFDN`.

    A :class:`pyFDN.FDNBuild` with:

    * delay lines organised into **groups** (N groups × Q channels),
    * Kronecker-product feedback ``A ⊗ I_Q``,
    * optional directional weighting ``T_i`` for anisotropic decay,
    * Ambisonics-encoded I/O.

    Parameters
    ----------
    sample_rate : float
        Sample rate in Hz.
    num_delay_line_groups : int
        *N* — number of delay-line groups.
    ambisonics_order : int
        Ambisonics order (1 → 4 ch, 2 → 9 ch, 3 → 16 ch).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        num_delay_line_groups: int = 4,
        ambisonics_order: int = 1,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.ambisonics_order = ambisonics_order
        self.num_delay_line_groups = num_delay_line_groups  # N
        self.Q = (ambisonics_order + 1) ** 2               # channels per group
        self.num_delay_lines = num_delay_line_groups * self.Q
        self.num_outputs = self.Q  # one output per Ambisonics channel
        self.seed = seed

        self.delay_length_samps: NDArray | None = None
        self.absorption_gains: NDArray | None = None  # broadband, per line
        self.absorption_sos: NDArray | None = None    # (sections, 6, N*Q)
        self.transform: DirectionalWeighting | None = None
        self._kronecker_matrix: NDArray | None = None
        self._base_feedback_matrix: NDArray | None = None

    # ==================================================================
    # High-level convenience
    # ==================================================================

    def setup(
        self,
        delay_range_ms: tuple[float, float] = (10.0, 30.0),
        des_t60_ms: float | NDArray = 800.0,
        *,
        feedback_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
        rng: np.random.Generator | None = None,
    ) -> None:
        """One-call initialisation of all FDN components.

        Parameters
        ----------
        delay_range_ms : (float, float)
            Range for coprime delay-line lengths.
        des_t60_ms : float or NDArray
            Desired T60 (ms).  Scalar → homogeneous; a ``(10,)`` array →
            frequency-dependent, on the graphic EQ's own grid (see
            :func:`~src.fdn.fdn_components.geq_band_frequencies`).
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        rng : np.random.Generator, optional
            Random generator for delay-length selection.
        """
        N, Q = self.num_delay_line_groups, self.Q

        # 1. Delay-line lengths
        if rng is None and self.seed is not None:
            rng = np.random.default_rng(self.seed)
        group_lengths = generate_delay_lengths(
            delay_range_ms, N, self.sample_rate, rng=rng
        )
        # Replicate for all Q channels per group
        self.delay_length_samps = np.repeat(group_lengths, Q).astype(np.int64)

        # 2. Feedback matrix  A ⊗ I_Q
        self._init_kronecker_feedback(feedback_type)

        # 3. Absorption
        self.init_absorption(des_t60_ms)

    def init_absorption(self, des_t60_ms: float | NDArray) -> None:
        """Set the absorption from a T60 target.

        Parameters
        ----------
        des_t60_ms : float or NDArray
            Broadband (scalar) **or** frequency-dependent T60 in ms.  An array
            is specified on the graphic EQ's own ten-point grid; see
            :func:`~src.fdn.fdn_components.geq_band_frequencies`.
        """
        if self.delay_length_samps is None:
            raise RuntimeError("Call setup() before init_absorption().")
        t60_s = np.asarray(des_t60_ms, dtype=float) * 1e-3
        if t60_s.ndim == 0:
            self.absorption_gains = pyFDN.db_to_lin(
                pyFDN.rt_to_slope(t60_s, self.sample_rate) * self.delay_length_samps
            )
            self.absorption_sos = None
        else:
            self.absorption_sos = design_absorption_sos(
                t60_s, self.delay_length_samps, self.sample_rate
            )
            self.absorption_gains = None

    def set_directional_decay(
        self,
        t60_at_directions_ms: NDArray,
        azimuths: NDArray,
        elevations: NDArray,
    ) -> None:
        """Configure direction-dependent decay (Section 3 of Alary et al. 2019).

        Builds the Directional Weighting Matrix (DWM) T_i for each delay-line
        group and reconfigures the base absorption to use T60_max (Eq. 16).

        Parameters
        ----------
        t60_at_directions_ms : NDArray, shape (K,)
            T60 in milliseconds at each sampling direction.
        azimuths : NDArray, shape (K,)
            Azimuth of each sampling direction in radians.
        elevations : NDArray, shape (K,)
            Elevation of each sampling direction in radians.
        """
        if self.delay_length_samps is None:
            raise RuntimeError("Call setup() before set_directional_decay().")

        N, Q = self.num_delay_line_groups, self.Q

        # Per-group delay lengths (one representative per group)
        group_lengths = self.delay_length_samps.reshape(N, Q)[:, 0]

        self.transform = DirectionalWeighting(
            num_groups=N,
            ambisonics_order=self.ambisonics_order,
            group_delay_lengths=group_lengths,
            sample_rate=self.sample_rate,
            t60_at_directions_ms=t60_at_directions_ms,
            azimuths=azimuths,
            elevations=elevations,
        )

        # Reconfigure base absorption with T60_max (Eq. 16)
        self.init_absorption(self.transform.t60_max_ms)

    def render(
        self,
        source_azimuth: float = 0.0,
        source_elevation: float = 0.0,
        length_seconds: float = 1.0,
    ) -> NDArray:
        """Generate an Ambisonics RIR for a point source.

        Input is omnidirectional per Eq. 17, so the late reverb starts diffuse
        and the DWM shapes anisotropy over time.  Sec. 2.1 puts the direct
        sound and the early reflections in a separate parallel path.

        Parameters
        ----------
        source_azimuth : float
            Source azimuth in radians.
        source_elevation : float
            Source elevation in radians.
        length_seconds : float
            Duration of the impulse response in seconds.

        Returns
        -------
        NDArray, shape (Q, num_samples)
            Ambisonics RIR in ACN/SN3D format.
        """
        num_samples = int(length_seconds * self.sample_rate)

        # Create impulse
        impulse = np.zeros((1, num_samples))
        impulse[0, 0] = 1.0

        return self.process_from_direction(
            impulse, source_azimuth, source_elevation
        )

    # ==================================================================
    # Core directional processing
    # ==================================================================

    def process_from_direction(
        self,
        input_data: NDArray,
        source_azimuth: float,
        source_elevation: float,
    ) -> NDArray:
        """Process audio through the directional FDN.

        Implements the full signal flow of Alary et al. 2019:

        1. Encode input omnidirectionally (Eq. 17).
        2. Feedback recursion with Kronecker product (Eq. 20).
        3. Optional Transform T_i for anisotropic decay (Section 3).
        4. Output extraction (Eq. 22).

        Parameters
        ----------
        input_data : NDArray, shape (1, num_samples)
            Mono input signal (typically an impulse).
        source_azimuth, source_elevation : float
            Source direction in radians.

        Returns
        -------
        NDArray, shape (num_outputs, num_samples)
            Ambisonics output in ACN/SN3D.
        """
        self._check_ready()
        if input_data.shape[0] != 1:
            raise ValueError("Input must be mono (1 channel).")

        build = self._fdn_build(source_azimuth, source_elevation)
        x = np.asarray(input_data, dtype=float)
        return pyFDN.build_to_td(build).process_signal(x.T).T

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _fdn_build(
        self, source_azimuth: float, source_elevation: float
    ) -> pyFDN.FDNBuild:
        """The network for one source direction, as a pyFDN build."""
        A = self._effective_feedback_matrix()
        C = self._build_output_gains()
        if self.absorption_gains is not None:
            # Broadband loss is diag(g) between the delays and both A and C.
            A = A * self.absorption_gains
            C = C * self.absorption_gains

        # Direct-path gain, Eq. 11: d = y(phi_0, theta_0), the plain SH
        # encoding of the source direction.  No 1/N -- the paper sets d from
        # the incident direction alone, and dividing by the group count would
        # drop the direct sound 6 dB every time N doubled.
        D = spatial.encode_directions(
            source_azimuth, source_elevation, self.ambisonics_order
        ).reshape(self.num_outputs, 1)

        return pyFDN.FDNBuild(
            A=A,
            B=self._build_input_gains(),
            C=C,
            D=D,
            delays=self.delay_length_samps,
            fs=float(self.sample_rate),
            post_delay=self.absorption_sos,
        )

    def _effective_feedback_matrix(self) -> NDArray:
        """Return the full N*Q x N*Q feedback matrix with T_i folded in.

        The recirculation path applies, per group, the Directional Weighting
        Matrix T_i, then the Kronecker feedback A kron I_Q.  Both are static,
        so the composition is a single matrix and the recursion needs no
        per-block hooks:

            A' = (A kron I_Q) @ blockdiag(T_0, ..., T_{N-1})

        Eq. 23 writes the recursion as ``sum_j A_ij g_j T_j s_j``, absorption
        between the transform and the feedback, where pyFDN applies
        absorption to the delay-line output and so evaluates ``A' g``.  The
        two agree here because every channel of a group shares that group's
        delay length (Sec. 2.2), so its absorption is one filter times I_Q and
        commutes with T_j -- verified identical to 0.0 across a group's
        channels.  It would *not* commute in a formulation with per-channel
        delays, which is why :class:`~src.fdn.directional_fdn.DirectionalFDN` has no
        transform to fold.
        """
        if self.transform is None:
            return self._kronecker_matrix
        return self._kronecker_matrix @ block_diag(
            *self.transform.transform_matrices
        )

    def _init_kronecker_feedback(
        self, matrix_type: FeedbackMatrixType
    ) -> None:
        """Build A ⊗ I_Q (Eq. 20)."""
        base = FeedbackMatrix(
            self.num_delay_line_groups,
            matrix_type,
            seed=self.seed,
        )
        self._base_feedback_matrix = base.matrix
        I_Q = np.eye(self.Q)
        self._kronecker_matrix = np.kron(self._base_feedback_matrix, I_Q)

        # Verify unitarity
        err = np.linalg.norm(
            self._kronecker_matrix.T @ self._kronecker_matrix
            - np.eye(self.num_delay_lines),
            "fro",
        )
        if err > 1e-10:
            raise RuntimeError(
                f"Kronecker feedback matrix unitarity error: {err:.2e}"
            )

    def _build_input_gains(self) -> NDArray:
        """Construct the input gain matrix b, per Eq. 17.

        Omnidirectional: each group is driven through its W channel alone, so
        the late reverb starts diffuse and the DWM shapes anisotropy over the
        decay.  The source direction does not enter here -- it reaches the
        output through the direct-path gain d of Eq. 11.
        """
        N, Q = self.num_delay_line_groups, self.Q
        b = np.zeros((self.num_delay_lines, 1))
        for g in range(N):
            b[g * Q, 0] = 1.0 / np.sqrt(N)
        return b

    def _build_output_gains(self) -> NDArray:
        """Construct the output gain matrix c (Eq. 22)."""
        N, Q = self.num_delay_line_groups, self.Q
        c = np.zeros((self.num_outputs, self.num_delay_lines))
        for ch in range(min(self.num_outputs, Q)):
            for g in range(N):
                c[ch, g * Q + ch] = 1.0 / np.sqrt(N)
        return c

    def _check_ready(self) -> None:
        """Check that the DFDN is fully initialised."""
        if self.delay_length_samps is None:
            raise RuntimeError("Call setup() before processing.")
        if self._kronecker_matrix is None:
            raise RuntimeError("Kronecker feedback matrix not initialised.")
        if self.absorption_gains is None and self.absorption_sos is None:
            raise RuntimeError("Absorption not initialised.")
