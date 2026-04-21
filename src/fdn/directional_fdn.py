"""Directional Feedback Delay Network (DFDN).

Implementation of Benoit et al. (2019) *"Directional Feedback Delay Network"*.

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
>>> from directional_fdn.fdn.directional_fdn import DirectionalFDN
>>> dfdn = DirectionalFDN(
...     sample_rate=48000,
...     num_delay_line_groups=4,
...     ambisonics_order=1,
...     frame_size=64,
... )
>>> dfdn.setup(delay_range_ms=(10, 30), des_t60_ms=800)
>>> rir = dfdn.render(source_azimuth=0.0, source_elevation=0.0,
...                    length_seconds=1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import spatial
from .delay_line import generate_coprime_delay_lengths
from .fdn import FDN
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .transform import DirectionalWeighting


class DirectionalFDN(FDN):
    """Directional Feedback Delay Network (Benoit et al. 2019).

    Extends :class:`FDN` with:

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
    frame_size : int
        Processing block size in samples.
    scattering_amount : float
        Controls inter-channel energy scattering in the feedback path.
        ``0.0`` (default) preserves the paper-faithful ``A ⊗ I_Q`` design
        where SH channels remain independent.  Values ``> 0`` blend toward
        random orthogonal per-group matrices that couple SH channels,
        modelling acoustic scattering that diffuses a directional input
        toward omnidirectional over time.  Not described in the papers;
        only needed when ``use_directional_input=True``.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        num_delay_line_groups: int = 4,
        ambisonics_order: int = 1,
        frame_size: int = 64,
        scattering_amount: float = 0.0,
        seed: int | None = None,
    ) -> None:
        Q = spatial.num_channels(ambisonics_order)
        total = num_delay_line_groups * Q
        num_outputs = Q  # one output per Ambisonics channel

        super().__init__(
            sample_rate=sample_rate,
            num_inputs=1,           # mono point-source input
            num_outputs=num_outputs,
            num_delay_lines=total,
            frame_size=frame_size,
            seed=seed,
        )

        self.ambisonics_order = ambisonics_order
        self.num_delay_line_groups = num_delay_line_groups  # N
        self.Q = Q                                          # channels per group
        self.scattering_amount = float(np.clip(scattering_amount, 0.0, 1.0))

        # Optional modules
        self.transform: DirectionalWeighting | None = None
        self._kronecker_matrix: NDArray | None = None
        self._base_feedback_matrix: NDArray | None = None
        self._scattering_matrices: list[NDArray] | None = None
        self._use_scattering: bool = False

    # ==================================================================
    # High-level convenience
    # ==================================================================

    def setup(
        self,
        delay_range_ms: tuple[float, float] = (10.0, 30.0),
        des_t60_ms: float | NDArray = 800.0,
        *,
        feedback_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
        freqs: NDArray | None = None,
        filter_order: int | None = None,
        warp_factor: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """One-call initialisation of all FDN components.

        Parameters
        ----------
        delay_range_ms : (float, float)
            Range for coprime delay-line lengths.
        des_t60_ms : float or NDArray
            Desired T60 (ms).  Scalar → homogeneous; array → frequency-dependent.
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        freqs, filter_order, warp_factor
            Frequency-dependent absorption parameters (see :class:`AbsorptionFilters`).
        rng : np.random.Generator, optional
            Random generator for delay-length selection.
        """
        N, Q = self.num_delay_line_groups, self.Q

        # 1. Delay-line lengths
        if rng is None and self.seed is not None:
            rng = np.random.default_rng(self.seed)
        group_lengths = generate_coprime_delay_lengths(
            delay_range_ms, N, self.sample_rate, rng=rng
        )
        # Replicate for all Q channels per group
        all_lengths = np.repeat(group_lengths, Q)
        self.init_delay_line_lengths(all_lengths)

        # 2. Feedback matrix  A ⊗ I_Q
        self._init_kronecker_feedback(feedback_type)

        # 3. Absorption
        self.init_absorption(
            des_t60_ms,
            freqs=freqs,
            filter_order=filter_order,
            warp_factor=warp_factor,
        )

        # 4. Delay lines
        self.init_delay_lines()

        # 5. I/O gains (paper defaults)
        self._init_paper_io_gains()

        # 6. Per-group scattering matrices (Q×Q, identity when amount=0)
        self._init_scattering_matrices(self.scattering_amount)

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
        *,
        use_directional_input: bool = False,
    ) -> NDArray:
        """Generate an Ambisonics RIR for a point source.

        Parameters
        ----------
        source_azimuth : float
            Source azimuth in radians.
        source_elevation : float
            Source elevation in radians.
        length_seconds : float
            Duration of the impulse response in seconds.
        use_directional_input : bool
            If *False* (default), use omnidirectional input per Eq. 17.
            This is the paper-correct mode: late reverb starts diffuse
            and the DWM shapes anisotropy over time.  Early reflections
            should be handled in a separate parallel path.
            If *True*, encode the source direction into all SH channels.
            Requires ``scattering_amount > 0`` to produce a plausible
            diffuse tail; otherwise energy stays locked in the source
            direction (by design of ``A ⊗ I_Q``).

        Returns
        -------
        NDArray, shape (Q, num_samples)
            Ambisonics RIR in ACN/SN3D format.
        """
        num_samples = int(length_seconds * self.sample_rate)
        # Ensure length is a multiple of frame_size
        num_samples = (num_samples // self.frame_size) * self.frame_size

        # Create impulse
        impulse = np.zeros((1, num_samples))
        impulse[0, 0] = 1.0

        return self.process_from_direction(
            impulse,
            source_azimuth,
            source_elevation,
            use_directional_input=use_directional_input,
        )

    # ==================================================================
    # Core directional processing
    # ==================================================================

    def process_from_direction(
        self,
        input_data: NDArray,
        source_azimuth: float,
        source_elevation: float,
        *,
        use_directional_input: bool = False,
    ) -> NDArray:
        """Process audio through the directional FDN.

        Implements the full signal flow of Benoit et al. 2019:

        1. Encode input (Eq. 17: omnidirectional, or directional variant).
        2. Feedback recursion with Kronecker product (Eq. 20).
        3. Optional Transform T_i for anisotropic decay (Section 3).
        4. Output extraction (Eq. 22).

        Parameters
        ----------
        input_data : NDArray, shape (1, num_samples)
            Mono input signal (typically an impulse).
        source_azimuth, source_elevation : float
            Source direction in radians.
        use_directional_input : bool
            Use directional (non-paper) input encoding.

        Returns
        -------
        NDArray, shape (num_outputs, num_samples)
            Ambisonics output in ACN/SN3D.
        """
        self._check_ready()
        if input_data.shape[0] != 1:
            raise ValueError("Input must be mono (1 channel).")

        N, Q, F = self.num_delay_line_groups, self.Q, self.frame_size
        ir_len = input_data.shape[-1]

        # Source encoding
        src_enc = spatial.encode_direction(
            source_azimuth, source_elevation, self.ambisonics_order
        )

        # Build input gains b  (Eq. 17)
        b = self._build_input_gains(src_enc, directional=use_directional_input)

        # Build output gains c  (Eq. 22)
        c = self._build_output_gains()

        # Direct-path gain d — encodes source direction per Eq. 11:
        #   d = y(φ₀, θ₀)  (SH encoding of the source direction).
        # The paper does not specify additional scaling; the 1/N factor
        # here matches the combined input/output gain (1/√N × 1/√N).
        direct_gain = src_enc.reshape(self.num_outputs, 1) / N

        # Feedback matrix A ⊗ I_Q  (Eq. 20)
        A = self._kronecker_matrix

        # Pre-computed DWM T_i matrices (Section 3)
        use_ti = self.transform is not None
        if use_ti:
            T_mats = self.transform.transform_matrices

        # Allocate buffers
        output = np.zeros((self.num_outputs, ir_len))
        fb_out = np.zeros((self.num_delay_lines, F))
        dl_out = np.zeros((self.num_delay_lines, F))

        sample = 0
        while sample + F <= ir_len:
            idx = slice(sample, sample + F)
            x = input_data[:, idx]

            # Input mixing + feedback  (Eq. 23 RHS)
            dl_in = b @ x + fb_out

            # Read / write delay lines  (Eq. 19)
            for i in range(self.num_delay_lines):
                dl_out[i] = self.delay_lines[i].read()
                self.delay_lines[i].write(dl_in[i])

            # Apply base absorption g_i  (Eq. 16 / Eq. 5)
            absorbed = self.absorption.process(dl_out)

            # Output extraction  (Eq. 22): y = Σ g_i c_i ⊙ s_i + d x
            output[:, idx] += c @ absorbed + direct_gain @ x

            # Feedback path  (Eq. 23): optionally scatter then transform
            if self._use_scattering:
                scattered = np.zeros_like(absorbed)
                for g in range(N):
                    s, e = g * Q, (g + 1) * Q
                    scattered[s:e] = self._scattering_matrices[g] @ absorbed[s:e]
            else:
                scattered = absorbed

            if use_ti:
                transformed = np.zeros_like(scattered)
                for g in range(N):
                    s, e = g * Q, (g + 1) * Q
                    transformed[s:e] = T_mats[g] @ scattered[s:e]
                fb_out = A @ transformed
            else:
                fb_out = A @ scattered

            sample += F

        return output

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _init_scattering_matrices(self, amount: float = 0.0) -> None:
        """Generate per-group Q×Q scattering matrices.

        When *amount* is 0 (default), each matrix is the identity —
        preserving the paper-faithful ``A ⊗ I_Q`` where SH channels
        stay independent.  When *amount* > 0, a convex blend toward a
        random orthogonal matrix is used and then re-orthogonalised
        via QR to keep the feedback lossless.  This models acoustic
        scattering that gradually diffuses directional energy across
        SH channels (not described in the papers; useful when
        ``use_directional_input=True``).
        """
        Q = self.Q
        self._scattering_matrices = []

        if amount <= 0.0:
            self._use_scattering = False
            for _ in range(self.num_delay_line_groups):
                self._scattering_matrices.append(np.eye(Q))
            return

        self._use_scattering = True
        rng = np.random.default_rng(
            self.seed + 9999 if self.seed is not None else None
        )
        for _ in range(self.num_delay_line_groups):
            H = rng.standard_normal((Q, Q))
            R, _ = np.linalg.qr(H)
            blended = (1.0 - amount) * np.eye(Q) + amount * R
            S, _ = np.linalg.qr(blended)
            self._scattering_matrices.append(S)

    def _init_kronecker_feedback(
        self, matrix_type: FeedbackMatrixType
    ) -> None:
        """Build A ⊗ I_Q (Eq. 20)."""
        base = FeedbackMatrix(
            self.num_delay_line_groups,
            matrix_type,
            self.frame_size,
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

        # Also store as the base FDN feedback for compatibility
        self.feedback = base

    def _init_paper_io_gains(self) -> None:
        """Set I/O gains per paper Eqs. 17 and 22.

        The stored direct-path gain is zero here because ``render()``
        and ``process_from_direction()`` compute ``d`` per-call from
        the actual source direction (Eq. 11).
        """
        dummy_enc = np.zeros(self.Q)
        b = self._build_input_gains(dummy_enc, directional=False)
        c = self._build_output_gains()
        self.init_io_gains(b, c)
        self.init_direct_gain(np.zeros((self.num_outputs, 1)))

    def _build_input_gains(
        self, source_encoding: NDArray, *, directional: bool
    ) -> NDArray:
        """Construct the input gain matrix b for a given direction.

        Parameters
        ----------
        source_encoding : NDArray, shape (Q,)
        directional : bool
            If *True*, each group receives the full Ambisonics encoding
            (better auralization but departs from Eq. 17).
        """
        N, Q = self.num_delay_line_groups, self.Q
        b = np.zeros((self.num_delay_lines, 1))
        if directional:
            for g in range(N):
                s = g * Q
                b[s : s + Q, 0] = source_encoding / np.sqrt(N)
        else:
            # Paper Eq. 17: omnidirectional (W only)
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
        if self.delay_lines is None:
            raise RuntimeError("Call setup() before processing.")
        if self._kronecker_matrix is None:
            raise RuntimeError("Kronecker feedback matrix not initialised.")
        if self.absorption is None:
            raise RuntimeError("Absorption not initialised.")
