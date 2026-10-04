"""Directional Feedback Delay Network on a spatial grid.

Implementation of Alary & Politis, *"Frequency-Dependent Directional Feedback
Delay Network"*, ICASSP 2020, 176-180: "a multichannel representation of
delay-line groups where each channel corresponds to an incident direction on a
spatial grid of Q directions" (Sec. 3). This is the package's working
reverberator. The spherical-harmonic formulation it replaces (Alary et al.,
J. Audio Eng. Soc. 67(10), 2019) is kept for comparison as
:class:`~src.fdn.dfdn_2019.DFDN2019`.

Architecture
------------
1. **Per-direction absorption filters**: each of the Q channels in each
   delay-line group gets its own filter for T60(w, phi, theta) (Eqs. 6-10).
2. **Independent delay lengths** within a group, which raises echo density
   (Fig. 4).
3. **No directional weighting transform**: the feedback ``A kron I_Q`` never
   mixes directions, so each keeps the decay its filter was designed for.
4. **Early reflection module**: reflections VBAP-panned onto the grid
   (Eqs. 3-5) and added to delay-line group ``lambda`` after its delay
   (Eq. 12). The paper taps them into the line at ``m_i - l_r``; adding a
   signal that already carries its ``l_r`` offsets is equivalent.
5. **Output encoder E** (Eq. 13): Ambisonics, loudspeakers or binaural.

Terminology
-----------
The paper names its channels incident directions on a grid, never plane waves.
The same authors later describe them as "incident plane-waves distributed
around a sphere" (Alary & Valimaki, I3DA 2021) and "a discrete set of incident
plane waves represented on a spherical grid" (Alary, Politis & Bilbao, AES
AVAR 2024). That holds for the late field under the Ambisonic encoder
``E = Y(L, U_grid)``: each channel is encoded exactly as a far-field plane wave,
and the channels are mutually uncorrelated. It is not a plane-wave
decomposition. Q exceeds (L+1)^2 and E carries no quadrature weights, so the
grid holds directional detail above order L that the encoded output discards.
The direct sound and the early reflections are VBAP-panned onto grid triplets
(Eqs. 3-5): unless they arrive on a grid direction they are spread sources, not
plane waves.

Usage
-----
>>> from src.fdn.directional_fdn import DirectionalFDN
>>> pw = DirectionalFDN(
...     sample_rate=48000,
...     num_delay_line_groups=8,
...     sh_order=2,  # Q ≥ (L+1)² grid points
... )
>>> pw.setup(
...     delay_range_ms=(10, 50),
...     t60_per_direction_ms=t60_map,  # (Q, 10), (10,), (Q,) or scalar
... )

The frequency axis is the graphic EQ's own design grid; build targets against
:func:`~src.fdn.fdn_components.geq_band_frequencies`.
>>> rir = pw.render(source_azimuth=0.0, source_elevation=0.0,
...                  length_seconds=1.0)
"""

from __future__ import annotations

import numpy as np
import pyFDN
from numpy.typing import NDArray
from pyFDN.eq import gain_to_geq
from pyFDN.td import SOSBank
from scipy.signal import sosfilt

from ..utils import db2lin
from . import spatial
from .air import AirBudget, Atmosphere
from .fdn_components import (
    GEQ_NUM_BANDS,
    FeedbackMatrix,
    FeedbackMatrixType,
    design_absorption_sos,
    generate_delay_lengths,
    geq_band_frequencies,
)
from .room import Reflection

# ---------------------------------------------------------------------------
# Early Reflection Module (Eqs. 3–5)
# ---------------------------------------------------------------------------

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
    """Early reflection set for the ICASSP 2020 module (Eqs. 3-5).

    Holds the reflections; the panning happens in
    :meth:`DirectionalFDN._build_er_grid_signal`, which places each one on the
    grid with VBAP exactly as Eqs. 4-5 specify.  The reflections
    therefore share the grid with the direct sound and the recirculating
    signals, and reach the output through the single encoder ``E`` of Eq. 13.

    Parameters
    ----------
    reflections : list[Reflection]
        Reflection definitions.
    sh_order : int
        SH order of the render this module belongs to.  Recorded for
        introspection; the encoding itself is ``E``'s job.
    """

    def __init__(
        self,
        reflections: list[Reflection],
        sh_order: int,
    ) -> None:
        self.reflections = reflections
        self.sh_order = sh_order



# ---------------------------------------------------------------------------
# Directional FDN
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

    pyFDN applies its ``post_delay`` hook to the delay-line output,
    which is precisely that point, and calls it once per block in order, so the
    running position below stays aligned with the render.
    """

    def __init__(
        self,
        absorption: SOSBank | None,
        er_grid: NDArray,
        lambda_group: int,
        num_channels: int,
    ) -> None:
        self._absorption = absorption
        self._er = er_grid
        self._slice = slice(lambda_group * num_channels,
                            (lambda_group + 1) * num_channels)
        self._position = 0

    def process_block(self, block: NDArray) -> NDArray:
        block = np.array(block, dtype=float, copy=True)
        n = block.shape[0]
        segment = self._er[:, self._position : self._position + n]
        block[: segment.shape[1], self._slice] += segment.T
        self._position += n
        if self._absorption is None:
            return block
        return self._absorption.process_block(block)


class DirectionalFDN:
    """Directional FDN on a spatial grid (Alary & Politis, ICASSP 2020).

    Each channel of a delay-line group corresponds to an incident direction
    on a Q-point grid.  Per-channel absorption filters set both the frequency-
    and the direction-dependent T60.

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
        Setting this lower than *sh_order* lets you keep a dense
        grid (better directional resolution inside the FDN) while emitting
        a lower-order Ambisonics output (e.g. ``sh_order=3``,
        ``output_sh_order=1`` to render a FOA RIR).
        Must satisfy ``output_sh_order <= sh_order``.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    realised_t60_s : NDArray or None
        The T60 actually designed for, shape ``(N*Q, n_bands)``.  Equals the
        requested target unless ``air_absorption`` shortened it.
    air_budget : AirBudget or None
        Per-round-trip attenuation in dB attributed to the medium and to the
        boundary, shape ``(N*Q, n_bands)``, which sum to the loss the realised
        target requires.  Set only when ``air_absorption`` is enabled.
        :attr:`air_attenuation_db` and :attr:`boundary_attenuation_db` read its
        two shares.
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        num_delay_line_groups: int = 8,
        sh_order: int = 2,
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
        self.Q = (sh_order + 1) ** 2  # grid points
        self.seed = seed

        self.total_delay_lines = num_delay_line_groups * self.Q  # N × Q
        self.num_outputs = (output_sh_order + 1) ** 2  # Ambisonics channels

        # State
        self.delay_length_samps: NDArray | None = None
        self.absorption_gains: NDArray | None = None  # broadband, per line
        self.absorption_sos: NDArray | None = None
        self.air_budget: AirBudget | None = None
        self.realised_t60_s: NDArray | None = None
        self._kronecker_matrix: NDArray | None = None
        self._base_feedback: FeedbackMatrix | None = None
        self._encoding_matrix: NDArray | None = None  # E (Eq. 13)
        self._early_reflections: EarlyReflectionModule | None = None
        self._input_distribution: NDArray | None = None  # b_i, Sec. 3
        self._er_injected = True
        self._p0_sos: NDArray | None = None

        # Grid directions
        self.grid_azimuths: NDArray | None = None
        self.grid_elevations: NDArray | None = None

    @property
    def air_attenuation_db(self) -> NDArray | None:
        """Medium share of the round-trip attenuation, ``(N*Q, n_bands)``."""
        return None if self.air_budget is None else self.air_budget.air_db

    @property
    def boundary_attenuation_db(self) -> NDArray | None:
        """Boundary share of the round-trip attenuation, ``(N*Q, n_bands)``."""
        return None if self.air_budget is None else self.air_budget.boundary_db

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
        feedback_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
        air_absorption: bool | Atmosphere = False,
        lambda_group: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise all components of the network.

        Parameters
        ----------
        delay_range_ms : (float, float)
            Range for randomly-chosen delay lengths (per channel).
        t60_per_direction_ms : float or NDArray
            Controls the target reverberation time.  Accepted formats:

            * **scalar** — same T60 for all directions and frequencies
              (isotropic, broadband).
            * **shape (10,)** — frequency-dependent T60, same for all
              directions.
            * **shape (Q,)** — direction-dependent T60, broadband.
            * **shape (Q, 10)** — direction- *and* frequency-dependent.
              Row *q* gives the T60 for grid direction *q*.

            The ten frequency values are read at
            :func:`~src.fdn.fdn_components.geq_band_frequencies`, the grid
            the graphic EQ designs on: octave centres 31.25 Hz to 16 kHz,
            interpolated linearly in hertz between them and held beyond.
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        air_absorption : bool or Atmosphere, optional
            Treat *t60_per_direction_ms* as the **boundary-only** target and
            add atmospheric absorption (ISO 9613-1) on top of it: ``True`` for
            the default :class:`~src.fdn.air.Atmosphere`, or an instance for
            other conditions.  The realised decay is
            ``1/T = 1/T_boundary + 1/T_air`` -- the ``4mV`` term of Sabine and
            Eyring.  The realised T60 is therefore shorter than requested,
            appreciably so above a few kHz, and is recorded in
            :attr:`realised_t60_s`.  The medium and boundary shares of each
            round trip are exposed as :attr:`air_budget`.  Requires a
            frequency-dependent T60; ignored with a warning for a broadband one,
            whose single value cannot carry a frequency-dependent medium.
            Default ``False``.
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

        # 2. Feedback matrix A ⊗ I_Q (ICASSP 2020, Sec. 3)
        self._init_kronecker_feedback(feedback_type)

        # 3. Per-direction absorption
        t60 = np.asarray(t60_per_direction_ms, dtype=float)
        freqs = geq_band_frequencies()

        # A (Q,) target is broadband per direction and a (GEQ_NUM_BANDS,) one
        # is frequency-dependent per band, so a grid of exactly that many
        # points would be ambiguous.  No grid this class builds has one --
        # quadrature designs give 14, 42, 86 or 146, the Fibonacci fallback
        # (L+1)^2 -- but say so rather than resolve it silently.
        if Q == GEQ_NUM_BANDS:
            raise ValueError(
                f"A {Q}-point grid collides with the {GEQ_NUM_BANDS}-band GEQ "
                "target, leaving a 1-D T60 ambiguous. Pass a 2-D (Q, "
                f"{GEQ_NUM_BANDS}) target."
            )

        if t60.ndim == 0:
            # Scalar → same broadband T60 for all channels
            t60_expanded = np.full(N * Q, float(t60))

        elif t60.ndim == 1 and t60.shape[0] == Q:
            # (Q,) → direction-dependent broadband
            t60_expanded = np.tile(t60, N)

        elif t60.ndim == 1 and t60.shape[0] == GEQ_NUM_BANDS:
            # (GEQ_NUM_BANDS,) → frequency-dependent, same for all directions
            t60_expanded = np.tile(t60[np.newaxis, :], (N * Q, 1))

        elif t60.ndim == 2 and t60.shape == (Q, GEQ_NUM_BANDS):
            # (Q, GEQ_NUM_BANDS) → direction- and frequency-dependent
            t60_expanded = np.tile(t60, (N, 1))

        else:
            raise ValueError(
                f"t60_per_direction_ms must be scalar, ({GEQ_NUM_BANDS},), "
                f"(Q={Q},), or (Q={Q}, {GEQ_NUM_BANDS}), got shape "
                f"{t60.shape}. The frequency axis is the GEQ design grid; see "
                "src.fdn.fdn_components.geq_band_frequencies."
            )
        # Branch on the shape actually computed above: a (Q,) T60 is
        # broadband-per-channel and must take the scalar-gain path.
        self.air_budget = None
        self.realised_t60_s = None
        if t60_expanded.ndim == 1:
            # Broadband: one scalar gain per channel, round trip = m exactly.
            t60_s = np.maximum(t60_expanded, 1.0) * 1e-3
            self.absorption_gains = db2lin(
                -60.0 * all_lengths / (self.sample_rate * t60_s)
            )
            self.absorption_sos = None
            if air_absorption:
                import warnings

                warnings.warn(
                    "air_absorption is ignored for a broadband T60: the "
                    "absorption coefficient is frequency-dependent, so the "
                    "medium/boundary split cannot be evaluated. Pass a "
                    f"({GEQ_NUM_BANDS},) or (Q, {GEQ_NUM_BANDS}) T60 to use "
                    "it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            t60_2d = t60_expanded * 1e-3
            if air_absorption:
                atmosphere = (
                    air_absorption
                    if isinstance(air_absorption, Atmosphere)
                    else Atmosphere()
                )
                self.air_budget = atmosphere.budget(
                    t60_2d, freqs, all_lengths, self.sample_rate
                )
                t60_2d = atmosphere.limit(t60_2d, freqs)
            self.realised_t60_s = np.broadcast_to(
                t60_2d, (N * Q, GEQ_NUM_BANDS)
            ).copy()
            self.absorption_sos = design_absorption_sos(
                t60_2d, all_lengths, self.sample_rate
            )
            self.absorption_gains = None

        # 5. Output encoding matrix E (Eq. 13) — default: Ambisonics
        self._init_ambisonics_encoding()

    def set_early_reflections(
        self, reflections: list[Reflection], *, injected: bool = True,
    ) -> None:
        """Configure the early reflection module (Eqs. 3-5).

        Must be called after :meth:`setup` so that the grid is available.

        Parameters
        ----------
        reflections : list[Reflection]
        injected : bool
            ``True`` (default): join group lambda after its delay and
            recirculate, ICASSP 2020 Eq. 12. ``False``: add them to the grid
            alongside the network, Alary and Valimaki (I3DA 2021, Sec. III B),
            leaving group lambda's input in place. Required for
            :meth:`set_initial_power`, which would otherwise filter them.
        """
        if self.grid_azimuths is None:
            raise RuntimeError("Call setup() before set_early_reflections().")
        self._early_reflections = EarlyReflectionModule(
            reflections, self.output_sh_order,
        )
        self._er_injected = bool(injected)

    def set_initial_power(self, p0_db: NDArray | float | None) -> None:
        """Shape the network's initial power per grid direction and band.

        Power gain in dB relative to the network's natural, flat initial power
        (Gallien et al., DAFx26, Sec. 3.2.1 and 3.2.3). Applied to the network
        output on the grid only: the direct sound and parallel early
        reflections are not filtered, and the decay is unchanged.

        Parameters
        ----------
        p0_db : float, NDArray or None
            Scalar, ``(10,)``, ``(Q,)`` or ``(Q, 10)`` on
            :func:`~src.fdn.fdn_components.geq_band_frequencies`. ``None``
            removes the shaping.
        """
        if self.grid_azimuths is None:
            raise RuntimeError("Call setup() before set_initial_power().")
        if p0_db is None:
            self._p0_sos = None
            return
        v = np.asarray(p0_db, dtype=float)
        Q = self.Q
        if v.ndim == 0:
            gains = np.full((GEQ_NUM_BANDS, Q), float(v))
        elif v.shape == (GEQ_NUM_BANDS,):
            gains = np.tile(v[:, np.newaxis], (1, Q))
        elif v.shape == (Q,):
            gains = np.tile(v[np.newaxis, :], (GEQ_NUM_BANDS, 1))
        elif v.shape == (Q, GEQ_NUM_BANDS):
            gains = v.T
        else:
            raise ValueError(
                f"p0_db must be scalar, ({GEQ_NUM_BANDS},), (Q={Q},) or "
                f"(Q={Q}, {GEQ_NUM_BANDS}), got shape {v.shape}."
            )
        #|H|^2 = P, so the magnitude target in dB is the power gain in dB
        self._p0_sos = np.asarray(gain_to_geq(gains, self.sample_rate))

    def match_late_level(
        self,
        target_energy: NDArray,
        bands: NDArray,
        *,
        window_start_s: float,
        length_s: float,
        iterations: int = 3,
    ) -> NDArray:
        """Match the late W-channel energy per octave band to a reference.

        Synthetic counterpart of the hybrid-reverb transition filter of
        Carpentier, Noisternig and Warusfel (DAFx 2014, Eq. 3): the reference
        is the energy from *window_start_s* to the end of a *length_s* render
        rather than the EDR at t_mix. The tail is rendered alone (no direct
        sound, no reflections) *iterations* times, refining one isotropic
        per-band correction applied through :meth:`set_initial_power`; any
        shaping set before is replaced.

        Parameters
        ----------
        target_energy : NDArray, shape (B,)
            Reference late energy per band, in the network's level convention
            (``direct_gain=1`` is a unit direct sound in W). Converting a
            measured or simulated reference into it is the caller's job.
        bands : NDArray, shape (B,)
            Consecutive octave centres from
            :func:`~src.fdn.fdn_components.geq_band_frequencies`.
        window_start_s : float
            Window start, relative to the render's first sample.
        length_s : float
            Render length; the same as the reference's.
        iterations : int
            Measure-and-correct passes (3 brings jagged targets within 0.3 dB).

        Returns
        -------
        NDArray, shape (B,)
            Applied correction in dB.
        """
        #local import: src.analysis imports src.fdn
        from ..analysis import band_energy

        if self._early_reflections is not None and self._er_injected:
            raise ValueError(
                "The injected early reflections recirculate with the tail; call "
                "set_early_reflections(..., injected=False) before matching."
            )
        bands = np.asarray(bands, dtype=float)
        command = geq_band_frequencies()
        if not np.all(np.isin(bands, command)):
            raise ValueError(
                "bands must be consecutive octave centres from geq_band_frequencies()."
            )
        target = np.asarray(target_energy, dtype=float)
        correction = np.zeros(bands.size)
        for _ in range(iterations):
            self.set_initial_power(
                np.interp(np.log2(command), np.log2(bands), correction)
            )
            w = self.render(0.0, 0.0, length_s, direct_gain=0.0, er_gain=0.0)[0]
            measured = band_energy(w, self.sample_rate, bands, window_start_s)[:, 0]
            correction = correction - 10.0 * np.log10(measured / target)
        self.set_initial_power(np.interp(np.log2(command), np.log2(bands), correction))
        return correction

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

    @property
    def encoding_matrix(self) -> NDArray:
        """Grid-to-Ambisonics encoder E (Eq. 13), shape ((L_out+1)^2, Q)."""
        if self._encoding_matrix is None:
            raise RuntimeError("Call setup() before reading encoding_matrix.")
        return self._encoding_matrix

    # ==================================================================
    # Kronecker feedback
    # ==================================================================

    def _init_kronecker_feedback(
        self, matrix_type: FeedbackMatrixType,
    ) -> None:
        """Build the ``A kron I_Q`` feedback matrix (ICASSP 2020, Sec. 3).

        The identity is the whole point: it "prevents recirculation between
        different channels", so a grid direction only ever meets itself and
        keeps the T60 its absorption filter was designed for.  Any mixing
        across directions lets energy equalise between them, and the
        direction-dependent decay this method exists to produce relaxes away.
        """
        self._base_feedback = FeedbackMatrix(
            self.num_delay_line_groups,
            matrix_type,
            seed=self.seed,
        )
        self._kronecker_matrix = np.kron(
            self._base_feedback.matrix, np.eye(self.Q)
        )
        if not pyFDN.is_unilossless(self._kronecker_matrix, tol=1e-8):
            raise RuntimeError(
                "Kronecker feedback matrix is not lossless; the recursion "
                "would not preserve energy."
            )

    # ==================================================================
    # Input encoding
    # ==================================================================

    def set_input_distribution(self, weights_per_direction: NDArray) -> None:
        """Set ``b_i`` from an initial directional energy distribution.

        ICASSP 2020 Sec. 3 describes ``b_i`` as a ``Q x 1`` vector that "sets
        an initial gain for each channel, corresponding to the distribution of
        early energy during the scattering phase of the reverberation".  Alary
        and Valimaki (I3DA 2021) write it as ``b_i (x) x(z)`` in their Eq. 7
        and specify it in Sec. III C from ``EDC_0``, the energy-decay curve's
        first sample, measured per direction on a captured SRIR.

        Uniform weights are the isotropic special case and remain the default,
        so leaving this unset reproduces the previous behaviour exactly.

        Parameters
        ----------
        weights_per_direction : NDArray, shape (Q,)
            Non-negative amplitude weight per grid direction, in the order of
            :attr:`grid_azimuths`.  Scaled internally so the total input
            energy stays unity: changing the *shape* of the distribution must
            not change how hard the network is driven.
        """
        weights = np.asarray(weights_per_direction, dtype=float).ravel()
        if weights.size != self.Q:
            raise ValueError(
                f"Input distribution must have length Q={self.Q}, got "
                f"{weights.size}."
            )
        if np.any(weights < 0.0):
            raise ValueError(
                "Input distribution must be non-negative; it is an energy "
                "distribution, so a negative weight has no meaning."
            )
        total = float(np.sum(weights**2))
        if total <= 0.0:
            raise ValueError(
                "Input distribution must have positive total energy; an "
                "all-zero vector would drive nothing."
            )
        self._input_distribution = weights

    def _build_input_gains(self) -> NDArray:
        """Build input vector b_i (ICASSP 2020 Sec. 3, Eq. 12).

        Omnidirectional by default -- every one of the ``N*Q`` channels gets
        ``1 / sqrt(N*Q)`` -- or shaped per direction when
        :meth:`set_input_distribution` has been called.

        Group ``lambda`` is zeroed once early reflections are set.  Eq. 12
        gives that group ``s'(z) + Z_lambda[...]`` with **no** ``b_lambda x``
        term, unlike every other group: the reflections replace its input
        injection rather than adding to it.  Driving it from both would
        double-count, and would leave the ER module unable to set the early
        part on its own.

        The source direction does not enter: it reaches the output through the
        direct-path gain ``d`` of Eq. 3 and, when present, through the early
        reflections of Eqs. 4-5.
        """
        N, Q = self.num_delay_line_groups, self.Q
        weights = self._input_distribution
        if weights is None:
            b = np.full((N, Q), 1.0 / np.sqrt(N * Q))
        else:
            scale = np.sqrt(N * float(np.sum(weights**2)))
            b = np.tile(weights / scale, N).reshape(N, Q)
        if self._early_reflections is not None and self._er_injected:
            b = b.copy()
            b[self._lambda_group] = 0.0
        return b.reshape(N * Q, 1)

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
        grid_output: bool = False,
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
        grid_output : bool, optional
            Return the ``Q`` grid signals ``y'`` (Eq. 11) instead of the
            encoded output ``y = E y'`` (Eq. 13).  Analysis only: the
            encoder is a spatial low-pass, so a directional decay measured
            through it is smoothed by ``output_sh_order`` and reads shallower
            than the one the network actually synthesised.  Taking the grid
            signals leaves that out and separates a design error from an
            encoding one.  Default ``False``.

        Returns
        -------
        NDArray, shape ((L_out+1)², num_samples)
            Ambisonics RIR in ACN/SN3D format, or ``(Q, num_samples)`` on the
            grid when *grid_output* is set.
        """
        num_samples = int(length_seconds * self.sample_rate)

        impulse = np.zeros((1, num_samples))
        impulse[0, 0] = 1.0

        return self.process(
            impulse, source_azimuth, source_elevation,
            direct_gain=direct_gain, er_gain=er_gain,
            reverb_gain=reverb_gain, grid_output=grid_output,
        )

    def process(
        self,
        input_data: NDArray,
        source_azimuth: float = 0.0,
        source_elevation: float = 0.0,
        direct_gain: float = 1.0,
        er_gain: float = 1.0,
        reverb_gain: float = 1.0,
        grid_output: bool = False,
    ) -> NDArray:
        """Process mono input through the network.

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
        grid_output : bool, optional
            Stop at ``y'`` (Eq. 11) and skip the encoder of Eq. 13.  See
            :meth:`render`.

        Returns
        -------
        NDArray, shape ((L+1)², num_samples), or (Q, num_samples) on the grid
        """
        self._check_ready()
        if input_data.shape[0] != 1:
            raise ValueError("Input must be mono (1 channel).")
        parallel_er = self._early_reflections is not None and not self._er_injected
        injected_er = self._early_reflections is not None and self._er_injected
        if self._p0_sos is not None and injected_er:
            raise ValueError(
                "Initial-power shaping would also filter the injected early "
                "reflections; call set_early_reflections(..., injected=False)."
            )
        if self._p0_sos is not None or parallel_er:
            return self._process_parallel(
                input_data, source_azimuth, source_elevation,
                direct_gain, er_gain, reverb_gain, grid_output,
            )

        Q = self.Q
        ir_len = input_data.shape[-1]

        b = self._build_input_gains() * float(reverb_gain)   # (NQ, 1)
        c = self._build_output_gains()               # (Q, NQ)
        d = self._build_direct_gain(source_azimuth, source_elevation) * float(
            direct_gain
        )                                            # (n_out, 1)
        E = np.eye(Q) if grid_output else self._encoding_matrix

        A = self._kronecker_matrix
        if self.absorption_gains is not None:
            # Broadband loss is diag(g) between the delays and both A and C.
            A = A * self.absorption_gains
            c = c * self.absorption_gains

        # Eq. 11: y' = sum_i C_i G_i s_i + d x, then y = E y'.  Everything
        # lives on the grid, so E is the only encoder in the signal path.
        post_delay: object = self.absorption_sos
        if self._early_reflections is not None:
            # Eq. 12: the reflections replace this group's input injection and
            # join its state after the delay, so they are wrapped around the
            # absorption hook rather than routed through B and D.
            post_delay = _EarlyReflectionInjection(
                None if self.absorption_sos is None else SOSBank(self.absorption_sos),
                self._build_er_grid_signal(input_data, ir_len, er_gain),
                self._lambda_group,
                Q,
            )

        graph = pyFDN.dss_to_td(
            self.delay_length_samps, A, b, E @ c, E @ d, post_delay=post_delay
        )
        x = np.asarray(input_data, dtype=float)
        return graph.process_signal(x.T).T

    def _process_parallel(
        self,
        input_data: NDArray,
        source_azimuth: float,
        source_elevation: float,
        direct_gain: float,
        er_gain: float,
        reverb_gain: float,
        grid_output: bool,
    ) -> NDArray:
        """Network on the grid, P0 shaping, then direct sound and reflections.

        Alary and Valimaki (I3DA 2021, Sec. III B): the early part runs beside
        the recursion. The network is rendered with no direct term so that the
        per-direction filters touch the late field alone.
        """
        Q = self.Q
        x = np.asarray(input_data, dtype=float)
        b = self._build_input_gains() * float(reverb_gain)
        c = self._build_output_gains()
        A = self._kronecker_matrix
        if self.absorption_gains is not None:
            A = A * self.absorption_gains
            c = c * self.absorption_gains
        graph = pyFDN.dss_to_td(
            self.delay_length_samps, A, b, c, np.zeros((Q, 1)),
            post_delay=self.absorption_sos,
        )
        grid = graph.process_signal(x.T).T
        if self._p0_sos is not None:
            grid = np.stack([
                sosfilt(np.ascontiguousarray(self._p0_sos[:, :, q]), grid[q])
                for q in range(Q)
            ])
        grid = grid + (
            self._build_direct_gain(source_azimuth, source_elevation) * float(direct_gain)
        ) @ x
        if self._early_reflections is not None:
            grid = grid + self._build_er_grid_signal(x, x.shape[-1], er_gain)
        return grid if grid_output else self._encoding_matrix @ grid

    # ==================================================================
    # Early reflections buffer
    # ==================================================================

    def _build_er_grid_signal(
        self, input_data: NDArray, ir_len: int, er_gain: float = 1.0,
    ) -> NDArray:
        """Early-reflection signal on the grid, per Eqs. 4-5.

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

    def _check_ready(self) -> None:
        if self.delay_length_samps is None:
            raise RuntimeError("Call setup() before processing.")
        if self._kronecker_matrix is None:
            raise RuntimeError("Kronecker feedback matrix not initialised.")
        if self.absorption_gains is None and self.absorption_sos is None:
            raise RuntimeError("Absorption not initialised.")
        if self._encoding_matrix is None:
            raise RuntimeError("Encoding matrix not initialised.")
