"""Frequency-Dependent Directional FDN (Plane-Wave Domain).

Implementation of Alary & Politis (2020) *"Frequency-Dependent Directional
Feedback Delay Network"*, ICASSP 2020.

Architecture
------------
Unlike the JAES 2019 DFDN (which operates in the SH domain with DWM
transforms), this formulation works in the **plane-wave domain** where
each channel of a delay-line group corresponds directly to an incident
direction on a spatial grid of Q points.

Key differences from :class:`DirectionalFDN`:

1. **Per-direction absorption filters** — each of the Q channels in each
   delay-line group gets its own IIR filter parameterised by
   T60(ω, φ, θ) (Eqs. 6–10).
2. **Independent delay lengths** within a group — each of the Q channels
   has a randomly-chosen delay length, improving echo density (Fig. 4).
3. **No DWM transform** — directional shaping is handled entirely by
   the per-channel filters.
4. **Early reflection module** — VBAP-encoded reflections injected at
   tap-in points along the delay lines (Eqs. 3–5).
5. **Output encoding matrix E** — converts grid signals to Ambisonics,
   loudspeaker, or binaural format (Eq. 13).

Usage
-----
>>> from directional_fdn.fdn.planewave_dfdn import PlaneWaveDFDN
>>> pw = PlaneWaveDFDN(
...     sample_rate=48000,
...     num_delay_line_groups=8,
...     sh_order=2,  # Q ≥ (L+1)² grid points
...     frame_size=64,
... )
>>> pw.setup(
...     delay_range_ms=(10, 50),
...     t60_per_direction_ms=t60_map,  # shape (Q, num_bands) or (Q,)
...     freqs=np.array([125, 250, 500, 1000, 2000, 4000, 8000]),
... )
>>> rir = pw.render(source_azimuth=0.0, source_elevation=0.0,
...                  length_seconds=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter, tf2zpk, zpk2tf

from scipy.linalg import hadamard

from . import spatial
from .delay_line import DelayLine, generate_coprime_delay_lengths
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .filter_fitting import (
    interpolate_magnitude_spectrum,
    prony_warped,
    tf2minphase,
)
from ..utils import db2lin


# Speed of sound constant (m/s) — used for air absorption and delay range
_SPEED_OF_SOUND = 343.0


def air_absorption_db_per_meter(freqs: NDArray) -> NDArray:
    """ISO 9613-1 atmospheric absorption at 20 °C, 50 %% humidity.

    Parameters
    ----------
    freqs : NDArray
        Frequencies in Hz.

    Returns
    -------
    NDArray
        Absorption coefficient in dB/m (positive values = attenuation).
    """
    # Simplified ISO 9613-1 model (T=20°C, h=50%)
    return 5.5e-4 * (50.0 / 50.0) * (np.asarray(freqs) / 1000.0) ** 1.7


def delay_range_from_room(
    room: tuple[float, float, float],
    speed_of_sound: float = _SPEED_OF_SOUND,
) -> tuple[float, float]:
    """Compute FDN delay range (ms) from room dimensions.

    The range spans from the shortest axis to the room diagonal,
    converted to milliseconds.  This ensures delay lines cover the
    full range of path lengths between successive wall reflections.

    Parameters
    ----------
    room : (Lx, Ly, Lz)
        Room dimensions in metres.
    speed_of_sound : float
        Speed of sound in m/s.

    Returns
    -------
    (min_ms, max_ms) : (float, float)
        Delay range in milliseconds.
    """
    Lx, Ly, Lz = room
    d_min = min(Lx, Ly, Lz)
    d_max = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    return (
        1000.0 * d_min / speed_of_sound,
        1000.0 * d_max / speed_of_sound,
    )


# ---------------------------------------------------------------------------
# Early Reflection Module (Eqs. 3–5)
# ---------------------------------------------------------------------------

@dataclass
class Reflection:
    """One early reflection with its gain, direction, and delay."""
    gain: float
    azimuth: float      # radians
    elevation: float    # radians
    delay_samples: int  # l_r — tap-in offset into the delay lines


def shoebox_reflections(
    room: tuple[float, float, float],
    source: tuple[float, float, float],
    receiver: tuple[float, float, float],
    sample_rate: float,
    wall_reflectivity: float | tuple[float, float, float, float, float, float] = 0.5,
    speed_of_sound: float = 343.0,
) -> list[Reflection]:
    """Compute first-order image-source reflections for a shoebox room.

    Returns 6 :class:`Reflection` objects (one per wall), sorted by
    arrival time, suitable for :meth:`PlaneWaveDFDN.set_early_reflections`.

    Parameters
    ----------
    room : (Lx, Ly, Lz)
        Room dimensions in metres.
    source : (sx, sy, sz)
        Source position in metres.
    receiver : (rx, ry, rz)
        Receiver position in metres.
    sample_rate : float
        Sample rate (Hz) — used to convert distances to delay_samples.
    wall_reflectivity : float or 6-tuple of floats
        Pressure reflection coefficient per wall (0–1).  If scalar, all
        six walls share the same value.  If a tuple, order is:
        (x=0, x=Lx, y=0, y=Ly, z=0, z=Lz).
    speed_of_sound : float
        Speed of sound in m/s.

    Returns
    -------
    list[Reflection]
        Six first-order reflections sorted by delay (earliest first).
    """
    Lx, Ly, Lz = room
    sx, sy, sz = source
    rx, ry, rz = receiver

    if isinstance(wall_reflectivity, (int, float)):
        refl = (float(wall_reflectivity),) * 6
    else:
        refl = tuple(wall_reflectivity)

    # Image sources: mirror the source across each wall
    image_sources = [
        ((-sx, sy, sz),       refl[0]),   # x = 0
        ((2*Lx - sx, sy, sz), refl[1]),   # x = Lx
        ((sx, -sy, sz),       refl[2]),   # y = 0
        ((sx, 2*Ly - sy, sz), refl[3]),   # y = Ly
        ((sx, sy, -sz),       refl[4]),   # z = 0 (floor)
        ((sx, sy, 2*Lz - sz), refl[5]),   # z = Lz (ceiling)
    ]

    # Direct-sound distance (for gain normalisation)
    d_direct = np.sqrt((sx - rx)**2 + (sy - ry)**2 + (sz - rz)**2)
    d_direct = max(d_direct, 1e-6)

    reflections: list[Reflection] = []
    for (ix, iy, iz), r in image_sources:
        dx = ix - rx
        dy = iy - ry
        dz = iz - rz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = max(dist, 1e-6)

        # Gain: wall reflectivity × inverse-distance law (relative to direct)
        gain = r * (d_direct / dist)

        # Direction from receiver to image source
        azimuth = float(np.arctan2(dy, dx))
        elevation = float(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))

        # Delay relative to direct sound
        delay_s = (dist - d_direct) / speed_of_sound
        delay_samples = max(1, int(np.round(delay_s * sample_rate)))

        reflections.append(Reflection(
            gain=gain,
            azimuth=azimuth,
            elevation=elevation,
            delay_samples=delay_samples,
        ))

    # Sort by arrival time
    reflections.sort(key=lambda r: r.delay_samples)
    return reflections


class EarlyReflectionModule:
    """Early reflection generator per ICASSP 2020 Eqs. 3–5.

    Each reflection is SH-encoded directly into the Ambisonics domain,
    bypassing the plane-wave grid.  This avoids directional quantisation
    artefacts from nearest-neighbour VBAP on coarse grids.

    Parameters
    ----------
    reflections : list[Reflection]
        Reflection definitions.
    sh_order : int
        Ambisonics order for SH encoding.
    """

    def __init__(
        self,
        reflections: list[Reflection],
        sh_order: int,
    ) -> None:
        self.reflections = reflections
        self.sh_order = sh_order
        # Pre-compute per-reflection SH encoding vectors (gain-weighted)
        self.sh_vectors: list[NDArray] = []
        for ref in reflections:
            y = spatial.encode_direction(ref.azimuth, ref.elevation, sh_order)
            self.sh_vectors.append(y * ref.gain)


# ---------------------------------------------------------------------------
# Per-Direction Absorption Filters (Eqs. 6–10)
# ---------------------------------------------------------------------------

class DirectionalAbsorptionFilters:
    """Per-channel IIR absorption for the plane-wave DFDN.

    Each of the total_delay_lines channels gets its own IIR filter fitted
    to the target T60(ω, φ_q, θ_q) and delay length m.

    Parameters
    ----------
    total_delay_lines : int
        N × Q — total number of delay lines.
    delay_lengths : NDArray, shape (N*Q,)
        Per-channel delay lengths in samples.
    sample_rate : float
    frame_size : int
    t60_per_channel_ms : NDArray, shape (N*Q, n_bands) or (N*Q,)
        Target T60 in ms for each delay line at each frequency band.
        If 1D, broadband (scalar per channel).
    freqs : NDArray, optional
        Frequency points (Hz) for frequency-dependent T60.
    filter_order : int
        IIR filter order per channel.
    warp_factor : float
        All-pass warping coefficient.
    """

    def __init__(
        self,
        total_delay_lines: int,
        delay_lengths: NDArray,
        sample_rate: float,
        frame_size: int,
        t60_per_channel_ms: NDArray,
        freqs: NDArray | None = None,
        filter_order: int = 6,
        warp_factor: float = 0.0,
        air_absorption: bool = False,
    ) -> None:
        self.N_total = total_delay_lines
        self.delay_lengths = np.asarray(delay_lengths, dtype=np.int64)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_order = filter_order
        self.rho = warp_factor
        self.air_absorption = air_absorption

        t60 = np.asarray(t60_per_channel_ms, dtype=float)
        if t60.ndim == 1:
            # Shape (N_total,) — broadband scalar per channel
            self._use_filters = False
            t60_s = np.maximum(t60, 1.0) * 1e-3
            self._gains = db2lin(
                -60.0 * self.delay_lengths / (self.sample_rate * t60_s)
            )
        elif t60.ndim == 2 and t60.shape[0] == total_delay_lines:
            # Shape (N_total, n_bands) — frequency-dependent per channel
            self._use_filters = True
            if freqs is None:
                raise ValueError("freqs required for frequency-dependent T60.")
            self.freqs = np.asarray(freqs, dtype=float)
            self._num_coeffs = np.zeros((total_delay_lines, filter_order + 1))
            self._den_coeffs = np.zeros_like(self._num_coeffs)
            self._zi = np.zeros((total_delay_lines, filter_order))
            self._fit_filters(t60)
        else:
            raise ValueError(
                f"t60_per_channel_ms must be shape ({total_delay_lines},) or "
                f"({total_delay_lines}, n_bands), got {t60.shape}."
            )

    def _fit_filters(self, t60_ms: NDArray, n_fft: int = 4096) -> None:
        """Fit one IIR filter per channel from T60(ω) and delay length."""
        # Pre-compute air absorption if enabled
        air_atten_db = None
        if self.air_absorption:
            air_atten_db = air_absorption_db_per_meter(self.freqs)

        for i in range(self.N_total):
            # Eq. 6-8: target gain spectrum for this channel
            t60_s = np.maximum(t60_ms[i], 1.0) * 1e-3
            # Compensate for filter group delay (ceil(order/2))
            effective_len = (
                self.delay_lengths[i] + int(np.ceil(self.filter_order / 2))
            )
            target_gains = db2lin(
                -60.0 * effective_len / (self.sample_rate * t60_s)
            )
            # Apply air absorption: extra attenuation per delay-line path
            if air_atten_db is not None:
                path_m = self.delay_lengths[i] / self.sample_rate * _SPEED_OF_SOUND
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
                z, p, k = tf2zpk(b, a)
                p = p * (0.999 / max_pole)
                b, a = zpk2tf(z, p, k)
            self._num_coeffs[i], self._den_coeffs[i] = b, a

    def process(self, x: NDArray) -> NDArray:
        """Apply per-channel absorption to a (N_total, frame_size) block."""
        if not self._use_filters:
            return x * self._gains[:, np.newaxis]
        out = np.empty_like(x)
        for i in range(self.N_total):
            out[i], self._zi[i] = lfilter(
                self._num_coeffs[i], self._den_coeffs[i], x[i], zi=self._zi[i]
            )
        return out

    def reset(self) -> None:
        if self._use_filters:
            self._zi[:] = 0.0


# ---------------------------------------------------------------------------
# Plane-Wave DFDN
# ---------------------------------------------------------------------------

class PlaneWaveDFDN:
    """Frequency-Dependent Directional FDN (ICASSP 2020).

    Operates in the plane-wave domain: each channel of a delay-line group
    corresponds to a direction on a Q-point spatial grid.  Per-channel
    absorption filters provide both frequency- and direction-dependent
    T60 control.

    Parameters
    ----------
    sample_rate : float
        Audio sample rate in Hz.
    num_delay_line_groups : int
        *N* — number of delay-line groups.
    sh_order : int
        SH order *L* controlling grid density: Q ≥ (L+1)².
        Also defines the Ambisonics output order.
    frame_size : int
        Processing block size in samples.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: float = 48000,
        num_delay_line_groups: int = 8,
        sh_order: int = 2,
        frame_size: int = 64,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_delay_line_groups = num_delay_line_groups  # N
        self.sh_order = sh_order
        self.Q = spatial.num_channels(sh_order)  # grid points
        self.frame_size = frame_size
        self.seed = seed

        self.total_delay_lines = num_delay_line_groups * self.Q  # N × Q
        self.num_outputs = self.Q  # Ambisonics channels (L+1)²

        # State
        self.delay_lines: list[DelayLine] | None = None
        self.delay_length_samps: NDArray | None = None
        self.absorption: DirectionalAbsorptionFilters | None = None
        self._kronecker_matrix: NDArray | None = None
        self._base_feedback: FeedbackMatrix | None = None
        self._encoding_matrix: NDArray | None = None  # E (Eq. 13)
        self._early_reflections: EarlyReflectionModule | None = None

        # Grid directions
        self.grid_azimuths: NDArray | None = None
        self.grid_elevations: NDArray | None = None

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
        freqs: NDArray | None = None,
        filter_order: int = 6,
        warp_factor: float = 0.0,
        feedback_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
        air_absorption: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise all components of the plane-wave DFDN.

        Parameters
        ----------
        delay_range_ms : (float, float)
            Range for randomly-chosen delay lengths (per channel).
        t60_per_direction_ms : float or NDArray
            Controls the target reverberation time.  Accepted formats:

            * **scalar** — same T60 for all directions and frequencies
              (isotropic, broadband).
            * **shape (n_bands,)** — frequency-dependent T60, same for all
              directions (isotropic, freq-dependent).  Requires *freqs*.
            * **shape (Q,)** — direction-dependent T60, broadband.
            * **shape (Q, n_bands)** — direction- *and* frequency-dependent.
              Row *q* gives the T60 for grid direction *q*; columns match
              *freqs*.  Requires *freqs*.
        freqs : NDArray, optional
            Frequency bands (Hz) — required when T60 is frequency-dependent.
        filter_order : int
            IIR filter order for absorption.
        warp_factor : float
            All-pass warping coefficient for Prony fitting.
        feedback_type : FeedbackMatrixType
            Type of base N×N feedback matrix.
        rng : np.random.Generator, optional
        """
        # 0. Spatial grid (may update self.Q for quadrature grids)
        self._init_grid()

        N, F = self.num_delay_line_groups, self.frame_size
        Q = self.Q  # read AFTER grid init, since Q may have changed

        # 1. Delay-line lengths — independent per channel (ICASSP Sec. 3)
        if rng is None and self.seed is not None:
            rng = np.random.default_rng(self.seed)
        all_lengths = generate_coprime_delay_lengths(
            delay_range_ms, N * Q, self.sample_rate, rng=rng
        )
        self.delay_length_samps = all_lengths

        # 2. Feedback matrix A ⊗ I_Q (same Kronecker structure)
        self._init_kronecker_feedback(feedback_type)

        # 3. Per-direction absorption
        t60 = np.asarray(t60_per_direction_ms, dtype=float)
        n_bands = len(freqs) if freqs is not None else 0

        if t60.ndim == 0:
            # Scalar → same broadband T60 for all channels
            t60_expanded = np.full(N * Q, float(t60))

        elif t60.ndim == 1 and t60.shape[0] == Q and freqs is None:
            # (Q,) → direction-dependent broadband
            t60_expanded = np.tile(t60, N)

        elif t60.ndim == 1 and freqs is not None and t60.shape[0] == n_bands:
            # (n_bands,) → frequency-dependent, same for all directions
            # Tile to (N*Q, n_bands)
            t60_expanded = np.tile(t60[np.newaxis, :], (N * Q, 1))

        elif t60.ndim == 1 and t60.shape[0] == Q and freqs is not None:
            # (Q,) with freqs given → direction-dependent broadband
            # (treat as broadband even though freqs is passed)
            t60_expanded = np.tile(t60, N)

        elif t60.ndim == 2 and t60.shape[0] == Q:
            # (Q, n_bands) → direction- and frequency-dependent
            t60_expanded = np.tile(t60, (N, 1))

        else:
            raise ValueError(
                f"t60_per_direction_ms must be scalar, (n_bands,), (Q={Q},), "
                f"or (Q={Q}, n_bands), got shape {t60.shape}."
            )
        self.absorption = DirectionalAbsorptionFilters(
            N * Q,
            all_lengths,
            self.sample_rate,
            F,
            t60_expanded,
            freqs=freqs,
            filter_order=filter_order,
            warp_factor=warp_factor,
            air_absorption=air_absorption,
        )

        # 4. Delay lines
        self.delay_lines = [
            DelayLine(int(m), F) for m in all_lengths
        ]

        # 5. Output encoding matrix E (Eq. 13) — default: Ambisonics
        self._init_ambisonics_encoding()

    def set_early_reflections(
        self, reflections: list[Reflection]
    ) -> None:
        """Configure the early reflection module (Eqs. 3–5).

        Must be called after :meth:`setup` so that the grid is available.
        """
        if self.grid_azimuths is None:
            raise RuntimeError("Call setup() before set_early_reflections().")
        self._early_reflections = EarlyReflectionModule(
            reflections, self.sh_order,
        )

    # ==================================================================
    # Output encoding (Eq. 13)
    # ==================================================================

    def _init_ambisonics_encoding(self) -> None:
        """Build Ambisonics encoding matrix E.

        Shape: ((L+1)², Q) — maps Q grid signals to Ambisonics.

        With a proper quadrature grid (t-design of degree 2L) and SN3D
        normalisation, we use ``E = Y^T · √(4π / Q)``.  This is the
        energy-preserving encoding for Q uncorrelated equal-power grid
        signals:  ``E[|a_lm|²] = σ² / (2l+1)`` — the correct SN3D
        diffuse-field power spectrum.
        """
        # Y matrix: (Q, (L+1)²) in SN3D
        Y = spatial.encode_directions(
            self.grid_azimuths, self.grid_elevations, self.sh_order
        )
        # Energy-preserving SN3D encoding for uncorrelated grid signals
        self._encoding_matrix = Y.T * np.sqrt(4.0 * np.pi / self.Q)  # ((L+1)², Q)

    # ==================================================================
    # Kronecker feedback
    # ==================================================================

    def _init_kronecker_feedback(
        self, matrix_type: FeedbackMatrixType
    ) -> None:
        """Build A ⊗ H_Q — Kronecker feedback with inter-direction mixing.

        Uses A ⊗ H_Q instead of A ⊗ I_Q so that directions are coupled
        through the feedback loop.  This diffuses directional energy
        over time, eliminating directional spikes in late reverb while
        preserving per-channel absorption for early directional T60
        control.

        H_Q is a Q×Q normalized Hadamard matrix (when Q is a power of
        2) or a random orthogonal matrix otherwise.
        """
        self._base_feedback = FeedbackMatrix(
            self.num_delay_line_groups,
            matrix_type,
            self.frame_size,
            seed=self.seed,
        )
        Q = self.Q
        # Build Q×Q unitary mixing matrix
        if Q > 0 and (Q & (Q - 1)) == 0:
            H_Q = hadamard(Q).astype(float) / np.sqrt(Q)
        else:
            rng = np.random.default_rng(
                self.seed + 7777 if self.seed is not None else None
            )
            M = rng.standard_normal((Q, Q))
            H_Q, _ = np.linalg.qr(M)
        self._kronecker_matrix = np.kron(self._base_feedback.matrix, H_Q)
        # Verify unitarity
        NQ = self.total_delay_lines
        err = np.linalg.norm(
            self._kronecker_matrix.T @ self._kronecker_matrix - np.eye(NQ),
            "fro",
        )
        if err > 1e-10:
            raise RuntimeError(
                f"Kronecker feedback matrix unitarity error: {err:.2e}"
            )

    # ==================================================================
    # Input encoding
    # ==================================================================

    def _build_input_gains(self, source_azimuth: float, source_elevation: float) -> NDArray:
        """Build input vector b_i — omnidirectional per ICASSP 2020 Sec. 3.

        Each of the N×Q delay-line channels receives equal gain
        ``1 / √(N·Q)`` so that the total input energy is unity.
        """
        N, Q = self.num_delay_line_groups, self.Q
        b = np.full((N * Q, 1), 1.0 / np.sqrt(N * Q))
        return b

    def _build_direct_gain(
        self, source_azimuth: float, source_elevation: float,
    ) -> NDArray:
        """Build direct-path gain vector in Ambisonics domain.

        Scales by ``1 / √N`` to match the output gain structure where
        ``c`` sums N groups at unit gain.  This yields a DRR consistent
        with the reverb level (approximately −10 dB for typical setups).

        Returns shape ((L+1)², 1).
        """
        N = self.num_delay_line_groups
        y_src = spatial.encode_direction(
            source_azimuth, source_elevation, self.sh_order,
        )  # ((L+1)²,)
        return (y_src / np.sqrt(N)).reshape(-1, 1)

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
    ) -> NDArray:
        """Generate an Ambisonics RIR.

        Parameters
        ----------
        source_azimuth, source_elevation : float
            Source direction in radians.
        length_seconds : float
            Duration of the impulse response.

        Returns
        -------
        NDArray, shape ((L+1)², num_samples)
            Ambisonics RIR in ACN/SN3D format.
        """
        num_samples = int(length_seconds * self.sample_rate)
        F = self.frame_size
        num_samples = (num_samples // F) * F

        impulse = np.zeros((1, num_samples))
        impulse[0, 0] = 1.0

        return self.process(impulse, source_azimuth, source_elevation)

    def process(
        self,
        input_data: NDArray,
        source_azimuth: float = 0.0,
        source_elevation: float = 0.0,
    ) -> NDArray:
        """Process mono input through the plane-wave DFDN.

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

        Returns
        -------
        NDArray, shape ((L+1)², num_samples)
        """
        self._check_ready()
        if input_data.shape[0] != 1:
            raise ValueError("Input must be mono (1 channel).")

        N, Q, F = self.num_delay_line_groups, self.Q, self.frame_size
        NQ = N * Q
        ir_len = input_data.shape[-1]

        # Build gains
        b = self._build_input_gains(source_azimuth, source_elevation)
        c = self._build_output_gains()  # (Q, NQ)
        d = self._build_direct_gain(source_azimuth, source_elevation)  # ((L+1)², 1)
        E = self._encoding_matrix  # ((L+1)², Q)
        A = self._kronecker_matrix  # (NQ, NQ)

        # Pre-compute early-reflection buffer in Ambisonics domain
        has_er = self._early_reflections is not None
        er_buffer = None
        if has_er:
            er_buffer = self._build_er_buffer(input_data, ir_len)

        # Allocate
        output_ambi = np.zeros((self.num_outputs, ir_len))
        fb_out = np.zeros((NQ, F))
        dl_out = np.zeros((NQ, F))

        sample = 0
        while sample + F <= ir_len:
            idx = slice(sample, sample + F)
            x = input_data[:, idx]

            # Input mixing + feedback → delay lines (Eq. 12)
            dl_in = b @ x + fb_out

            # Read / write delay lines
            for i in range(NQ):
                dl_out[i] = self.delay_lines[i].read()
                self.delay_lines[i].write(dl_in[i])

            # Per-direction absorption G_i(z) (Eqs. 6-10)
            absorbed = self.absorption.process(dl_out)

            # Output: sum across groups in grid domain (Eq. 11 reverb part)
            grid_output = c @ absorbed  # (Q, F)

            # Encode reverb to Ambisonics (Eq. 13) + direct path in SH domain
            output_ambi[:, idx] += E @ grid_output + d @ x

            # Add early reflections directly in Ambisonics domain
            if has_er:
                output_ambi[:, idx] += er_buffer[:, idx]

            # Feedback path: A ⊗ H_Q
            fb_out = A @ absorbed

            sample += F

        return output_ambi

    # ==================================================================
    # Early reflections buffer
    # ==================================================================

    def _build_er_buffer(
        self, input_data: NDArray, ir_len: int,
    ) -> NDArray:
        """Pre-compute the early-reflection signal in Ambisonics domain.

        Each reflection is SH-encoded at its true direction and delayed
        by its arrival time.  Scaled by ``1 / √N`` to match the
        direct-path gain convention.

        Returns shape ((L+1)², ir_len) — Ambisonics ER signal added
        directly to the output, bypassing the plane-wave grid.
        """
        N = self.num_delay_line_groups
        n_ambi = (self.sh_order + 1) ** 2
        er_buf = np.zeros((n_ambi, ir_len))

        if self._early_reflections is None:
            return er_buf

        for ref, sh_vec in zip(
            self._early_reflections.reflections,
            self._early_reflections.sh_vectors,
        ):
            l_r = ref.delay_samples
            if l_r >= ir_len:
                continue
            end = ir_len - l_r
            er_buf[:, l_r:l_r + end] += (
                sh_vec[:, np.newaxis] / np.sqrt(N) * input_data[0:1, :end]
            )

        return er_buf

    # ==================================================================
    # State management
    # ==================================================================

    def reset(self) -> None:
        """Clear all internal state."""
        if self.delay_lines is not None:
            for dl in self.delay_lines:
                dl.reset()
        if self.absorption is not None:
            self.absorption.reset()

    def _check_ready(self) -> None:
        if self.delay_lines is None:
            raise RuntimeError("Call setup() before processing.")
        if self._kronecker_matrix is None:
            raise RuntimeError("Kronecker feedback matrix not initialised.")
        if self.absorption is None:
            raise RuntimeError("Absorption not initialised.")
        if self._encoding_matrix is None:
            raise RuntimeError("Encoding matrix not initialised.")
