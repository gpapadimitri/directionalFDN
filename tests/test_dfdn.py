"""Test suite for the Directional Feedback Delay Network."""

from __future__ import annotations

import numpy as np
import pyFDN
import pytest

from src.dfdn_2019 import DFDN2019
from src.directional_fdn import DirectionalFDN
from src.fdn_components import (
    FeedbackMatrix,
    FeedbackMatrixType,
    generate_distinct_delay_lengths,
    geq_band_frequencies,
)
from src.room import Reflection
from src.spatial import (
    encode_directions,
    uniform_directions,
)
from src.transform import DirectionalWeighting
from src.utils import db2lin, lin2db, ms_to_samps

# ======================================================================
# Utilities
# ======================================================================

class TestUtils:
    def test_db2lin_roundtrip(self):
        for val in [0.0, -6.0, -20.0, 6.0]:
            assert abs(lin2db(db2lin(val)) - val) < 1e-10

    def test_ms_to_samps(self):
        assert ms_to_samps(1000.0, 48000.0) == 48000
        assert ms_to_samps(0.0, 48000.0) == 0


# ======================================================================
# Spatial encoding
# ======================================================================

class TestSpatial:
    # AmbiX SN3D: Y_0^0 = 1 in every direction.
    _W = 1.0

    def test_encode_front(self):
        """Front direction (0, 0) -> W active, X active (ACN 3)."""
        c = encode_directions(0.0, 0.0, order=1)
        np.testing.assert_allclose(
            c, [self._W, 0.0, 0.0, self._W], atol=1e-12,
        )

    def test_encode_left(self):
        """Left direction (pi/2, 0) -> W active, Y active (ACN 1)."""
        c = encode_directions(np.pi / 2, 0.0, order=1)
        np.testing.assert_allclose(
            c, [self._W, self._W, 0.0, 0.0], atol=1e-12,
        )

    def test_encode_up(self):
        """Up direction (0, pi/2) -> W active, Z active (ACN 2)."""
        c = encode_directions(0.0, np.pi / 2, order=1)
        np.testing.assert_allclose(
            c, [self._W, 0.0, self._W, 0.0], atol=1e-12,
        )

    def test_encode_normalisation(self):
        """First-order SN3D: ||[Y,Z,X]|| = 1 for any direction."""
        for az in np.linspace(0, 2 * np.pi, 13):
            for el in np.linspace(-np.pi / 2, np.pi / 2, 7):
                c = encode_directions(az, el, order=1)
                assert abs(np.linalg.norm(c[1:]) - self._W) < 1e-12

    def test_encode_shape(self):
        assert encode_directions(0.3, 0.1, order=2).shape == (9,)
        az = np.linspace(0, np.pi, 5)
        assert encode_directions(az, 0.0, order=2).shape == (5, 9)

    def test_uniform_directions(self):
        az, el = uniform_directions(50)
        assert len(az) == 50
        assert np.all(el >= -np.pi / 2) and np.all(el <= np.pi / 2)


# ======================================================================
# Delay lengths
# ======================================================================

class TestDelayLengths:
    def test_coprime_lengths(self):
        lengths = generate_distinct_delay_lengths(
            (10, 30), 4, 48000, rng=np.random.default_rng(42)
        )
        assert len(lengths) == 4
        assert all(v >= ms_to_samps(10, 48000) for v in lengths)


# ======================================================================
# Feedback matrix
# ======================================================================

class TestFeedbackMatrix:
    def test_random_is_lossless(self):
        fb = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, seed=0)
        assert pyFDN.is_unilossless(fb.matrix)

    def test_hadamard_is_lossless(self):
        fb = FeedbackMatrix(8, FeedbackMatrixType.HADAMARD)
        assert pyFDN.is_unilossless(fb.matrix)

    def test_hadamard_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            FeedbackMatrix(6, FeedbackMatrixType.HADAMARD)

    def test_process_preserves_energy(self):
        fb = FeedbackMatrix(4, FeedbackMatrixType.RANDOM, seed=1)
        x = np.random.default_rng(2).standard_normal((4, 32))
        y = fb.matrix @ x
        np.testing.assert_allclose(
            np.sum(x**2), np.sum(y**2), rtol=1e-10
        )


# ======================================================================
# Directional FDN
# ======================================================================

class TestDFDN2019:
    def _make_dfdn(self, seed=42):
        dfdn = DFDN2019(
            sample_rate=48000,
            num_delay_line_groups=4,
            ambisonics_order=1,
            seed=seed,
        )
        dfdn.setup(delay_range_ms=(10, 30), des_t60_ms=800)
        return dfdn

    def test_output_shape(self):
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=0.5)
        expected_len = int(0.5 * 48000 / 64) * 64
        assert rir.shape == (4, expected_len)

    def test_w_channel_active_with_omni_input(self):
        """Eq. 17 drives W, so the W channel carries the late field."""
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=0.5)
        w_energy = np.sum(rir[0] ** 2)
        assert w_energy > 1e-6

    def test_direct_path_encodes_direction(self):
        """Direct path d should encode source direction (Eq. 11)."""
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=0.1)
        # Front source: first sample should have X > 0 (ACN 3)
        assert rir[3, 0] > 0

    def test_repeat_render_is_deterministic(self):
        dfdn = self._make_dfdn()
        rir1 = dfdn.render(0.0, 0.0, length_seconds=0.2)
        rir2 = dfdn.render(0.0, 0.0, length_seconds=0.2)
        np.testing.assert_allclose(rir1, rir2, atol=1e-12)

    def test_different_directions_differ(self):
        """Directional decay + DWM: different source directions should differ."""
        dfdn = self._make_dfdn()
        K = 60
        az_map, el_map = uniform_directions(K)
        t60_map = 400.0 + 400.0 * np.abs(np.cos(az_map))
        dfdn.set_directional_decay(t60_map, az_map, el_map)

        rir_front = dfdn.render(0.0, 0.0, length_seconds=0.3)
        rir_left = dfdn.render(np.pi / 2, 0.0, length_seconds=0.3)
        assert not np.allclose(rir_front, rir_left, atol=1e-10)

    def test_kronecker_unitarity(self):
        dfdn = self._make_dfdn()
        kron = dfdn._kronecker_matrix
        eye = np.eye(kron.shape[0])
        np.testing.assert_allclose(kron.T @ kron, eye, atol=1e-10)

    def test_energy_decays(self):
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=1.0)
        n = rir.shape[1]
        e_first = np.sum(rir[:, : n // 4] ** 2)
        e_last = np.sum(rir[:, 3 * n // 4 :] ** 2)
        assert e_first > e_last


# ======================================================================
# Directional weighting T_i
# ======================================================================

class TestDirectionalWeighting:
    def test_matrices_correct_shape(self):
        K = 60
        azimuths, elevations = uniform_directions(K)
        t60_ms = 600.0 + 200.0 * np.abs(np.cos(azimuths))
        group_lengths = np.array([521, 601, 701, 809])

        tw = DirectionalWeighting(
            num_groups=4,
            ambisonics_order=1,
            group_delay_lengths=group_lengths,
            sample_rate=48000,
            t60_at_directions_ms=t60_ms,
            azimuths=azimuths,
            elevations=elevations,
        )
        assert len(tw.transform_matrices) == 4
        for m in tw.transform_matrices:
            assert m.shape == (4, 4)

    def test_isotropic_t60_gives_identity(self):
        """When T60 is the same everywhere, T_i should be near identity."""
        K = 120
        azimuths, elevations = uniform_directions(K)
        t60_ms = np.full(K, 800.0)  # uniform T60
        group_lengths = np.array([521, 601])

        tw = DirectionalWeighting(
            num_groups=2,
            ambisonics_order=1,
            group_delay_lengths=group_lengths,
            sample_rate=48000,
            t60_at_directions_ms=t60_ms,
            azimuths=azimuths,
            elevations=elevations,
        )
        for m in tw.transform_matrices:
            np.testing.assert_allclose(m, np.eye(4), atol=1e-10)

    def test_anisotropic_off_diagonal(self):
        """Anisotropic T60 should produce non-diagonal T_i (cross-SH coupling)."""
        K = 60
        azimuths, elevations = uniform_directions(K)
        # 3:1, but bounded away from zero: a T60 that reaches 0 makes the decay
        # gain non-band-limited, and no finite grid satisfies Eq. 35 for it.
        t60_ms = 600.0 + 300.0 * np.cos(azimuths)
        group_lengths = np.array([521])

        tw = DirectionalWeighting(
            num_groups=1,
            ambisonics_order=1,
            group_delay_lengths=group_lengths,
            sample_rate=48000,
            t60_at_directions_ms=t60_ms,
            azimuths=azimuths,
            elevations=elevations,
        )
        T = tw.transform_matrices[0]
        off_diag = T - np.diag(np.diag(T))
        assert np.max(np.abs(off_diag)) > 1e-3


# ======================================================================
# Plane-Wave DFDN (ICASSP 2020)
# ======================================================================

class TestDirectionalFDN:
    """Tests for the ICASSP 2020 grid-based DirectionalFDN."""

    @staticmethod
    def _make(
        N: int = 4,
        L: int = 1,
        t60: float = 800.0,
        seed: int = 42,
    ) -> DirectionalFDN:
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=N,
            sh_order=L,
            seed=seed,
        )
        pw.setup(
            delay_range_ms=(10, 60),
            t60_per_direction_ms=t60,
        )
        return pw

    def test_basic_setup(self):
        pw = self._make()
        Q = pw.Q  # grid size (t-design may differ from (L+1)^2)
        assert pw.total_delay_lines == 4 * Q
        assert len(pw.delay_length_samps) == 4 * Q
        assert pw.delay_length_samps is not None

    def test_independent_delay_lengths(self):
        """All N*Q delay lengths should be distinct (coprime primes)."""
        pw = self._make()
        lengths = pw.delay_length_samps
        assert len(np.unique(lengths)) == len(lengths)

    def test_kronecker_unitarity(self):
        pw = self._make()
        A = pw._kronecker_matrix
        NQ = pw.total_delay_lines
        np.testing.assert_allclose(A.T @ A, np.eye(NQ), atol=1e-10)

    def test_encoding_matrix_shape(self):
        pw = self._make(L=2)
        E = pw._encoding_matrix
        Q_ambi = (2 + 1) ** 2  # 9 Ambisonics channels
        Q_grid = pw.Q            # actual grid size (may be > Q_ambi)
        assert E.shape == (Q_ambi, Q_grid)

    def test_render_shape(self):
        pw = self._make()
        rir = pw.render(0.0, 0.0, length_seconds=0.5)
        expected_len = int(0.5 * 48000 // 64) * 64
        assert rir.shape == (4, expected_len)

    def test_render_order2(self):
        pw = self._make(L=2, N=3)
        rir = pw.render(0.0, 0.0, length_seconds=0.3)
        Q_ambi = 9
        assert rir.shape[0] == Q_ambi

    def test_energy_decays(self):
        pw = self._make()
        rir = pw.render(0.0, 0.0, length_seconds=1.0)
        n = rir.shape[1]
        e_first = np.sum(rir[:, : n // 4] ** 2)
        e_last = np.sum(rir[:, 3 * n // 4 :] ** 2)
        assert e_first > e_last

    def test_directional_t60(self):
        """Direction-dependent T60 should produce anisotropic output."""
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            seed=42,
        )
        # Init grid first to learn actual Q (t-design may differ from (L+1)^2)
        pw._init_grid()
        Q = pw.Q
        t60_dir = np.linspace(400.0, 1000.0, Q)
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=t60_dir,
        )
        Q_ambi = (pw.sh_order + 1) ** 2
        rir = pw.render(0.0, 0.0, length_seconds=1.0)
        assert rir.shape[0] == Q_ambi
        # Energy should vary across channels
        channel_energies = np.sum(rir ** 2, axis=1)
        assert np.std(channel_energies) > 0

    def test_freq_dependent_isotropic(self):
        """Frequency-dependent but isotropic T60 (n_bands,) -- most common case."""
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            seed=42,
        )
        # On the GEQ grid: [DC, 63, 125, 250, 500, 1k, 2k, 4k, 8k, fs].
        t60_ms = np.array(
            [1200.0, 1200, 1200, 1100, 1000, 900, 800, 700, 600, 300]
        )
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=t60_ms,
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5)
        assert rir.shape[0] == 4

    def test_freq_dependent_t60(self):
        """Frequency-dependent + direction-dependent T60."""
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            seed=42,
        )
        # Init grid first to learn actual Q
        pw._init_grid()
        Q = pw.Q
        Q_ambi = (pw.sh_order + 1) ** 2
        t60_2d = np.linspace(
            [1000, 1000, 1000, 900, 800, 700, 600, 500, 400, 200],
            [700, 700, 700, 620, 540, 460, 380, 300, 220, 100],
            Q,
        ).astype(float)  # (Q, 10)
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=t60_2d,
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5)
        assert rir.shape[0] == Q_ambi

    def test_repeat_render_is_deterministic(self):
        pw = self._make()
        rir1 = pw.render(0.0, 0.0, length_seconds=0.2)
        rir2 = pw.render(0.0, 0.0, length_seconds=0.2)
        np.testing.assert_allclose(rir1, rir2, atol=1e-12)


class TestEarlyReflections:
    def test_er_grid_signal_is_vbap_panned(self):
        """Eqs. 4-5: each reflection lands on the grid, at its own delay,
        panned toward its own direction."""
        pw = DirectionalFDN(
            sample_rate=48000, num_delay_line_groups=4, sh_order=1,
            seed=42,
        )
        pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=800.0)

        refs = [
            Reflection(gain=0.8, azimuth=0.0, elevation=0.0, delay_samples=100),
            Reflection(gain=0.5, azimuth=np.pi, elevation=0.0, delay_samples=200),
        ]
        pw.set_early_reflections(refs)

        impulse = np.zeros((1, 400))
        impulse[0, 0] = 1.0
        grid = pw._build_er_grid_signal(impulse, 400)

        assert grid.shape == (pw.Q, 400)

        # Energy appears only at the two arrival times.
        active = np.flatnonzero(np.any(np.abs(grid) > 1e-12, axis=0))
        np.testing.assert_array_equal(active, [100, 200])

        # VBAP puts unit-sum weights on the hull face containing the target,
        # so each column sums to the reflection's gain.
        assert grid[:, 100].sum() == pytest.approx(0.8)
        assert grid[:, 200].sum() == pytest.approx(0.5)

        # ...and weights the grid point nearest the target most heavily.
        for ref, sample in zip(refs, (100, 200), strict=True):
            target = np.array([
                np.cos(ref.elevation) * np.cos(ref.azimuth),
                np.cos(ref.elevation) * np.sin(ref.azimuth),
                np.sin(ref.elevation),
            ])
            grid_vecs = np.stack([
                np.cos(pw.grid_elevations) * np.cos(pw.grid_azimuths),
                np.cos(pw.grid_elevations) * np.sin(pw.grid_azimuths),
                np.sin(pw.grid_elevations),
            ])
            assert np.argmax(grid[:, sample]) == np.argmax(target @ grid_vecs)

    def test_er_injection(self):
        """Early reflections should appear at their expected delay times."""
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            seed=42,
        )
        pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=800.0)
        rir_no_er = pw.render(0.0, 0.0, length_seconds=0.5)

        er_delay = 200  # samples
        pw.set_early_reflections([
            Reflection(gain=0.9, azimuth=0.0, elevation=0.0, delay_samples=er_delay),
        ])
        rir_with_er = pw.render(0.0, 0.0, length_seconds=0.5)

        # Energy around the ER arrival should be higher than without ERs
        window = slice(er_delay, er_delay + 64)
        e_no = np.sum(rir_no_er[0, window] ** 2)
        e_with = np.sum(rir_with_er[0, window] ** 2)
        assert e_with > e_no


# ======================================================================
# Air absorption accounting
# ======================================================================

class TestAirAbsorptionAccounting:
    """Air absorption is part of the target T60 budget, not an extra loss.

    Enabling ``air_absorption`` must not change the realised decay: it
    reallocates the required attenuation between boundary and medium and
    reports when the target is not achievable in air.
    """

    freqs = geq_band_frequencies()
    t60_ms = np.array(
        [1000.0, 1000, 1000, 1100, 1000, 1000, 950, 900, 900, 400]
    )

    @staticmethod
    def _realised_t60_ms(pw, freqs):
        """Per-band T60 implied by the fitted absorption filters."""
        from scipy.signal import sosfreqz

        sos = pw.absorption_sos            # (n_sections, 6, N)
        delays = np.asarray(pw.delay_length_samps, dtype=float)
        w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / pw.sample_rate
        out = np.zeros((sos.shape[2], len(freqs)))
        for i in range(sos.shape[2]):
            _, h = sosfreqz(sos[:, :, i], worN=w)
            att_db = -20.0 * np.log10(np.abs(h))
            out[i] = 60.0 * delays[i] / (pw.sample_rate * att_db)
        return out.mean(axis=0) * 1e3

    def _make(self, air, t60_ms=None):
        pw = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            seed=42,
        )
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=self.t60_ms if t60_ms is None else t60_ms,
            air_absorption=air,
        )
        return pw

    def test_air_absorption_shortens_realised_t60(self):
        """Air must shorten the decay where it acts, and leave the bottom alone.

        Measured at the eight peaking-band centres rather than the full grid:
        the outer two points are interpolation anchors, and ``sosfreqz`` at fs
        wraps to DC, so probing there reports the DC value again.

        The claim is deliberately not "shorter in every band". The GEQ is a
        single global least-squares fit, so steepening the high shelf -- air
        takes the fs anchor from 400 ms to under 90 ms -- shifts neighbouring
        bands by a few percent in either direction. What must hold is that the
        bands where air actually acts get clearly shorter while the bottom of
        the range does not move much.

        Compared against air=False rather than the target so that the
        (separate) filter-fitting error cancels out.
        """
        bands = self.freqs[1:9]                    # 63 Hz to 8 kHz
        off = self._realised_t60_ms(self._make(False), bands)
        on = self._realised_t60_ms(self._make(True), bands)
        assert on[-1] < off[-1] * 0.85             # 8 kHz
        assert on[-2] < off[-2] * 0.90             # 4 kHz
        assert on[0] == pytest.approx(off[0], rel=0.05)   # 63 Hz

    def test_shares_account_for_the_realised_decay(self):
        """air_db + boundary_db must equal the loss the realised T60 needs."""
        pw = self._make(True)
        required = (
            60.0
            * np.asarray(pw.delay_length_samps, dtype=float)[:, None]
            / (pw.sample_rate * pw.realised_t60_s)
        )
        np.testing.assert_allclose(
            pw.air_attenuation_db + pw.boundary_attenuation_db, required, rtol=1e-10
        )

    def test_target_beyond_the_air_budget_is_clamped_toward_it(self):
        """Air is a ceiling: a 5 s target at 8 kHz cannot survive it.

        This replaces an earlier warning-based check.  Under the combined
        model an unreachable target is not an error to report, it is simply
        pulled down toward what the medium allows.
        """
        from src.air import Atmosphere

        t60 = self.t60_ms.copy()
        t60[-1] = 5000.0
        pw = self._make(True, t60_ms=t60)
        ceiling = Atmosphere().t60(self.freqs)[-1]
        realised_top = pw.realised_t60_s[:, -1].mean()
        assert realised_top < 5.0 * 0.5
        assert realised_top < ceiling


@pytest.mark.parametrize(
    "t60", [-800.0, np.full(10, -1200.0), np.full(10, np.nan), np.zeros(10), 0.0],
    ids=["neg-scalar", "neg-bands", "nan", "zero-bands", "zero-scalar"],
)
def test_setup_rejects_a_non_positive_or_non_finite_t60(t60):
    """A negative T60 designs a gain above one: 10 bands of -1200 ms grew 280 dB."""
    pw = DirectionalFDN(48000, 4, sh_order=1, seed=1)
    with pytest.raises(ValueError, match="finite and positive"):
        pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=t60)


def test_zero_length_render_is_empty_not_an_error():
    pw = DirectionalFDN(48000, 4, sh_order=1, seed=1)
    pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=800.0)
    assert pw.render(0.0, 0.0, 0.0).shape == (4, 0)
    assert pw.render(0.0, 0.0, 0.0, grid_output=True).shape == (pw.Q, 0)


@pytest.mark.parametrize("az, el", [(np.nan, 0.0), (0.0, np.nan), (np.inf, 0.0)])
def test_non_finite_source_direction_raises_a_clear_error(az, el):
    pw = DirectionalFDN(48000, 4, sh_order=1, seed=1)
    pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=800.0)
    with pytest.raises(ValueError, match="finite direction"):
        pw.render(az, el, 0.1)
