"""Test suite for the Directional Feedback Delay Network."""

from __future__ import annotations

import numpy as np
import pytest

from src.fdn.absorption import AbsorptionGains, create_absorption
from src.fdn.delay_line import DelayLine, generate_coprime_delay_lengths
from src.fdn.directional_fdn import DirectionalFDN
from src.fdn.fdn import FDN
from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from src.fdn.spatial import (
    acn_to_lm,
    encode_direction,
    encode_directions,
    num_channels,
    uniform_directions,
)
from src.fdn.transform import DirectionalWeighting
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
    # SN3D normalisation constant: Y_0^0 = 1/sqrt(4π)
    _W = 1.0 / np.sqrt(4.0 * np.pi)

    def test_acn_to_lm(self):
        assert acn_to_lm(0) == (0, 0)
        assert acn_to_lm(1) == (1, -1)
        assert acn_to_lm(2) == (1, 0)
        assert acn_to_lm(3) == (1, 1)
        assert acn_to_lm(4) == (2, -2)

    def test_num_channels(self):
        assert num_channels(0) == 1
        assert num_channels(1) == 4
        assert num_channels(2) == 9
        assert num_channels(3) == 16

    def test_encode_front(self):
        """Front direction (0, 0) → W active, X active (ACN 3)."""
        c = encode_direction(0.0, 0.0, order=1)
        np.testing.assert_allclose(
            c, [self._W, 0.0, 0.0, self._W], atol=1e-12,
        )

    def test_encode_left(self):
        """Left direction (π/2, 0) → W active, Y active (ACN 1)."""
        c = encode_direction(np.pi / 2, 0.0, order=1)
        np.testing.assert_allclose(
            c, [self._W, self._W, 0.0, 0.0], atol=1e-12,
        )

    def test_encode_up(self):
        """Up direction (0, π/2) → W active, Z active (ACN 2)."""
        c = encode_direction(0.0, np.pi / 2, order=1)
        np.testing.assert_allclose(
            c, [self._W, 0.0, self._W, 0.0], atol=1e-12,
        )

    def test_encode_normalisation(self):
        """First-order SN3D: ||[Y,Z,X]|| = 1/√(4π) for any direction."""
        for az in np.linspace(0, 2 * np.pi, 13):
            for el in np.linspace(-np.pi / 2, np.pi / 2, 7):
                c = encode_direction(az, el, order=1)
                assert abs(np.linalg.norm(c[1:]) - self._W) < 1e-12

    def test_encode_directions_shape(self):
        az = np.linspace(0, np.pi, 5)
        el = np.zeros(5)
        out = encode_directions(az, el, order=2)
        assert out.shape == (5, 9)

    def test_uniform_directions(self):
        az, el = uniform_directions(50)
        assert len(az) == 50
        assert np.all(el >= -np.pi / 2) and np.all(el <= np.pi / 2)


# ======================================================================
# Delay line
# ======================================================================

class TestDelayLine:
    def test_basic_delay(self):
        dl = DelayLine(100, 10)
        # Write an impulse
        buf = np.zeros(10)
        buf[0] = 1.0
        dl.write(buf)
        # Should not appear yet (delay = 100)
        out = dl.read()
        assert np.allclose(out, 0.0)

    def test_fifo_order(self):
        dl = DelayLine(20, 10)
        buf1 = np.arange(10, dtype=float)
        dl.write(buf1)
        dl.write(np.zeros(10))
        out = dl.read()
        np.testing.assert_allclose(out, buf1)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            DelayLine(5, 10)

    def test_reset(self):
        dl = DelayLine(100, 10)
        dl.write(np.ones(10))
        dl.reset()
        assert np.allclose(dl.data, 0.0)

    def test_coprime_lengths(self):
        lengths = generate_coprime_delay_lengths(
            (10, 30), 4, 48000, rng=np.random.default_rng(42)
        )
        assert len(lengths) == 4
        assert all(v >= ms_to_samps(10, 48000) for v in lengths)


# ======================================================================
# Feedback matrix
# ======================================================================

class TestFeedbackMatrix:
    def test_random_is_unitary(self):
        fb = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, 64, seed=0)
        assert FeedbackMatrix.is_unitary(fb.matrix)

    def test_hadamard_is_unitary(self):
        fb = FeedbackMatrix(8, FeedbackMatrixType.HADAMARD, 64)
        assert FeedbackMatrix.is_unitary(fb.matrix)

    def test_hadamard_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            FeedbackMatrix(6, FeedbackMatrixType.HADAMARD, 64)

    def test_process_preserves_energy(self):
        fb = FeedbackMatrix(4, FeedbackMatrixType.RANDOM, 32, seed=1)
        x = np.random.default_rng(2).standard_normal((4, 32))
        y = fb.process(x)
        np.testing.assert_allclose(
            np.sum(x**2), np.sum(y**2), rtol=1e-10
        )


# ======================================================================
# Absorption
# ======================================================================

class TestAbsorption:
    def test_gains_attenuate(self):
        lengths = np.array([500, 600, 700, 800])
        abs_g = AbsorptionGains(4, lengths, 48000, 32, 500.0)
        x = np.ones((4, 32))
        y = abs_g.process(x)
        # Output should be attenuated
        assert np.all(y < x)
        assert np.all(y > 0)

    def test_factory_scalar(self):
        abs_obj = create_absorption(4, np.array([500, 600, 700, 800]), 48000, 32, 500.0)
        assert isinstance(abs_obj, AbsorptionGains)


# ======================================================================
# Base FDN
# ======================================================================

class TestFDN:
    def _make_fdn(self, seed=42):
        fdn = FDN(48000, 1, 1, 4, 64, seed=seed)
        fdn.init_delay_line_lengths(np.array([521, 601, 701, 809]))
        fdn.init_feedback_matrix(FeedbackMatrixType.RANDOM)
        fdn.init_absorption(500.0)
        fdn.init_delay_lines()
        b = np.ones((4, 1)) / 2.0
        c = np.ones((1, 4)) / 2.0
        fdn.init_io_gains(b, c)
        fdn.init_direct_gain(np.zeros((1, 1)))
        return fdn

    def test_process_shape(self):
        fdn = self._make_fdn()
        imp = np.zeros((1, 48000))
        imp[0, 0] = 1.0
        out = fdn.process(imp)
        assert out.shape == (1, 48000)

    def test_output_not_silent(self):
        fdn = self._make_fdn()
        imp = np.zeros((1, 48000))
        imp[0, 0] = 1.0
        out = fdn.process(imp)
        assert np.max(np.abs(out)) > 1e-6

    def test_output_decays(self):
        fdn = self._make_fdn()
        imp = np.zeros((1, 48000))
        imp[0, 0] = 1.0
        out = fdn.process(imp)
        # Energy in first quarter should exceed last quarter
        e1 = np.sum(out[0, :12000] ** 2)
        e4 = np.sum(out[0, 36000:] ** 2)
        assert e1 > e4

    def test_reset_clears_state(self):
        fdn = self._make_fdn()
        imp = np.zeros((1, 4800))
        imp[0, 0] = 1.0
        fdn.process(imp)
        fdn.reset()
        # After reset, processing silence should be silent
        out = fdn.process(np.zeros((1, 4800)))
        assert np.max(np.abs(out)) < 1e-12


# ======================================================================
# Directional FDN
# ======================================================================

class TestDirectionalFDN:
    def _make_dfdn(self, seed=42):
        dfdn = DirectionalFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            ambisonics_order=1,
            frame_size=64,
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
        """With omnidirectional input (paper mode), W channel should be active."""
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=0.5, use_directional_input=False)
        w_energy = np.sum(rir[0] ** 2)
        assert w_energy > 1e-6

    def test_directional_input_activates_xyz(self):
        """Directional input (extension) should create energy in X/Y/Z channels."""
        dfdn = self._make_dfdn()
        rir = dfdn.render(
            np.pi / 4, 0.0, length_seconds=0.5, use_directional_input=True
        )
        # X and Y should have energy for az=π/4
        x_energy = np.sum(rir[3] ** 2)
        y_energy = np.sum(rir[1] ** 2)
        assert x_energy > 1e-8
        assert y_energy > 1e-8

    def test_direct_path_encodes_direction(self):
        """Direct path d should encode source direction (Eq. 11)."""
        dfdn = self._make_dfdn()
        rir = dfdn.render(0.0, 0.0, length_seconds=0.1)
        # Front source: first sample should have X > 0 (ACN 3)
        assert rir[3, 0] > 0

    def test_reset_between_renders(self):
        dfdn = self._make_dfdn()
        rir1 = dfdn.render(0.0, 0.0, length_seconds=0.2)
        dfdn.reset()
        rir2 = dfdn.render(0.0, 0.0, length_seconds=0.2)
        np.testing.assert_allclose(rir1, rir2, atol=1e-12)

    def test_different_directions_differ(self):
        """Directional decay + DWM: different source directions should differ."""
        dfdn = self._make_dfdn()
        K = 60
        az_map = np.linspace(0, 2 * np.pi, K, endpoint=False)
        el_map = np.zeros(K)
        t60_map = 400.0 + 400.0 * np.abs(np.cos(az_map))
        dfdn.set_directional_decay(t60_map, az_map, el_map)

        rir_front = dfdn.render(0.0, 0.0, length_seconds=0.3)
        dfdn.reset()
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
        azimuths = np.linspace(0, 2 * np.pi, K, endpoint=False)
        elevations = np.zeros(K)
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
        azimuths = np.linspace(0, 2 * np.pi, K, endpoint=False)
        elevations = np.zeros(K)
        t60_ms = 400.0 + 400.0 * np.cos(azimuths)  # strong anisotropy
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

from src.fdn.planewave_dfdn import (
    DirectionalAbsorptionFilters,
    EarlyReflectionModule,
    PlaneWaveDFDN,
    Reflection,
)


class TestPlaneWaveDFDN:
    """Tests for the ICASSP 2020 Plane-Wave Domain DFDN."""

    @staticmethod
    def _make(
        N: int = 4,
        L: int = 1,
        t60: float = 800.0,
        seed: int = 42,
    ) -> PlaneWaveDFDN:
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=N,
            sh_order=L,
            frame_size=64,
            seed=seed,
        )
        pw.setup(
            delay_range_ms=(10, 60),
            t60_per_direction_ms=t60,
        )
        return pw

    def test_basic_setup(self):
        pw = self._make()
        Q = pw.Q  # grid size (t-design may differ from (L+1)²)
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
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            frame_size=64,
            seed=42,
        )
        # Init grid first to learn actual Q (t-design may differ from (L+1)²)
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
        """Frequency-dependent but isotropic T60 (n_bands,) — most common case."""
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            frame_size=64,
            seed=42,
        )
        freqs = np.array([125, 500, 2000, 8000], dtype=float)
        t60_ms = np.array([1200, 1000, 800, 600], dtype=float)  # (n_bands,)
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=t60_ms,
            freqs=freqs,
            filter_order=4,
            warp_factor=0.6,
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5)
        assert rir.shape[0] == 4

    def test_freq_dependent_t60(self):
        """Frequency-dependent + direction-dependent T60."""
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            frame_size=64,
            seed=42,
        )
        # Init grid first to learn actual Q
        pw._init_grid()
        Q = pw.Q
        Q_ambi = (pw.sh_order + 1) ** 2
        freqs = np.array([125, 500, 2000, 8000], dtype=float)
        t60_2d = np.linspace(
            [1000, 800, 600, 400],
            [700, 500, 300, 150],
            Q,
        ).astype(float)  # (Q, n_bands)
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=t60_2d,
            freqs=freqs,
            filter_order=4,
            warp_factor=0.7,
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5)
        assert rir.shape[0] == Q_ambi

    def test_reset(self):
        pw = self._make()
        rir1 = pw.render(0.0, 0.0, length_seconds=0.2)
        pw.reset()
        rir2 = pw.render(0.0, 0.0, length_seconds=0.2)
        np.testing.assert_allclose(rir1, rir2, atol=1e-12)


class TestEarlyReflections:
    def test_er_module_shape(self):
        sh_order = 1
        n_ambi = (sh_order + 1) ** 2
        refs = [
            Reflection(gain=0.8, azimuth=0.0, elevation=0.0, delay_samples=100),
            Reflection(gain=0.5, azimuth=np.pi, elevation=0.0, delay_samples=200),
        ]
        er = EarlyReflectionModule(refs, sh_order)
        assert len(er.sh_vectors) == 2
        for sh_vec in er.sh_vectors:
            assert sh_vec.shape == (n_ambi,)

    def test_er_injection(self):
        """Early reflections should appear at their expected delay times."""
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            frame_size=64,
            seed=42,
        )
        pw.setup(delay_range_ms=(10, 30), t60_per_direction_ms=800.0)
        rir_no_er = pw.render(0.0, 0.0, length_seconds=0.5)

        pw.reset()
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


class TestDirectionalAbsorptionFilters:
    def test_scalar_t60(self):
        """Broadband scalar T60 should use simple gains."""
        filt = DirectionalAbsorptionFilters(
            total_delay_lines=4,
            delay_lengths=np.array([500, 600, 700, 800]),
            sample_rate=48000,
            frame_size=64,
            t60_per_channel_ms=np.full(4, 800.0),
        )
        assert not filt._use_filters
        x = np.random.randn(4, 64)
        y = filt.process(x)
        assert y.shape == x.shape
        # Energy should decrease
        assert np.sum(y ** 2) < np.sum(x ** 2)

    def test_freq_dependent_t60(self):
        """Multi-band T60 should use IIR filters."""
        freqs = np.array([125, 500, 2000, 8000], dtype=float)
        t60 = np.array([
            [1000, 800, 600, 400],
            [900, 700, 500, 300],
        ], dtype=float)
        filt = DirectionalAbsorptionFilters(
            total_delay_lines=2,
            delay_lengths=np.array([500, 600]),
            sample_rate=48000,
            frame_size=64,
            t60_per_channel_ms=t60,
            freqs=freqs,
            filter_order=4,
            warp_factor=0.7,
        )
        assert filt._use_filters
        x = np.random.randn(2, 64)
        y = filt.process(x)
        assert y.shape == x.shape


# ======================================================================
# Air absorption accounting
# ======================================================================

class TestAirAbsorptionAccounting:
    """Air absorption is part of the target T60 budget, not an extra loss.

    Enabling ``air_absorption`` must not change the realised decay: it
    reallocates the required attenuation between boundary and medium and
    reports when the target is not achievable in air.
    """

    freqs = np.array([125.0, 500.0, 2000.0, 8000.0])
    t60_ms = np.array([1000.0, 1100.0, 1000.0, 900.0])

    @staticmethod
    def _realised_t60_ms(pw, freqs):
        """Per-band T60 implied by the fitted absorption filters."""
        from scipy.signal import freqz, group_delay

        absn = pw.absorption
        w = 2.0 * np.pi * freqs / pw.sample_rate
        out = np.zeros((absn.N_total, len(freqs)))
        for i in range(absn.N_total):
            b, a = absn._num_coeffs[i], absn._den_coeffs[i]
            _, h = freqz(b, a, worN=w)
            _, gd = group_delay((b, a), w=w)
            loop = absn.delay_lengths[i] + absn.frame_size + gd
            att_db = -20.0 * np.log10(np.abs(h))
            out[i] = 60.0 * loop / (pw.sample_rate * att_db)
        return out.mean(axis=0) * 1e3

    def _make(self, air, t60_ms=None):
        pw = PlaneWaveDFDN(
            sample_rate=48000,
            num_delay_line_groups=4,
            sh_order=1,
            frame_size=64,
            seed=42,
        )
        pw.setup(
            delay_range_ms=(10, 30),
            t60_per_direction_ms=self.t60_ms if t60_ms is None else t60_ms,
            freqs=self.freqs,
            filter_order=6,
            air_absorption=air,
        )
        return pw

    def test_air_absorption_preserves_realised_t60(self):
        """Toggling air absorption must not shift the realised T60.

        Compared against air=False rather than the target so that the
        (separate) filter-fitting error cancels out.
        """
        off = self._realised_t60_ms(self._make(False), self.freqs)
        on = self._realised_t60_ms(self._make(True), self.freqs)
        np.testing.assert_allclose(on, off, rtol=0.01)

    def test_infeasible_target_warns(self):
        """A T60 longer than air absorption alone permits must warn."""
        t60 = np.array([1000.0, 1100.0, 1000.0, 5000.0])
        with pytest.warns(RuntimeWarning, match="air absorption"):
            self._make(True, t60_ms=t60)
