"""Regression tests for the pyFDN migration."""

from __future__ import annotations

import numpy as np
import pytest

from src.fdn.delay_line import generate_delay_lengths


class TestDelayLengths:
    def test_count_and_range(self):
        lengths = generate_delay_lengths((20.0, 60.0), 16, 48000.0, rng=0)
        assert len(lengths) == 16
        assert lengths.dtype == np.int64
        assert np.all(lengths >= int(0.020 * 48000))
        assert np.all(lengths <= int(0.060 * 48000))

    def test_distinct(self):
        lengths = generate_delay_lengths((20.0, 60.0), 16, 48000.0, rng=0)
        assert len(np.unique(lengths)) == 16

    def test_coprime_when_feasible(self):
        from math import gcd

        lengths = generate_delay_lengths((20.0, 60.0), 16, 48000.0, rng=0)
        for i in range(len(lengths)):
            for j in range(i + 1, len(lengths)):
                assert gcd(int(lengths[i]), int(lengths[j])) == 1

    def test_reproducible(self):
        a = generate_delay_lengths((20.0, 60.0), 16, 48000.0, rng=7)
        b = generate_delay_lengths((20.0, 60.0), 16, 48000.0, rng=7)
        np.testing.assert_array_equal(a, b)

    def test_high_channel_count_stays_in_range(self):
        """NQ=336 makes pairwise coprimality infeasible; must fall back."""
        lengths = generate_delay_lengths((20.0, 60.0), 336, 48000.0, rng=0)
        assert len(lengths) == 336
        assert len(np.unique(lengths)) == 336
        assert np.all(lengths >= int(0.020 * 48000))
        assert np.all(lengths <= int(0.060 * 48000))


class TestAbsorptionDesign:
    FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    T60_S = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45])
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def test_grid_has_ten_points(self):
        from src.fdn.absorption_design import t60_on_geq_grid

        grid = t60_on_geq_grid(self.T60_S, self.FREQS, 48000.0)
        assert grid.shape == (10,)

    def test_grid_clamps_at_edges(self):
        from src.fdn.absorption_design import t60_on_geq_grid

        grid = t60_on_geq_grid(self.T60_S, self.FREQS, 48000.0)
        # DC point (1 Hz) clamps to the lowest supplied band, Nyquist to the highest
        assert grid[0] == pytest.approx(self.T60_S[0])
        assert grid[-1] == pytest.approx(self.T60_S[-1])

    def test_shape_and_layout(self):
        from src.fdn.absorption_design import design_absorption_sos

        sos = design_absorption_sos(self.T60_S, self.FREQS, self.DELAYS, 48000.0)
        assert sos.shape == (11, 6, 4)
        np.testing.assert_allclose(sos[:, 3, :], 1.0)  # a0 normalised

    def test_matches_pyfdn_absorption_geq_for_shared_rt(self):
        """With one shared RT curve this must equal pyFDN.absorption_geq."""
        import pyFDN

        from src.fdn.absorption_design import design_absorption_sos, t60_on_geq_grid

        rt10 = t60_on_geq_grid(self.T60_S, self.FREQS, 48000.0)
        expected = pyFDN.absorption_geq(rt10, self.DELAYS, 48000.0)
        actual = design_absorption_sos(self.T60_S, self.FREQS, self.DELAYS, 48000.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)

    def test_per_channel_rt_gives_different_filters(self):
        from src.fdn.absorption_design import design_absorption_sos

        t60 = np.vstack([self.T60_S, self.T60_S * 0.5, self.T60_S, self.T60_S * 2.0])
        sos = design_absorption_sos(t60, self.FREQS, self.DELAYS, 48000.0)
        assert sos.shape == (11, 6, 4)
        assert not np.allclose(sos[:, :, 0], sos[:, :, 1])

    def test_rejects_wrong_channel_count(self):
        from src.fdn.absorption_design import design_absorption_sos

        t60 = np.vstack([self.T60_S, self.T60_S])  # 2 rows for 4 delays
        with pytest.raises(ValueError, match="one T60 curve per delay line"):
            design_absorption_sos(t60, self.FREQS, self.DELAYS, 48000.0)


class TestAbsorptionGEQ:
    FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    T60_MS = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def _make(self):
        from src.fdn.absorption import AbsorptionGEQ

        return AbsorptionGEQ(4, self.DELAYS, 48000.0, 64, self.T60_MS, self.FREQS)

    def test_shape_preserved(self):
        absorber = self._make()
        x = np.ones((4, 64))
        y = absorber.process(x)
        assert y.shape == x.shape

    def test_attenuates(self):
        absorber = self._make()
        x = np.ones((4, 512))
        y = absorber.process(x)
        assert np.sum(y**2) < np.sum(x**2)

    def test_state_persists_across_blocks(self):
        """Two 64-sample blocks must equal one 128-sample block."""
        a = self._make()
        b = self._make()
        x = np.random.default_rng(0).standard_normal((4, 128))
        one_shot = b.process(x)
        chunked = np.concatenate([a.process(x[:, :64]), a.process(x[:, 64:])], axis=1)
        np.testing.assert_allclose(chunked, one_shot, atol=1e-12)

    def test_reset_clears_state(self):
        absorber = self._make()
        x = np.random.default_rng(0).standard_normal((4, 128))
        first = absorber.process(x)
        absorber.reset()
        second = absorber.process(x)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_factory_returns_geq_by_default(self):
        from src.fdn.absorption import AbsorptionGEQ, create_absorption

        absorber = create_absorption(
            4, self.DELAYS, 48000.0, 64, self.T60_MS, freqs=self.FREQS
        )
        assert isinstance(absorber, AbsorptionGEQ)

    def test_factory_scalar_still_returns_gains(self):
        from src.fdn.absorption import AbsorptionGains, create_absorption

        absorber = create_absorption(4, self.DELAYS, 48000.0, 64, 500.0)
        assert isinstance(absorber, AbsorptionGains)


class TestRenderAdapter:
    def test_orientation_single_output(self):
        from src.fdn.render import ScalarGains, render_fdn

        delays = np.array([1039, 1213, 1381, 1523])
        A = np.linalg.qr(np.random.default_rng(0).standard_normal((4, 4)))[0]
        b = np.ones((4, 1)) / 2.0
        c = np.ones((1, 4)) / 2.0
        x = np.zeros((1, 4096))
        x[0, 0] = 1.0
        y = render_fdn(x, delays, A, b, c, np.zeros((1, 1)),
                       absorption=ScalarGains(np.full(4, 0.99)))
        assert y.shape == (1, 4096)

    def test_orientation_multi_output(self):
        from src.fdn.render import ScalarGains, render_fdn

        delays = np.array([1039, 1213, 1381, 1523])
        A = np.eye(4)
        b = np.ones((4, 1)) / 2.0
        c = np.eye(4)
        x = np.zeros((1, 4096))
        x[0, 0] = 1.0
        y = render_fdn(x, delays, A, b, c, np.zeros((4, 1)),
                       absorption=ScalarGains(np.full(4, 0.99)))
        assert y.shape == (4, 4096)

    def test_direct_path_passes_through(self):
        from src.fdn.render import ScalarGains, render_fdn

        delays = np.array([1039, 1213, 1381, 1523])
        x = np.zeros((1, 2048))
        x[0, 0] = 1.0
        y = render_fdn(x, delays, np.eye(4), np.zeros((4, 1)), np.zeros((1, 4)),
                       np.array([[0.5]]), absorption=ScalarGains(np.full(4, 0.99)))
        assert y[0, 0] == pytest.approx(0.5)
        assert np.allclose(y[0, 1:], 0.0)

    def test_round_trip_is_exactly_m_samples(self, measured_t60):
        """The defining property: no frame_size compensation needed."""
        from src.fdn.render import ScalarGains, render_fdn

        fs = 48000.0
        target_t60 = 1.5
        delays = np.array([1039, 1213, 1381, 1523, 1699, 1877, 2039, 2213])
        gains = 10.0 ** (-3.0 * delays / (fs * target_t60))
        A = np.linalg.qr(np.random.default_rng(0).standard_normal((8, 8)))[0]
        x = np.zeros((1, int(4 * fs)))
        x[0, 0] = 1.0
        y = render_fdn(x, delays, A, np.ones((8, 1)) / np.sqrt(8),
                       np.ones((1, 8)) / np.sqrt(8), np.zeros((1, 1)),
                       absorption=ScalarGains(gains))

        rt = measured_t60(y[0], fs)
        # 62 Hz band is noisy on a 4 s IR; check 125 Hz upward.
        np.testing.assert_allclose(rt[1:], target_t60, rtol=0.03)

    def test_impulse_arrives_at_the_delay_length(self):
        from src.fdn.render import ScalarGains, render_fdn

        delays = np.array([1039, 1213])
        b = np.array([[1.0], [0.0]])
        c = np.array([[1.0, 0.0]])
        x = np.zeros((1, 4096))
        x[0, 0] = 1.0
        y = render_fdn(x, delays, np.zeros((2, 2)), b, c, np.zeros((1, 1)),
                       absorption=ScalarGains(np.ones(2)))
        assert np.argmax(np.abs(y[0])) == 1039

    def test_single_sample_render_keeps_both_axes(self):
        """process_fdn squeezes every singleton axis; the adapter must not guess."""
        from src.fdn.render import ScalarGains, render_fdn

        delays = np.array([5, 7, 11])
        x = np.ones((1, 1))
        gains = ScalarGains(np.ones(3))

        multi = render_fdn(x, delays, np.eye(3), np.ones((3, 1)), np.eye(3),
                           np.zeros((3, 1)), absorption=gains)
        assert multi.shape == (3, 1)

        single = render_fdn(x, delays, np.eye(3), np.ones((3, 1)),
                            np.ones((1, 3)), np.zeros((1, 1)), absorption=gains)
        assert single.shape == (1, 1)


class TestFDNCorrectedLoop:
    FS = 48000.0
    DELAYS = np.array([1039, 1213, 1381, 1523, 1699, 1877, 2039, 2213])

    def _make(self, t60_ms, freqs=None):
        from src.fdn.fdn import FDN
        from src.fdn.feedback_matrix import FeedbackMatrixType

        fdn = FDN(self.FS, 1, 1, 8, 64, seed=42)
        fdn.init_delay_line_lengths(self.DELAYS)
        fdn.init_feedback_matrix(FeedbackMatrixType.HADAMARD)
        fdn.init_absorption(t60_ms, freqs=freqs)
        fdn.init_delay_lines()
        fdn.init_io_gains(np.ones((8, 1)) / np.sqrt(8), np.ones((1, 8)) / np.sqrt(8))
        fdn.init_direct_gain(np.zeros((1, 1)))
        return fdn

    def _impulse(self, seconds=4.0):
        x = np.zeros((1, int(seconds * self.FS)))
        x[0, 0] = 1.0
        return x

    def test_broadband_t60_is_accurate(self, measured_t60):
        fdn = self._make(1500.0)
        rt = measured_t60(fdn.process(self._impulse())[0], self.FS)
        np.testing.assert_allclose(rt[1:], 1.5, rtol=0.05)

    def test_frame_size_does_not_change_t60(self, measured_t60):
        """The old loop's T60 depended on frame_size; the new one must not."""
        from src.fdn.fdn import FDN
        from src.fdn.feedback_matrix import FeedbackMatrixType

        results = []
        for frame in (32, 256):
            fdn = FDN(self.FS, 1, 1, 8, frame, seed=42)
            fdn.init_delay_line_lengths(self.DELAYS)
            fdn.init_feedback_matrix(FeedbackMatrixType.HADAMARD)
            fdn.init_absorption(1500.0)
            fdn.init_delay_lines()
            fdn.init_io_gains(
                np.ones((8, 1)) / np.sqrt(8), np.ones((1, 8)) / np.sqrt(8)
            )
            fdn.init_direct_gain(np.zeros((1, 1)))
            results.append(measured_t60(fdn.process(self._impulse())[0], self.FS))
        np.testing.assert_allclose(results[0][1:], results[1][1:], rtol=1e-6)

    def test_frequency_dependent_t60_is_accurate(self, measured_t60):
        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
        t60_ms = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3

        fdn = self._make(t60_ms, freqs=freqs)
        rt = measured_t60(fdn.process(self._impulse())[0], self.FS)
        target = t60_ms * 1e-3
        # 62 Hz and 8 kHz bands sit at the edge of the design grid; check the middle.
        np.testing.assert_allclose(rt[1:6], target[1:6], rtol=0.10)

    def test_silence_in_gives_silence_out(self):
        """Zero input produces zero output; reset() is incidental here."""
        fdn = self._make(1500.0)
        fdn.process(self._impulse(0.5))
        fdn.reset()
        out = fdn.process(np.zeros((1, 4096)))
        assert np.max(np.abs(out)) < 1e-12

    def test_reset_clears_absorption_filter_state(self):
        """reset() must clear the SOS memory that persists between process() calls."""
        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
        t60_ms = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
        fdn = self._make(t60_ms, freqs=freqs)
        x = self._impulse(0.2)

        first = fdn.process(x)
        without_reset = fdn.process(x)
        fdn.reset()
        after_reset = fdn.process(x)

        # Absorption filter memory carries into the second call...
        assert not np.allclose(first, without_reset, atol=1e-12)
        # ...and reset() puts it back to the initial state.
        np.testing.assert_allclose(first, after_reset, atol=1e-12)


class TestDirectionalFDNCorrectedLoop:
    FS = 48000.0

    def _make(self, **kwargs):
        from src.fdn.directional_fdn import DirectionalFDN

        dfdn = DirectionalFDN(self.FS, 8, ambisonics_order=1, frame_size=64, seed=42,
                              **kwargs)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
        return dfdn

    def test_effective_matrix_is_kronecker_when_plain(self):
        dfdn = self._make()
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), dfdn._kronecker_matrix, atol=1e-12
        )

    def test_effective_matrix_folds_transform(self):
        from scipy.linalg import block_diag

        from src.fdn.spatial import uniform_directions

        dfdn = self._make()
        az, el = uniform_directions(24)
        t60 = np.full(24, 1500.0)
        t60[az < np.pi] = 900.0
        dfdn.set_directional_decay(t60, az, el)

        expected = dfdn._kronecker_matrix @ block_diag(
            *[T @ S for T, S in zip(dfdn.transform.transform_matrices,
                                    dfdn._scattering_matrices)]
        )
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), expected, atol=1e-12
        )

    def test_effective_matrix_folds_scattering(self):
        from scipy.linalg import block_diag

        dfdn = self._make(scattering_amount=0.4)
        expected = dfdn._kronecker_matrix @ block_diag(*dfdn._scattering_matrices)
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), expected, atol=1e-12
        )

    def test_rir_shape_and_decay(self):
        dfdn = self._make()
        rir = dfdn.render(0.0, 0.0, length_seconds=2.0)
        assert rir.shape[0] == 4
        first = np.sum(rir[0, : rir.shape[1] // 4] ** 2)
        last = np.sum(rir[0, -rir.shape[1] // 4 :] ** 2)
        assert first > last

    def test_broadband_t60_is_accurate(self, measured_t60):
        dfdn = self._make()
        rir = dfdn.render(0.0, 0.0, length_seconds=4.0)
        np.testing.assert_allclose(measured_t60(rir[0], self.FS)[1:], 1.5, rtol=0.08)

    def test_source_direction_still_changes_the_rir(self):
        front = self._make().render(0.0, 0.0, length_seconds=1.0)
        left = self._make().render(np.pi / 2, 0.0, length_seconds=1.0)
        assert not np.allclose(front, left, atol=1e-10)


class TestPlaneWaveDFDNCorrectedLoop:
    FS = 48000.0

    def _make(self, **setup_kwargs):
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, 8, sh_order=1, frame_size=64, seed=42)
        kwargs = dict(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        kwargs.update(setup_kwargs)
        pw.setup(**kwargs)
        return pw

    def test_rir_shape(self):
        pw = self._make()
        rir = pw.render(0.0, 0.0, length_seconds=1.0)
        assert rir.shape[0] == 4
        assert rir.shape[1] == int(1.0 * self.FS)

    def test_broadband_t60_is_accurate(self, measured_t60):
        pw = self._make()
        rir = pw.render(0.0, 0.0, length_seconds=4.0)
        np.testing.assert_allclose(measured_t60(rir[0], self.FS)[1:], 1.5, rtol=0.08)

    def test_direction_dependent_t60(self, measured_t60):
        """Per-direction T60 must reach the filters and shape the decay."""
        grid = self._make()
        mixed_t60 = np.where(grid.grid_azimuths < np.pi, 2000.0, 600.0)

        long_rir = self._make(t60_per_direction_ms=2000.0).render(
            0.0, 0.0, length_seconds=4.0)
        short_rir = self._make(t60_per_direction_ms=600.0).render(
            0.0, 0.0, length_seconds=4.0)
        mixed_rir = self._make(t60_per_direction_ms=mixed_t60).render(
            0.0, 0.0, length_seconds=4.0)

        # A uniformly short room decays faster than a uniformly long one.
        assert measured_t60(short_rir[0], self.FS)[3] < measured_t60(
            long_rir[0], self.FS)[3]
        # A mixed field is not collapsed to either uniform case...
        assert not np.allclose(mixed_rir, long_rir, atol=1e-12)
        assert not np.allclose(mixed_rir, short_rir, atol=1e-12)
        # ...and its total energy lands strictly between them.
        e_short = np.sum(short_rir**2)
        e_mixed = np.sum(mixed_rir**2)
        e_long = np.sum(long_rir**2)
        assert e_short < e_mixed < e_long

    def test_direct_path_amplitude(self):
        pw = self._make()
        rir = pw.render(0.0, 0.0, length_seconds=0.5, direct_gain=0.5,
                        reverb_gain=0.0)
        # With no reverb feed the first sample is the direct path only.
        expected = 0.5 / np.sqrt(4.0 * np.pi)
        assert rir[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_early_reflections_arrive_on_time(self):
        from src.fdn.planewave_dfdn import Reflection

        pw = self._make()
        pw.set_early_reflections(
            [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5, direct_gain=0.0,
                        reverb_gain=0.0)
        assert np.abs(rir[0, 200]) > 1e-3

    def test_early_reflections_add_energy(self):
        from src.fdn.planewave_dfdn import Reflection

        without = self._make().render(0.0, 0.0, length_seconds=1.0)
        pw = self._make()
        pw.set_early_reflections(
            [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
        )
        with_er = pw.render(0.0, 0.0, length_seconds=1.0)
        assert np.sum(with_er**2) > np.sum(without**2)


class TestAirAbsorptionSplit:
    """The medium/boundary attribution from commit 0554ddd must survive."""

    FS = 48000.0
    FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def test_shares_sum_to_required_attenuation(self):
        from src.fdn.planewave_dfdn import air_absorption_split

        t60 = np.tile(np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]), (4, 1))
        air, boundary = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        required = 60.0 * self.DELAYS[:, None] / (self.FS * t60)
        np.testing.assert_allclose(air + boundary, required, rtol=1e-12)

    def test_air_share_grows_with_frequency(self):
        from src.fdn.planewave_dfdn import air_absorption_split

        t60 = np.tile(np.full(8, 1.5), (4, 1))
        air, _ = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        assert np.all(np.diff(air[0]) > 0.0)

    def test_warns_when_target_exceeds_air_budget(self):
        from src.fdn.planewave_dfdn import air_absorption_split

        # A very long T60 at high frequency leaves no boundary share.
        t60 = np.tile(np.full(8, 60.0), (4, 1))
        with pytest.warns(RuntimeWarning, match="air absorption alone permits"):
            air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)

    def test_realised_t60_is_unaffected(self):
        """Attribution must not change the filters, hence not the decay."""
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        t60 = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
        rirs = []
        for air in (False, True):
            pw = PlaneWaveDFDN(self.FS, 8, sh_order=1, frame_size=64, seed=42)
            pw.setup(
                delay_range_ms=(20.0, 60.0),
                t60_per_direction_ms=t60,
                freqs=self.FREQS,
                air_absorption=air,
            )
            rirs.append(pw.render(0.0, 0.0, length_seconds=1.0))
        np.testing.assert_allclose(rirs[0], rirs[1], atol=1e-12)

    def test_attributes_exposed_on_the_class(self):
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        t60 = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
        pw = PlaneWaveDFDN(self.FS, 8, sh_order=1, frame_size=64, seed=42)
        pw.setup(
            delay_range_ms=(20.0, 60.0),
            t60_per_direction_ms=t60,
            freqs=self.FREQS,
            air_absorption=True,
        )
        assert pw.air_attenuation_db.shape == (pw.total_delay_lines, 8)
        assert pw.boundary_attenuation_db.shape == (pw.total_delay_lines, 8)


class TestAnalysisHelpers:
    FS = 48000.0

    def _synthetic_ir(self, t60=1.2, seconds=3.0):
        n = int(seconds * self.FS)
        t = np.arange(n) / self.FS
        rng = np.random.default_rng(0)
        return rng.standard_normal(n) * 10.0 ** (-3.0 * t / t60)

    def test_estimate_t60_bands(self):
        from src.utils import estimate_t60_bands

        rt, fc = estimate_t60_bands(self._synthetic_ir(1.2), self.FS)
        assert rt.shape == fc.shape
        np.testing.assert_allclose(rt[1:-1], 1.2, rtol=0.10)

    def test_mixing_time_of_diffuse_noise_is_finite(self):
        from src.utils import mixing_time_ms

        t_mix = mixing_time_ms(self._synthetic_ir(), self.FS)
        assert np.isfinite(t_mix)
        assert t_mix >= 0.0

    def test_sparse_response_never_reaches_the_threshold(self):
        """A pulse train is not diffuse; the wrapper must say so, not say 0."""
        from src.utils import mixing_time_ms

        dense = self._synthetic_ir()
        sparse = np.zeros_like(dense)
        sparse[::400] = dense[::400] * 20.0
        assert np.isnan(mixing_time_ms(sparse, self.FS))
