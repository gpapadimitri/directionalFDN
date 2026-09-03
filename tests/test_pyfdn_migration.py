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

    def test_repeat_render_is_deterministic(self):
        """A render is a pure function of setup and input; no state carries over."""
        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
        t60_ms = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
        fdn = self._make(t60_ms, freqs=freqs)
        x = self._impulse(0.2)
        np.testing.assert_allclose(fdn.process(x), fdn.process(x), atol=1e-12)


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
                                    dfdn._scattering_matrices, strict=True)]
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

    def test_effective_matrix_folds_transform_after_scattering(self):
        """T_g @ S_g, in that order; both non-identity so the order is pinned."""
        from scipy.linalg import block_diag

        from src.fdn.spatial import uniform_directions

        dfdn = self._make(scattering_amount=0.4)
        az, el = uniform_directions(24)
        t60 = np.full(24, 1500.0)
        t60[az < np.pi] = 900.0
        dfdn.set_directional_decay(t60, az, el)

        blocks = [
            T @ S
            for T, S in zip(
                dfdn.transform.transform_matrices, dfdn._scattering_matrices,
                strict=True,
            )
        ]
        expected = dfdn._kronecker_matrix @ block_diag(*blocks)
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), expected, atol=1e-12
        )
        # The reversed order must actually differ, or this test proves nothing.
        reversed_blocks = [
            S @ T
            for T, S in zip(
                dfdn.transform.transform_matrices, dfdn._scattering_matrices,
                strict=True,
            )
        ]
        assert not np.allclose(
            block_diag(*blocks), block_diag(*reversed_blocks), atol=1e-12
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

    def test_repeat_render_is_deterministic(self):
        dfdn = self._make()
        first = dfdn.render(0.0, 0.0, length_seconds=0.5)
        second = dfdn.render(0.0, 0.0, length_seconds=0.5)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_render_length_is_not_truncated(self):
        """render_fdn processes every sample; no frame_size rounding."""
        rir = self._make().render(0.0, 0.0, length_seconds=0.501)
        assert rir.shape[1] == int(0.501 * self.FS)


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

        odd = pw.render(0.0, 0.0, length_seconds=0.501)
        assert odd.shape[1] == int(0.501 * self.FS)

    def test_broadband_t60_accepts_freqs(self):
        """A scalar or per-direction broadband T60 must not break when freqs is given."""
        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
        scalar = self._make(t60_per_direction_ms=1500.0, freqs=freqs)
        assert scalar.absorption_sos is None

        grid = self._make()
        per_dir = self._make(
            t60_per_direction_ms=np.full(grid.Q, 1500.0), freqs=freqs
        )
        assert per_dir.absorption_sos is None
        assert per_dir.render(0.0, 0.0, length_seconds=0.5).shape[0] == 4

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

    def test_repeat_render_is_deterministic(self):
        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
        t60_ms = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3
        pw = self._make(t60_per_direction_ms=t60_ms, freqs=freqs)
        first = pw.render(0.0, 0.0, length_seconds=0.5)
        second = pw.render(0.0, 0.0, length_seconds=0.5)
        np.testing.assert_allclose(first, second, atol=1e-12)


class TestAirAbsorption:
    """Air absorption enters through the T60 target, the standardized route.

    No FDN implementation in this lineage models a propagation medium:
    fdnToolbox's absorptionGEQ/onePoleAbsorption/firstOrderAbsorption,
    artificial-audio/pyFDN's equivalents, and the DAFx24 tutorial's own
    absorption_filter all take reverberation time and delay lengths and
    nothing else.  The canonical place for the medium is therefore the target
    itself -- the 4mV term of Sabine/Eyring,
    ``T60 = 0.161 V / (S alpha + 4 m V)``, which is algebraically the same as
    summing decay rates, ``1/T = 1/T_boundary + 1/T_air``.
    """

    FS = 48000.0
    FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    DELAYS = np.array([1039, 1213, 1381, 1523])
    T60 = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45])

    def test_air_only_t60_falls_with_frequency(self):
        from src.fdn.planewave_dfdn import air_absorption_t60

        t_air = air_absorption_t60(self.FREQS)
        assert np.all(np.diff(t_air) < 0.0)
        # 16 kHz at 20 C / 50% RH is around half a second in air alone.
        assert 0.3 < air_absorption_t60(np.array([16000.0]))[0] < 0.7

    def test_air_limited_target_matches_the_sabine_4mv_term(self):
        """The rate sum must equal the textbook air term, not merely resemble it."""
        from src.fdn.planewave_dfdn import (
            air_absorption_db_per_meter,
            air_limited_t60,
        )

        volume, surface, alpha_boundary = 320.0, 304.0, 0.1857
        sabine_dry = 0.161 * volume / (surface * alpha_boundary)
        m = air_absorption_db_per_meter(self.FREQS) / (10 * np.log10(np.e))
        sabine_with_air = 0.161 * volume / (surface * alpha_boundary + 4 * m * volume)

        combined = air_limited_t60(np.full(self.FREQS.size, sabine_dry), self.FREQS)
        np.testing.assert_allclose(combined, sabine_with_air, rtol=1e-3)

    def test_air_can_only_shorten(self):
        from src.fdn.planewave_dfdn import air_limited_t60

        limited = air_limited_t60(self.T60, self.FREQS)
        assert np.all(limited <= self.T60)
        assert limited[-1] < self.T60[-1]          # 8 kHz is measurably shorter
        assert limited[0] == pytest.approx(self.T60[0], rel=5e-3)   # 63 Hz barely

    def test_shares_sum_to_the_required_attenuation(self):
        """air_db and boundary_db must account for the realised round trip."""
        from src.fdn.planewave_dfdn import air_absorption_split, air_limited_t60

        t60 = np.tile(self.T60, (4, 1))
        air, boundary = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        realised = air_limited_t60(t60, self.FREQS)
        required = 60.0 * self.DELAYS[:, None] / (self.FS * realised)
        np.testing.assert_allclose(air + boundary, required, rtol=1e-10)

    def test_both_shares_are_positive(self):
        """The old model could produce a negative boundary share; this cannot."""
        from src.fdn.planewave_dfdn import air_absorption_split

        t60 = np.tile(np.full(8, 60.0), (4, 1))    # absurdly long target
        air, boundary = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        assert np.all(air > 0.0)
        assert np.all(boundary > 0.0)

    def test_enabling_air_shortens_the_decay(self):
        """The point of the change: the flag must now affect the sound."""
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        rt = {}
        for air in (False, True):
            pw = PlaneWaveDFDN(self.FS, 8, sh_order=1, frame_size=64, seed=42)
            pw.setup(
                delay_range_ms=(20.0, 60.0),
                t60_per_direction_ms=self.T60 * 1e3,
                freqs=self.FREQS,
                air_absorption=air,
            )
            w = pw.render(0.0, 0.0, length_seconds=4.0,
                          direct_gain=0.0, er_gain=0.0)[0]
            import pyFDN

            rt[air], fc = pyFDN.estimate_rt_bands(w, self.FS, n=8, start=-4)
        top = int(np.argmin(np.abs(fc - 8000.0)))
        assert rt[True][top] < rt[False][top]      # 8 kHz decays faster
        low = int(np.argmin(np.abs(fc - 125.0)))
        assert rt[True][low] == pytest.approx(rt[False][low], rel=0.05)

    def test_attributes_exposed_on_the_class(self):
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, 8, sh_order=1, frame_size=64, seed=42)
        pw.setup(
            delay_range_ms=(20.0, 60.0),
            t60_per_direction_ms=self.T60 * 1e3,
            freqs=self.FREQS,
            air_absorption=True,
        )
        assert pw.air_attenuation_db.shape == (pw.total_delay_lines, 8)
        assert pw.boundary_attenuation_db.shape == (pw.total_delay_lines, 8)
        assert pw.realised_t60_s.shape == (pw.total_delay_lines, 8)
        assert np.all(pw.realised_t60_s <= self.T60[None, :] + 1e-12)


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


class TestFeedbackMatrixFromPyFDN:
    """The generators are pyFDN's, with reproducibility preserved."""

    def test_hadamard_is_unchanged(self):
        """The gallery's Hadamard must be bit-identical to the old one."""
        import pyFDN

        from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, FeedbackMatrixType.HADAMARD, 64)
        np.testing.assert_allclose(
            fb.matrix, pyFDN.fdn_matrix_gallery(8, "Hadamard"), atol=1e-15
        )

    def test_householder_stays_the_canonical_all_ones(self):
        """fdn_matrix_gallery randomises Householder; we want the all-ones one."""
        import pyFDN

        from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, FeedbackMatrixType.HOUSEHOLDER, 64)
        np.testing.assert_allclose(
            fb.matrix, pyFDN.householder_matrix(np.ones(8)), atol=1e-15
        )

    def test_random_is_reproducible(self):
        from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType

        a = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, 64, seed=42).matrix
        b = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, 64, seed=42).matrix
        c = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, 64, seed=7).matrix
        np.testing.assert_allclose(a, b, atol=1e-15)
        assert not np.allclose(a, c)

    def test_random_does_not_disturb_the_global_stream(self):
        """pyFDN's gallery draws from global np.random; we must restore it."""
        from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType

        np.random.seed(0)
        expected = np.random.rand()
        np.random.seed(0)
        FeedbackMatrix(8, FeedbackMatrixType.RANDOM, 64, seed=42)
        assert np.random.rand() == expected

    @pytest.mark.parametrize(
        "kind",
        [
            "RANDOM",
            "HADAMARD",
            "HOUSEHOLDER",
            "CIRCULANT",
            "PERMUTATION",
            "DIAGONALLY_SIMILAR",
            "ANDERSON",
            "TINY_ROTATION",
        ],
    )
    def test_every_type_builds_and_is_lossless(self, kind):
        import pyFDN

        from src.fdn.feedback_matrix import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, getattr(FeedbackMatrixType, kind), 64, seed=42)
        assert fb.matrix.shape == (8, 8)
        assert pyFDN.is_unilossless(fb.matrix)

    def test_unknown_type_still_raises(self):
        from src.fdn.feedback_matrix import FeedbackMatrix

        with pytest.raises(ValueError, match="Unsupported feedback matrix type"):
            FeedbackMatrix(8, "not-a-type", 64)

    def test_matrix_convolve_is_gone(self):
        from src.fdn.feedback_matrix import FeedbackMatrix

        assert not hasattr(FeedbackMatrix, "matrix_convolve")


class TestDelayLengthSpread:
    """Global spread across all N*Q lines, not per-sub-network coprimality.

    A ``kron`` feedback matrix never mixes across the channel index, so it is
    tempting to draw the N lengths of each channel as its own coprime FDN.
    Measured, that is much worse: on an N=8, Q=86 grid it takes every one of
    the 86 sub-networks from non-coprime to coprime but moves the composite
    mixing time from 4.9 ms to 57 ms.  The audible output is the sum of all Q
    sub-networks through the encoder, so what makes it dense is that all N*Q
    lengths are distinct and spread -- the classic criterion applies to the
    whole pool here, not to each sub-network.
    """

    FS = 48000.0

    def _make(self, sh_order=3, groups=8):
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, groups, sh_order=sh_order, frame_size=64, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        return pw

    def test_all_lengths_are_distinct(self):
        pw = self._make()
        lengths = pw.delay_length_samps
        assert len(np.unique(lengths)) == len(lengths)

    def test_lengths_stay_in_range(self):
        pw = self._make()
        lo, hi = int(0.020 * self.FS), int(0.060 * self.FS)
        assert np.all(pw.delay_length_samps >= lo)
        assert np.all(pw.delay_length_samps <= hi)

    def test_tail_mixes_soon_after_the_shortest_delay(self):
        """The property the spread actually buys: an early mixing time.

        Nothing can mix before the shortest delay line returns, so the bound is
        relative to that rather than absolute.  Drawing per sub-network instead
        pushes this well past 2x.
        """
        import pyFDN

        pw = self._make()
        w = pw.render(0.0, 0.0, 2.0, direct_gain=0.0, er_gain=0.0, reverb_gain=1.0)[0]
        t_mix, _ = pyFDN.echo_density(w, fs=self.FS)
        shortest_ms = pw.delay_length_samps.min() / self.FS * 1000.0
        assert 0.0 < t_mix < 2.0 * shortest_ms

    def test_reproducible(self):
        np.testing.assert_array_equal(
            self._make().delay_length_samps, self._make().delay_length_samps
        )


class TestLosslessnessChecks:
    """Both directional classes verify the feedback matrix with pyFDN."""

    FS = 48000.0

    def test_planewave_kronecker_is_lossless(self):
        import pyFDN

        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, 4, sh_order=1, frame_size=64, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        assert pyFDN.is_unilossless(pw._kronecker_matrix)

    def test_directional_kronecker_is_lossless(self):
        import pyFDN

        from src.fdn.directional_fdn import DirectionalFDN

        dfdn = DirectionalFDN(self.FS, 4, ambisonics_order=1, frame_size=64, seed=42)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
        assert pyFDN.is_unilossless(dfdn._kronecker_matrix)

    @pytest.mark.xfail(
        reason="pre-existing: lateral_mix's matrix-log interpolation "
        "real(expm(t*logm(H_Q))) is orthogonal only when H_Q has no -1 "
        "eigenvalues. Forcing det(H_Q)=+1 fixes odd Q (14, 42) but not "
        "Hadamard sizes, where H is a symmetric involution with eigenvalues "
        "{1,1,-1,-1} and no real principal log. Needs a real-Schur angle-scaling "
        "construction; pyFDN.interpolate_orthogonal has the same limitation.",
        raises=RuntimeError,
        strict=True,
    )
    def test_lateral_mix_stays_lossless(self):
        import pyFDN

        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, 4, sh_order=1, frame_size=64, seed=42)
        pw.setup(
            delay_range_ms=(20.0, 60.0),
            t60_per_direction_ms=1500.0,
            lateral_mix=0.5,
        )
        assert pyFDN.is_unilossless(pw._kronecker_matrix)


class TestPoleBounds:
    """SOS-native reimplementation of pyFDN.pole_boundaries.

    pyFDN's own version needs a direct-form (b, a) per channel; flattening an
    11-section cascade with sos2tf is numerically indefensible (measured max
    |H| 5.69 against the cascade's true 0.92, plus a spurious pole at 1.13).
    This works from the cascade directly.
    """

    FS = 48000.0
    FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def _sos(self, t60_s):
        from src.fdn.absorption_design import design_absorption_sos

        return design_absorption_sos(t60_s, self.FREQS, self.DELAYS, self.FS)

    def _A(self):
        return np.linalg.qr(np.random.default_rng(0).standard_normal((4, 4)))[0]

    def test_shapes_and_frequency_axis(self):
        from src.utils import pole_bounds

        lo, hi, f = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 1.2)), self._A(), self.FS, nfft=512
        )
        assert lo.shape == hi.shape == f.shape == (512,)
        assert f[0] == pytest.approx(0.0)
        assert f[-1] == pytest.approx(self.FS / 2, rel=1e-3)

    def test_lossy_fdn_stays_below_unity(self):
        """An orthogonal A has all singular values 1, so the bound is the
        absorption -- which must be lossy at every frequency."""
        from src.utils import pole_bounds

        _, hi, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 1.2)), self._A(), self.FS
        )
        assert np.all(hi < 1.0)

    def test_shorter_t60_lowers_the_bound(self):
        from src.utils import pole_bounds

        _, hi_long, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 2.0)), self._A(), self.FS
        )
        _, hi_short, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 0.5)), self._A(), self.FS
        )
        assert np.all(hi_short < hi_long)

    def test_min_curve_does_not_exceed_max(self):
        from src.utils import pole_bounds

        lo, hi, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 1.2)), self._A(), self.FS
        )
        assert np.all(lo <= hi + 1e-12)

    def test_accepts_a_2d_feedback_matrix(self):
        from src.utils import pole_bounds

        lo, hi, f = pole_bounds(
            self.DELAYS, self._sos(np.full(8, 1.2)), np.eye(4), self.FS, nfft=256
        )
        assert lo.shape == (256,)


class TestHelpersComeFromPyFDN:
    def test_db_helpers_are_pyfdn(self):
        import pyFDN

        from src import utils

        assert utils.lin2db is pyFDN.lin_to_db
        assert utils.db2lin is pyFDN.db_to_lin
        assert utils.ms_to_samps is pyFDN.ms_to_smp

    def test_db_roundtrip_still_holds(self):
        from src.utils import db2lin, lin2db

        for value in (0.0, -6.0, -20.0, 6.0):
            assert abs(float(lin2db(db2lin(value))) - value) < 1e-10

    def test_ms_conversion(self):
        from src.utils import ms_to_samps

        assert ms_to_samps(1000.0, 48000.0) == 48000
        assert ms_to_samps(0.0, 48000.0) == 0
        np.testing.assert_array_equal(
            ms_to_samps(np.array([10.0, 60.0]), 48000.0), [480, 2880]
        )

    def test_dead_db_helper_is_gone(self):
        from src import utils

        assert not hasattr(utils, "db")


class TestHigherOrderReflections:
    """Image sources beyond first order, with per-band wall and air loss.

    A shoebox has 4n^2+2 images at reflection order n, so 6, 18, 38, 66 for
    orders 1 to 4.  First order alone leaves the 0-50 ms window nearly empty,
    which is what makes the individual arrivals stand out.
    """

    FS = 48000.0
    ROOM = (10.0, 8.0, 4.0)
    SRC = (3.0, 7.0, 1.5)
    REC = (8.0, 2.0, 1.5)
    FREQS = np.array([125.0, 250, 500, 1000, 2000, 4000, 8000, 16000])
    T60 = np.array([0.9, 1.0, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5])

    def _refl(self, **kw):
        from src.fdn.planewave_dfdn import shoebox_reflections

        args = dict(room=self.ROOM, source=self.SRC, receiver=self.REC,
                    sample_rate=self.FS, t60_broadband_s=0.825)
        args.update(kw)
        return shoebox_reflections(**args)

    def test_default_is_still_first_order(self):
        assert len(self._refl()) == 6

    @pytest.mark.parametrize("order,expected", [(1, 6), (2, 24), (3, 62), (4, 128)])
    def test_image_counts_follow_4n2_plus_2(self, order, expected):
        """Cumulative count is sum of 4n^2+2 over n = 1..order."""
        assert len(self._refl(order=order)) == expected

    def test_sorted_by_arrival(self):
        delays = [r.delay_samples for r in self._refl(order=3)]
        assert delays == sorted(delays)

    def test_higher_orders_are_quieter(self):
        """More wall bounces and a longer path must both cost amplitude."""
        first = self._refl(order=1)
        third = self._refl(order=3)
        assert max(r.gain for r in third[-10:]) < min(r.gain for r in first)

    def test_no_band_gains_without_freqs(self):
        """Backward compatible: a broadband call behaves exactly as before."""
        for r in self._refl(order=2):
            assert r.band_gains is None

    def test_band_gains_populated_with_freqs(self):
        refl = self._refl(order=2, freqs=self.FREQS, t60_per_band_s=self.T60)
        for r in refl:
            assert r.band_gains is not None
            assert r.band_gains.shape == self.FREQS.shape
            assert np.all(r.band_gains > 0.0)

    def test_air_absorption_darkens_reflections(self):
        """HF must be attenuated relative to LF, and more so on longer paths."""
        refl = self._refl(order=2, freqs=self.FREQS, t60_per_band_s=self.T60,
                          air_absorption=True)
        by_path = sorted(refl, key=lambda r: r.delay_samples)
        near, far = by_path[0], by_path[-1]
        tilt = lambda r: r.band_gains[-1] / r.band_gains[0]  # noqa: E731
        assert tilt(near) < 1.0
        assert tilt(far) < tilt(near)

    def test_air_absorption_can_be_disabled(self):
        wet = self._refl(order=1, freqs=self.FREQS, t60_per_band_s=self.T60,
                         air_absorption=True)
        dry = self._refl(order=1, freqs=self.FREQS, t60_per_band_s=self.T60,
                         air_absorption=False)
        assert wet[-1].band_gains[-1] < dry[-1].band_gains[-1]

    def test_frequency_dependent_wall_loss(self):
        """A T60 curve that falls with frequency must darken the walls too."""
        refl = self._refl(order=1, freqs=self.FREQS, t60_per_band_s=self.T60,
                          air_absorption=False)
        assert refl[0].band_gains[-1] < refl[0].band_gains[0]

    def test_band_gains_render_as_a_filtered_arrival(self):
        """With band gains a reflection is a short impulse response, not a spike."""
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        pw = PlaneWaveDFDN(self.FS, 4, sh_order=1, frame_size=64, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
        pw.set_early_reflections(
            self._refl(order=1, freqs=self.FREQS, t60_per_band_s=self.T60)
        )
        w = pw.render(0.0, 0.0, length_seconds=0.5,
                      direct_gain=0.0, reverb_gain=0.0)[0]
        first = min(r.delay_samples for r in
                    self._refl(order=1, freqs=self.FREQS, t60_per_band_s=self.T60))
        window = w[first - 40 : first + 40]
        assert np.count_nonzero(np.abs(window) > 1e-6) > 1

    def test_higher_order_fills_the_early_window(self):
        from src.fdn.planewave_dfdn import PlaneWaveDFDN

        energy = {}
        for order in (1, 3):
            pw = PlaneWaveDFDN(self.FS, 4, sh_order=1, frame_size=64, seed=42)
            pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
            pw.set_early_reflections(self._refl(order=order))
            w = pw.render(0.0, 0.0, length_seconds=0.5,
                          direct_gain=0.0, reverb_gain=0.0)[0]
            energy[order] = np.sum(w[: int(0.05 * self.FS)] ** 2)
        assert energy[3] > 1.5 * energy[1]
