"""Regression tests for the pyFDN migration."""

from __future__ import annotations

import numpy as np
import pytest

from src.fdn.fdn_components import generate_delay_lengths, geq_band_frequencies


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
    FS = 48000.0
    T60_S = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.1])
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def test_grid_is_the_designs_own(self):
        from src.fdn.fdn_components import (
            GEQ_NUM_BANDS,
            geq_band_frequencies,
        )

        grid = geq_band_frequencies()
        assert grid.shape == (GEQ_NUM_BANDS,)
        # pyFDN >= 0.5.0: octave centres 31.25 Hz to 16 kHz.
        np.testing.assert_allclose(
            grid, [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        )

    def test_shape_and_layout(self):
        from src.fdn.fdn_components import design_absorption_sos

        sos = design_absorption_sos(self.T60_S, self.DELAYS, self.FS)
        assert sos.shape == (11, 6, 4)
        np.testing.assert_allclose(sos[:, 3, :], 1.0)  # a0 normalised

    def test_matches_pyfdn_decay_to_geq(self):
        """Rows of an (N, 10) target map to pyFDN's (10, N) columns."""
        import pyFDN

        from src.fdn.fdn_components import design_absorption_sos

        expected = pyFDN.decay_to_geq(self.T60_S, self.DELAYS, self.FS)
        actual = design_absorption_sos(self.T60_S, self.DELAYS, self.FS)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)

        t60 = self.T60_S[np.newaxis, :] * np.array([[1.0], [0.5], [1.5], [2.0]])
        actual = design_absorption_sos(t60, self.DELAYS, self.FS)
        for i, m in enumerate(self.DELAYS):
            np.testing.assert_allclose(
                actual[:, :, i],
                pyFDN.decay_to_geq(t60[i], np.array([m]), self.FS)[:, :, 0],
                rtol=1e-8, atol=1e-12,
            )

    def test_per_channel_rt_gives_different_filters(self):
        from src.fdn.fdn_components import design_absorption_sos

        t60 = np.vstack([self.T60_S, self.T60_S * 0.5, self.T60_S, self.T60_S * 2.0])
        sos = design_absorption_sos(t60, self.DELAYS, self.FS)
        assert sos.shape == (11, 6, 4)
        assert not np.allclose(sos[:, :, 0], sos[:, :, 1])

    def test_rejects_wrong_channel_count(self):
        from src.fdn.fdn_components import design_absorption_sos

        t60 = np.vstack([self.T60_S, self.T60_S])  # 2 rows for 4 delays
        with pytest.raises(ValueError, match="one T60 curve per delay line"):
            design_absorption_sos(t60, self.DELAYS, self.FS)

    def test_rejects_a_target_off_the_design_grid(self):
        """A curve on any other frequency axis must be refused, not resampled.

        Silently accepting eight values and stretching them onto the grid
        would misplace every one of them.
        """
        from src.fdn.fdn_components import design_absorption_sos

        eight_bands = np.array([0.9, 1.0, 1.1, 1.0, 0.8, 0.7, 0.6, 0.5])
        with pytest.raises(ValueError, match="10 values per delay line"):
            design_absorption_sos(eight_bands, self.DELAYS, self.FS)

    def test_realised_t60_at_every_command_point(self):
        """Each command value is realised at its own frequency, 16 kHz included."""
        from scipy.signal import sosfreqz

        from src.fdn.fdn_components import design_absorption_sos

        m = 1500.0
        t60 = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.3])
        sos = design_absorption_sos(t60, np.array([m]), self.FS)

        probe = geq_band_frequencies()
        _, h = sosfreqz(sos[:, :, 0], worN=2 * np.pi * probe / self.FS)
        realised = -60.0 * m / (self.FS * 20 * np.log10(np.abs(h)))
        np.testing.assert_allclose(realised, t60, rtol=0.05)


class TestPlainFDNOnPyFDN:
    """A plain pyFDN build with this package's absorption meets its T60."""

    FS = 48000.0
    DELAYS = np.array([1039, 1213, 1381, 1523, 1699, 1877, 2039, 2213])

    def _render(self, *, gains=None, post_delay=None, seconds=4.0):
        import pyFDN

        A = pyFDN.fdn_matrix_gallery(8, "hadamard")
        C = np.ones((1, 8)) / np.sqrt(8)
        if gains is not None:
            A, C = A * gains, C * gains
        build = pyFDN.FDNBuild(
            A=A, B=np.ones((8, 1)) / np.sqrt(8), C=C, D=np.zeros((1, 1)),
            delays=self.DELAYS, fs=self.FS, post_delay=post_delay,
        )
        x = np.zeros(int(seconds * self.FS))
        x[0] = 1.0
        return pyFDN.process_fdn(x, build)

    def test_broadband_gains_folded_into_a_and_c(self, measured_t60):
        """diag(g) after the delays is A @ diag(g) and C @ diag(g)."""
        import pyFDN

        g = pyFDN.db_to_lin(pyFDN.rt_to_slope(1.5, self.FS) * self.DELAYS)
        rt = measured_t60(self._render(gains=g), self.FS)
        np.testing.assert_allclose(rt[1:], 1.5, rtol=0.05)

    def test_frequency_dependent_t60_is_accurate(self, measured_t60):
        from src.fdn.fdn_components import design_absorption_sos

        t60_s = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.1])
        sos = design_absorption_sos(t60_s, self.DELAYS, self.FS)
        rt = measured_t60(self._render(post_delay=sos), self.FS)
        # measured_t60 reports 62.5 Hz to 8 kHz, i.e. grid entries 1 to 8.
        target = t60_s[1:9]
        # The edge bands sit against the shelves; check the middle.
        np.testing.assert_allclose(rt[1:6], target[1:6], rtol=0.10)


class TestDFDN2019CorrectedLoop:
    FS = 48000.0

    def _make(self, **kwargs):
        from src.fdn.dfdn_2019 import DFDN2019

        dfdn = DFDN2019(self.FS, 8, ambisonics_order=1, seed=42,
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
            *dfdn.transform.transform_matrices
        )
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), expected, atol=1e-12
        )
        # The DWM must actually be doing something, or this proves nothing.
        assert not np.allclose(
            dfdn._effective_feedback_matrix(), dfdn._kronecker_matrix, atol=1e-8
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
        """pyFDN processes every sample; the length is not rounded to a block."""
        rir = self._make().render(0.0, 0.0, length_seconds=0.501)
        assert rir.shape[1] == int(0.501 * self.FS)


class TestDirectionalFDNCorrectedLoop:
    FS = 48000.0

    def _make(self, **setup_kwargs):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 8, sh_order=1, seed=42)
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

    def test_broadband_t60_takes_the_scalar_gain_path(self):
        """A scalar or per-direction broadband T60 must not become a filter."""
        scalar = self._make(t60_per_direction_ms=1500.0)
        assert scalar.absorption_sos is None

        grid = self._make()
        per_dir = self._make(t60_per_direction_ms=np.full(grid.Q, 1500.0))
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
        # With no reverb feed the first sample is the direct path only, and
        # SN3D puts W = 1 in every direction, so it is the gain unchanged.
        assert rir[0, 0] == pytest.approx(0.5, rel=1e-6)

    def test_early_reflections_arrive_on_time(self):
        from src.fdn.room import Reflection

        pw = self._make()
        pw.set_early_reflections(
            [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
        )
        rir = pw.render(0.0, 0.0, length_seconds=0.5, direct_gain=0.0,
                        reverb_gain=0.0)
        assert np.abs(rir[0, 200]) > 1e-3

    def test_early_reflections_add_energy(self):
        from src.fdn.room import Reflection

        without = self._make().render(0.0, 0.0, length_seconds=1.0)
        pw = self._make()
        pw.set_early_reflections(
            [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
        )
        with_er = pw.render(0.0, 0.0, length_seconds=1.0)
        assert np.sum(with_er**2) > np.sum(without**2)

    def test_repeat_render_is_deterministic(self):
        t60_ms = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.1]) * 1e3
        pw = self._make(t60_per_direction_ms=t60_ms)
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
    # The GEQ design grid: [DC, 63, 125, 250, 500, 1k, 2k, 4k, 8k, fs].
    FREQS = geq_band_frequencies()
    DELAYS = np.array([1039, 1213, 1381, 1523])
    T60 = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.2])

    @pytest.mark.parametrize(
        ("temperature_c", "humidity_percent"),
        [(20.0, 50.0), (20.0, 1.0), (20.0, 90.0), (0.0, 50.0),
         (40.0, 50.0), (-15.0, 30.0), (35.0, 80.0), (50.0, 10.0)],
    )
    def test_matches_pyfars_independent_iso9613(
        self, temperature_c, humidity_percent
    ):
        """Pinned against another implementation of the same standard.

        pyfar.constants.air_attenuation is an independent ISO 9613-1, and it
        takes relative humidity as a 0-1 fraction where this module takes
        percent -- so the conversion below is also the convention check.
        """
        import pyfar

        from src.fdn.room import air_absorption_db_per_meter

        freqs = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
        ours = air_absorption_db_per_meter(
            freqs, temperature_c=temperature_c,
            humidity_percent=humidity_percent,
        )
        # Returns (alpha, m, accuracy); alpha is the dB/m coefficient.
        theirs, _, _ = pyfar.constants.air_attenuation(
            temperature_c, freqs, humidity_percent / 100.0
        )
        np.testing.assert_allclose(ours, theirs, rtol=1e-3)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"humidity_percent": 150.0}, "humidity"),
            ({"humidity_percent": -10.0}, "humidity"),
            ({"temperature_c": -273.15}, "absolute zero"),
            ({"temperature_c": -300.0}, "absolute zero"),
            ({"pressure_kpa": 0.0}, "pressure"),
            ({"pressure_kpa": -5.0}, "pressure"),
        ],
    )
    def test_rejects_unphysical_atmospheres(self, kwargs, match):
        """Out-of-range inputs returned plausible numbers rather than failing."""
        from src.fdn.room import air_absorption_db_per_meter

        with pytest.raises(ValueError, match=match):
            air_absorption_db_per_meter(np.array([1000.0]), **kwargs)

    def test_rejects_negative_frequencies(self):
        """alpha goes as f^2, so a sign error would pass silently."""
        from src.fdn.room import air_absorption_db_per_meter

        with pytest.raises(ValueError, match="[Ff]requenc"):
            air_absorption_db_per_meter(np.array([-1000.0]))

    def test_zero_frequency_gives_an_infinite_air_t60(self):
        """DC loses nothing to the medium, so it caps no boundary target.

        The GEQ grid's first point is an anchor near DC, so this is reached in
        normal use and must not raise or warn its way to a nan.
        """
        import warnings

        from src.fdn.room import air_absorption_t60

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t60 = air_absorption_t60(np.array([0.0, 1000.0]))
        assert np.isinf(t60[0])
        assert np.isfinite(t60[1])

    def test_air_only_t60_falls_with_frequency(self):
        from src.fdn.room import air_absorption_t60

        t_air = air_absorption_t60(self.FREQS)
        assert np.all(np.diff(t_air) < 0.0)
        # 16 kHz at 20 C / 50% RH is around half a second in air alone.
        assert 0.3 < air_absorption_t60(np.array([16000.0]))[0] < 0.7

    def test_air_limited_target_matches_the_sabine_4mv_term(self):
        """The rate sum must equal the textbook air term, not merely resemble it."""
        from src.fdn.room import air_absorption_db_per_meter, air_limited_t60

        volume, surface, alpha_boundary = 320.0, 304.0, 0.1857
        sabine_dry = 0.161 * volume / (surface * alpha_boundary)
        m = air_absorption_db_per_meter(self.FREQS) / (10 * np.log10(np.e))
        sabine_with_air = 0.161 * volume / (surface * alpha_boundary + 4 * m * volume)

        combined = air_limited_t60(np.full(self.FREQS.size, sabine_dry), self.FREQS)
        np.testing.assert_allclose(combined, sabine_with_air, rtol=1e-3)

    def test_air_can_only_shorten(self):
        from src.fdn.room import air_limited_t60

        limited = air_limited_t60(self.T60, self.FREQS)
        assert np.all(limited <= self.T60)
        assert limited[-1] < self.T60[-1]      # the fs anchor is much shorter
        assert limited[0] == pytest.approx(self.T60[0], rel=5e-3)   # DC barely

    def test_shares_sum_to_the_required_attenuation(self):
        """air_db and boundary_db must account for the realised round trip."""
        from src.fdn.room import air_absorption_split, air_limited_t60

        t60 = np.tile(self.T60, (4, 1))
        air, boundary = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        realised = air_limited_t60(t60, self.FREQS)
        required = 60.0 * self.DELAYS[:, None] / (self.FS * realised)
        np.testing.assert_allclose(air + boundary, required, rtol=1e-10)

    def test_both_shares_are_positive(self):
        """The old model could produce a negative boundary share; this cannot."""
        from src.fdn.room import air_absorption_split

        t60 = np.tile(np.full(self.FREQS.size, 60.0), (4, 1))  # absurd target
        air, boundary = air_absorption_split(t60, self.FREQS, self.DELAYS, self.FS)
        assert np.all(air > 0.0)
        assert np.all(boundary > 0.0)

    def test_enabling_air_shortens_the_decay(self):
        """The point of the change: the flag must now affect the sound."""
        from src.fdn.directional_fdn import DirectionalFDN

        rt = {}
        for air in (False, True):
            pw = DirectionalFDN(self.FS, 8, sh_order=1, seed=42)
            pw.setup(
                delay_range_ms=(20.0, 60.0),
                t60_per_direction_ms=self.T60 * 1e3,
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
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 8, sh_order=1, seed=42)
        pw.setup(
            delay_range_ms=(20.0, 60.0),
            t60_per_direction_ms=self.T60 * 1e3,
            air_absorption=True,
        )
        n = self.FREQS.size
        assert pw.air_attenuation_db.shape == (pw.total_delay_lines, n)
        assert pw.boundary_attenuation_db.shape == (pw.total_delay_lines, n)
        assert pw.realised_t60_s.shape == (pw.total_delay_lines, n)
        assert np.all(pw.realised_t60_s <= self.T60[None, :] + 1e-12)


class TestFeedbackMatrixFromPyFDN:
    """The generators are pyFDN's, with reproducibility preserved."""

    def test_hadamard_is_unchanged(self):
        """The gallery's Hadamard must be bit-identical to the old one."""
        import pyFDN

        from src.fdn.fdn_components import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, FeedbackMatrixType.HADAMARD)
        np.testing.assert_allclose(
            fb.matrix, pyFDN.fdn_matrix_gallery(8, "Hadamard"), atol=1e-15
        )

    def test_householder_stays_the_canonical_all_ones(self):
        """fdn_matrix_gallery randomises Householder; we want the all-ones one."""
        import pyFDN

        from src.fdn.fdn_components import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, FeedbackMatrixType.HOUSEHOLDER)
        np.testing.assert_allclose(
            fb.matrix, pyFDN.householder_matrix(np.ones(8)), atol=1e-15
        )

    def test_random_is_reproducible(self):
        from src.fdn.fdn_components import FeedbackMatrix, FeedbackMatrixType

        a = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, seed=42).matrix
        b = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, seed=42).matrix
        c = FeedbackMatrix(8, FeedbackMatrixType.RANDOM, seed=7).matrix
        np.testing.assert_allclose(a, b, atol=1e-15)
        assert not np.allclose(a, c)

    def test_random_does_not_disturb_the_global_stream(self):
        """pyFDN's gallery draws from global np.random; we must restore it."""
        from src.fdn.fdn_components import FeedbackMatrix, FeedbackMatrixType

        np.random.seed(0)
        expected = np.random.rand()
        np.random.seed(0)
        FeedbackMatrix(8, FeedbackMatrixType.RANDOM, seed=42)
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

        from src.fdn.fdn_components import FeedbackMatrix, FeedbackMatrixType

        fb = FeedbackMatrix(8, getattr(FeedbackMatrixType, kind), seed=42)
        assert fb.matrix.shape == (8, 8)
        assert pyFDN.is_unilossless(fb.matrix)

    def test_unknown_type_still_raises(self):
        from src.fdn.fdn_components import FeedbackMatrix

        with pytest.raises(ValueError, match="Unsupported feedback matrix type"):
            FeedbackMatrix(8, "not-a-type")

    def test_matrix_convolve_is_gone(self):
        from src.fdn.fdn_components import FeedbackMatrix

        assert not hasattr(FeedbackMatrix, "matrix_convolve")


class TestDelayLengthSpread:
    """Global spread across all N*Q lines, not per-sub-network coprimality.

    A ``kron`` feedback matrix never mixes across the channel index, so it is
    tempting to draw the N lengths of each channel as its own coprime FDN, or
    -- as the SH formulation of JAES 2019 does -- to share one length across a
    group's Q channels.  Either way the composite loses density: the audible
    output is the sum of all Q sub-networks through the encoder, so what makes
    it dense is that all N*Q lengths are distinct and spread.  The classic
    coprimality criterion applies to the whole pool here, not to each
    sub-network.  This is the argument of ICASSP 2020 Fig. 4, where sharing a
    length within a group leaves N distinct delays "thus not contributing to a
    higher integrated sum".

    Mixing time therefore tracks N*Q rather than N.  Measured on the degree-2L
    grid: N=8, L=7 gives 912 lines and mixes 1.1x after the shortest delay,
    while N=8, L=3 gives 208 and takes 14x.  An absolute bound would only be
    pinning one grid size, so the assertion below is the comparison instead.
    """

    FS = 48000.0

    def _make(self, sh_order=3, groups=8):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, groups, sh_order=sh_order, seed=42)
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

    def _mixing_time(self, pw):
        import pyFDN

        w = pw.render(0.0, 0.0, 2.0, direct_gain=0.0, er_gain=0.0,
                      reverb_gain=1.0)[0]
        t_mix, _ = pyFDN.echo_density(w, fs=self.FS)
        return t_mix

    def test_distinct_lengths_mix_sooner_than_shared_ones(self):
        """ICASSP 2020 Fig. 4: N*Q distinct delays beat N shared ones."""
        pw = self._make()
        distinct = self._mixing_time(pw)

        # The JAES 2019 draw: one length per group, repeated across its Q
        # channels, so N*Q lines carry only N distinct delays.
        shared = self._make()
        per_group = shared.delay_length_samps.reshape(
            shared.num_delay_line_groups, shared.Q
        )[:, 0]
        shared.delay_length_samps = np.repeat(per_group, shared.Q)

        assert 0.0 < distinct < self._mixing_time(shared)

    def test_mixing_time_improves_with_more_lines(self):
        """Density tracks N*Q, which is why the grid degree matters."""
        sparse = self._mixing_time(self._make(sh_order=3, groups=8))
        dense = self._mixing_time(self._make(sh_order=7, groups=8))
        assert 0.0 < dense < sparse

    def test_reproducible(self):
        np.testing.assert_array_equal(
            self._make().delay_length_samps, self._make().delay_length_samps
        )


class TestLosslessnessChecks:
    """Both directional classes verify the feedback matrix with pyFDN."""

    FS = 48000.0

    def test_directional_fdn_kronecker_is_lossless(self):
        import pyFDN

        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        assert pyFDN.is_unilossless(pw._kronecker_matrix)

    def test_dfdn_2019_kronecker_is_lossless(self):
        import pyFDN

        from src.fdn.dfdn_2019 import DFDN2019

        dfdn = DFDN2019(self.FS, 4, ambisonics_order=1, seed=42)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
        assert pyFDN.is_unilossless(dfdn._kronecker_matrix)

    def test_directional_fdn_kronecker_matrix_stays_lossless(self):
        """A kron I_Q is lossless whenever A is (ICASSP 2020, Sec. 3)."""
        import pyFDN

        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        assert pyFDN.is_unilossless(pw._kronecker_matrix)

    def test_directional_fdn_feedback_never_couples_directions(self):
        """Eq. of Sec. 3: recirculation between different channels is zero."""
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0)
        Q = pw.Q
        for i in range(pw.num_delay_line_groups):
            for j in range(pw.num_delay_line_groups):
                block = pw._kronecker_matrix[i * Q:(i + 1) * Q, j * Q:(j + 1) * Q]
                off = block[~np.eye(Q, dtype=bool)]
                np.testing.assert_allclose(off, 0.0, atol=1e-12)


class TestPoleBounds:
    """pole_bounds is pyFDN.pole_boundaries, fed the package's SOS banks."""

    FS = 48000.0
    FREQS = geq_band_frequencies()
    DELAYS = np.array([1039, 1213, 1381, 1523])

    def _sos(self, t60_s):
        from src.fdn.fdn_components import design_absorption_sos

        return design_absorption_sos(t60_s, self.DELAYS, self.FS)

    def _A(self):
        return np.linalg.qr(np.random.default_rng(0).standard_normal((4, 4)))[0]

    def test_shapes_and_frequency_axis(self):
        from src.utils import pole_bounds

        lo, hi, f = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 1.2)),
            self._A(), self.FS, nfft=512,
        )
        assert lo.shape == hi.shape == f.shape == (512,)
        assert f[0] == pytest.approx(0.0)
        assert f[-1] == pytest.approx(self.FS / 2 * 511 / 512)

    def test_lossy_fdn_stays_below_unity(self):
        """An orthogonal A has all singular values 1, so the bound is the
        absorption -- which must be lossy at every frequency."""
        from src.utils import pole_bounds

        _, hi, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 1.2)), self._A(), self.FS
        )
        assert np.all(hi < 1.0)

    def test_shorter_t60_lowers_the_bound(self):
        from src.utils import pole_bounds

        _, hi_long, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 2.0)), self._A(), self.FS
        )
        _, hi_short, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 0.5)), self._A(), self.FS
        )
        assert np.all(hi_short < hi_long)

    def test_min_curve_does_not_exceed_max(self):
        from src.utils import pole_bounds

        lo, hi, _ = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 1.2)), self._A(), self.FS
        )
        assert np.all(lo <= hi + 1e-12)

    def test_accepts_a_2d_feedback_matrix(self):
        from src.utils import pole_bounds

        lo, hi, f = pole_bounds(
            self.DELAYS, self._sos(np.full(self.FREQS.size, 1.2)),
            np.eye(4), self.FS, nfft=256,
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
        from src.fdn.room import shoebox_reflections

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
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, seed=42)
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
        from src.fdn.directional_fdn import DirectionalFDN

        energy = {}
        for order in (1, 3):
            pw = DirectionalFDN(self.FS, 4, sh_order=1, seed=42)
            pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
            pw.set_early_reflections(self._refl(order=order))
            w = pw.render(0.0, 0.0, length_seconds=0.5,
                          direct_gain=0.0, reverb_gain=0.0)[0]
            energy[order] = np.sum(w[: int(0.05 * self.FS)] ** 2)
        assert energy[3] > 1.5 * energy[1]


class TestVBAP:
    """Vector-base amplitude panning onto the plane-wave grid.

    Alary and Politis (ICASSP 2020) pan the direct sound (Eq. 3) and every
    early reflection (Eq. 4) onto the grid with VBAP, then encode the grid to
    the output with E (Eq. 13).  Gains are normalised to sum to one so that
    E @ g reproduces y(theta): the l=0 harmonic is constant over the sphere,
    so its encoded value is (sum g) / sqrt(4 pi).
    """

    def _grid(self, sh_order=2):
        from src.fdn import spatial

        return spatial.quadrature_grid(sh_order)

    def test_gains_are_nonnegative_and_sum_to_one(self):
        from src.fdn.spatial import vbap_gains

        az, el = self._grid()
        for direction in [(0.0, 0.0), (1.3, 0.4), (-2.0, -0.8), (np.pi, 0.0)]:
            g = vbap_gains(az, el, *direction)
            assert np.all(g >= -1e-12)
            assert g.sum() == pytest.approx(1.0)

    def test_at_most_three_active_gains(self):
        from src.fdn.spatial import vbap_gains

        az, el = self._grid()
        g = vbap_gains(az, el, 0.9, 0.2)
        assert np.count_nonzero(g > 1e-9) <= 3

    def test_a_grid_direction_pans_to_itself(self):
        from src.fdn.spatial import vbap_gains

        az, el = self._grid()
        g = vbap_gains(az, el, float(az[5]), float(el[5]))
        assert g[5] == pytest.approx(1.0, abs=1e-6)

    def test_encoding_recovers_the_direction(self):
        """E @ vbap(theta) must approximate y(theta) -- the paper's premise."""
        from src.fdn import spatial

        az, el = self._grid(sh_order=3)
        Y = spatial.encode_directions(az, el, 1)          # E = Y^T per Eq. 13
        for direction in [(0.0, 0.0), (1.3, 0.4), (-2.0, -0.5)]:
            g = spatial.vbap_gains(az, el, *direction)
            encoded = Y.T @ g
            target = spatial.encode_directions(*direction, 1)
            # W is exact by construction; the first-order axes are a
            # three-point approximation on a finite grid.  The tolerance is a
            # fraction of W, which SN3D fixes at 1 regardless of direction.
            assert encoded[0] == pytest.approx(target[0], rel=1e-9)
            np.testing.assert_allclose(encoded[1:], target[1:], atol=0.22)

    def test_rejects_a_degenerate_grid(self):
        from src.fdn.spatial import vbap_gains

        with pytest.raises(ValueError, match="at least 4"):
            vbap_gains(np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0, 0.0)


class TestDFDN2019CarriesNoExtraKnobs:
    """Only what JAES 2019 describes: A kron I_Q, the DWM, and Eq. 17 input.

    ``scattering_amount`` blended the per-group recirculation toward a random
    orthogonal matrix and ``use_directional_input`` encoded the source
    direction into every group instead of Eq. 17's omnidirectional gains.
    Both were marked "not described in the papers" in their own docstrings.
    """

    FS = 48000.0

    def _dfdn(self):
        from src.fdn.dfdn_2019 import DFDN2019

        dfdn = DFDN2019(self.FS, 4, ambisonics_order=1,
                              seed=42)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
        return dfdn

    def test_constructor_takes_no_scattering_amount(self):
        from src.fdn.dfdn_2019 import DFDN2019

        with pytest.raises(TypeError, match="scattering_amount"):
            DFDN2019(self.FS, 4, ambisonics_order=1,
                           scattering_amount=0.5)

    def test_render_takes_no_directional_input_flag(self):
        with pytest.raises(TypeError, match="use_directional_input"):
            self._dfdn().render(0.0, 0.0, 0.5, use_directional_input=True)

    def test_recirculation_is_kronecker_times_the_dwm_alone(self):
        """No per-group factor survives between A kron I_Q and T_i."""
        dfdn = self._dfdn()
        np.testing.assert_allclose(
            dfdn._effective_feedback_matrix(), dfdn._kronecker_matrix,
            atol=1e-12,
        )

    def test_input_gains_are_omnidirectional_per_eq_17(self):
        """Eq. 17 drives W only, with no directional variant to opt into."""
        dfdn = self._dfdn()
        b = dfdn._build_input_gains().ravel()

        N, Q = dfdn.num_delay_line_groups, dfdn.Q
        expected = np.zeros(N * Q)
        expected[::Q] = 1.0 / np.sqrt(N)
        np.testing.assert_allclose(b, expected, atol=1e-12)


class TestAmbisonicNormalization:
    """ACN/SN3D as AmbiX defines it: W = 1 in every direction.

    spharpy offers both conventions and the names do not mean what they
    look like -- its ``"SN3D"`` carries the ``1/sqrt(4 pi)`` of the
    orthonormal basis, while ``"SNM"`` is the Schmidt semi-normalisation
    AmbiX calls SN3D.  Picking the wrong one left every render 11 dB low
    with the docstrings still claiming SN3D.
    """

    @staticmethod
    def _sn3d_reference(azimuth, elevation, order):
        """Independent SN3D, ACN-ordered, built from scipy."""
        from scipy.special import sph_harm_y

        out = []
        for degree in range(order + 1):
            for m in range(-degree, degree + 1):
                Y = sph_harm_y(degree, abs(m), np.pi / 2 - elevation, azimuth)
                if m > 0:
                    value = np.sqrt(2) * (-1) ** m * Y.real
                elif m < 0:
                    value = np.sqrt(2) * (-1) ** m * Y.imag
                else:
                    value = Y.real
                out.append(value * np.sqrt(4 * np.pi / (2 * degree + 1)))
        return np.array(out)

    @pytest.mark.parametrize(
        ("azimuth", "elevation"),
        [(0.0, 0.0), (0.9, 0.4), (-2.1, -1.1), (np.pi, np.pi / 2)],
    )
    def test_matches_sn3d_across_orders(self, azimuth, elevation):
        from src.fdn import spatial

        for order in (1, 2, 3):
            np.testing.assert_allclose(
                spatial.encode_directions(azimuth, elevation, order),
                self._sn3d_reference(azimuth, elevation, order),
                atol=1e-12,
            )

    @pytest.mark.parametrize(
        ("azimuth", "elevation"),
        [(0.0, 0.0), (0.9, 0.4), (-2.1, -1.1), (np.pi, np.pi / 2)],
    )
    def test_w_is_unity_everywhere(self, azimuth, elevation):
        """The defining property: an omnidirectional source encodes to W = 1."""
        from src.fdn import spatial

        assert spatial.encode_directions(azimuth, elevation, 3)[0] == pytest.approx(1.0)

    def test_array_input_agrees_with_scalar_calls(self):
        from src.fdn import spatial

        az = np.array([0.0, 0.9, -2.1])
        el = np.array([0.0, 0.4, -1.1])
        stacked = spatial.encode_directions(az, el, 3)
        for i, (a, e) in enumerate(zip(az, el, strict=True)):
            np.testing.assert_allclose(
                stacked[i], spatial.encode_directions(a, e, 3), atol=1e-12
            )


class TestQuadratureGrid:
    """ICASSP 2020 Sec. 3: "spherical t-designs of degree t = 2L"."""

    @staticmethod
    def _design_size(degree: int) -> int:
        """Point count of spharpy's t-design, for even *degree*."""
        return -(-((degree + 1) ** 2) // 2) + 1

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
    def test_grid_is_the_degree_2L_design(self, order):
        """Not degree 4L: spharpy's n_max already doubles to a degree."""
        from src.fdn import spatial

        azimuths, _ = spatial.quadrature_grid(order)
        assert len(azimuths) == self._design_size(2 * order)

    @pytest.mark.parametrize("order", [1, 2, 3, 5, 7])
    def test_grid_satisfies_sn3d_orthogonality(self, order):
        """Y^T Y (4 pi / Q) = diag(4 pi / (2l + 1)) -- what the degree is for."""
        from src.fdn import spatial

        azimuths, elevations = spatial.quadrature_grid(order)
        Y = spatial.encode_directions(azimuths, elevations, order)
        gram = Y.T @ Y * (4 * np.pi / len(azimuths))
        target = np.diag([
            4 * np.pi / (2 * degree + 1)
            for degree in range(order + 1)
            for _ in range(2 * degree + 1)
        ])
        np.testing.assert_allclose(gram, target, atol=1e-12)

    def test_grid_is_large_enough_for_the_order(self, ):
        """Sec. 3: "at least Q >= (L + 1)^2 uniform points"."""
        from src.fdn import spatial

        for order in range(1, 8):
            azimuths, _ = spatial.quadrature_grid(order)
            assert len(azimuths) >= (order + 1) ** 2


class TestInputDistribution:
    """b_i is a Q-vector carrying the initial directional energy.

    ICASSP 2020 Sec. 3: the scaling vector "sets an initial gain for each
    channel, corresponding to the distribution of early energy during the
    scattering phase".  Alary and Valimaki, I3DA 2021, Eq. 7 writes it as
    ``b_i (x) x(z)`` and Sec. III C specifies it from the measured EDC_0 of
    a captured SRIR.  Uniform is the isotropic special case, not the only one.
    """

    FS = 48000.0

    def _make(self):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=2, output_sh_order=2,
                           seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1200.0)
        return pw

    def test_default_is_uniform(self):
        """Unset, b stays exactly what it was: 1 / sqrt(N Q) everywhere."""
        pw = self._make()
        b = pw._build_input_gains().ravel()
        expected = 1.0 / np.sqrt(pw.num_delay_line_groups * pw.Q)
        np.testing.assert_allclose(b, expected, rtol=1e-12)

    def test_distribution_shapes_b_per_direction(self):
        """Each group gets the same Q-vector, proportional to the weights."""
        pw = self._make()
        weights = np.linspace(0.2, 1.0, pw.Q)
        pw.set_input_distribution(weights)
        b = pw._build_input_gains().ravel()

        per_group = b.reshape(pw.num_delay_line_groups, pw.Q)
        for group in per_group[1:]:
            np.testing.assert_allclose(group, per_group[0], rtol=1e-12)
        ratio = per_group[0] / weights
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)

    @pytest.mark.parametrize("weights", [None, "ramp", "spike"])
    def test_total_input_energy_is_unity(self, weights):
        """Changing the distribution must not change the drive level."""
        pw = self._make()
        if weights == "ramp":
            pw.set_input_distribution(np.linspace(0.1, 1.0, pw.Q))
        elif weights == "spike":
            spike = np.full(pw.Q, 0.01)
            spike[0] = 1.0
            pw.set_input_distribution(spike)
        b = pw._build_input_gains()
        assert float(np.sum(b**2)) == pytest.approx(1.0, rel=1e-12)

    def test_concentrated_distribution_favours_that_direction(self):
        """The audible consequence: that grid channel carries more energy."""
        pw = self._make()
        spike = np.full(pw.Q, 0.05)
        spike[0] = 1.0
        pw.set_input_distribution(spike)
        grid = pw.render(0.0, 0.0, 0.4, direct_gain=0.0, grid_output=True)
        energy = np.sum(grid**2, axis=1)
        assert energy[0] > 5.0 * np.median(energy[1:])

    def test_rejects_wrong_length(self):
        pw = self._make()
        with pytest.raises(ValueError, match="length"):
            pw.set_input_distribution(np.ones(pw.Q + 1))

    def test_rejects_negative_weights(self):
        """EDC_0 is an energy, so a negative weight is meaningless."""
        pw = self._make()
        weights = np.ones(pw.Q)
        weights[2] = -0.5
        with pytest.raises(ValueError, match="negative"):
            pw.set_input_distribution(weights)

    def test_rejects_an_all_zero_distribution(self):
        pw = self._make()
        with pytest.raises(ValueError, match="positive"):
            pw.set_input_distribution(np.zeros(pw.Q))


class TestLambdaGroupInput:
    """Eq. 12: for i = lambda the reflections replace ``b_lambda x``.

        s_i = s'(z) + Z_i[sum_j A_ij G_j s_j],          i = lambda
        s_i = Z_i[b_i x(z) + sum_j A_ij G_j s_j],       else

    There is no ``b_lambda x`` term in the first branch.  Driving that group
    from both the impulse and the reflections double-counts its input.
    """

    FS = 48000.0

    def _make(self, lambda_group=0):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, output_sh_order=1,
                           seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1200.0,
                 lambda_group=lambda_group)
        return pw

    def _reflection(self):
        from src.fdn.room import Reflection

        return [Reflection(gain=0.5, azimuth=0.3, elevation=0.1,
                           delay_samples=150)]

    def test_without_reflections_every_group_is_driven(self):
        pw = self._make()
        b = pw._build_input_gains().ravel()
        assert np.count_nonzero(b) == pw.num_delay_line_groups * pw.Q

    @pytest.mark.parametrize("lambda_group", [0, 2])
    def test_lambda_group_loses_its_input_injection(self, lambda_group):
        pw = self._make(lambda_group)
        pw.set_early_reflections(self._reflection())
        b = pw._build_input_gains().ravel().reshape(
            pw.num_delay_line_groups, pw.Q)

        np.testing.assert_allclose(b[lambda_group], 0.0, atol=1e-15)
        others = [g for g in range(pw.num_delay_line_groups)
                  if g != lambda_group]
        assert np.all(b[others] > 0.0)

    def test_other_groups_keep_their_original_gain(self):
        """Only lambda changes; the rest are untouched by the ER module."""
        pw = self._make(lambda_group=1)
        before = pw._build_input_gains().ravel().reshape(
            pw.num_delay_line_groups, pw.Q)
        pw.set_early_reflections(self._reflection())
        after = pw._build_input_gains().ravel().reshape(
            pw.num_delay_line_groups, pw.Q)

        for group in range(pw.num_delay_line_groups):
            if group == 1:
                continue
            np.testing.assert_allclose(after[group], before[group], rtol=1e-12)

    def test_starving_lambda_changes_the_render(self):
        """The audible consequence, not just the vector."""
        pw = self._make()
        plain = pw.render(0.0, 0.0, 0.5, direct_gain=0.0)
        pw.set_early_reflections(self._reflection())
        with_er = pw.render(0.0, 0.0, 0.5, direct_gain=0.0, er_gain=0.0)
        # er_gain=0 injects nothing, so the only difference is the group that
        # no longer receives b_lambda x.
        assert np.sum(with_er**2) < np.sum(plain**2)


class TestPaperFaithfulEncoding:
    """One grid, one encoder -- Alary and Politis (ICASSP 2020) Eqs. 3, 11-13.

    The paper puts the direct sound, the early reflections and the
    recirculating signals all on the plane-wave grid, and encodes them with a
    single matrix E = Y(L, U_grid).  Every path then shares one convention, so
    their relative levels do not depend on the grid size.
    """

    FS = 48000.0
    FREQS = np.array([125.0, 250, 500, 1000, 2000, 4000, 8000])
    T60 = np.array([0.9, 1.0, 1.1, 1.0, 0.8, 0.7, 0.6])

    def _make(self, sh_order=2, **kw):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 8, sh_order=sh_order, output_sh_order=1,
                           seed=42, **kw)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
        return pw

    def test_encoder_is_the_plain_sh_matrix(self):
        """Eq. 13: E = Y(L, U_grid), with no quadrature scaling."""
        from src.fdn import spatial

        pw = self._make()
        Y = spatial.encode_directions(
            pw.grid_azimuths, pw.grid_elevations, pw.output_sh_order
        )
        np.testing.assert_allclose(pw._encoding_matrix, Y.T, atol=1e-12)

    def test_one_grid_channel_encodes_to_its_own_direction(self):
        from src.fdn import spatial

        pw = self._make()
        q = 5
        s = np.zeros(pw.Q)
        s[q] = 1.0
        expected = spatial.encode_directions(
            float(pw.grid_azimuths[q]), float(pw.grid_elevations[q]),
            pw.output_sh_order,
        )
        np.testing.assert_allclose(pw._encoding_matrix @ s, expected, atol=1e-12)

    def test_direct_sound_is_panned_onto_the_grid(self):
        """Eq. 3: d = vbap(U_grid, (phi, theta)), then encoded by E."""
        from src.fdn import spatial

        pw = self._make()
        az, el = 0.8, 0.3
        d = pw._build_direct_gain(az, el)
        assert d.shape == (pw.Q, 1)
        assert np.count_nonzero(d > 1e-9) <= 3
        encoded = (pw._encoding_matrix @ d).ravel()
        target = spatial.encode_directions(az, el, pw.output_sh_order)
        assert encoded[0] == pytest.approx(target[0], rel=1e-9)
        np.testing.assert_allclose(encoded[1:], target[1:], atol=0.36)

    def test_drr_no_longer_depends_on_grid_size(self):
        """The leak this fixes: sh_order changed DRR by 7.3 dB."""
        from src.fdn.directional_fdn import DirectionalFDN
        from src.fdn.room import shoebox_reflections

        room, src, rec = (10.0, 8.0, 4.0), (3.0, 7.0, 1.5), (8.0, 2.0, 1.5)
        dist = float(np.linalg.norm(np.array(src) - np.array(rec)))
        refl = shoebox_reflections(room=room, source=src, receiver=rec,
                                   sample_rate=self.FS, t60_broadband_s=0.825)
        drrs = []
        for sh in (1, 2, 3):
            pw = DirectionalFDN(self.FS, 8, sh_order=sh, output_sh_order=1,
                               seed=42)
            pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
            pw.set_early_reflections(refl)
            w = pw.render(0.0, 0.0, 2.0, direct_gain=1.0 / dist,
                          reverb_gain=0.8)[0]
            drrs.append(10 * np.log10(w[0] ** 2 / np.sum(w[1:] ** 2)))
        assert max(drrs) - min(drrs) < 2.0

    def test_reflections_pass_through_the_absorption_filter(self):
        """Eq. 12: s' joins group lambda after the delay, so G_lambda shapes it.

        A reflection alone must therefore come out darker than it went in.
        """
        from src.fdn.directional_fdn import DirectionalFDN
        from src.fdn.room import Reflection

        pw = DirectionalFDN(self.FS, 8, sh_order=2, output_sh_order=1,
                           seed=42)
        # A frequency-varying target on the GEQ grid, so G_lambda actually
        # colours: a flat one leaves the reflection a bare impulse.
        pw.setup(delay_range_ms=(20.0, 60.0),
                 t60_per_direction_ms=np.array(
                     [900.0, 900, 1000, 1100, 1000, 800, 700, 600, 500, 300]))
        pw.set_early_reflections(
            [Reflection(gain=1.0, azimuth=0.0, elevation=0.0, delay_samples=500)]
        )
        w = pw.render(0.0, 0.0, 1.0, direct_gain=0.0, reverb_gain=0.0)[0]
        assert np.max(np.abs(w[400:700])) > 1e-6      # the reflection arrives
        assert np.count_nonzero(np.abs(w[400:700]) > 1e-9) > 1   # and is filtered

    def test_reflections_still_excite_the_late_field(self):
        from src.fdn.directional_fdn import DirectionalFDN
        from src.fdn.room import Reflection

        pw = DirectionalFDN(self.FS, 8, sh_order=2, output_sh_order=1,
                           seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
        pw.set_early_reflections(
            [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
        )
        w = pw.render(0.0, 0.0, 1.0, direct_gain=0.0, reverb_gain=0.0)[0]
        assert np.sum(w[:200] ** 2) == pytest.approx(0.0, abs=1e-20)
        assert np.sum(w[5000:] ** 2) > 1e-10


class TestDFDN2019AgainstJAES2019:
    """Alary, Politis, Schlecht and Valimaki, JAES 67(10), 2019.

    Grounds DFDN2019 in the paper it implements.  Verified as already
    correct and left alone: Eq. 20 (A kron I_Q), Eq. 21 (block-diagonal T),
    Eq. 16 (base absorption from T60_max), Eqs. 24-26 (directional gain), and
    the ordering of Eq. 23 -- the paper applies T before g, which commutes
    because g is one scalar per group.
    """

    FS = 48000.0

    def test_direct_gain_is_the_sh_encoding_of_the_direction(self):
        """Eq. 11: d = y(phi_0, theta_0), with no dependence on N.

        A 1/N factor made the direct sound drop 6 dB per doubling of the
        delay-line group count.
        """
        from src.fdn.dfdn_2019 import DFDN2019
        from src.fdn.spatial import encode_directions

        expected = encode_directions(0.0, 0.0, 1)[0]
        for groups in (4, 8, 16):
            dfdn = DFDN2019(self.FS, groups, ambisonics_order=1,
                                  seed=0)
            dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1000.0)
            rir = dfdn.render(0.0, 0.0, length_seconds=0.2)
            assert rir[0, 0] == pytest.approx(expected, rel=1e-9)

    def test_weighting_effective_order(self):
        """Sec 3.2: L' is the order holding 95% of the weighting's energy.

        Sampled over a sphere, as the paper specifies -- a ring of constant
        elevation is rank deficient in the SH basis and reports nonsense.
        """
        from src.fdn import spatial
        from src.fdn.transform import weighting_effective_order

        az, el = spatial.uniform_directions(64)
        assert weighting_effective_order(np.ones(64), az, el) == 0
        dipole = 1.0 + 0.9 * np.cos(az) * np.cos(el)
        assert weighting_effective_order(dipole, az, el) >= 1
        # A realistic decay weighting is dominated by its mean, so L' = 0 and
        # the order-2L grid quadrature_grid() builds is already adequate.
        decay = np.where(np.cos(az) > 0, 0.999, 0.997)
        assert weighting_effective_order(decay, az, el) == 0

    def test_transform_does_not_add_energy(self):
        """Sec 2.4: the transform must not inject energy, or the loop grows."""
        from src.fdn import spatial
        from src.fdn.transform import DirectionalWeighting

        az, el = spatial.uniform_directions(64)
        t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
        dwm = DirectionalWeighting(4, 2, np.array([1039, 1213, 1381, 1523]),
                                   self.FS, t60, az, el)
        for T in dwm.transform_matrices:
            assert np.max(np.abs(np.linalg.eigvals(T))) <= 1.0 + 1e-9

    def test_warns_on_a_degenerate_sampling(self):
        """Sec 3.1 samples over a sphere; a ring spans only part of the basis.

        The channels a ring misses are exactly the ones null_complement passes
        through unweighted, so those directions get no decay control at all.
        """
        from src.fdn.transform import DirectionalWeighting

        az = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        el = np.zeros(12)                     # a ring, not a sphere
        t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
        with pytest.warns(RuntimeWarning, match="span only"):
            DirectionalWeighting(2, 1, np.array([1039, 1213]), self.FS,
                                 t60, az, el)

    def test_a_spherical_sampling_does_not_warn(self):
        import warnings as _warnings

        from src.fdn import spatial
        from src.fdn.transform import DirectionalWeighting

        az, el = spatial.uniform_directions(64)
        t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
        with _warnings.catch_warnings():
            _warnings.simplefilter("error", RuntimeWarning)
            DirectionalWeighting(2, 1, np.array([1039, 1213]), self.FS,
                                 t60, az, el)

    def test_anisotropy_is_realised(self):
        """Fig. 3: the realised directional decay tracks the target."""
        import pyFDN

        from src.fdn import spatial
        from src.fdn.dfdn_2019 import DFDN2019

        order = 2
        # Sec 3.1 samples T60 "using a uniform set of points on a sphere, such
        # as spherical t-designs of high degrees (t >= 20)".  A sparse grid
        # cannot carry a sharp target: the per-line weighting is g_lin**m, so a
        # long delay raises the directional function to a high power and
        # sharpens it, pushing L' up with it.
        az, el = spatial.quadrature_grid(10)
        t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
        dfdn = DFDN2019(self.FS, 8, ambisonics_order=order,
                              seed=42)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
        dfdn.set_directional_decay(t60, az, el)
        rir = dfdn.render(0.0, 0.0, length_seconds=6.0)

        def beam_t60(direction):
            y = spatial.encode_directions(direction, 0.0, order)
            e = pyFDN.edc(y @ rir)
            e = e[e > 0]
            db = 10 * np.log10(e / e[0])
            i0, i1 = np.argmax(db < -5), np.argmax(db < -35)
            t = np.arange(i1 - i0) / self.FS
            return -60 / np.polyfit(t, db[i0:i1] - db[i0], 1)[0]

        assert beam_t60(0.0) > 1.7               # front stays long
        assert beam_t60(np.pi) < 0.9             # back decays fast
        assert beam_t60(0.0) > 3.0 * beam_t60(np.pi)


class TestEnergyDecayDeviation:
    """EDD of Alary et al., J. Acoust. Soc. Am. 149(5), 2021, Eqs. 4-7."""

    FS = 48000.0

    def _decaying_noise(self, t60, seconds=2.0, seed=0):
        n = int(seconds * self.FS)
        t = np.arange(n) / self.FS
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) * 10.0 ** (-3.0 * t / t60)

    def test_deviations_sum_to_zero_at_every_instant(self):
        """EDD is a deviation from the across-direction mean, by definition."""
        from src.utils import energy_decay_deviation

        irs = np.stack([
            self._decaying_noise(t60, seed=i)
            for i, t60 in enumerate((0.6, 1.0, 1.4))
        ])
        edd = energy_decay_deviation(irs)
        np.testing.assert_allclose(edd.sum(axis=0), 0.0, atol=1e-9)

    def test_shape_matches_input(self):
        from src.utils import energy_decay_deviation

        irs = np.stack([self._decaying_noise(1.0, seed=i) for i in range(4)])
        assert energy_decay_deviation(irs).shape == irs.shape

    def test_isotropic_field_has_no_deviation(self):
        from src.utils import energy_decay_deviation

        irs = np.stack([self._decaying_noise(1.0, seed=i) for i in range(6)])
        edd = energy_decay_deviation(irs)
        half = edd.shape[1] // 2
        assert np.max(np.abs(edd[:, :half])) < 1.0

    def test_slower_direction_gains_energy_over_time(self):
        """A longer T60 deviates further from the mean as the tail develops."""
        from src.utils import energy_decay_deviation

        irs = np.stack([
            self._decaying_noise(0.5, seed=0),
            self._decaying_noise(1.5, seed=1),
        ])
        edd = energy_decay_deviation(irs)
        quarter, half = edd.shape[1] // 4, edd.shape[1] // 2
        assert edd[1, quarter] > 0.0
        assert edd[1, half] > edd[1, quarter]
        assert edd[0, half] < edd[0, quarter]

    def test_rejects_a_single_direction(self):
        """One direction is its own mean, so the deviation is undefined."""
        from src.utils import energy_decay_deviation

        with pytest.raises(ValueError, match="two directions"):
            energy_decay_deviation(self._decaying_noise(1.0)[None, :])


class TestGridOutput:
    """Raw grid channels, to tell synthesis error from encoder smoothing."""

    FS = 48000.0

    def _dfdn(self):
        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 4, sh_order=1, output_sh_order=1,
                           seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1000.0)
        return pw

    def test_returns_one_channel_per_grid_direction(self):
        pw = self._dfdn()
        assert pw.render(0.0, 0.0, 0.5, grid_output=True).shape[0] == pw.Q

    def test_encoding_the_grid_output_reproduces_the_ambisonic_render(self):
        """The diagnostic must tap the real signal path, not a parallel one."""
        pw = self._dfdn()
        grid = pw.render(0.0, 0.0, 0.5, grid_output=True)
        ambi = pw.render(0.0, 0.0, 0.5)
        np.testing.assert_allclose(
            pw._encoding_matrix @ grid, ambi, atol=1e-9
        )

    def test_grid_channel_decay_matches_a_direction_dependent_target(self):
        """Without the encoder, each grid channel keeps its own T60."""
        import pyFDN

        from src.fdn.directional_fdn import DirectionalFDN

        pw = DirectionalFDN(self.FS, 8, sh_order=1, output_sh_order=1,
                           seed=42)
        pw._init_grid()
        scale = 1.0 - 0.5 * np.clip(np.sin(pw.grid_elevations), 0.0, None)
        pw.setup(delay_range_ms=(20.0, 60.0),
                 t60_per_direction_ms=1200.0 * scale)
        grid = pw.render(0.0, 0.0, 3.0, direct_gain=0.0, grid_output=True)

        slowest, fastest = int(np.argmax(scale)), int(np.argmin(scale))
        rt_slow, _ = pyFDN.estimate_rt_bands(grid[slowest], self.FS, start=-3)
        rt_fast, _ = pyFDN.estimate_rt_bands(grid[fastest], self.FS, start=-3)
        ratio = np.median(rt_fast[1:-1]) / np.median(rt_slow[1:-1])
        assert ratio == pytest.approx(scale[fastest] / scale[slowest], rel=0.2)
