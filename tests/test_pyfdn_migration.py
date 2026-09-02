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
