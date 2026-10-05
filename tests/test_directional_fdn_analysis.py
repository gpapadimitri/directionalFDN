"""DirectionalFDN measured with the analysis toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from src import spatial
from src.analysis import (
    band_correlation,
    decay_fit,
    mean_offdiag_abs,
    predicted_beam_decay,
    sh_to_beams,
)
from src.directional_fdn import DirectionalFDN

FS = 48000
BANDS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def _make(t60_ms, seed=3, sh_order=3, groups=4):
    pw = DirectionalFDN(FS, groups, sh_order=sh_order, seed=seed)
    pw.setup(
        delay_range_ms=(5.0, 25.0), t60_per_direction_ms=t60_ms, feedback_type="hadamard"
    )
    return pw


def test_encoding_matrix_property():
    pw = _make(800.0)
    assert pw.encoding_matrix.shape == (16, pw.Q)
    with pytest.raises(RuntimeError, match="setup"):
        _ = DirectionalFDN(FS, 4, sh_order=3, seed=0).encoding_matrix


def test_beam_t30_matches_prediction_not_the_map():
    az, el = spatial.quadrature_grid(3)
    front = np.cos(az) * np.cos(el) > 0
    t60_q = np.where(front, 600.0, 1500.0)
    pw = _make(t60_q)
    sh = pw.render(0.0, 0.0, 3.0, direct_gain=0.0)
    measured, _ = decay_fit(sh_to_beams(sh, az, el), FS, BANDS, "T30")
    predicted = predicted_beam_decay(
        sh_to_beams(pw.encoding_matrix, az, el),
        t60_q[:, np.newaxis],
        np.zeros((az.size, 1)),
        FS,
        sh.shape[-1] / FS,
    )
    rel = measured[:, 4:7] / predicted - 1.0
    #probe at N = 4, seeds 3-5: median 0.6-0.9 %, p90 2.2-3.0 %
    assert np.median(np.abs(rel)) < 0.03, np.median(np.abs(rel))
    #order-3 front beams read about 1.38 s against their 0.6 s target
    assert np.median(predicted[front, 0]) > 1.2


T60_CMD = np.array([1200.0, 1200, 1200, 1100, 1000, 900, 800, 700, 600, 600])


def test_grid_channels_are_uncorrelated():
    #Gallien et al. 2026 Sec. 3.2.2 matching to uncorrelated plane waves would be
    #the identity: the per-direction loops sit at the estimation floor
    grid = _make(T60_CMD).render(0.0, 0.0, 1.5, direct_gain=0.0, grid_output=True)
    rng = np.random.default_rng(7)
    n = np.arange(grid.shape[1])
    floor = rng.standard_normal(grid.shape) * 10.0 ** (-3.0 * n / FS)
    ratio = mean_offdiag_abs(band_correlation(grid, FS, BANDS)) / mean_offdiag_abs(
        band_correlation(floor, FS, BANDS)
    )
    #probe, seeds 3-5: 0.79-1.35
    assert np.all(ratio < 1.5), ratio


def test_natural_initial_power_is_flat():
    grid = _make(T60_CMD).render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    _, p0 = decay_fit(grid, FS, BANDS, "T20")
    med = np.nanmedian(p0, axis=0)[2:]
    #probe, seeds 3-5: 0.58-1.04 dB from 250 Hz
    assert np.ptp(med) < 1.5, med


def test_grid_t30_matches_target():
    grid = _make(T60_CMD).render(0.0, 0.0, 2.5, direct_gain=0.0, grid_output=True)
    t30, _ = decay_fit(grid, FS, BANDS, "T30")
    rel = np.median(t30, axis=0) / (T60_CMD[1:9] * 1e-3) - 1.0
    #probe, seeds 3-5: within 3 %; the 8 kHz octave integrates a falling target
    assert np.all(np.abs(rel) < 0.05), rel


def test_same_seed_is_bit_identical():
    a = _make(T60_CMD, seed=11).render(0.3, 0.1, 0.5)
    b = _make(T60_CMD, seed=11).render(0.3, 0.1, 0.5)
    assert np.array_equal(a, b)
