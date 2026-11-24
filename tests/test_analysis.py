"""Directional decay, level and correlation analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src import spatial
from src.analysis import (
    band_correlation,
    decay_fit,
    energy_decay_deviation,
    mean_offdiag_abs,
    predicted_beam_decay,
    sh_to_beams,
)

FS = 48000
BANDS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
#below 250 Hz a single octave-band realisation is too noisy to pin
FIT = slice(2, None)


def _decaying_noise(n_channels, t60_s, length_s, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(int(length_s * FS))
    return rng.standard_normal((n_channels, n.size)) * 10.0 ** (-3.0 * n / (FS * t60_s))


#EDT fits only 10 dB, so its medians scatter most (probe: <= 10.2 % from 125 Hz)
@pytest.mark.parametrize(
    ("parameter", "tol"), [("EDT", 0.12), ("T20", 0.06), ("T30", 0.03)]
)
def test_decay_fit_recovers_t60(parameter, tol):
    t60, _ = decay_fit(_decaying_noise(32, 1.0, 2.0, seed=1), FS, BANDS, parameter)
    rel = np.median(t60, axis=0)[FIT] - 1.0
    assert np.all(np.abs(rel) < tol), rel


def test_decay_fit_p0_is_the_psd_of_white_noise():
    _, p0 = decay_fit(_decaying_noise(32, 1.0, 2.0, seed=2), FS, BANDS)
    err = np.median(p0, axis=0)[FIT] - 10.0 * np.log10(2.0 / FS)
    assert np.all(np.abs(err) < 1.0), err


def test_decay_fit_is_nan_when_truncation_dominates():
    t60, p0 = decay_fit(_decaying_noise(2, 5.0, 0.5, seed=3), FS, BANDS, "T30")
    assert np.all(np.isnan(t60)) and np.all(np.isnan(p0))


def test_band_correlation_of_identical_and_independent_channels():
    x = _decaying_noise(1, 1.0, 1.0, seed=6)
    assert np.allclose(band_correlation(np.vstack([x, x]), FS, BANDS)[:, 0, 1], 1.0)
    m = mean_offdiag_abs(band_correlation(_decaying_noise(8, 1.0, 1.0, seed=7), FS, BANDS))
    assert np.all(m[4:] < 0.1), m


def test_sh_to_beams_has_unit_gain_toward_a_plane_wave():
    az, el = spatial.quadrature_grid(3)
    sh = spatial.encode_directions(0.7, 0.3, 3)[:, np.newaxis]
    beams = sh_to_beams(sh, np.r_[0.7, az], np.r_[0.3, el])
    assert np.isclose(beams[0, 0], 1.0)
    assert np.all(beams[1:, 0] < 1.0)


def test_predicted_beam_decay_with_identity_beams_returns_target():
    t60 = np.array([[500.0], [1000.0], [2000.0]])
    pred = predicted_beam_decay(np.eye(3), t60, np.zeros_like(t60), FS, 6.0)
    assert np.allclose(pred[:, 0], [0.5, 1.0, 2.0], rtol=0.01)


def test_energy_decay_deviation_sums_to_zero_and_tracks_the_slower_direction():
    fast, slow = _decaying_noise(1, 0.5, 2.0, seed=0), _decaying_noise(1, 1.5, 2.0, seed=1)
    irs = np.vstack([fast, slow])
    edd = energy_decay_deviation(irs)
    np.testing.assert_allclose(edd.sum(axis=0), 0.0, atol=1e-9)
    quarter, half = edd.shape[1] // 4, edd.shape[1] // 2
    assert 0.0 < edd[1, quarter] < edd[1, half]
