"""Tests for the directional analysis toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from src import spatial
from src.analysis import (
    band_correlation,
    band_edc_db,
    band_energy,
    decay_fit,
    mean_offdiag_abs,
    octave_bands,
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
    assert t60.shape == (32, 8)
    rel = np.median(t60, axis=0)[FIT] - 1.0
    assert np.all(np.abs(rel) < tol), rel


def test_decay_fit_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="parameter"):
        decay_fit(_decaying_noise(1, 1.0, 1.0), FS, BANDS, "C80")


def test_decay_fit_p0_is_flat_psd_of_white_noise():
    _, p0 = decay_fit(_decaying_noise(32, 1.0, 2.0, seed=2), FS, BANDS)
    err = np.median(p0, axis=0)[FIT] - 10.0 * np.log10(2.0 / FS)
    assert np.all(np.abs(err) < 1.0), err


@pytest.mark.parametrize("parameter", ["T20", "T30"])
def test_decay_fit_is_nan_when_truncation_dominates(parameter):
    t60, p0 = decay_fit(_decaying_noise(2, 5.0, 0.5, seed=3), FS, BANDS, parameter)
    assert np.all(np.isnan(t60)) and np.all(np.isnan(p0))


def test_non_octave_bands_raise():
    with pytest.raises(ValueError, match="consecutive octave"):
        decay_fit(_decaying_noise(1, 1.0, 1.0), FS, np.array([125.0, 500.0, 1000.0]))


def test_band_edc_db_starts_at_zero_and_decays():
    edc = band_edc_db(_decaying_noise(3, 1.0, 1.0), FS, BANDS)
    assert edc.shape == (8, 3, FS)
    assert np.allclose(edc[..., 0], 0.0)
    assert np.all(edc[FIT, :, FS // 2] < -20.0)


def test_band_energy_of_a_scaled_copy_scales():
    x = _decaying_noise(1, 1.0, 1.0, seed=4)
    e1, e2 = band_energy(x, FS, BANDS, 0.08), band_energy(2.0 * x, FS, BANDS, 0.08)
    assert e1.shape == (8, 1)
    assert np.allclose(e2, 4.0 * e1)


def test_band_energy_ignores_everything_before_the_window():
    x = _decaying_noise(1, 1.0, 1.0, seed=5)
    y = x.copy()
    y[:, :100] *= 50.0
    assert np.allclose(band_energy(x, FS, BANDS, 0.08), band_energy(y, FS, BANDS, 0.08))


def test_band_correlation_identical_and_independent_channels():
    x = _decaying_noise(1, 1.0, 1.0, seed=6)
    assert np.allclose(band_correlation(np.vstack([x, x]), FS, BANDS)[:, 0, 1], 1.0)
    m = mean_offdiag_abs(band_correlation(_decaying_noise(8, 1.0, 1.0, seed=7), FS, BANDS))
    assert m.shape == (8,)
    assert np.all(m[4:] < 0.1), m


def test_sh_to_beams_has_unit_gain_toward_a_plane_wave():
    az, el = spatial.quadrature_grid(3)
    sh = spatial.encode_directions(0.7, 0.3, 3)[:, np.newaxis]
    beams = sh_to_beams(sh, np.r_[0.7, az], np.r_[0.3, el])
    assert beams.shape == (1 + az.size, 1)
    assert np.isclose(beams[0, 0], 1.0)
    assert np.all(beams[1:, 0] < 1.0)


def test_predicted_beam_decay_with_identity_beams_returns_target():
    t60 = np.array([[500.0], [1000.0], [2000.0]])
    pred = predicted_beam_decay(np.eye(3), t60, np.zeros_like(t60), FS, 6.0)
    assert np.allclose(pred[:, 0], [0.5, 1.0, 2.0], rtol=0.01)


def test_predicted_beam_decay_sidelobe_pulls_toward_slow_direction():
    t60 = np.array([[600.0], [1500.0]])
    pred = predicted_beam_decay(np.array([[1.0, 0.3]]), t60, np.zeros_like(t60), FS, 4.0)
    #a -10 dB sidelobe on the slow direction owns most of the T30 window
    assert 0.9 < pred[0, 0] < 1.5


def test_single_band_is_supported():
    x = _decaying_noise(2, 1.0, 1.0, seed=8)
    assert octave_bands(x, FS, np.array([1000.0])).shape == (1, 2, FS)
    assert band_energy(x, FS, np.array([1000.0]), 0.01).shape == (1, 2)


def test_silent_channel_gives_nan_not_an_exception():
    x = np.vstack([np.zeros(FS * 2), _decaying_noise(1, 1.0, 2.0, seed=9)[0]])
    t60, p0 = decay_fit(x, FS, BANDS, "T30")
    assert np.all(np.isnan(t60[0])) and np.all(np.isnan(p0[0]))
    assert np.all(np.isfinite(t60[1, 2:]))


def test_predicted_beam_decay_rejects_a_non_positive_t60():
    with pytest.raises(ValueError, match="t60_ms"):
        predicted_beam_decay(np.eye(2), np.zeros((2, 1)), np.zeros((2, 1)), FS, 2.0)


def test_sh_to_beams_rejects_a_channel_count_that_is_not_a_square():
    with pytest.raises(ValueError, match=r"\(L\+1\)\^2"):
        sh_to_beams(np.zeros((10, 100)), np.zeros(3), np.zeros(3))
