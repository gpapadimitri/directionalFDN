"""DirectionalFDN measured with the analysis toolkit."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import decay_fit, predicted_beam_decay, sh_to_beams
from src.fdn import spatial
from src.fdn.directional_fdn import DirectionalFDN

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
