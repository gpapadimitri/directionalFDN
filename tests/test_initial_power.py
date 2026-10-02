"""Parallel early-reflection path and per-direction initial-power shaping."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import decay_fit
from src.fdn.directional_fdn import DirectionalFDN
from src.fdn.fdn_components import geq_band_frequencies
from src.fdn.room import Reflection

FS = 48000
BANDS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
T60_CMD = np.array([1200.0, 1200, 1200, 1100, 1000, 900, 800, 700, 600, 600])
TILT = np.linspace(6.0, -6.0, 10)
REFLECTIONS = [
    Reflection(gain=0.3, azimuth=1.0, elevation=0.1, delay_samples=150),
    Reflection(gain=0.2, azimuth=-2.0, elevation=-0.3, delay_samples=400),
]


def _make(seed=3):
    pw = DirectionalFDN(FS, 4, sh_order=2, seed=seed)
    pw.setup(
        delay_range_ms=(5.0, 25.0), t60_per_direction_ms=T60_CMD, feedback_type="hadamard"
    )
    return pw


def _impulse(n):
    x = np.zeros((1, n))
    x[0, 0] = 1.0
    return x


def test_parallel_reflections_add_alongside_the_network():
    plain = _make().render(0.3, 0.1, 0.5)
    pw = _make()
    pw.set_early_reflections(REFLECTIONS, injected=False)
    out = pw.render(0.3, 0.1, 0.5)
    er = pw.encoding_matrix @ pw._build_er_grid_signal(_impulse(out.shape[1]), out.shape[1])
    assert np.allclose(out, plain + er)


def test_zero_db_initial_power_is_identity():
    pw = _make()
    pw.set_initial_power(0.0)
    assert np.allclose(pw.render(0.3, 0.1, 0.5), _make().render(0.3, 0.1, 0.5), atol=1e-12)


def test_tilt_is_realised_without_changing_decay():
    base = _make().render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    pw = _make()
    pw.set_initial_power(TILT)
    shaped = pw.render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    t0, p0 = decay_fit(base, FS, BANDS, "T20")
    t1, p1 = decay_fit(shaped, FS, BANDS, "T20")
    expected = np.interp(np.log2(BANDS), np.log2(geq_band_frequencies()), TILT)
    #probe: within 0.12 dB per band, T20 within 0.4 %
    assert np.all(np.abs(np.median(p1 - p0, axis=0) - expected) < 0.5)
    assert np.all(np.abs(np.median(t1 / t0, axis=0) - 1.0) < 0.02)


def test_broadband_gain_per_direction():
    pw = _make()
    az, el = pw.grid_azimuths, pw.grid_elevations
    front = np.cos(az) * np.cos(el) > 0
    pw.set_initial_power(np.where(front, 6.0, 0.0))
    shaped = pw.render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    base = _make().render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    _, p_s = decay_fit(shaped, FS, BANDS, "T20")
    _, p_b = decay_fit(base, FS, BANDS, "T20")
    change = (p_s - p_b)[:, 4]
    assert np.allclose(change[front], 6.0, atol=0.2)
    assert np.allclose(change[~front], 0.0, atol=0.2)


def test_initial_power_leaves_direct_and_reflections_untouched():
    outs = []
    for p0 in (None, -20.0):
        pw = _make()
        pw.set_early_reflections(REFLECTIONS, injected=False)
        pw.set_initial_power(p0)
        outs.append(pw.render(0.3, 0.1, 0.3, direct_gain=0.2))
    #delay lines are >= 5 ms (240 samples), so the first 200 samples hold only
    #the direct sound and the reflections
    assert np.allclose(outs[0][:, :200], outs[1][:, :200])
    assert not np.allclose(outs[0], outs[1])


def test_initial_power_with_injected_reflections_raises():
    pw = _make()
    pw.set_early_reflections(REFLECTIONS)
    pw.set_initial_power(3.0)
    with pytest.raises(ValueError, match="injected"):
        pw.render(0.0, 0.0, 0.2)


def test_grid_output_encodes_to_sh_output_on_the_new_path():
    pw = _make()
    pw.set_early_reflections(REFLECTIONS, injected=False)
    pw.set_initial_power(TILT)
    grid = pw.render(0.3, 0.1, 0.3, grid_output=True)
    assert np.allclose(pw.encoding_matrix @ grid, pw.render(0.3, 0.1, 0.3))


def test_set_initial_power_shapes():
    pw = _make()
    for p0 in (2.0, TILT, np.zeros(pw.Q), np.zeros((pw.Q, 10))):
        pw.set_initial_power(p0)
    with pytest.raises(ValueError, match="p0_db"):
        pw.set_initial_power(np.zeros(7))
    with pytest.raises(RuntimeError, match="setup"):
        DirectionalFDN(FS, 4, sh_order=2, seed=0).set_initial_power(0.0)
