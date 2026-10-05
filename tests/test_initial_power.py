"""Parallel early-reflection path and per-direction initial-power shaping."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import band_energy, decay_fit
from src.directional_fdn import DirectionalFDN
from src.fdn_components import geq_band_frequencies
from src.room import Reflection

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


def _late_w(pw, start_s=0.08, length_s=3.0):
    w = pw.render(0.0, 0.0, length_s, direct_gain=0.0, er_gain=0.0)[0]
    return band_energy(w, FS, BANDS, start_s)[:, 0]


@pytest.mark.parametrize(
    ("gains", "tol"),
    [
        (np.array([8.0, 6, 2, 0, 0, -1, -2, -4]), 0.2),
        #band-to-band jumps; probe after 3 passes: 0.26 dB
        (np.array([3.0, -2, 0, 4, -1, 2, -3, 1]), 0.5),
    ],
)
def test_matching_reaches_reference(gains, tol):
    target = _late_w(_make()) * 10.0 ** (gains / 10.0)
    pw = _make()
    pw.match_late_level(target, BANDS, window_start_s=0.08, length_s=3.0)
    err = 10.0 * np.log10(_late_w(pw) / target)
    assert np.all(np.abs(err) < tol), err


def test_rematching_does_not_stack():
    target = _late_w(_make()) * 2.0
    pw = _make()
    first = pw.match_late_level(target, BANDS, window_start_s=0.08, length_s=3.0)
    second = pw.match_late_level(target, BANDS, window_start_s=0.08, length_s=3.0)
    assert np.allclose(first, second)


def test_parallel_reflections_do_not_bias_matching():
    target = _late_w(_make()) * 2.0
    pw = _make()
    pw.set_early_reflections(REFLECTIONS, injected=False)
    corr = pw.match_late_level(target, BANDS, window_start_s=0.08, length_s=3.0)
    assert np.allclose(corr, 10.0 * np.log10(2.0), atol=0.05)


def test_matching_with_injected_reflections_raises():
    pw = _make()
    pw.set_early_reflections(REFLECTIONS)
    with pytest.raises(ValueError, match="injected"):
        pw.match_late_level(np.ones(8), BANDS, window_start_s=0.08, length_s=1.0)


def test_matching_rejects_off_grid_bands():
    with pytest.raises(ValueError, match="geq_band_frequencies"):
        _make().match_late_level(
            np.ones(3), np.array([100.0, 200.0, 400.0]), window_start_s=0.08, length_s=1.0
        )


def test_setup_resets_initial_power_shaping():
    pw = _make()
    pw.set_initial_power(6.0)
    pw.setup(
        delay_range_ms=(5.0, 25.0), t60_per_direction_ms=T60_CMD, feedback_type="hadamard"
    )
    assert np.allclose(pw.render(0.3, 0.1, 0.3), _make().render(0.3, 0.1, 0.3))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_set_initial_power_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        _make().set_initial_power(bad)


@pytest.mark.parametrize(
    "target", [np.zeros(8), -np.ones(8), np.full(8, np.nan), np.ones(5)],
    ids=["zero", "negative", "nan", "wrong-shape"],
)
def test_matching_rejects_unusable_targets(target):
    with pytest.raises(ValueError, match="target_energy"):
        _make().match_late_level(target, BANDS, window_start_s=0.08, length_s=1.0)


def test_matching_rejects_a_window_past_the_render():
    pw = _make()
    with pytest.raises(ValueError, match="no energy"):
        pw.match_late_level(np.full(8, 1e-4), BANDS, window_start_s=5.0, length_s=1.0)
    assert pw._p0_sos is None


def test_matching_warns_on_an_implausible_level_convention():
    with pytest.warns(RuntimeWarning, match="level convention"):
        _make().match_late_level(
            np.full(8, 1e30), BANDS, window_start_s=0.08, length_s=1.0
        )


def test_matching_supports_a_single_band():
    pw = _make()
    target = _late_w(_make())[4:5] * 2.0
    pw.match_late_level(target, BANDS[4:5], window_start_s=0.08, length_s=3.0)
    err = 10.0 * np.log10(_late_w(pw)[4] / target[0])
    assert abs(err) < 0.3
