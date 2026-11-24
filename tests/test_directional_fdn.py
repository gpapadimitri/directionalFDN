"""DirectionalFDN, the grid network of Alary and Politis (ICASSP 2020)."""

from __future__ import annotations

import numpy as np
import pyFDN
import pytest

from src import spatial
from src.air import Atmosphere
from src.analysis import band_energy, decay_fit, predicted_beam_decay, sh_to_beams
from src.directional_fdn import DirectionalFDN
from src.fdn_components import FeedbackMatrixType, geq_band_frequencies
from src.room import Reflection

FS = 48000
BANDS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
T60_CMD = np.array([1200.0, 1200, 1200, 1100, 1000, 900, 800, 700, 600, 600])
REFLECTIONS = [
    Reflection(gain=0.3, azimuth=1.0, elevation=0.1, delay_samples=150),
    Reflection(gain=0.2, azimuth=-2.0, elevation=-0.3, delay_samples=400),
]


def _make(t60_ms=T60_CMD, seed=3, sh_order=3, groups=4, **setup):
    pw = DirectionalFDN(FS, groups, sh_order=sh_order, seed=seed)
    pw.setup(
        delay_range_ms=(5.0, 25.0), t60_per_direction_ms=t60_ms, feedback_type="hadamard",
        **setup,
    )
    return pw


def _late_w(pw, start_s=0.08, length_s=3.0):
    w = pw.render(0.0, 0.0, length_s, direct_gain=0.0, er_gain=0.0)[0]
    return band_energy(w, FS, BANDS, start_s)[:, 0]


def test_grid_t30_matches_target():
    grid = _make().render(0.0, 0.0, 2.5, direct_gain=0.0, grid_output=True)
    t30, _ = decay_fit(grid, FS, BANDS, "T30")
    rel = np.median(t30, axis=0) / (T60_CMD[1:9] * 1e-3) - 1.0
    #probe, seeds 3-5: within 3 %; the 8 kHz octave integrates a falling target
    assert np.all(np.abs(rel) < 0.05), rel


def test_each_grid_channel_keeps_its_own_t60():
    pw = DirectionalFDN(FS, 8, sh_order=1, seed=42)
    pw._init_grid()
    scale = 1.0 - 0.5 * np.clip(np.sin(pw.grid_elevations), 0.0, None)
    pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1200.0 * scale)
    grid = pw.render(0.0, 0.0, 3.0, direct_gain=0.0, grid_output=True)
    slowest, fastest = int(np.argmax(scale)), int(np.argmin(scale))
    rt_slow, _ = pyFDN.estimate_rt_bands(grid[slowest], FS, lowest_octave=-3)
    rt_fast, _ = pyFDN.estimate_rt_bands(grid[fastest], FS, lowest_octave=-3)
    ratio = np.median(rt_fast[1:-1]) / np.median(rt_slow[1:-1])
    assert ratio == pytest.approx(scale[fastest] / scale[slowest], rel=0.2)


def test_beam_t30_matches_prediction_not_the_map():
    az, el = spatial.quadrature_grid(3)
    front = np.cos(az) * np.cos(el) > 0
    t60_q = np.where(front, 600.0, 1500.0)
    pw = _make(t60_q)
    sh = pw.render(0.0, 0.0, 3.0, direct_gain=0.0)
    measured, _ = decay_fit(sh_to_beams(sh, az, el), FS, BANDS, "T30")
    predicted = predicted_beam_decay(
        sh_to_beams(pw.encoding_matrix, az, el),
        t60_q[:, np.newaxis], np.zeros((az.size, 1)), FS, sh.shape[-1] / FS,
    )
    rel = measured[:, 4:7] / predicted - 1.0
    #probe at N = 4, seeds 3-5: median 0.6-0.9 %, p90 2.2-3.0 %
    assert np.median(np.abs(rel)) < 0.03, np.median(np.abs(rel))
    #order-3 front beams read about 1.38 s against their 0.6 s target
    assert np.median(predicted[front, 0]) > 1.2


@pytest.mark.parametrize("kind", list(FeedbackMatrixType))
def test_feedback_is_lossless_and_never_couples_directions(kind):
    pw = DirectionalFDN(FS, 4, sh_order=1, seed=42)
    pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=1500.0, feedback_type=kind)
    assert pyFDN.is_unilossless(pw._base_feedback.matrix, tol=1e-8)
    Q, A = pw.Q, pw._kronecker_matrix
    for i in range(pw.num_delay_line_groups):
        for j in range(pw.num_delay_line_groups):
            block = A[i * Q:(i + 1) * Q, j * Q:(j + 1) * Q]
            np.testing.assert_allclose(block[~np.eye(Q, dtype=bool)], 0.0, atol=1e-12)


def test_grid_output_encodes_to_the_ambisonic_render():
    pw = _make(sh_order=1)
    grid = pw.render(0.0, 0.0, 0.5, grid_output=True)
    ambi = pw.render(0.0, 0.0, 0.5)
    np.testing.assert_allclose(pw.encoding_matrix @ grid, ambi, atol=1e-9)


def test_direct_sound_is_the_gain_in_w():
    #with no reverb feed sample 0 is the direct path alone, and SN3D W is 1 everywhere
    rir = _make(sh_order=1).render(0.7, 0.2, 0.5, direct_gain=0.5, reverb_gain=0.0)
    assert rir[0, 0] == pytest.approx(0.5, rel=1e-6)


def test_same_seed_is_bit_identical():
    a = _make(seed=11).render(0.3, 0.1, 0.5)
    b = _make(seed=11).render(0.3, 0.1, 0.5)
    assert np.array_equal(a, b)


def test_injected_reflections_arrive_on_time():
    pw = _make(sh_order=1)
    pw.set_early_reflections(
        [Reflection(gain=0.5, azimuth=0.0, elevation=0.0, delay_samples=200)]
    )
    rir = pw.render(0.0, 0.0, 0.5, direct_gain=0.0, reverb_gain=0.0)
    assert np.abs(rir[0, 200]) > 1e-3
    assert np.all(rir[:, :200] == 0.0)


def test_parallel_reflections_add_alongside_the_network():
    plain = _make(sh_order=2).render(0.3, 0.1, 0.5)
    pw = _make(sh_order=2)
    pw.set_early_reflections(REFLECTIONS, injected=False)
    out = pw.render(0.3, 0.1, 0.5)
    impulse = np.zeros((1, out.shape[1]))
    impulse[0, 0] = 1.0
    er = pw.encoding_matrix @ pw._build_er_grid_signal(impulse, out.shape[1])
    assert np.allclose(out, plain + er)


def test_initial_power_tilt_leaves_the_decay_alone():
    tilt = np.linspace(6.0, -6.0, 10)
    base = _make(sh_order=2).render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    pw = _make(sh_order=2)
    pw.set_initial_power(tilt)
    shaped = pw.render(0.0, 0.0, 2.0, direct_gain=0.0, grid_output=True)
    t0, p0 = decay_fit(base, FS, BANDS, "T20")
    t1, p1 = decay_fit(shaped, FS, BANDS, "T20")
    expected = np.interp(np.log2(BANDS), np.log2(geq_band_frequencies()), tilt)
    #probe: within 0.12 dB per band, T20 within 0.4 %
    assert np.all(np.abs(np.median(p1 - p0, axis=0) - expected) < 0.5)
    assert np.all(np.abs(np.median(t1 / t0, axis=0) - 1.0) < 0.02)


def test_late_level_matching_reaches_the_reference():
    gains = np.array([8.0, 6, 2, 0, 0, -1, -2, -4])
    target = _late_w(_make(sh_order=2)) * 10.0 ** (gains / 10.0)
    pw = _make(sh_order=2)
    pw.match_late_level(target, BANDS, window_start_s=0.08, length_s=3.0)
    err = 10.0 * np.log10(_late_w(pw) / target)
    assert np.all(np.abs(err) < 0.2), err


def test_air_absorption_follows_the_sabine_4mv_term():
    freqs = geq_band_frequencies()
    volume, surface, alpha_boundary = 320.0, 304.0, 0.1857
    sabine_dry = 0.161 * volume / (surface * alpha_boundary)
    m = Atmosphere().alpha_db_per_m(freqs) / (10.0 * np.log10(np.e))
    sabine_with_air = 0.161 * volume / (surface * alpha_boundary + 4.0 * m * volume)
    pw = _make(np.full(freqs.size, sabine_dry * 1e3), sh_order=1, air_absorption=True)
    np.testing.assert_allclose(pw.realised_t60_s[0], sabine_with_air, rtol=1e-3)
    required = 60.0 * pw.delay_length_samps[:, np.newaxis] / (FS * pw.realised_t60_s)
    np.testing.assert_allclose(
        pw.air_attenuation_db + pw.boundary_attenuation_db, required, rtol=1e-10
    )


def test_air_absorption_shortens_the_high_band_decay():
    t60 = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.2]) * 1e3
    rt = {}
    for air in (False, True):
        pw = DirectionalFDN(FS, 8, sh_order=1, seed=42)
        pw.setup(delay_range_ms=(20.0, 60.0), t60_per_direction_ms=t60, air_absorption=air)
        w = pw.render(0.0, 0.0, 4.0, direct_gain=0.0)[0]
        rt[air], fc = pyFDN.estimate_rt_bands(w, FS, num_bands=8, lowest_octave=-4)
    top, low = int(np.argmin(np.abs(fc - 8000.0))), int(np.argmin(np.abs(fc - 125.0)))
    assert rt[True][top] < rt[False][top]
    assert rt[True][low] == pytest.approx(rt[False][low], rel=0.05)


@pytest.mark.parametrize(
    "t60", [-800.0, np.full(10, -1200.0), np.full(10, np.nan), np.zeros(10)],
    ids=["neg-scalar", "neg-bands", "nan", "zero-bands"],
)
def test_setup_rejects_a_non_positive_or_non_finite_t60(t60):
    with pytest.raises(ValueError, match="finite and positive"):
        DirectionalFDN(FS, 4, sh_order=1, seed=1).setup((10, 30), t60)
