"""Delay lengths, absorption, spherical harmonics, VBAP and image sources."""

from __future__ import annotations

from math import gcd

import numpy as np
import pytest
from scipy.signal import sosfreqz

from src import spatial
from src.fdn_components import (
    design_absorption_sos,
    generate_delay_lengths,
    geq_band_frequencies,
)
from src.room import shoebox_reflections

FS = 48000.0


def test_delay_lengths_are_coprime_and_in_range():
    lengths = generate_delay_lengths((20.0, 60.0), 16, FS, rng=0)
    assert lengths.size == 16
    assert np.all((lengths >= 960) & (lengths <= 2880))
    pairs = [(int(a), int(b)) for i, a in enumerate(lengths) for b in lengths[i + 1:]]
    assert all(gcd(a, b) == 1 for a, b in pairs)


def test_delay_lengths_fall_back_to_distinct_when_coprime_is_infeasible():
    lengths = generate_delay_lengths((20.0, 60.0), 336, FS, rng=0)
    assert np.unique(lengths).size == 336
    assert np.all((lengths >= 960) & (lengths <= 2880))


def test_absorption_realises_the_t60_at_every_command_frequency():
    m = 1500.0
    t60 = np.array([1.8, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45, 0.3])
    sos = design_absorption_sos(t60, np.array([m]), FS)
    _, h = sosfreqz(sos[:, :, 0], worN=2 * np.pi * geq_band_frequencies() / FS)
    realised = -60.0 * m / (FS * 20 * np.log10(np.abs(h)))
    np.testing.assert_allclose(realised, t60, rtol=0.05)


@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected"),
    [
        (0.0, 0.0, [1.0, 0.0, 0.0, 1.0]),          # front: X
        (np.pi / 2, 0.0, [1.0, 1.0, 0.0, 0.0]),    # left: Y
        (0.0, np.pi / 2, [1.0, 0.0, 1.0, 0.0]),    # up: Z
    ],
)
def test_encoding_is_acn_sn3d(azimuth, elevation, expected):
    np.testing.assert_allclose(
        spatial.encode_directions(azimuth, elevation, 1), expected, atol=1e-12
    )
    w = spatial.encode_directions(azimuth + 0.4, elevation - 0.3, 3)[0]
    assert w == pytest.approx(1.0)


@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_grid_satisfies_sn3d_orthogonality(order):
    az, el = spatial.quadrature_grid(order)
    Y = spatial.encode_directions(az, el, order)
    gram = Y.T @ Y * (4 * np.pi / az.size)
    degrees = [n for n in range(order + 1) for _ in range(2 * n + 1)]
    target = np.diag(4 * np.pi / (2 * np.array(degrees) + 1))
    np.testing.assert_allclose(gram, target, atol=1e-12)


def test_vbap_gains_are_a_triplet_summing_to_one():
    az, el = spatial.quadrature_grid(2)
    for direction in [(0.0, 0.0), (1.3, 0.4), (-2.0, -0.8), (np.pi, 0.0)]:
        g = spatial.vbap_gains(az, el, *direction)
        assert np.all(g >= -1e-12)
        assert g.sum() == pytest.approx(1.0)
        assert np.count_nonzero(g > 1e-9) <= 3


def test_vbap_encoding_recovers_the_direction():
    #E @ vbap(theta) approximates y(theta): W exactly, the axes within a triplet
    az, el = spatial.quadrature_grid(3)
    Y = spatial.encode_directions(az, el, 1)
    for direction in [(0.0, 0.0), (1.3, 0.4), (-2.0, -0.5)]:
        encoded = Y.T @ spatial.vbap_gains(az, el, *direction)
        target = spatial.encode_directions(*direction, 1)
        assert encoded[0] == pytest.approx(target[0], rel=1e-9)
        np.testing.assert_allclose(encoded[1:], target[1:], atol=0.22)


@pytest.mark.parametrize(("order", "expected"), [(1, 6), (2, 24), (3, 62), (4, 128)])
def test_image_source_counts_follow_4n2_plus_2(order, expected):
    reflections = shoebox_reflections(
        (10.0, 8.0, 4.0), (3.0, 7.0, 1.5), (8.0, 2.0, 1.5), FS, 0.825, order=order
    )
    assert len(reflections) == expected
    delays = [r.delay_samples for r in reflections]
    assert delays == sorted(delays)
