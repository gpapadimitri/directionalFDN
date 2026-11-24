"""DFDN2019, the spherical-harmonic network of Alary et al. (JAES 2019)."""

from __future__ import annotations

import numpy as np
import pyFDN
import pytest

from src import spatial
from src.dfdn_2019 import DFDN2019
from src.transform import DirectionalWeighting

FS = 48000.0


def test_broadband_t60_is_accurate(measured_t60):
    dfdn = DFDN2019(FS, 8, ambisonics_order=1, seed=42)
    dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
    rir = dfdn.render(0.0, 0.0, length_seconds=4.0)
    np.testing.assert_allclose(measured_t60(rir[0], FS)[1:], 1.5, rtol=0.08)


def test_direct_sound_is_the_sh_encoding_for_any_group_count():
    #Eq. 11: d = y(phi_0, theta_0), no 1/N
    for groups in (4, 8, 16):
        dfdn = DFDN2019(FS, groups, ambisonics_order=1, seed=0)
        dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1000.0)
        assert dfdn.render(0.0, 0.0, length_seconds=0.2)[0, 0] == pytest.approx(1.0)


def test_weighting_is_identity_for_an_isotropic_target():
    az, el = spatial.uniform_directions(120)
    dwm = DirectionalWeighting(2, 1, np.array([521, 601]), FS, np.full(120, 800.0), az, el)
    for T in dwm.transform_matrices:
        np.testing.assert_allclose(T, np.eye(4), atol=1e-10)


def test_weighting_does_not_add_energy():
    #Sec. 2.4: a spectral radius above one would make the loop grow
    az, el = spatial.uniform_directions(64)
    t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
    dwm = DirectionalWeighting(4, 2, np.array([1039, 1213, 1381, 1523]), FS, t60, az, el)
    for T in dwm.transform_matrices:
        assert np.max(np.abs(np.linalg.eigvals(T))) <= 1.0 + 1e-9


def test_anisotropy_is_realised():
    #Sec. 3.1 samples the target on a dense t-design (t >= 20)
    order = 2
    az, el = spatial.quadrature_grid(10)
    t60 = np.where(np.cos(az) > 0, 2.0, 0.6) * 1e3
    dfdn = DFDN2019(FS, 8, ambisonics_order=order, seed=42)
    dfdn.setup(delay_range_ms=(20.0, 60.0), des_t60_ms=1500.0)
    dfdn.set_directional_decay(t60, az, el)
    rir = dfdn.render(0.0, 0.0, length_seconds=6.0)

    def beam_t60(direction):
        e = pyFDN.edc(spatial.encode_directions(direction, 0.0, order) @ rir)
        e = e[e > 0]
        db = 10 * np.log10(e / e[0])
        i0, i1 = np.argmax(db < -5), np.argmax(db < -35)
        return -60 / np.polyfit(np.arange(i1 - i0) / FS, db[i0:i1] - db[i0], 1)[0]

    assert beam_t60(0.0) > 1.7
    assert beam_t60(np.pi) < 0.9
