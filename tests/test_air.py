"""Atmosphere: ISO 9613-1 through pyfar, and its place in a T60 target."""

from __future__ import annotations

import numpy as np
import pytest

from src.air import Atmosphere
from src.room import shoebox_reflections

FS = 48000.0
FREQS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
#from an independent ISO 9613-1 implementation
GOLDEN_20C_50PCT = np.array(
    [1.20560478e-04, 4.39738348e-04, 1.30968126e-03, 2.72819773e-03, 4.66480705e-03,
     9.88633221e-03, 2.96615601e-02, 1.05275496e-01, 3.64500802e-01]
)
GOLDEN_35C_80PCT = np.array(
    [4.83115962e-05, 1.91995059e-04, 7.48620927e-04, 2.72177770e-03, 8.03335699e-03,
     1.64611286e-02, 2.74732698e-02, 5.65030937e-02, 1.66629094e-01]
)


def test_coefficient_matches_an_independent_implementation():
    np.testing.assert_allclose(
        Atmosphere().alpha_db_per_m(FREQS), GOLDEN_20C_50PCT, rtol=1e-3
    )
    hot = Atmosphere(temperature=35.0, relative_humidity=0.8)
    np.testing.assert_allclose(hot.alpha_db_per_m(FREQS), GOLDEN_35C_80PCT, rtol=1e-3)


def test_below_50_hz_alpha_is_continued_as_f_squared():
    atm = Atmosphere()
    a50 = atm.alpha_db_per_m(np.array([50.0]))[0]
    low = atm.alpha_db_per_m(np.array([0.0, 25.0, 31.25, 49.999]))
    assert low[0] == 0.0
    assert low[1] == pytest.approx(a50 * 0.25)
    assert low[3] == pytest.approx(a50, rel=1e-3)
    #the exact ISO 9613-1 value at 31.25 Hz is 3.09e-5; the continuation stays within 40 %
    assert low[2] == pytest.approx(3.09e-5, rel=0.4)
    assert np.all(np.diff(atm.alpha_db_per_m(np.linspace(0.0, 200.0, 50))) > 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"relative_humidity": 1.5}, "humidity"),
        ({"relative_humidity": -0.1}, "humidity"),
        ({"temperature": -300.0}, "Temperature"),
        ({"atmospheric_pressure": 3e5}, "pressure"),
        ({"atmospheric_pressure": -5e3}, "vapour"),
    ],
)
def test_unphysical_atmospheres_raise_on_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Atmosphere(**kwargs)


def _reflections(**kw):
    return shoebox_reflections(
        (10.0, 8.0, 4.0), (3.0, 7.0, 1.5), (8.0, 2.0, 1.5), FS, 0.8,
        order=1, freqs=FREQS, t60_per_band_s=np.full(FREQS.size, 0.8), **kw,
    )


def test_air_in_reflections_is_off_by_default_and_takes_an_atmosphere():
    dry = _reflections()
    default = _reflections(air_absorption=True)
    humid = _reflections(air_absorption=Atmosphere(relative_humidity=0.95))
    off = _reflections(air_absorption=False)
    assert np.allclose(dry[-1].band_gains, off[-1].band_gains)
    assert default[-1].band_gains[-1] < dry[-1].band_gains[-1]
    assert humid[-1].band_gains[-1] != pytest.approx(default[-1].band_gains[-1], rel=1e-3)


def test_unsorted_duplicate_and_nd_frequencies_agree_with_a_sorted_call():
    atm = Atmosphere()
    grid = np.array([125.0, 1000.0, 4000.0])
    ref = dict(zip(grid, atm.alpha_db_per_m(grid), strict=True))
    shuffled = np.array([[4000.0, 125.0, 125.0], [1000.0, 4000.0, 1000.0]])
    out = atm.alpha_db_per_m(shuffled)
    assert out.shape == shuffled.shape
    np.testing.assert_allclose(out, np.vectorize(ref.get)(shuffled))
    assert atm.alpha_db_per_m(1000.0).shape == ()
