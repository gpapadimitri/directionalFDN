"""Smoke tests for the directional plot helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import spatial  # noqa: E402
from src.utils import (  # noqa: E402
    plot_band_correlation,
    plot_directional_map,
    plot_directional_t60,
)

BANDS = np.array([62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def test_plot_directional_t60_lines():
    t60 = np.random.default_rng(0).uniform(0.8, 1.2, (26, 8))
    plot_directional_t60(t60, BANDS, target_ms=np.full(8, 1000.0), ylabel="T30 (s)")
    ax = plt.gcf().axes[0]
    assert len(ax.lines) == 26 + 2
    assert ax.get_ylabel() == "T30 (s)"
    plt.close("all")


def test_plot_directional_map_points():
    az, el = spatial.quadrature_grid(3)
    plot_directional_map(az, el, np.linspace(0.5, 1.5, az.size), "T30 (s)")
    assert plt.gcf().axes[0].collections[0].get_offsets().shape == (az.size, 2)
    plt.close("all")


def test_plot_band_correlation_with_floor():
    plot_band_correlation(np.full(8, 0.05), BANDS, floor=np.full(8, 0.04))
    assert len(plt.gcf().axes[0].lines) == 2
    plt.close("all")
