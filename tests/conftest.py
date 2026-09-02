"""Shared fixtures for the directional_fdn test suite."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

FS = 48000.0
OCTAVE_FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
T60_MS = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3


@pytest.fixture(scope="session")
def fs() -> float:
    return FS


@pytest.fixture(scope="session")
def golden():
    """Load a pre-migration reference RIR by name: 'fdn', 'dfdn', 'pwdfdn'."""

    def _load(name: str) -> np.ndarray:
        path = FIXTURES / f"golden_{name}.npz"
        if not path.exists():
            pytest.skip(f"missing {path}; run scripts/capture_golden.py")
        return np.load(path)["rir"]

    return _load


@pytest.fixture(scope="session")
def measured_t60():
    """Octave-band T60 of a single channel, via pyFDN's Schroeder fit.

    Returns a callable ``(ir, sample_rate) -> NDArray`` of 8 values at 62.5 Hz
    to 8 kHz.  A fixture rather than an importable helper: several installed
    packages ship their own top-level ``tests`` package into site-packages,
    which shadows this directory for ``import``.  pytest loads conftest by
    path, so fixtures reach test modules that imports cannot.
    """
    import pyFDN

    def _measure(ir: np.ndarray, sample_rate: float) -> np.ndarray:
        rt, _ = pyFDN.estimate_rt_bands(
            np.asarray(ir).ravel(), sample_rate, n=8, start=-4
        )
        return rt

    return _measure
