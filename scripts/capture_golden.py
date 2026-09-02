"""Capture reference RIRs from the pre-migration implementation.

Run once, before any pyFDN swap. The .npz files it writes are the
regression baseline for every task in the migration plan.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Running this as a script puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.fdn.directional_fdn import DirectionalFDN  # noqa: E402
from src.fdn.fdn import FDN  # noqa: E402
from src.fdn.feedback_matrix import FeedbackMatrixType  # noqa: E402
from src.fdn.planewave_dfdn import PlaneWaveDFDN  # noqa: E402

FS = 48000.0
OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FREQS = np.array([63.0, 125, 250, 500, 1000, 2000, 4000, 8000])
T60_MS = np.array([1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.7, 0.45]) * 1e3


def capture_fdn() -> np.ndarray:
    fdn = FDN(FS, 1, 1, 8, 64, seed=42)
    fdn.init_delay_line_lengths(
        np.array([1039, 1213, 1381, 1523, 1699, 1877, 2039, 2213])
    )
    fdn.init_feedback_matrix(FeedbackMatrixType.HADAMARD)
    fdn.init_absorption(T60_MS, freqs=FREQS, filter_order=8, warp_factor=0.0)
    fdn.init_delay_lines()
    fdn.init_io_gains(np.ones((8, 1)) / np.sqrt(8), np.ones((1, 8)) / np.sqrt(8))
    fdn.init_direct_gain(np.zeros((1, 1)))
    imp = np.zeros((1, int(3 * FS)))
    imp[0, 0] = 1.0
    return fdn.process(imp)


def capture_dfdn() -> np.ndarray:
    dfdn = DirectionalFDN(FS, 8, ambisonics_order=1, frame_size=64, seed=42)
    dfdn.setup(
        delay_range_ms=(20.0, 60.0),
        des_t60_ms=T60_MS,
        freqs=FREQS,
        filter_order=8,
        warp_factor=0.0,
    )
    return dfdn.render(0.0, 0.0, length_seconds=3.0)


def capture_pwdfdn() -> np.ndarray:
    pw = PlaneWaveDFDN(FS, 8, sh_order=1, frame_size=64, seed=42)
    pw.setup(
        delay_range_ms=(20.0, 60.0),
        t60_per_direction_ms=T60_MS,
        freqs=FREQS,
        filter_order=8,
    )
    return pw.render(0.0, 0.0, length_seconds=3.0)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("fdn", capture_fdn),
        ("dfdn", capture_dfdn),
        ("pwdfdn", capture_pwdfdn),
    ):
        rir = fn()
        np.savez_compressed(OUT / f"golden_{name}.npz", rir=rir, fs=FS)
        print(f"{name}: shape={rir.shape} peak={np.max(np.abs(rir)):.4e}")
