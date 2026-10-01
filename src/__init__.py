"""
directional_fdn — Directional Feedback Delay Network (DFDN)

Two published formulations for Ambisonics room impulse response synthesis:

* :class:`~src.fdn.directional_fdn.DirectionalFDN` — Alary, Politis, Schlecht
  & Valimaki, J. Audio Eng. Soc. 67(10), 752-762 (2019). Spherical-harmonic
  state with a directional weighting matrix.
* :class:`~src.fdn.planewave_dfdn.PlaneWaveDFDN` — Alary & Politis, ICASSP
  2020, 176-180. Plane-wave grid with per-direction absorbent filters.
"""

__version__ = "0.1.0"

from .fdn import DirectionalFDN, FeedbackMatrixType, PlaneWaveDFDN

__all__ = ["DirectionalFDN", "FeedbackMatrixType", "PlaneWaveDFDN"]
