"""
directional_fdn -- Directional Feedback Delay Network (DFDN)

Two published formulations for Ambisonic room impulse response synthesis:

* :class:`~src.directional_fdn.DirectionalFDN` -- Alary & Politis, ICASSP
  2020, 176-180. One channel per incident direction on a spatial grid, with
  per-direction absorbent filters. The working reverberator.
* :class:`~src.dfdn_2019.DFDN2019` -- Alary, Politis,
  Schlecht & Valimaki, J. Audio Eng. Soc. 67(10), 752-762 (2019).
  Spherical-harmonic state with a directional weighting matrix. Archival.
"""


__version__ = "0.1.0"

from .dfdn_2019 import DFDN2019
from .directional_fdn import DirectionalFDN
from .fdn_components import FeedbackMatrixType

__all__ = ["DirectionalFDN", "DFDN2019", "FeedbackMatrixType"]
