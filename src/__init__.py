"""
directional_fdn — Directional Feedback Delay Network (DFDN)

Clean implementation of the Directional FDN for Ambisonics room impulse
response synthesis, following Benoit et al. 2019.
"""

__version__ = "0.1.0"

from .fdn import FDN, DirectionalFDN, FeedbackMatrixType, PlaneWaveDFDN

__all__ = ["DirectionalFDN", "FDN", "FeedbackMatrixType", "PlaneWaveDFDN"]
