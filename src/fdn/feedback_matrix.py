"""Feedback matrices for the FDN: scalar (unitary) and filter (para-unitary).

Supported types
---------------
* ``RANDOM``   — random orthonormal matrix via QR decomposition.
* ``HADAMARD`` — normalised Hadamard matrix (size must be a power of 2).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import hadamard, qr


class FeedbackMatrixType(str, Enum):
    """Feedback matrix flavour."""

    RANDOM = "random"
    HADAMARD = "hadamard"
    HOUSEHOLDER = "householder"

    def __repr__(self) -> str:  # pragma: no cover
        return self.value



# ---------------------------------------------------------------------------
# FeedbackMatrix — scalar feedback matrix for the FDN
# ---------------------------------------------------------------------------

class FeedbackMatrix:
    """Scalar feedback matrix for the FDN.

    Parameters
    ----------
    size : int
        Matrix dimension (*N* x *N*).
    matrix_type : FeedbackMatrixType
        ``RANDOM``, ``HADAMARD`` or ``HOUSEHOLDER``.
    frame_size : int
        Accepted for signature compatibility; unused since the recursion moved
        to :func:`src.fdn.render.render_fdn`.
    seed : int, optional
        Random seed (used for ``RANDOM`` type).

    Attributes
    ----------
    matrix : NDArray
        The *N* x *N* unitary feedback matrix.
    """

    def __init__(
        self,
        size: int,
        matrix_type: FeedbackMatrixType,
        frame_size: int,
        *,
        seed: int | None = None,
    ) -> None:
        self.num_rows = size
        self.num_cols = size
        self.frame_size = frame_size
        self._rng = np.random.default_rng(seed)

        if matrix_type == FeedbackMatrixType.RANDOM:
            self.matrix = self._random_unitary(size)
        elif matrix_type == FeedbackMatrixType.HADAMARD:
            self.matrix = self._hadamard(size)
        elif matrix_type == FeedbackMatrixType.HOUSEHOLDER:
            self.matrix = self._householder(size)
        else:
            raise ValueError(f"Unsupported feedback matrix type: {matrix_type}")

    # -- Static generators --------------------------------------------------

    def _random_unitary(self, n: int) -> NDArray:
        """Random orthogonal (unitary) matrix via QR decomposition."""
        Q, R = qr(self._rng.standard_normal((n, n)))
        return Q @ np.diag(np.sign(np.diag(R)))

    @staticmethod
    def _hadamard(n: int) -> NDArray:
        """Normalised Hadamard matrix (size must be a power of 2)."""
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError(f"Hadamard size must be a power of 2, got {n}.")
        return hadamard(n).astype(float) / np.sqrt(n)

    @staticmethod
    def _householder(n: int) -> NDArray:
        """Householder reflection matrix."""
        v = np.ones((n, 1))
        H = np.eye(n) - 2 * (v @ v.T) / (v.T @ v)
        return H

    # -- Verification -------------------------------------------------------

    @staticmethod
    def is_unitary(M: NDArray, tol: float = 1e-9) -> bool:
        """Check *M*ᵀ *M* ≈ *I*."""
        return float(np.linalg.norm(M.T @ M - np.eye(M.shape[0]), "fro")) < tol

    @staticmethod
    def matrix_convolve(A: NDArray, B: NDArray) -> NDArray:
        """Polynomial matrix multiplication (3-D)."""
        from pyFDN import matrix_convolution

        return matrix_convolution(A, B)
