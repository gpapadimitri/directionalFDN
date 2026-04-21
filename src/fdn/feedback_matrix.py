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
from scipy.signal import fftconvolve


class FeedbackMatrixType(str, Enum):
    """Feedback matrix flavour."""

    RANDOM = "random"
    HADAMARD = "hadamard"
    HOUSEHOLDER = "householder"

    def __repr__(self) -> str:  # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# FilterMatrix — stateful overlap-add convolution engine
# ---------------------------------------------------------------------------

class FilterMatrix:
    """Matrix of FIR filters processed with stateful overlap-add.

    For a scalar (2-D) matrix the filter order is 1 and the ``process``
    method reduces to a plain matrix multiply.

    Parameters
    ----------
    num_rows, num_cols : int
        Matrix dimensions.
    frame_size : int
        Block size in samples.
    """

    def __init__(self, num_rows: int, num_cols: int, frame_size: int) -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.frame_size = frame_size
        self._matrix: NDArray | None = None
        self.filter_order: int = 1
        self._overlap: NDArray = np.zeros((num_rows, 0))

    def set_matrix(self, matrix: NDArray) -> None:
        """Load a 2-D or 3-D filter matrix.

        * 2-D ``(R, C)``       → scalar matrix (order 1).
        * 3-D ``(R, C, L)``    → polynomial matrix (order *L*).
        """
        self._matrix = matrix.copy()
        if matrix.ndim == 2:
            self.filter_order = 1
            self._overlap = np.zeros((self.num_rows, 0))
        elif matrix.ndim == 3:
            self.filter_order = matrix.shape[-1]
            tail = self.filter_order - 1
            self._overlap = np.zeros((self.num_rows, max(tail, 0)))
        else:
            raise ValueError("Matrix must be 2-D or 3-D.")

    def process(self, x: NDArray) -> NDArray:
        """Apply the filter matrix to a (num_cols, frame_size) block.

        Returns
        -------
        NDArray of shape (num_rows, frame_size)
        """
        if self._matrix is None:
            raise RuntimeError("Call set_matrix() before process().")
        if self.filter_order == 1:
            return self._matrix @ x

        # Full OLA convolution for polynomial (FIR) matrix
        out_len = self.frame_size + self.filter_order - 1
        buf = np.zeros((self.num_rows, out_len))
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                buf[i] += fftconvolve(self._matrix[i, j, :], x[j])

        # Add overlap from previous frame
        n_ov = self._overlap.shape[1]
        if n_ov:
            buf[:, :n_ov] += self._overlap

        # Split: current output | save overlap for next frame
        result = buf[:, : self.frame_size]
        self._overlap = buf[:, self.frame_size :]
        return result

    def reset(self) -> None:
        """Clear the overlap buffer."""
        n = self._overlap.shape[1] if self._overlap.ndim == 2 else 0
        self._overlap = np.zeros((self.num_rows, n))


# ---------------------------------------------------------------------------
# FeedbackMatrix — generates a (para-)unitary matrix of the chosen type
# ---------------------------------------------------------------------------

class FeedbackMatrix(FilterMatrix):
    """Scalar feedback matrix for the FDN.

    Parameters
    ----------
    size : int
        Matrix dimension (*N* × *N*).
    matrix_type : FeedbackMatrixType
        ``RANDOM`` or ``HADAMARD``.
    frame_size : int
        Block size in samples.
    seed : int, optional
        Random seed (used for ``RANDOM`` type).

    Attributes
    ----------
    matrix : NDArray
        The *N* × *N* unitary feedback matrix.
    """

    def __init__(
        self,
        size: int,
        matrix_type: FeedbackMatrixType,
        frame_size: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(size, size, frame_size)
        self._rng = np.random.default_rng(seed)

        if matrix_type == FeedbackMatrixType.RANDOM:
            self.matrix = self._random_unitary(size)
        elif matrix_type == FeedbackMatrixType.HADAMARD:
            self.matrix = self._hadamard(size)
        elif matrix_type == FeedbackMatrixType.HOUSEHOLDER:
            self.matrix = self._householder(size)
        else:
            raise ValueError(f"Unsupported feedback matrix type: {matrix_type}")

        # Register with the OLA engine (scalar ⇒ order 1)
        self.set_matrix(self.matrix)

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
        assert A.shape[1] == B.shape[0]
        C = np.zeros((A.shape[0], B.shape[1], A.shape[2] + B.shape[2] - 1))
        for r in range(A.shape[0]):
            for c in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[r, c] += np.convolve(A[r, k], B[k, c])
        return C
