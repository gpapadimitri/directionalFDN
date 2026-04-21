from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyfar import Coordinates
import spharpy


def acn_to_lm(acn: int) -> tuple[int, int]:
    """Convert an ACN index to degree *l* and order *m*.

    ACN 0 → (0, 0), 1 → (1, −1), 2 → (1, 0), 3 → (1, 1), …
    """
    n, m = spharpy.spherical.acn_to_nm(acn)
    return int(n), int(m)


def num_channels(order: int) -> int:
    """Number of Ambisonics channels for a given order: *(L+1)²*."""
    return (order + 1) ** 2


def encode_direction(
    azimuth: float,
    elevation: float,
    order: int = 1,
) -> NDArray:
    """Encode a single direction as ACN/SN3D Ambisonics coefficients.

    Uses spharpy's real spherical harmonics with SN3D normalisation.
    The zeroth-order coefficient (W) equals ``1 / √(4π)`` for any
    direction, consistent with the AmbiX convention.

    Parameters
    ----------
    azimuth : float
        Azimuth in **radians** (0 = front, π/2 = left).
    elevation : float
        Elevation in **radians** (0 = horizontal, π/2 = up).
    order : int
        Maximum Ambisonics order.

    Returns
    -------
    NDArray, length (order+1)²
        SN3D coefficient vector.
    """
    coords = Coordinates.from_spherical_elevation(azimuth, elevation, 1.0)
    Y = spharpy.spherical.spherical_harmonic_basis_real(
        order, coords, normalization="SN3D"
    )
    return Y.ravel()


def encode_directions(
    azimuths: NDArray,
    elevations: NDArray,
    order: int = 1,
) -> NDArray:
    """Encode an array of directions.

    Parameters
    ----------
    azimuths, elevations : NDArray
        Angles in radians, same length *D*.
    order : int
        Maximum Ambisonics order.

    Returns
    -------
    NDArray, shape (D, (order+1)²)
        SN3D coefficient matrix.
    """
    azimuths = np.asarray(azimuths, dtype=float)
    elevations = np.asarray(elevations, dtype=float)
    coords = Coordinates.from_spherical_elevation(
        azimuths, elevations, np.ones_like(azimuths)
    )
    Y = spharpy.spherical.spherical_harmonic_basis_real(
        order, coords, normalization="SN3D"
    )
    return Y


def uniform_directions(n: int) -> tuple[NDArray, NDArray]:
    """Generate *n* approximately-uniform directions on the sphere (Fibonacci spiral).

    Returns
    -------
    azimuths, elevations : NDArray
        Angles in radians.
    """
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)            # colatitude
    theta = np.pi * (1.0 + np.sqrt(5.0)) * idx      # golden-angle azimuth
    return np.mod(theta, 2.0 * np.pi), np.pi / 2.0 - phi


def quadrature_grid(order: int) -> tuple[NDArray, NDArray]:
    """Return a spherical quadrature grid suitable for SH order *order*.

    Uses ``spharpy.samplings.t_design`` with ``n_max=2*order`` so that
    the grid provides exact quadrature for products of order-*L*
    spherical harmonics (needed for the discrete SN3D orthogonality
    condition ``Y^T Y · (4π/Q) = diag(1/(2l+1))``).

    Parameters
    ----------
    order : int
        SH order (≥ 1).

    Returns
    -------
    azimuths, elevations : NDArray
        Grid directions in radians.  Length Q ≥ (order+1)².
    """
    if order < 1:
        raise ValueError(f"quadrature_grid requires order >= 1, got {order}")
    coords = spharpy.samplings.t_design(n_max=2 * order)
    return np.asarray(coords.azimuth), np.asarray(coords.elevation)
