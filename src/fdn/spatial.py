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


def _unit_vectors(azimuths: NDArray, elevations: NDArray) -> NDArray:
    """Cartesian unit vectors for spherical angles, shape ``(n, 3)``."""
    azimuths = np.asarray(azimuths, dtype=float).ravel()
    elevations = np.asarray(elevations, dtype=float).ravel()
    cos_el = np.cos(elevations)
    return np.column_stack(
        (cos_el * np.cos(azimuths), cos_el * np.sin(azimuths), np.sin(elevations))
    )


def vbap_gains(
    grid_azimuths: NDArray,
    grid_elevations: NDArray,
    azimuth: float,
    elevation: float,
) -> NDArray:
    """Vector-base amplitude panning of one direction onto a spherical grid.

    Finds the triangle of grid points enclosing the direction and solves for
    the three non-negative gains whose weighted sum points at it (Pulkki 1997).
    All other gains are zero.

    Alary and Politis (ICASSP 2020) use this for the direct sound (Eq. 3) and
    for every early reflection (Eq. 4), so that those signals live on the same
    plane-wave grid as the recirculating ones and reach the output through the
    single encoding matrix ``E`` of Eq. 13.

    Gains are normalised to **sum** to one rather than to unit norm.  The
    zeroth harmonic is constant over the sphere, so ``E @ g`` reproduces its
    value exactly only when the gains sum to one; unit-norm gains would make
    the encoded level depend on where between grid points the direction falls.

    Parameters
    ----------
    grid_azimuths, grid_elevations : NDArray, shape (Q,)
        Grid directions in radians.  At least four, not all coplanar.
    azimuth, elevation : float
        Direction to pan, in radians.

    Returns
    -------
    NDArray, shape (Q,)
        Non-negative gains summing to one, at most three of them non-zero.
    """
    from scipy.spatial import ConvexHull

    grid = _unit_vectors(grid_azimuths, grid_elevations)
    if grid.shape[0] < 4:
        raise ValueError(
            f"VBAP needs at least 4 grid directions, got {grid.shape[0]}."
        )
    target = _unit_vectors(np.array([azimuth]), np.array([elevation]))[0]

    # The grid lies on the unit sphere, so its convex hull triangulates it.
    hull = ConvexHull(grid)
    best_gains: NDArray | None = None
    best_margin = -np.inf
    for triplet in hull.simplices:
        try:
            gains = np.linalg.solve(grid[triplet].T, target)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate triangle
            continue
        margin = float(np.min(gains))
        if margin > best_margin:
            best_margin, best_gains = margin, (triplet, gains)
        if margin >= 0.0:
            break

    triplet, gains = best_gains
    gains = np.maximum(gains, 0.0)
    total = gains.sum()
    if total <= 0.0:  # pragma: no cover - target opposite every triangle
        raise ValueError("VBAP found no enclosing triangle for this direction.")

    out = np.zeros(grid.shape[0])
    out[triplet] = gains / total
    return out
