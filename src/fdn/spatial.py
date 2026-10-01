from __future__ import annotations

import numpy as np
import spharpy
from numpy.typing import NDArray
from pyfar import Coordinates


def encode_directions(
    azimuths: float | NDArray,
    elevations: float | NDArray,
    order: int = 1,
) -> NDArray:
    """Encode one or more directions as ACN/SN3D Ambisonics coefficients.

    The zeroth-order coefficient (W) equals ``1`` for any direction, which
    is what AmbiX means by SN3D.

    Note the spharpy spelling: its ``"SN3D"`` is rooted in the orthonormal
    basis and carries a ``1 / sqrt(4 pi)`` that AmbiX does not, so every
    coefficient would come back 11 dB low.  ``"SNM"`` is the Schmidt
    semi-normalisation without it, and is the one AmbiX calls SN3D.

    Parameters
    ----------
    azimuths : float or NDArray
        Azimuth in **radians** (0 = front, π/2 = left).
    elevations : float or NDArray
        Elevation in **radians** (0 = horizontal, π/2 = up).  Broadcast
        against *azimuths*.
    order : int
        Maximum Ambisonics order.

    Returns
    -------
    NDArray, shape ``np.shape(azimuths) + ((order+1)²,)``
        SN3D coefficients: a vector for one direction, a ``(D, (order+1)²)``
        matrix for *D*.
    """
    azimuths, elevations = np.broadcast_arrays(
        np.asarray(azimuths, dtype=float), np.asarray(elevations, dtype=float)
    )
    coords = Coordinates.from_spherical_elevation(
        azimuths.ravel(), elevations.ravel(), 1.0
    )
    Y = spharpy.spherical.spherical_harmonic_basis_real(
        order, coords, normalization="SNM"
    )
    return Y.reshape(azimuths.shape + (-1,))


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

    A t-design of degree ``t = 2L``, which is what ICASSP 2020 Sec. 3 asks
    for -- "at least Q >= (L+1)^2 uniform points, such as spherical t-designs
    of degree t = 2L" -- and what the discrete SN3D orthogonality condition
    ``Y^T Y · (4π/Q) = diag(4π/(2l+1))`` needs to hold exactly.

    ``spharpy``'s *n_max* is already an SH order, doubled to a degree inside
    the library, so it takes *order* rather than ``2 * order``.  Passing the
    doubled value asks for degree ``4L``: at order 3 that is 86 points where
    26 suffice, and the delay-line count of a plane-wave DFDN scales with it.
    The orthogonality error is identical either way, and slightly worse on
    the larger grid, there being more points to sum over.

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
    coords = spharpy.samplings.t_design(n_max=order)
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
