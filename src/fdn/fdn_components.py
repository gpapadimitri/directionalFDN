"""The three FDN ingredients a :class:`pyFDN.FDNBuild` needs from this package.

* **Delay lengths** -- :func:`generate_delay_lengths`, over
  :func:`pyFDN.sample_delay_lengths`.
* **Feedback matrix** -- :class:`FeedbackMatrix`, over
  :func:`pyFDN.fdn_matrix_gallery`.
* **Absorption** -- :func:`design_absorption_sos`, over
  :func:`pyFDN.decay_to_geq`, passed to pyFDN as ``post_delay``.

Feedback matrices
-----------------
* ``RANDOM``             -- Haar-distributed orthogonal matrix.
* ``HADAMARD``           -- normalised Hadamard (size must be a power of 2).
* ``HOUSEHOLDER``        -- reflection about the all-ones vector.
* ``CIRCULANT``          -- random unit-modulus circulant.
* ``PERMUTATION``        -- random permutation (parallel comb filters, permuted).
* ``DIAGONALLY_SIMILAR`` -- ``D^-1 Q D``; lossless but not orthogonal.
* ``ANDERSON``           -- Anderson's construction.
* ``TINY_ROTATION``      -- near-identity rotation.

All generators come from ``pyFDN``.  ``pyFDN`` draws from NumPy's *global*
random state and exposes no seed parameter, so :func:`_seeded` saves that
state, seeds it, and restores it -- keeping this package reproducible without
disturbing the caller's own stream.

Absorption
----------
Graphic-EQ absorption (Schlecht and Habets, DAFx 2017).
:func:`pyFDN.decay_to_geq` already takes one T60 curve per delay line -- what
the plane-wave DFDN needs, one per direction.  :func:`design_absorption_sos`
adds the ``(N, 10)`` row-per-line orientation used throughout the package and
refuses a target on any other frequency axis.

The design grid belongs to the method, not to the caller.  The equaliser is
eight peaking bands at 63 Hz to 8 kHz plus low and high shelves crossing over
at 46 Hz and 11360 Hz (Valimaki and Reiss, *Applied Sciences* 6(5) 129, 2016),
and a target is ten command values on the octave centres 31.25 Hz to 16 kHz,
:data:`pyFDN.eq.COMMAND_FREQUENCIES` -- the grid :func:`geq_band_frequencies`
returns.  The design interpolates linearly in hertz between command points
and holds the end values below 31.25 Hz and above 16 kHz.

pyFDN before 0.5.0 placed the outer two command points at 1 Hz and ``fs``
instead, so a 16 kHz value was read as if it applied at ``fs``.  Targets
written for that grid keep their shape but now mean what their position
suggests: the last value is the T60 at 16 kHz.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import pyFDN
from numpy.typing import NDArray
from pyFDN import sample_delay_lengths
from pyFDN.eq import COMMAND_FREQUENCIES, decay_to_geq

from ..utils import ms_to_samps

# ---------------------------------------------------------------------------
# Delay lengths
# ---------------------------------------------------------------------------

def generate_distinct_delay_lengths(
    delay_range_ms: tuple[float, float],
    num_delay_lines: int,
    fs: float,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Return *num_delay_lines* distinct integer delay lengths (primes preferred).

    Strict pairwise coprimality is impractical when ``num_delay_lines`` is
    large (e.g. NQ \u2273 200) because the prime count in any usable audio range
    is much smaller.  This helper therefore returns *distinct* lengths,
    prioritising primes within the requested range and filling the
    remainder with distinct composites.  Distinct, well-spread lengths are
    the standard FDN choice in this regime; modal/echo density depends on
    the *spread* of the lengths, not on strict coprimality.

    Parameters
    ----------
    delay_range_ms : (float, float)
        Min and max delay in milliseconds.
    num_delay_lines : int
        Number of delay lines.
    fs : float
        Sample rate in Hz.
    rng : numpy.random.Generator, optional
        Random generator for shuffling.

    Returns
    -------
    NDArray of int
        Delay-line lengths in samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    lo, hi = ms_to_samps(np.asarray(delay_range_ms), fs)

    # Sieve of Eratosthenes — adequate for audio-range lengths
    sieve = np.ones(hi + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(hi)) + 1):
        if sieve[i]:
            sieve[i * i :: i] = False
    primes = np.nonzero(sieve)[0]
    primes_in_range = primes[(primes >= lo) & (primes <= hi)]

    if len(primes_in_range) >= num_delay_lines:
        rng.shuffle(primes_in_range)
        return primes_in_range[:num_delay_lines].astype(np.int64)

    # Not enough primes — fall back to distinct integer lengths in [lo, hi],
    # prioritizing primes. Strict pairwise coprimality is impractical for
    # large delay-line counts; distinct, spread lengths are the standard
    # FDN choice in this regime.
    all_lengths = np.arange(lo, hi + 1, dtype=np.int64)
    if len(all_lengths) < num_delay_lines:
        raise ValueError(
            f"Range [{lo}, {hi}] contains only {len(all_lengths)} integers, "
            f"need {num_delay_lines}. Widen delay_range_ms."
        )

    is_prime = np.zeros(len(all_lengths), dtype=bool)
    is_prime[np.isin(all_lengths, primes_in_range)] = True

    primes_arr = all_lengths[is_prime].copy()
    composites = all_lengths[~is_prime].copy()
    rng.shuffle(primes_arr)
    rng.shuffle(composites)

    needed = num_delay_lines - len(primes_arr)
    selected = np.concatenate([primes_arr, composites[:needed]])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def generate_delay_lengths(
    delay_range_ms: tuple[float, float],
    num_delay_lines: int,
    fs: float,
    *,
    distribution: str = "geometric",
    rng: np.random.Generator | int | None = None,
) -> NDArray:
    """Return *num_delay_lines* distinct delay lengths in samples.

    Draws from ``pyFDN.sample_delay_lengths`` with pairwise-coprime snapping,
    which is the classic FDN choice for avoiding coinciding echoes.  Coprimality
    is only achievable while the requested range holds enough mutually coprime
    integers; at high channel counts (e.g. ``PlaneWaveDFDN`` with N*Q >= 300 in a
    20-60 ms range) the snapping pushes values far outside the range, so this
    falls back to :func:`generate_distinct_delay_lengths`, which guarantees
    distinctness within the range but not coprimality.

    Parameters
    ----------
    delay_range_ms : (float, float)
        Min and max delay in milliseconds.
    num_delay_lines : int
        Number of delay lines.
    fs : float
        Sample rate in Hz.
    distribution : str
        ``"geometric"`` (log-uniform, default), ``"uniform"`` or ``"lognormal"``.
    rng : numpy.random.Generator or int, optional
        Generator or integer seed.

    Returns
    -------
    NDArray of int64
        Delay-line lengths in samples.
    """
    lo, hi = (int(v) for v in ms_to_samps(np.asarray(delay_range_ms, dtype=float), fs))

    candidate = sample_delay_lengths(
        num_delay_lines,
        (lo, hi),
        distribution=distribution,
        coprime=True,
        rng=rng,
    ).astype(np.int64)

    in_range = bool(np.all((candidate >= lo) & (candidate <= hi)))
    distinct = len(np.unique(candidate)) == num_delay_lines
    if in_range and distinct:
        return candidate

    # Coprimality is infeasible in this range at this channel count.
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return generate_distinct_delay_lengths(
        delay_range_ms, num_delay_lines, fs, rng=generator
    )


# ---------------------------------------------------------------------------
# Feedback matrix
# ---------------------------------------------------------------------------

class FeedbackMatrixType(str, Enum):
    """Feedback matrix flavour."""

    RANDOM = "random"
    HADAMARD = "hadamard"
    HOUSEHOLDER = "householder"
    CIRCULANT = "circulant"
    PERMUTATION = "permutation"
    DIAGONALLY_SIMILAR = "diagonally_similar"
    ANDERSON = "anderson"
    TINY_ROTATION = "tiny_rotation"

    def __repr__(self) -> str:  # pragma: no cover
        return self.value


#: Types this package delegates straight to ``pyFDN.fdn_matrix_gallery``,
#: mapped to the gallery's own type strings.  ``HOUSEHOLDER`` is deliberately
#: absent: the gallery builds a *random* reflection, while the FDN literature
#: means the one about the all-ones vector.
_GALLERY_TYPES: dict[FeedbackMatrixType, str] = {
    FeedbackMatrixType.RANDOM: "orthogonal",
    FeedbackMatrixType.HADAMARD: "hadamard",
    FeedbackMatrixType.CIRCULANT: "circulant",
    FeedbackMatrixType.PERMUTATION: "permutation",
    FeedbackMatrixType.DIAGONALLY_SIMILAR: "diagonally_similar_to_orthogonal",
    FeedbackMatrixType.ANDERSON: "anderson",
    FeedbackMatrixType.TINY_ROTATION: "tiny_rotation",
}


def _seeded(fn: Callable[..., Any], seed: int | None, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* with NumPy's global random state seeded, then restore it.

    ``pyFDN``'s matrix generators draw from ``np.random`` and take no seed.
    Saving and restoring the global state around the call makes them
    reproducible here while leaving the caller's stream exactly as it was.
    """
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return fn(*args, **kwargs)
    finally:
        np.random.set_state(state)


class FeedbackMatrix:
    """Scalar feedback matrix for the FDN.

    Parameters
    ----------
    size : int
        Matrix dimension (*N* x *N*).
    matrix_type : FeedbackMatrixType
        Any member of :class:`FeedbackMatrixType`.
    seed : int, optional
        Random seed.  Deterministic types ignore it.

    Attributes
    ----------
    matrix : NDArray
        The *N* x *N* lossless feedback matrix.
    """

    def __init__(
        self,
        size: int,
        matrix_type: FeedbackMatrixType,
        *,
        seed: int | None = None,
    ) -> None:
        if matrix_type == FeedbackMatrixType.HOUSEHOLDER:
            # Not via the gallery: pyFDN's "householder" reflects about a random
            # vector, whereas the FDN literature means the all-ones one.
            self.matrix = np.asarray(pyFDN.householder_matrix(np.ones(size)), float)
            return

        gallery_type = _GALLERY_TYPES.get(matrix_type)
        if gallery_type is None:
            raise ValueError(f"Unsupported feedback matrix type: {matrix_type}")
        if matrix_type == FeedbackMatrixType.HADAMARD and (
            size < 1 or (size & (size - 1)) != 0
        ):
            raise ValueError(f"Hadamard size must be a power of 2, got {size}.")

        matrix = _seeded(pyFDN.fdn_matrix_gallery, seed, size, gallery_type)
        self.matrix = np.asarray(matrix, dtype=float)


# ---------------------------------------------------------------------------
# Absorption (graphic EQ)
# ---------------------------------------------------------------------------

#: Number of command values a target must supply.
GEQ_NUM_BANDS = len(COMMAND_FREQUENCIES)


def geq_band_frequencies() -> NDArray:
    """The ten frequencies a GEQ absorption target is specified at.

    Returns
    -------
    NDArray, shape (10,)
        ``[31.25, 62.5, 125, 250, 500, 1k, 2k, 4k, 8k, 16k]`` Hz, independent
        of the sample rate.
    """
    return COMMAND_FREQUENCIES.copy()


def design_absorption_sos(
    t60_s: NDArray,
    delays: NDArray,
    fs: float,
) -> NDArray:
    """Design one GEQ absorption cascade per delay line.

    Parameters
    ----------
    t60_s : NDArray
        Shape ``(10,)`` for one curve shared by all delay lines, or
        ``(N, 10)`` for one curve per delay line.  Seconds, specified at
        :func:`geq_band_frequencies`.
    delays : NDArray, shape (N,)
        Delay-line lengths in samples.
    fs : float
        Sample rate in Hz.

    Returns
    -------
    NDArray, shape (11, 6, N)
        Per-channel SOS bank in pyFDN's canonical layout, ready for
        :class:`pyFDN.td.SOSBank`.  All sections are normalised to a0 = 1.
    """
    delays = np.asarray(delays, dtype=float).ravel()
    num_delays = delays.size
    if num_delays == 0:
        raise ValueError("delays must not be empty")

    curves = np.atleast_2d(np.asarray(t60_s, dtype=float))
    if curves.shape[-1] != GEQ_NUM_BANDS:
        raise ValueError(
            f"t60_s must give {GEQ_NUM_BANDS} values per delay line, at "
            f"geq_band_frequencies(); got {curves.shape[-1]}."
        )
    if curves.shape[0] not in (1, num_delays):
        raise ValueError(
            f"t60_s must supply one T60 curve per delay line: expected "
            f"{num_delays} rows, got {curves.shape[0]}"
        )
    return decay_to_geq(curves.T, delays, fs)
