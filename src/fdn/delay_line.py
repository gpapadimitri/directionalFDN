"""Circular delay-line buffer and coprime prime-length generator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyFDN import sample_delay_lengths

from ..utils import ms_to_samps

# ---------------------------------------------------------------------------
# Delay-line lengths
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
# Circular FIFO delay line
# ---------------------------------------------------------------------------

class DelayLine:
    """Fixed-length circular FIFO delay buffer.

    Reads the oldest *frame_size* samples, writes the newest.
    Uses a write-pointer circular buffer for O(frame_size) writes
    instead of O(length) array copies.

    Parameters
    ----------
    length : int
        Delay-line length in samples.
    frame_size : int
        Buffer (block) size in samples.

    Raises
    ------
    ValueError
        If *length* < *frame_size*.
    """

    __slots__ = ("length", "frame_size", "data", "_write_ptr")

    def __init__(self, length: int, frame_size: int) -> None:
        if length < frame_size:
            raise ValueError(
                f"Delay-line length ({length}) must be ≥ frame_size ({frame_size})."
            )
        self.length = int(length)
        self.frame_size = int(frame_size)
        self.data = np.zeros(self.length)
        self._write_ptr = 0

    # -- I/O ----------------------------------------------------------------

    def read(self) -> NDArray:
        """Return the oldest *frame_size* samples (FIFO order)."""
        # The oldest samples start at _write_ptr (the next position to be
        # overwritten is the oldest data).
        n = self.frame_size
        start = self._write_ptr
        end = start + n
        if end <= self.length:
            return self.data[start:end].copy()
        else:
            wrap = end - self.length
            return np.concatenate((self.data[start:], self.data[:wrap]))

    def write(self, buf: NDArray) -> None:
        """Push *buf* (length *frame_size*) onto the delay line."""
        n = len(buf)
        # Write pointer advances *before* the new data region.
        # We write at the current write_ptr position (which held the oldest
        # data just returned by read()).
        start = self._write_ptr
        end = start + n
        if end <= self.length:
            self.data[start:end] = buf
        else:
            split = self.length - start
            self.data[start:] = buf[:split]
            self.data[: n - split] = buf[split:]
        self._write_ptr = end % self.length

    # -- State management ---------------------------------------------------

    def reset(self) -> None:
        """Zero out the internal buffer."""
        self.data[:] = 0.0
        self._write_ptr = 0
