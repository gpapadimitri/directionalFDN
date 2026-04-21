"""Circular delay-line buffer and coprime prime-length generator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..utils import ms_to_samps

# ---------------------------------------------------------------------------
# Delay-line lengths
# ---------------------------------------------------------------------------

def generate_coprime_delay_lengths(
    delay_range_ms: tuple[float, float],
    num_delay_lines: int,
    fs: float,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Return *num_delay_lines* coprime prime-number delay lengths.

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
    primes = primes[(primes >= lo) & (primes <= hi)]

    if len(primes) < num_delay_lines:
        raise ValueError(
            f"Not enough primes in [{lo}, {hi}] for {num_delay_lines} delay lines."
        )

    rng.shuffle(primes)
    return primes[:num_delay_lines].astype(np.int64)


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
