"""Base Feedback Delay Network (FDN).

Implements the standard time-domain FDN with:

* Input mixing (b)
* Circular delay lines
* Absorption (scalar gains **or** frequency-dependent IIR)
* Unitary feedback matrix
* Output mixing (c)
* Direct path (d)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyFDN import SOSFilterBank

from .absorption import (
    Absorption,
    AbsorptionGains,
    AbsorptionGEQ,
    AbsorptionSOS,
    create_absorption,
)
from .delay_line import DelayLine
from .feedback_matrix import FeedbackMatrix, FeedbackMatrixType
from .render import ScalarGains, render_fdn


class FDN:
    """Time-domain Feedback Delay Network.

    Parameters
    ----------
    sample_rate : float
        Audio sample rate in Hz.
    num_inputs : int
        Number of input channels.
    num_outputs : int
        Number of output channels.
    num_delay_lines : int
        Total number of delay lines.
    frame_size : int
        Processing block size in samples.
    seed : int, optional
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        sample_rate: float,
        num_inputs: int,
        num_outputs: int,
        num_delay_lines: int,
        frame_size: int,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_delay_lines = num_delay_lines
        self.frame_size = frame_size
        self.seed = seed

        # Initialised later via dedicated ``init_*`` methods
        self.input_gains: NDArray | None = None
        self.output_gains: NDArray | None = None
        self.direct_gain: NDArray | None = None
        self.delay_lines: list[DelayLine] | None = None
        self.absorption: Absorption | None = None
        self.feedback: FeedbackMatrix | None = None
        self.delay_length_samps: NDArray | None = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def init_io_gains(self, b: NDArray, c: NDArray) -> None:
        """Set input (*b*) and output (*c*) mixing matrices.

        Parameters
        ----------
        b : NDArray, shape (num_delay_lines, num_inputs)
        c : NDArray, shape (num_outputs, num_delay_lines)
        """
        if b.shape != (self.num_delay_lines, self.num_inputs):
            raise ValueError(f"b must be ({self.num_delay_lines}, {self.num_inputs})")
        if c.shape != (self.num_outputs, self.num_delay_lines):
            raise ValueError(f"c must be ({self.num_outputs}, {self.num_delay_lines})")
        self.input_gains = b
        self.output_gains = c

    def init_direct_gain(self, d: NDArray) -> None:
        """Set direct-path gain matrix.

        Parameters
        ----------
        d : NDArray, shape (num_outputs, num_inputs)
        """
        if d.shape != (self.num_outputs, self.num_inputs):
            raise ValueError(f"d must be ({self.num_outputs}, {self.num_inputs})")
        self.direct_gain = d

    def init_delay_line_lengths(self, lengths: NDArray) -> None:
        """Provide delay-line lengths (samples)."""
        lengths = np.asarray(lengths, dtype=np.int64)
        if len(lengths) != self.num_delay_lines:
            raise ValueError(
                f"Expected {self.num_delay_lines} lengths, got {len(lengths)}."
            )
        self.delay_length_samps = lengths

    def init_feedback_matrix(
        self,
        matrix_type: FeedbackMatrixType = FeedbackMatrixType.RANDOM,
    ) -> None:
        """Create and store the feedback matrix."""
        self.feedback = FeedbackMatrix(
            self.num_delay_lines, matrix_type, self.frame_size, seed=self.seed
        )

    def init_absorption(
        self,
        des_t60_ms: float | NDArray,
        *,
        freqs: NDArray | None = None,
        filter_order: int | None = None,
        warp_factor: float | None = None,
    ) -> None:
        """Initialise absorption gains / filters.

        Parameters
        ----------
        des_t60_ms : float or NDArray
            Broadband (scalar) **or** frequency-dependent (array) T60 in ms.
        freqs : NDArray, optional
            Frequency points for frequency-dependent T60.
        filter_order : int, optional
            Ignored. Retained for API compatibility; absorption is now
            designed as a graphic EQ (see :mod:`src.fdn.absorption_design`),
            which takes no filter order or warping coefficient.
        warp_factor : float, optional
            Ignored. Retained for API compatibility; absorption is now
            designed as a graphic EQ (see :mod:`src.fdn.absorption_design`),
            which takes no filter order or warping coefficient.
        """
        if self.delay_length_samps is None:
            raise RuntimeError("Initialise delay-line lengths first.")
        self.absorption = create_absorption(
            self.num_delay_lines,
            self.delay_length_samps,
            self.sample_rate,
            self.frame_size,
            des_t60_ms,
            freqs=freqs,
            filter_order=filter_order,
            warp_factor=warp_factor,
        )

    def init_absorption_sos(self, sos_coeffs: NDArray) -> None:
        """Initialise absorption from pre-computed SOS cascades.

        Parameters
        ----------
        sos_coeffs : NDArray, shape (N, sections, 6)
        """
        if self.delay_length_samps is None:
            raise RuntimeError("Initialise delay-line lengths first.")
        self.absorption = AbsorptionSOS(
            self.num_delay_lines,
            self.delay_length_samps,
            self.sample_rate,
            self.frame_size,
            sos_coeffs,
        )

    def init_delay_lines(self) -> None:
        """Allocate circular delay-line buffers."""
        if self.delay_length_samps is None:
            raise RuntimeError("Initialise delay-line lengths first.")
        self.delay_lines = [
            DelayLine(int(m), self.frame_size) for m in self.delay_length_samps
        ]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _absorption_hook(self) -> object:
        """Return the absorption as a ``process_fdn`` filter object.

        ``process_fdn`` applies its ``post_delay`` hook to the delay-line
        output, which is where this package applies absorption.  The hook only
        needs ``filter(block: (F, N)) -> (F, N)``.

        The returned hook is built fresh on every call and carries no state
        in: ``render_fdn`` allocates a fresh delay bank each call, so handing
        back a filter object with live memory would leave the network
        half-reset and make repeat renders differ.
        """
        if isinstance(self.absorption, AbsorptionGains):
            return ScalarGains(self.absorption._gains)
        if isinstance(self.absorption, AbsorptionGEQ):
            # A fresh bank per render: render_fdn allocates a fresh delay bank
            # each call, so carrying filter memory across calls would leave the
            # network half-reset and make repeat renders differ.
            return SOSFilterBank(self.absorption.sos, self.num_delay_lines)
        if isinstance(self.absorption, AbsorptionSOS):
            # AbsorptionSOS stores (N, sections, 6); pyFDN wants (sections, 6, N).
            return SOSFilterBank(
                self.absorption.sos_coeffs.transpose(1, 2, 0), self.num_delay_lines
            )
        raise TypeError(
            f"{type(self.absorption).__name__} is not supported by the pyFDN "
            "recursion; use AbsorptionGains, AbsorptionGEQ or AbsorptionSOS."
        )

    def process(self, input_data: NDArray) -> NDArray:
        """Run the FDN on *input_data* and return the output.

        Parameters
        ----------
        input_data : NDArray, shape (num_inputs, num_samples)

        Returns
        -------
        NDArray, shape (num_outputs, num_samples)
        """
        self._check_ready()
        if input_data.shape[0] != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} input channels, got {input_data.shape[0]}."
            )

        return render_fdn(
            input_data,
            self.delay_length_samps,
            self.feedback.matrix,
            self.input_gains,
            self.output_gains,
            self.direct_gain,
            absorption=self._absorption_hook(),
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear internal state: delay-line buffers and absorption filters.

        The feedback matrix is stateless, so there is nothing to reset there.
        The delay-line loop is vestigial for ``process``, which allocates a
        fresh delay bank per call through :func:`src.fdn.render.render_fdn`;
        it is kept because ``init_delay_lines`` and ``_check_ready`` still
        expose those buffers.

        ``process()`` now builds a fresh absorption filter bank on every
        call (see :meth:`_absorption_hook`), so this no longer affects
        subsequent ``process()`` output.  What it does reset is the
        ``absorption`` object's own internal state, for callers that invoke
        ``absorption.process()`` directly rather than through ``process()``.
        """
        if self.delay_lines is not None:
            for dl in self.delay_lines:
                dl.reset()
        if self.absorption is not None:
            self.absorption.reset()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_ready(self) -> None:
        """Raise if any component is uninitialised."""
        missing = []
        if self.input_gains is None:
            missing.append("io_gains (b, c)")
        if self.direct_gain is None:
            missing.append("direct_gain (d)")
        if self.delay_lines is None:
            missing.append("delay_lines")
        if self.absorption is None:
            missing.append("absorption")
        if self.feedback is None:
            missing.append("feedback_matrix")
        if missing:
            raise RuntimeError(
                "FDN not fully initialised. Missing: " + ", ".join(missing)
            )
