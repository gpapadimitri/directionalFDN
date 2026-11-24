# directional_fdn

Directional Feedback Delay Network (DFDN) for Ambisonic room impulse response synthesis (see References).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/gpapadimitri/directional_fdn.git
cd directional_fdn
pip install -e .          # or: pip install -e ".[dev]" for tests and linting
```

The package is imported as `src`.

## Usage

```python
import numpy as np
from src import DirectionalFDN

dfdn = DirectionalFDN(sample_rate=48000, num_delay_line_groups=8, sh_order=2, seed=0)

# T60 in ms per octave band, 31.25 Hz to 16 kHz.
t60_ms = np.array([800, 900, 900, 1000, 1100, 1000, 800, 700, 600, 600])
dfdn.setup(delay_range_ms=(10, 50), t60_per_direction_ms=t60_ms)

# Ambisonic RIR, ACN/SN3D
rir = dfdn.render(source_azimuth=0.0, source_elevation=0.0)
```

## References

1. B. Alary, A. Politis, S. J. Schlecht, and V. Välimäki, "Directional Feedback
   Delay Network," J. Audio Eng. Soc., vol. 67, no. 10, pp. 752-762, 2019.
2. B. Alary and A. Politis, "Frequency-Dependent Directional Feedback Delay
   Network," in Proc. IEEE ICASSP, Barcelona, Spain, 2020, pp. 176-180.
3. B. Alary, P. Massé, S. J. Schlecht, M. Noisternig, and V. Välimäki,
   "Perceptual Analysis of Directional Late Reverberation," J. Acoust. Soc.
   Am., vol. 149, no. 5, pp. 3189-3200, 2021.
4. B. Alary and V. Välimäki, "A Method for Capturing and Reproducing
   Directional Reverberation in Six Degrees of Freedom," in Proc. Immersive
   and 3D Audio: from Architecture to Automotive (I3DA), Bologna, Italy, 2021.
