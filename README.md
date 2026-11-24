# directional_fdn

Directional Feedback Delay Network (DFDN) for Ambisonic room impulse response
synthesis: a reverberator whose decay time depends on direction as well as on
frequency. Two formulations by Alary et al. are implemented on top of
[pyFDN](https://github.com/artificial-audio/pyFDN):

- `DirectionalFDN` (`src/directional_fdn.py`), after ICASSP 2020 [2]: one
  channel per incident direction on a spherical grid, with an absorption filter
  per direction. The working reverberator.
- `DFDN2019` (`src/dfdn_2019.py`), after JAES 2019 [1]: spherical-harmonic
  state with a directional weighting matrix. Kept for comparison.

Around them: shoebox image-source early reflections, ISO 9613-1 air
absorption, and directional decay, level and correlation analysis.

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

fdn = DirectionalFDN(sample_rate=48000, num_delay_line_groups=8, sh_order=2, seed=0)

# T60 in ms at the ten octave centres 31.25 Hz to 16 kHz
# (src.fdn_components.geq_band_frequencies). A (Q, 10) array sets it per
# grid direction as well.
t60_ms = np.array([900, 900, 900, 1000, 1100, 1000, 800, 700, 600, 600.0])
fdn.setup(delay_range_ms=(10, 50), t60_per_direction_ms=t60_ms)

# Ambisonic RIR, ACN/SN3D, shape ((sh_order + 1)^2, samples)
rir = fdn.render(source_azimuth=0.0, source_elevation=0.0, length_seconds=1.0)
```

## Examples

- `examples/dfdn_directional.ipynb`: the grid network. Decay per direction and
  frequency, echo density, output decorrelation, early reflections.
- `examples/dfdn_directional_2019.ipynb`: the same outline for the
  spherical-harmonic network.

## Tests

```bash
pytest
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

## License

MIT, see [LICENSE](LICENSE).
