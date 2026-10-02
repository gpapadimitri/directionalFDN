# directional_fdn — Directional Feedback Delay Network

Directional Feedback Delay Network (DFDN) for Ambisonic room impulse response synthesis.
Two formulations, both by Alary et al.:

- `DirectionalFDN` (`src/fdn/directional_fdn.py`) -- ICASSP 2020: one channel per
  incident direction on a spatial grid, per-direction absorption. The working
  reverberator.
- `DFDN2019` (`src/fdn/dfdn_2019.py`) -- JAES 2019:
  spherical-harmonic state with a directional weighting matrix. Archival.

Motivated from the following publications:

    - B. Alary, A. Politis, S. Schlecht, and V. Välimäki, "Directional Feedback Delay Network," Journal of the Audio Engineering Society, vol. 67, no. 10, pp. 752–762, 2019.

    - B. Alary and A. Politis, "Frequency-Dependent Directional Feedback Delay Network," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020.

    - B. Alary, P. Massé, S. J. Schlecht, M. Noisternig, and V. Välimäki, "Perceptual Analysis of Directional Late Reverberation," The Journal of the Acoustical Society of America, vol. 149, no. 5, pp. 3189–3200, 2021.

    - B. Alary and V. Välimäki, "A Method for Capturing and Reproducing Directional Reverberation in Six Degrees of Freedom," in Proc. 2021 Immersive and 3D Audio: from Architecture to Automotive (I3DA), Bologna, Italy, 2021.

