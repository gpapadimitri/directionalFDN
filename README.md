# directional_fdn — Directional Feedback Delay Network

Clean Python implementation of the **Directional Feedback Delay Network (DFDN)** for Ambisonics room impulse response synthesis.

Based on:

> Benoit, A., Musikka, M., & Välimäki, V. (2019). *Directional Feedback Delay Network*. Proc. DAFx.

## Features

| Feature | Description |
|---------|-------------|
| Kronecker feedback | `A ⊗ I_Q` preserves Ambisonics channel structure (Eq. 20) |
| Directional weighting T_i | Anisotropic T60 per delay-line group (Section 3) |
| Ambisonics I/O | ACN/SN3D encoding, orders 1–3 |
| Frequency-dependent T60 | IIR absorption via warped Prony's method |
| Paper-accurate modes | Omnidirectional input (Eq. 17) or directional variant |

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from directional_fdn import DirectionalFDN

dfdn = DirectionalFDN(
    sample_rate=48000,
    num_delay_line_groups=4,
    ambisonics_order=1,
    frame_size=64,
    seed=42,
)

dfdn.setup(delay_range_ms=(10, 30), des_t60_ms=800)

# Optional: enable anisotropic decay
dfdn.enable_directional_weighting(sharpness=0.5, diffusion=0.3)

# Render a 1-second Ambisonics RIR for a source at 45° azimuth
rir = dfdn.render(
    source_azimuth=0.785,  # π/4
    source_elevation=0.0,
    length_seconds=1.0,
)

print(rir.shape)  # (4, 48000) — W, Y, Z, X channels
```

## Architecture

```
x(n) ──► b_i ──► [ Delay-line groups s_i(n) ] ──► c_i ──► y(n)
                          │                          ▲
                          ▼                          │
                      T_i (opt.)                     │
                          │                          │
                          ▼                          │
                    Absorption g_i                   │
                          │                          │
                          ▼                          │
                    A ⊗ I_Q ─────────────────────────┘
```

## Project structure

```
directional_fdn/
├── pyproject.toml
├── README.md
├── src/directional_fdn/
│   ├── __init__.py
│   ├── utils.py
│   └── fdn/
│       ├── __init__.py
│       ├── absorption.py       # Gain / IIR / SOS absorption
│       ├── delay_line.py       # Circular FIFO delay buffer
│       ├── directional_fdn.py  # DirectionalFDN (main class)
│       ├── fdn.py              # Base FDN engine
│       ├── feedback_matrix.py  # Unitary / Hadamard matrices
│       ├── filter_fitting.py   # Warped Prony IIR fitting
│       ├── spatial.py          # ACN/SN3D Ambisonics encoding
│       └── transform.py        # Directional weighting T_i
├── examples/
│   └── basic_dfdn.py
└── tests/
    └── test_dfdn.py
```

## License

MIT
