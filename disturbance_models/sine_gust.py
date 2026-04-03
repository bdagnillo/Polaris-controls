"""sine_gust — sinusoidal roll-torque burst.

A single sine-wave lobe centred at T_CENTRE with total duration DURATION.
The moment is zero outside the burst window, and smoothly rises and falls
within it (no sharp edges, so less stiff for the integrator than a pulse).

Parameters (edit below):
    MAGNITUDE   — peak moment magnitude [N·m]
    T_CENTRE    — time of peak moment [s]
    DURATION    — full width of the burst (zero-to-zero) [s]
"""

import numpy as np

DESCRIPTION = "Sinusoidal roll-torque burst: 1.0 N·m peak centred at t=8.1 s, duration 0.4 s"

MAGNITUDE = 1.0    # N·m
T_CENTRE  = 8.1    # s
DURATION  = 0.4    # s


def disturbance(t: float) -> np.ndarray:
    t_start = T_CENTRE - DURATION / 2.0
    t_end   = T_CENTRE + DURATION / 2.0
    if t_start < t < t_end:
        phase = np.pi * (t - t_start) / DURATION   # 0 → π over the burst
        mx = MAGNITUDE * np.sin(phase)
        return np.array([mx, 0.0, 0.0])
    return np.zeros(3)
