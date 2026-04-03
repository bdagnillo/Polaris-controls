"""default — rectangular roll-torque pulse.

Applies a fixed-magnitude roll moment for 0.2 s starting at t = 8 s.
This reproduces the disturbance that was previously hardcoded in simulate.py.

Parameters (edit below):
    MAGNITUDE   — peak moment magnitude [N·m]
    T_START     — time the pulse begins [s]
    T_END       — time the pulse ends [s]
"""

import numpy as np

DESCRIPTION = "Rectangular roll-torque pulse: 0.5 N·m from t=8.0 to t=8.2 s"

MAGNITUDE = 0.5   # N·m
T_START   = 8.0   # s
T_END     = 8.2   # s


def disturbance(t: float) -> np.ndarray:
    if T_START < t < T_END:
        return np.array([MAGNITUDE, 0.0, 0.0])
    return np.zeros(3)
