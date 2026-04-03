"""wind_shear — sustained cross-wind shear producing a pitch+yaw moment.

Models flying into a wind-shear layer that creates a persistent pitching and
yawing disturbance (no direct roll component). The moment ramps up linearly
over RAMP_TIME, holds at peak for HOLD_TIME, then ramps back down.

Parameters (edit below):
    PITCH_MAG   — peak pitch moment [N·m]
    YAW_MAG     — peak yaw moment [N·m]
    T_START     — time the shear begins [s]
    RAMP_TIME   — duration of the ramp-up (and ramp-down) phase [s]
    HOLD_TIME   — duration of the sustained peak phase [s]
"""

import numpy as np

DESCRIPTION = "Wind-shear pitch+yaw disturbance: ramp-hold-ramp profile starting at t=6 s"

PITCH_MAG = 0.3   # N·m
YAW_MAG   = 0.2   # N·m
T_START   = 6.0   # s
RAMP_TIME = 0.5   # s
HOLD_TIME = 1.0   # s


def disturbance(t: float) -> np.ndarray:
    t_ramp_end  = T_START + RAMP_TIME
    t_hold_end  = t_ramp_end + HOLD_TIME
    t_decay_end = t_hold_end + RAMP_TIME

    if t < T_START or t > t_decay_end:
        scale = 0.0
    elif t < t_ramp_end:
        scale = (t - T_START) / RAMP_TIME
    elif t < t_hold_end:
        scale = 1.0
    else:
        scale = 1.0 - (t - t_hold_end) / RAMP_TIME

    return np.array([0.0, scale * PITCH_MAG, scale * YAW_MAG])
