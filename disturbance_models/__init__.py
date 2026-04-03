"""disturbance_models — loader for named disturbance functions.

Each model module in this package must expose:
    DESCRIPTION : str          — one-line human-readable description
    disturbance(t) -> ndarray  — 3-vector [Mx, My, Mz] in N·m at time t

Usage:
    from disturbance_models import load
    dist_fn = load("default")
    moment = dist_fn(t)
"""

import importlib
import numpy as np


def load(name: str):
    """Import disturbance_models.<name> and return its disturbance callable."""
    try:
        mod = importlib.import_module(f"disturbance_models.{name}")
    except ModuleNotFoundError:
        raise SystemExit(
            f"[disturbance] Unknown model '{name}'. "
            f"Add disturbance_models/{name}.py to define it."
        )
    if not hasattr(mod, "disturbance"):
        raise SystemExit(
            f"[disturbance] disturbance_models/{name}.py must define "
            f"a callable named 'disturbance'."
        )
    return mod.disturbance
