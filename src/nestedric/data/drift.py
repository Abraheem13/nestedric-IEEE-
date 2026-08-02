"""Controlled non-stationarity injection and drift measurement.

Needed both for the contingency plan (public traces may drift too little) and to
obtain the drift rate that enters the frequency-separation bound.

Status: STUB -- implemented on Day 10 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import numpy as np


def estimate_drift_rate(stream, metric: str = "wasserstein") -> np.ndarray:
    """Per-transition distribution distance between consecutive environments.

    This quantity is the empirical stand-in for the drift rate that appears in the
    frequency-separation bound of ``nestedric.theory.bound``.
    """
    raise NotImplementedError("Day 10")


def inject_drift(stream, magnitude: float, kind: str = "traffic_scale", seed: int = 0):
    """Return a copy of *stream* with synthetic drift of controlled magnitude."""
    raise NotImplementedError("Day 10")
