"""Continual-learning metrics computed from the T x T evaluation matrix R, where\nR[i, j] is performance on environment j after training through environment i.

Status: STUB -- implemented on Day 4 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import numpy as np


def average_performance(R: np.ndarray) -> float:
    """Mean performance over all environments after the final one."""
    raise NotImplementedError("Day 4")


def backward_transfer(R: np.ndarray) -> float:
    """BWT: mean change on earlier environments caused by later training."""
    raise NotImplementedError("Day 4")


def forward_transfer(R: np.ndarray, random_baseline: np.ndarray) -> float:
    """FWT: benefit on unseen environments relative to a random-init reference."""
    raise NotImplementedError("Day 4")


def forgetting_measure(R: np.ndarray) -> float:
    """Mean over environments of (best-ever performance - final performance)."""
    raise NotImplementedError("Day 4")


def adaptation_latency(curve: np.ndarray, target: float) -> float:
    """Steps (and wall-clock seconds) to recover *target* performance after a shift.

    This is the metric that speaks directly to the RANPilot-style operational cost of
    re-adaptation, and it is where multi-timescale nesting should win most clearly.
    """
    raise NotImplementedError("Day 4")
