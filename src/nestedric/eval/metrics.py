"""Continual-learning metrics over the T x T evaluation matrix.

``R[i, j]`` is performance on environment j after training through environment i, so
the lower triangle holds retention, the diagonal holds fresh performance, and the upper
triangle holds transfer to environments not yet seen.

Every metric here takes performance where **higher is better**. The benchmark's raw
losses are errors, so the evaluator converts once, at the boundary, rather than leaving
each metric to remember the sign -- which is how a backward-transfer number ends up
reported with the wrong sign in a table.
"""

from __future__ import annotations

import numpy as np


def _check(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype="float64")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got {R.shape}")
    return R


def average_performance(R: np.ndarray) -> float:
    """Mean performance over all environments after the final one."""
    R = _check(R)
    return float(np.mean(R[-1]))


def backward_transfer(R: np.ndarray) -> float:
    """BWT: mean change on earlier environments caused by later training.

    Negative means forgetting. This is the headline quantity the bound is about.
    """
    R = _check(R)
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))


def per_environment_bwt(R: np.ndarray) -> np.ndarray:
    """BWT per environment, not just the aggregate.

    docs/THEORY.md requires this: the bound is checked per transition, and an aggregate
    cannot be disaggregated afterwards without re-running everything.
    """
    R = _check(R)
    T = R.shape[0]
    return np.array([R[-1, j] - R[j, j] for j in range(T - 1)], dtype="float64")


def forward_transfer(R: np.ndarray, random_baseline: np.ndarray) -> float:
    """FWT: benefit on unseen environments relative to a random-init reference."""
    R = _check(R)
    b = np.asarray(random_baseline, dtype="float64")
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[i - 1, i] - b[i] for i in range(1, T)]))


def forgetting_measure(R: np.ndarray) -> float:
    """Mean over environments of (best-ever performance - final performance)."""
    R = _check(R)
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([np.max(R[: T - 1, j]) - R[-1, j] for j in range(T - 1)]))


def adaptation_latency(curve: np.ndarray, target: float) -> float:
    """Steps to reach *target* performance after a shift; ``inf`` if never reached.

    The operationally meaningful metric for a RIC: a method that eventually recovers but
    takes a thousand steps has still failed the control loop it was meant to serve.
    """
    curve = np.asarray(curve, dtype="float64")
    hits = np.nonzero(curve >= target)[0]
    return float(hits[0]) if len(hits) else float("inf")
