"""Compute/memory accounting against near-RT RIC feasibility (inference < 10 ms).

Status: STUB -- implemented on Day 4 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations


def measure_footprint(method, batch, device: str = "cuda") -> dict:
    """Return params, peak memory (MB), and p50/p99 inference latency (ms)."""
    raise NotImplementedError("Day 4")


def near_rt_feasible(footprint: dict, budget_ms: float = 10.0) -> bool:
    """Whether the method could actually run inside a near-RT RIC control loop."""
    raise NotImplementedError("Day 4")
