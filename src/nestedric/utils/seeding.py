"""Deterministic seeding across python/numpy/torch/cudnn.

Status: STUB -- implemented on Day 2 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and optionally force deterministic cuDNN kernels."""
    raise NotImplementedError("Day 2")
