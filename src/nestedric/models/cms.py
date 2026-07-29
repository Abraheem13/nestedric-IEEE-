"""Continuum Memory System (CMS).\n\nA stack of associative-memory blocks, each with its own update period. Block i is\nrefreshed every tau_i steps with tau_0 < tau_1 < ... < tau_{L-1}; the frequency\nseparation ratio tau_{i+1}/tau_i is the central knob of this paper.

Status: STUB -- implemented on Day 5 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AssociativeMemoryBlock(nn.Module):
    """A key-value associative memory updated by an inner objective.

    Following the Nested Learning view, the block's update rule is itself a learned
    optimisation step rather than a fixed write.
    """

    def __init__(self, dim: int, capacity: int, update_period: int) -> None:
        super().__init__()
        raise NotImplementedError("Day 5")

    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        raise NotImplementedError("Day 5")

    def read(self, queries: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Day 5")


class ContinuumMemory(nn.Module):
    """Frequency-tiered stack of :class:`AssociativeMemoryBlock`.

    Parameters
    ----------
    periods
        Update periods ``(tau_0, ..., tau_{L-1})`` in optimiser steps. In the RIC mapping
        these are anchored to the near-RT (10 ms - 1 s) and non-RT (> 1 s) control loops.
    """

    def __init__(self, dim: int, periods: tuple[int, ...], capacity: int) -> None:
        super().__init__()
        raise NotImplementedError("Day 5")

    @property
    def separation_ratios(self) -> tuple[float, ...]:
        """``(tau_1/tau_0, tau_2/tau_1, ...)`` -- reported with every run."""
        raise NotImplementedError("Day 5")
