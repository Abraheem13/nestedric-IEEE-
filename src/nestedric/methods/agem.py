"""Averaged Gradient Episodic Memory (Chaudhry et al., 2019).

Status: STUB -- implemented on Day 4 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from nestedric.methods import register


@register("agem")
class Agem:
    """Averaged Gradient Episodic Memory (Chaudhry et al., 2019)."""

    def __init__(self, model, cfg) -> None:
        raise NotImplementedError("Day 4")

    def begin_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 4")

    def observe(self, batch, step: int) -> dict:
        raise NotImplementedError("Day 4")

    def end_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 4")

    def predict(self, batch):
        raise NotImplementedError("Day 4")

    def footprint(self) -> dict:
        raise NotImplementedError("Day 4")
