"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from nestedric.methods import register


@register("ewc")
class Ewc:
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""

    def __init__(self, model, cfg) -> None:
        raise NotImplementedError("Day 3")

    def begin_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 3")

    def observe(self, batch, step: int) -> dict:
        raise NotImplementedError("Day 3")

    def end_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 3")

    def predict(self, batch):
        raise NotImplementedError("Day 3")

    def footprint(self) -> dict:
        raise NotImplementedError("Day 3")
