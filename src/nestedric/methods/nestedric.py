"""NestedRIC as a Method: wires models.nested.NestedRIC into the continual engine.

Status: STUB -- implemented on Day 5-7 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from nestedric.methods import register


@register("nestedric")
class NestedRICMethod:
    """Proposed method. Ablation axes are all exposed through ``cfg``."""

    def __init__(self, model, cfg) -> None:
        raise NotImplementedError("Day 5-7")

    def begin_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 5-7")

    def observe(self, batch, step: int) -> dict:
        raise NotImplementedError("Day 5-7")

    def end_environment(self, env, env_index: int) -> None:
        raise NotImplementedError("Day 5-7")

    def predict(self, batch):
        raise NotImplementedError("Day 5-7")

    def footprint(self) -> dict:
        raise NotImplementedError("Day 5-7")
