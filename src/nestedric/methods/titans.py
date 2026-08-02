"""Single-timescale test-time memory (Titans) -- frequency-separation ablation.

Status: STUB -- implemented on Day 4 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from nestedric.methods import register


@register("titans")
class Titans:
    """Single-timescale test-time memory (Titans) -- frequency-separation ablation."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
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
