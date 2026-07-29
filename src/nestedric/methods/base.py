"""The Method protocol every baseline and NestedRIC must satisfy.

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from typing import Protocol


class Method(Protocol):
    """Uniform interface so the engine treats all methods identically."""

    def begin_environment(self, env, env_index: int) -> None:
        """Hook called before training on a new environment."""

    def observe(self, batch, step: int) -> dict:
        """One optimisation step. Returns a dict of scalars to log."""

    def end_environment(self, env, env_index: int) -> None:
        """Hook called after finishing an environment (e.g. Fisher/importance update)."""

    def predict(self, batch):
        """Inference used by the evaluator."""

    def footprint(self) -> dict:
        """Parameter count, memory bytes and per-step latency -- for the RIC budget check."""
