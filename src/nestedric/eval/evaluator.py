"""Builds the T x T evaluation matrix and writes the canonical results record.

Status: STUB -- implemented on Day 4 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations


class ContinualEvaluator:
    """Evaluates on every seen (and unseen) environment after each environment."""

    def __init__(self, stream, cfg) -> None:
        raise NotImplementedError("Day 4")

    def evaluate_all(self, method, after_env_index: int) -> dict:
        raise NotImplementedError("Day 4")

    def finalise(self) -> dict:
        """Return all metrics plus the raw R matrix for the results artefact."""
        raise NotImplementedError("Day 4")
