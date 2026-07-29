"""The continual-learning loop shared by every method (no method-specific branches).

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations


class ContinualTrainer:
    """Iterate the stream, call the Method hooks, drive the ContinualEvaluator."""

    def __init__(self, method, stream, evaluator, cfg) -> None:
        raise NotImplementedError("Day 3")

    def run(self) -> dict:
        """Execute the full stream and return the results record."""
        raise NotImplementedError("Day 3")
