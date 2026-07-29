"""Continual-learning methods. All expose the same `Method` protocol.

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator adding a method to the registry used by the CLI."""

    def _wrap(cls):
        REGISTRY[name] = cls
        return cls

    return _wrap


def build_method(name: str, **kwargs):
    """Instantiate a registered method by name."""
    raise NotImplementedError("Day 3")
