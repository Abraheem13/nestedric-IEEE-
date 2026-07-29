"""YAML config loading with inheritance (`_base_`) and CLI overrides.

Status: STUB -- implemented on Day 2 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, resolve ``_base_`` inheritance, apply ``key=value`` overrides."""
    raise NotImplementedError("Day 2")
