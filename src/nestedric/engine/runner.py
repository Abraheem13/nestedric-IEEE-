"""Config resolution, seed sweeps, run directories and result serialisation.

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from pathlib import Path


def run_experiment(cfg: dict, out_dir: Path) -> Path:
    """Run one (method, stream, seed) experiment; write results.json + checkpoints."""
    raise NotImplementedError("Day 3")


def run_sweep(cfg: dict, out_dir: Path) -> list[Path]:
    """Run a grid over methods x streams x seeds."""
    raise NotImplementedError("Day 3")
