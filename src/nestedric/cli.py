"""Command line entry point. `nestedric <command> --config <path>`.

Status: STUB -- implemented on Day 2 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser.

    Sub-commands
    ------------
    prepare   : materialise raw datasets into the canonical KPI parquet schema.
    stream    : build an O-RAN-CL environment stream from a stream config.
    train     : run one (method, stream, seed) continual-learning experiment.
    evaluate  : recompute metrics from a finished run directory.
    ablate    : sweep one axis of the NestedRIC configuration.
    figures   : regenerate all paper figures and tables from results/.
    """
    raise NotImplementedError("Day 2")


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError("Day 2")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
