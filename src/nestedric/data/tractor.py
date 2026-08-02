"""Adapter for the TRACTOR 5G KPI traces (genesys-lab.org/tractor).

Status: STUB -- implemented on Day 1 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_URL = ""  # TODO(Day 1): pin the exact release / commit hash.
LICENCE = ""  # TODO(Day 1): record licence for the artefact release.


def download(dest: Path) -> Path:
    """Fetch the raw archive into ``dest`` and return the extracted root."""
    raise NotImplementedError("Day 1")


def load_raw(root: Path) -> pd.DataFrame:
    """Read the raw CSVs without harmonisation."""
    raise NotImplementedError("Day 1")


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw columns onto ``nestedric.data.schema`` columns and units."""
    raise NotImplementedError("Day 1")


def prepare(root: Path, out: Path) -> Path:
    """``download`` -> ``load_raw`` -> ``to_canonical`` -> parquet. Returns output path."""
    raise NotImplementedError("Day 1")
