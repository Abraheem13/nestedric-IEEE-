"""COMMAG dataset entry point.

Thin wrapper over :mod:`nestedric.data.colosseum`. COMMAG contributes the mobility
(static/slow), distance (close/medium/far) and slice-assignment (mixed/traffic) axes
that ColO-RAN holds fixed -- these are the source of genuine radio-condition shift.

Source: Bonati, D'Oro, Polese, Basagni, Melodia, "Intelligence and Learning in O-RAN
for Data-driven NextG Cellular Networks," IEEE Communications Magazine, vol. 59,
no. 10, pp. 21-27, October 2021.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from nestedric.data.colosseum import COMMAG_REPO, LICENCE
from nestedric.data.colosseum import prepare as _prepare

SOURCE_URL = COMMAG_REPO
DATASET = "commag"
__all__ = ["SOURCE_URL", "LICENCE", "DATASET", "prepare"]

prepare = partial(_prepare, dataset=DATASET)


def default_root(raw_dir: Path) -> Path:
    """Conventional location after ``scripts/download_data.sh``."""
    return raw_dir / "colosseum-oran-commag-dataset"
