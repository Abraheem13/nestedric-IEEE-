"""ColO-RAN dataset entry point.

Thin wrapper over :mod:`nestedric.data.colosseum`, which implements the shared
slice-metrics format. Kept as a separate module so configs can name datasets directly.

Source: Polese, Bonati, D'Oro, Basagni, Melodia, "ColO-RAN: Developing Machine
Learning-based xApps for Open RAN Closed-loop Control on Programmable Experimental
Platforms," IEEE Trans. Mobile Computing, vol. 22, no. 10, pp. 5787-5800, 2022.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from nestedric.data.colosseum import COLORAN_REPO, LICENCE
from nestedric.data.colosseum import prepare as _prepare

SOURCE_URL = COLORAN_REPO
DATASET = "coloran"
__all__ = ["SOURCE_URL", "LICENCE", "DATASET", "prepare"]

prepare = partial(_prepare, dataset=DATASET)


def default_root(raw_dir: Path) -> Path:
    """Conventional location after ``scripts/download_data.sh``."""
    return raw_dir / "colosseum-oran-coloran-dataset"
