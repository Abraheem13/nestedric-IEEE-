"""Make ``src/`` importable so ``pytest`` works in a fresh clone without installing.

Day 15 requires that a fresh clone reproduces the smoke run; requiring ``pip install -e .``
before the test suite will even collect is a papercut between a reviewer and that claim.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
