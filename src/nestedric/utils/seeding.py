"""Deterministic seeding across python/numpy/torch/cudnn.

Reproducibility is a release requirement (Day 15: a fresh clone must reproduce the
smoke run), and it is also a debugging tool -- a result that moves when nothing changed
is a bug hunt that cannot start until the run is deterministic.

``derive_seed`` exists because a stream, its environment splits and each method's
initialisation must be reproducible *independently*. Reusing one global counter would
mean that adding a method to a sweep silently changed the data splits of every other
method, which would invalidate the paired comparisons the statistics depend on.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

#: numpy's legacy seed ceiling, the tightest constraint in play.
_SEED_MAX = 0xFFFFFFFF


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG and optionally force deterministic cuDNN kernels.

    Determinism costs throughput on the L4 (cuDNN cannot pick the fastest algorithm),
    so it is a flag rather than a given -- but it defaults on, because a fast result
    that cannot be reproduced is not a result.
    """
    random.seed(seed)
    np.random.seed(seed % (_SEED_MAX + 1))
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # the data-side tooling runs without torch installed
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def derive_seed(*parts: object) -> int:
    """Derive a stable 32-bit seed from arbitrary components.

    Stable across processes and platforms, unlike ``hash()``, which is randomised per
    interpreter and must not decide anything whose value reaches the paper.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def seeded_rng(*parts: object) -> np.random.Generator:
    """A numpy Generator seeded by :func:`derive_seed` of *parts*."""
    return np.random.default_rng(derive_seed(*parts))
