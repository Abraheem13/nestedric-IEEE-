"""Statistical treatment.\n\nEnvironments within a dataset are correlated, so the unit of analysis is the\nstream/fold, not the sample. Provides paired bootstrap CIs over folds, Holm-\nBonferroni correction across methods, cluster-robust variance, and effect sizes.

Status: STUB -- implemented on Day 12 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05):
    """Bootstrap CI on the paired mean difference across folds."""
    raise NotImplementedError("Day 12")


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    raise NotImplementedError("Day 12")


def cluster_robust_se(values: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error with clusters = environments/traces."""
    raise NotImplementedError("Day 12")


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size computed on fold-level differences."""
    raise NotImplementedError("Day 12")
