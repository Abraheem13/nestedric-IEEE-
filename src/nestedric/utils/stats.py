"""Statistical treatment.

Environments within a dataset are correlated, so the unit of analysis is the
stream/fold, not the sample. Provides paired bootstrap CIs over folds, Holm-Bonferroni
correction across methods, cluster-robust variance, and effect sizes.

**Why the unit matters here specifically.** A cross-dataset environment holds ~450,000
rows drawn from 240 traces, and rows 250 ms apart within a trace are near-duplicates.
Treating rows as independent inflates the effective sample size by roughly the design
effect ``1 + (n_bar - 1) * rho_intra``; with ~1,900 rows per trace and even a modest
intra-trace correlation that is two orders of magnitude. Every p-value computed that way
is meaningless, and it is the easiest thing in the paper for a reviewer to attack,
because the resulting intervals are visibly too narrow to be true.

So nothing here consumes samples. These functions take **fold-level** numbers -- one
value per (stream, seed) cell -- and every comparison is paired within fold, because
every method saw the same environments and the same splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval and the fold count behind it."""

    estimate: float
    low: float
    high: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero."""
        return (self.low > 0) or (self.high < 0)

    def format(self, digits: int = 4) -> str:
        """Render as ``estimate [low, high]`` for a paper table."""
        return f"{self.estimate:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Bootstrap CI on the mean of a single fold-level quantity."""
    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        raise ValueError("need at least one fold")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(x.mean()), float(low), float(high), int(len(x)))


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI on the paired mean difference ``a - b`` across folds.

    Resamples *folds*, not samples, keeping each fold's pair together: both methods saw
    the same environments and splits, so pairing removes the between-fold variance that
    would otherwise swamp the comparison.

    With five seeds this is a small-sample procedure and the interval will be wide. That
    width is the honest uncertainty, not a defect to be engineered away by switching to
    a per-sample test.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal shapes, got {a.shape} and {b.shape}")
    if a.size == 0:
        raise ValueError("paired comparison needs at least one fold")

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(diff.mean()), float(low), float(high), int(len(diff)))


def paired_permutation_p(
    a: np.ndarray, b: np.ndarray, n_perm: int = 10_000, seed: int = 0
) -> float:
    """Two-sided p-value for a paired difference, by sign-flipping folds.

    Exact under the null that the two methods are exchangeable within each fold, which
    is the right null when the fold is the randomisation unit. Preferred over a t-test
    because five folds cannot support a normality assumption.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    diff = a - b
    if diff.size == 0:
        raise ValueError("need at least one fold")

    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    # +1 top and bottom: a finite permutation sample cannot attain p = 0, and reporting
    # zero would overstate the evidence.
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni correction across methods; returns per-key rejection decisions.

    The benchmark compares nine baselines against the proposed method across five
    streams. Uncorrected, one comparison in twenty clears 0.05 by chance, so at this
    many comparisons a spurious win is more likely than not.
    """
    if not pvalues:
        return {}
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    decisions: dict[str, bool] = {}
    still_rejecting = True
    for i, (key, p) in enumerate(ordered):
        threshold = alpha / (m - i)
        # Step-down: once one hypothesis fails, every larger p-value fails too,
        # whatever its own threshold would have allowed.
        still_rejecting = still_rejecting and (p <= threshold)
        decisions[key] = still_rejecting
    return decisions


def cluster_robust_se(values: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error with clusters = environments or traces.

    For quantities that must be computed at sample level, this at least prices in the
    within-cluster correlation instead of ignoring it. Fold-level statistics remain
    preferred; this exists for per-environment diagnostics where folds do not apply.
    """
    values = np.asarray(values, dtype="float64")
    clusters = np.asarray(clusters)
    if values.shape[0] != clusters.shape[0]:
        raise ValueError("values and clusters must align")

    unique = np.unique(clusters)
    g = len(unique)
    if g < 2:
        return float("nan")

    grand = values.mean()
    cluster_means = np.array([values[clusters == c].mean() for c in unique])
    cluster_sizes = np.array([np.sum(clusters == c) for c in unique], dtype="float64")

    weighted = np.sum(cluster_sizes * (cluster_means - grand) ** 2)
    correction = g / (g - 1)
    return float(np.sqrt(correction * weighted) / values.shape[0])


def design_effect(values: np.ndarray, clusters: np.ndarray) -> float:
    """``1 + (n_bar - 1) * rho_intra``: how much clustering inflates naive significance.

    Reported in the paper so a reader can see the size of the error the fold-level
    protocol avoids, rather than being asked to take it on trust.
    """
    values = np.asarray(values, dtype="float64")
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return float("nan")

    sizes = np.array([np.sum(clusters == c) for c in unique], dtype="float64")
    means = np.array([values[clusters == c].mean() for c in unique])

    between = means.var(ddof=1)
    within = float(
        np.mean(
            [
                values[clusters == c].var(ddof=1) if np.sum(clusters == c) > 1 else 0.0
                for c in unique
            ]
        )
    )
    total = between + within
    if total <= 0:
        return 1.0
    return float(1 + (sizes.mean() - 1) * (between / total))


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size (Cohen's d_z) on fold-level differences.

    Reported beside every p-value: with five folds, significance and practical
    relevance diverge in both directions, and a table of stars without magnitudes tells
    a reader nothing about whether a method is worth deploying.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    diff = a - b
    sd = diff.std(ddof=1) if len(diff) > 1 else 0.0
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float(np.sign(diff.mean()) * np.inf)
    return float(diff.mean() / sd)


def compare_methods(
    scores: dict[str, np.ndarray],
    reference: str,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, dict]:
    """Compare every method against *reference* at fold level, with correction.

    Returns per-method CI, effect size, raw p-value and Holm-corrected decision --
    computed once, so the results table cannot disagree with the text.
    """
    if reference not in scores:
        raise KeyError(f"reference {reference!r} not among {sorted(scores)}")

    ref = np.asarray(scores[reference], dtype="float64")
    raw: dict[str, float] = {}
    rows: dict[str, dict] = {}

    for name, values in scores.items():
        if name == reference:
            continue
        values = np.asarray(values, dtype="float64")
        ci = paired_bootstrap_ci(values, ref, n_boot=n_boot, alpha=alpha, seed=seed)
        p = paired_permutation_p(values, ref, seed=seed)
        raw[name] = p
        rows[name] = {
            "mean_difference": ci.estimate,
            "ci_low": ci.low,
            "ci_high": ci.high,
            "n_folds": ci.n,
            "effect_size": effect_size(values, ref),
            "p_value": p,
        }

    for name, rejected in holm_bonferroni(raw, alpha=alpha).items():
        rows[name]["significant_holm"] = rejected

    return rows
