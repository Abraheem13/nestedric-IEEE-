"""Fold-level statistics: the properties a reviewer will check.

Every test here is a claim the paper makes about its own inference procedure. They are
checked against constructed data where the right answer is known, because a statistics
module that is only exercised on real results cannot be distinguished from one that is
subtly wrong.
"""

import numpy as np
import pytest

from nestedric.utils.stats import (
    cluster_robust_se,
    compare_methods,
    design_effect,
    effect_size,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_permutation_p,
)


def test_paired_ci_covers_a_known_difference():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20)
    a, b = base + 0.5, base  # a constant paired advantage of 0.5
    ci = paired_bootstrap_ci(a, b, seed=0)
    assert ci.low <= 0.5 <= ci.high
    assert ci.excludes_zero


def test_paired_ci_includes_zero_when_methods_are_equivalent():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 20)
    ci = paired_bootstrap_ci(base, base.copy(), seed=0)
    assert ci.estimate == pytest.approx(0.0)
    assert not ci.excludes_zero


def test_pairing_beats_unpaired_when_folds_differ_wildly():
    """The reason comparisons are paired: fold variance dwarfs the method effect."""
    fold_effect = np.array([10.0, -8.0, 4.0, -6.0, 12.0])  # huge between-fold spread
    a = fold_effect + 0.2
    b = fold_effect
    ci = paired_bootstrap_ci(a, b, seed=0)
    assert ci.excludes_zero  # a 0.2 effect is detectable once pairing removes the fold


def test_permutation_p_is_small_for_a_consistent_advantage():
    base = np.arange(8, dtype="float64")
    p = paired_permutation_p(base + 1.0, base, seed=0)
    assert p < 0.05


def test_permutation_p_is_large_for_noise():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 10)
    b = rng.normal(0, 1, 10)
    assert paired_permutation_p(a, b, seed=0) > 0.05


def test_permutation_p_is_never_zero():
    """A finite permutation sample cannot attain zero; claiming it overstates evidence."""
    base = np.arange(12, dtype="float64")
    assert paired_permutation_p(base + 100.0, base, n_perm=100, seed=0) > 0.0


def test_holm_is_a_step_down_procedure():
    """Once one hypothesis fails, every larger p-value fails too.

    Sorted p-values 0.001, 0.030, 0.031 against thresholds 0.0167, 0.025, 0.05: the
    first is rejected, the second fails, and the third fails *despite* clearing its own
    threshold of 0.05. That last part is what makes Holm a step-down procedure rather
    than a per-hypothesis test.
    """
    decisions = holm_bonferroni({"a": 0.001, "b": 0.030, "c": 0.031}, alpha=0.05)
    assert decisions["a"] is True
    assert decisions["b"] is False  # 0.030 > 0.05/2
    assert decisions["c"] is False  # 0.031 < 0.05 but is blocked by b failing


def test_holm_is_more_conservative_than_no_correction():
    pvalues = {f"m{i}": 0.04 for i in range(10)}
    decisions = holm_bonferroni(pvalues, alpha=0.05)
    assert not any(decisions.values())  # each would pass alone; none survives correction


def test_holm_handles_the_empty_case():
    assert holm_bonferroni({}) == {}


def test_effect_size_scales_with_separation():
    base = np.arange(10, dtype="float64")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, 10)
    small = effect_size(base + 0.1 + noise, base)
    large = effect_size(base + 1.0 + noise, base)
    assert abs(large) > abs(small)


def test_design_effect_exceeds_one_when_clusters_are_real():
    """The number the paper quotes to justify fold-level inference."""
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    cluster_means = rng.normal(0, 5, 10)  # strong between-cluster variance
    values = cluster_means[clusters] + rng.normal(0, 0.1, len(clusters))
    assert design_effect(values, clusters) > 10


def test_design_effect_is_about_one_without_clustering():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    values = rng.normal(0, 1, len(clusters))  # cluster label carries no information
    assert design_effect(values, clusters) < 5


def test_cluster_robust_se_grows_with_between_cluster_variance():
    clusters = np.repeat(np.arange(8), 25)
    tight = np.tile(np.linspace(-0.1, 0.1, 25), 8)
    spread = np.repeat(np.linspace(-5, 5, 8), 25)
    assert cluster_robust_se(spread, clusters) > cluster_robust_se(tight, clusters)


def test_compare_methods_returns_everything_a_table_needs():
    rng = np.random.default_rng(0)
    folds = rng.normal(0, 1, 6)
    scores = {
        "nestedric": folds + 0.3,
        "replay": folds + 0.28,
        "finetune": folds,
    }
    rows = compare_methods(scores, reference="finetune", seed=0)

    assert set(rows) == {"nestedric", "replay"}
    for row in rows.values():
        assert {
            "mean_difference",
            "ci_low",
            "ci_high",
            "effect_size",
            "p_value",
            "significant_holm",
            "n_folds",
        } <= set(row)
        assert row["ci_low"] <= row["mean_difference"] <= row["ci_high"]
        assert row["n_folds"] == 6


def test_compare_methods_rejects_an_unknown_reference():
    with pytest.raises(KeyError, match="reference"):
        compare_methods({"a": np.zeros(3)}, reference="missing")


def test_mismatched_fold_counts_raise():
    with pytest.raises(ValueError, match="equal shapes"):
        paired_bootstrap_ci(np.zeros(5), np.zeros(4))
