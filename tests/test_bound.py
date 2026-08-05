"""The bound's stated properties, checked as arithmetic rather than asserted in prose.

Each test is a sentence the paper makes about Theorem 1 or Proposition 2. If one fails,
the corresponding sentence is wrong.
"""

import numpy as np
import pytest

from nestedric.theory.bound import (
    check_bound,
    fit_constants,
    forgetting_bound,
    optimal_ratio,
    crossover_drift,
    ratio_is_optimal_at,
)

CONSTS = {"C_r": 1.0, "C_a": 0.01, "C_n": 0.0}


def test_bound_degenerates_to_finetune_at_ratio_one():
    """rho = 1 is the single-timescale case the theorem must recover exactly."""
    delta = 0.3
    at_one = forgetting_bound(delta, 1.0, n_eff=np.inf, consts=CONSTS)
    expected = CONSTS["C_r"] * delta + CONSTS["C_a"] * delta**2
    assert at_one == pytest.approx(expected)


def test_retention_term_falls_with_separation():
    """The reason to separate at all."""
    small = forgetting_bound(0.1, 1.0, np.inf, {"C_r": 1.0, "C_a": 0.0, "C_n": 0.0})
    large = forgetting_bound(0.1, 64.0, np.inf, {"C_r": 1.0, "C_a": 0.0, "C_n": 0.0})
    assert large < small


def test_adaptation_term_grows_with_separation():
    """The term the Day 0 sketch omitted, and the reason more separation is not free."""
    small = forgetting_bound(0.1, 1.0, np.inf, {"C_r": 0.0, "C_a": 1.0, "C_n": 0.0})
    large = forgetting_bound(0.1, 64.0, np.inf, {"C_r": 0.0, "C_a": 1.0, "C_n": 0.0})
    assert large > small


def test_bound_has_an_interior_minimum_in_rho():
    """Proposition 2's content: neither extreme of rho is optimal."""
    rhos = np.linspace(1, 200, 400)
    values = forgetting_bound(0.05, rhos, np.inf, CONSTS)
    best = rhos[np.argmin(values)]
    assert 1 < best < 200
    assert best == pytest.approx(optimal_ratio(0.05, CONSTS), rel=0.05)


def test_optimal_ratio_decreases_with_drift():
    """High drift favours less separation -- the prediction the Day 10 sweep tests."""
    ratios = [optimal_ratio(d, CONSTS) for d in (0.01, 0.1, 0.5, 1.0, 4.0)]
    assert all(a > b for a, b in zip(ratios, ratios[1:], strict=False))


def test_optimal_ratio_never_inverts_the_hierarchy():
    """At drift high enough, the prediction is 'do not separate', not 'separate backwards'."""
    assert optimal_ratio(1e6, CONSTS) == 1.0


def test_no_drift_leaves_separation_costless():
    assert optimal_ratio(0.0, CONSTS) == float("inf")


def test_ratio_is_optimal_at_inverts_proposition_two():
    for ratio in (4.0, 32.0, 128.0):
        delta = ratio_is_optimal_at(CONSTS, ratio=ratio)
        assert optimal_ratio(delta, CONSTS) == pytest.approx(ratio, rel=1e-6)


def test_crossover_and_optimality_are_different_drift_levels():
    """Conflating them is a factor-of-rho error, and an earlier draft made it.

    A ratio stops being *optimal* long before it stops being *better than nothing*.
    """
    ratio = 32.0
    assert crossover_drift(CONSTS, ratio) == pytest.approx(
        ratio * ratio_is_optimal_at(CONSTS, ratio)
    )


def test_a_fixed_ratio_helps_below_the_reversal_and_hurts_above():
    """The shape the drift sweep measured: separation pays, then stops paying."""
    ratio = 32.0
    delta_star = crossover_drift(CONSTS, ratio)

    below, above = delta_star * 0.5, delta_star * 2.0
    for delta, expect_help in ((below, True), (above, False)):
        separated = forgetting_bound(delta, ratio, np.inf, CONSTS)
        single = forgetting_bound(delta, 1.0, np.inf, CONSTS)
        # bool() because the bound returns numpy scalars, and np.True_ is not True.
        assert bool(separated < single) is expect_help


def test_fitted_constants_bound_every_observation():
    """An upper bound fitted to the mean would be an approximation, not a bound."""
    rng = np.random.default_rng(0)
    drift = np.repeat([0.0, 0.25, 0.5, 1.0], 4)
    ratios = np.tile([1.0, 32.0], 8)
    measured = 0.05 * drift / ratios + 0.02 * drift**2 * ratios
    measured = measured + rng.normal(0, 1e-4, measured.shape)

    consts = fit_constants(drift, ratios, measured, n_eff=1e4)
    result = check_bound(measured, ratios, drift, consts, n_eff=1e4)

    assert result["holds"], result
    assert result["n_violations"] == 0


def test_check_bound_reports_a_violation_when_one_exists():
    measured = np.array([10.0])
    result = check_bound(measured, np.array([32.0]), np.array([0.1]), CONSTS, n_eff=1e4)
    assert not result["holds"]
    assert result["n_violations"] == 1
    assert result["worst_violation"] > 0


def test_slack_is_reported_so_a_loose_bound_cannot_pass_as_a_tight_one():
    tight = check_bound(np.array([0.0313]), np.array([1.0]), np.array([0.03]), CONSTS, np.inf)
    loose = check_bound(np.array([1e-6]), np.array([1.0]), np.array([0.03]), CONSTS, np.inf)
    assert loose["median_slack"] > tight["median_slack"]


def test_ratio_below_one_is_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        forgetting_bound(0.1, 0.5, np.inf, CONSTS)
