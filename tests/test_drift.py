"""Drift measures must separate covariate shift from concept shift.

The whole point of these measures is that they answer different questions, so the tests
are constructed controls where the right answer is known: inputs that move with a fixed
mapping, and a fixed input distribution under a changed mapping. If a measure cannot
tell those apart it cannot support the claim the Day 4 gate raised.
"""

import numpy as np

from nestedric.data.drift import concept_drift, covariate_drift, label_drift, transfer_gap

N, WINDOW, FEAT = 500, 4, 3


def _linear_env(rng, shift=0.0, weights=None):
    x = (rng.standard_normal((N, WINDOW, FEAT)) + shift).astype("float32")
    w = weights if weights is not None else rng.standard_normal((WINDOW * FEAT, 2))
    y = (x.reshape(N, -1) @ w).astype("float32")
    return x, y, w


def test_covariate_shift_is_seen_by_covariate_drift_only():
    """Inputs move two standard deviations; the mapping is untouched."""
    rng = np.random.default_rng(0)
    xa, ya, w = _linear_env(rng)
    xb, yb, _ = _linear_env(rng, shift=2.0, weights=w)

    assert covariate_drift(xa, xb) > 1.5
    assert concept_drift(xa, ya, xb, yb) < 0.01


def test_concept_shift_is_seen_by_concept_drift_only():
    """Same inputs; the mapping is replaced."""
    rng = np.random.default_rng(0)
    xa, ya, _ = _linear_env(rng)
    _, yb, _ = _linear_env(rng, weights=rng.standard_normal((WINDOW * FEAT, 2)))
    yb = (xa.reshape(N, -1) @ rng.standard_normal((WINDOW * FEAT, 2))).astype("float32")

    assert covariate_drift(xa, xa) < 1e-6
    assert concept_drift(xa, ya, xa, yb) > 0.5


def test_identical_environments_have_no_drift():
    rng = np.random.default_rng(0)
    xa, ya, _ = _linear_env(rng)
    assert covariate_drift(xa, xa) < 1e-6
    assert concept_drift(xa, ya, xa, ya) < 1e-6
    assert transfer_gap(xa, ya, xa, ya) < 1e-3


def test_transfer_gap_grows_with_concept_shift():
    rng = np.random.default_rng(0)
    xa, ya, w = _linear_env(rng)
    mild = (xa.reshape(N, -1) @ (w + 0.1 * rng.standard_normal(w.shape))).astype("float32")
    severe = (xa.reshape(N, -1) @ rng.standard_normal(w.shape)).astype("float32")
    assert transfer_gap(xa, ya, xa, mild) < transfer_gap(xa, ya, xa, severe)


def test_label_drift_is_a_total_variation_distance():
    assert label_drift(np.zeros(100, dtype=int), np.zeros(100, dtype=int)) == 0.0
    assert label_drift(np.zeros(100, dtype=int), np.ones(100, dtype=int)) == 1.0
    mixed = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    assert 0.0 < label_drift(np.zeros(100, dtype=int), mixed) < 1.0
