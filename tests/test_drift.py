"""Drift measures must separate covariate shift from concept shift.

The whole point of these measures is that they answer different questions, so the tests
are constructed controls where the right answer is known: inputs that move with a fixed
mapping, and a fixed input distribution under a changed mapping. If a measure cannot
tell those apart it cannot support the claim the Day 4 gate raised.
"""

import numpy as np
import pytest

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


def _windows(n=200, window=8, features=19, seed=0):
    from nestedric.data.loaders import WindowSet

    rng = np.random.default_rng(seed)
    return WindowSet(
        x=rng.standard_normal((n, window, features)).astype("float32"),
        y=rng.standard_normal((n, 2)).astype("float32"),
        actions=rng.integers(0, 3, size=n),
        trace_index=np.zeros(n, dtype="int32"),
    )


def test_concept_drift_moves_targets_and_leaves_inputs_alone():
    """Concept drift by construction: P(Y|X) moves, P(X) does not."""
    from nestedric.data.drift import inject_drift

    ws = _windows()
    out = inject_drift(ws, env_index=1, magnitude=1.0, kind="concept")
    assert np.array_equal(out.x, ws.x)
    assert not np.allclose(out.y, ws.y)


def test_covariate_drift_moves_inputs_and_leaves_the_mapping_alone():
    from nestedric.data.drift import inject_drift

    ws = _windows()
    out = inject_drift(ws, env_index=1, magnitude=1.0, kind="covariate")
    assert np.array_equal(out.y, ws.y)
    assert not np.allclose(out.x, ws.x)


def test_drift_magnitude_scales_the_perturbation():
    from nestedric.data.drift import inject_drift

    ws = _windows()
    small = inject_drift(ws, 1, 0.5, "concept")
    large = inject_drift(ws, 1, 2.0, "concept")
    assert np.abs(large.y - ws.y).mean() > 3 * np.abs(small.y - ws.y).mean()


def test_environments_disagree_but_each_is_deterministic():
    """The sweep needs the same environment perturbed identically at every magnitude."""
    from nestedric.data.drift import inject_drift

    ws = _windows()
    a1 = inject_drift(ws, 0, 1.0, "concept")
    a2 = inject_drift(ws, 0, 1.0, "concept")
    b = inject_drift(ws, 1, 1.0, "concept")
    assert np.array_equal(a1.y, a2.y)
    assert not np.allclose(a1.y, b.y)


def test_zero_magnitude_is_exactly_the_identity():
    from nestedric.data.drift import inject_drift

    ws = _windows()
    out = inject_drift(ws, 3, 0.0, "concept")
    assert out is ws


def test_constant_inputs_receive_no_concept_drift():
    """A documented degeneracy, not a bug.

    The perturbation is proportional to the input, so a constant-input environment sees
    none. The synthetic smoke corpus is constant after standardisation, which is why the
    smoke drift sweep shows a flat BWT and cannot validate this path -- only real data
    can.
    """
    from nestedric.data.drift import inject_drift
    from nestedric.data.loaders import WindowSet

    ws = WindowSet(
        x=np.zeros((50, 8, 19), dtype="float32"),
        y=np.ones((50, 2), dtype="float32"),
        actions=np.zeros(50, dtype="int64"),
        trace_index=np.zeros(50, dtype="int32"),
    )
    out = inject_drift(ws, 1, 4.0, "concept")
    assert np.array_equal(out.y, ws.y)


def test_unknown_drift_kind_raises():
    from nestedric.data.drift import inject_drift

    with pytest.raises(ValueError, match="unknown drift kind"):
        inject_drift(_windows(), 0, 1.0, "wishful")
