"""Drift measurement, and controlled non-stationarity injection.

The Day 4 gate produced a result the Day 1 design did not predict: ``radio-shift``,
built from COMMAG's mobility and distance axes precisely because they were expected to
be the strongest source of forgetting, showed |BWT| = 0.0006, while ``sched-shift``
showed 0.0443. Two orders of magnitude, in the opposite order to the prediction.

The explanation this module exists to test is that "drift" is not one quantity:

* **Covariate drift** moves P(X). The inputs look different; the mapping from inputs to
  targets is unchanged. A model that has learned the mapping is not harmed by it.
* **Concept drift** moves P(Y|X). The same inputs now imply different targets, so
  fitting the new environment necessarily overwrites the old mapping. This is what
  catastrophic forgetting *is*.

Distance and mobility change the radio conditions a UE experiences -- the marginal.
Scheduling policy changes how the scheduler responds to a given buffer and channel
state -- the conditional. If the measurements below show ``radio-shift`` with high
covariate drift and near-zero concept drift, the hypothesis holds and the paper gains a
sharp finding about *which* O-RAN shifts cause forgetting.

It also matters for Theorem 1. The bound is stated in a single drift rate delta, with
|BWT| increasing in it. If |BWT| tracks concept drift and ignores covariate drift, then
delta must be *defined* as concept drift, or the theorem is false in a way a reviewer
would find with one counterexample from our own benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from nestedric.data.loaders import Normaliser, build_windows

#: Windows sampled per environment for the drift probes. The estimates are stable well
#: below the full environment and this keeps the whole sweep to a few minutes.
DEFAULT_SAMPLE = 4000

#: Ridge penalty for the concept probes. Large enough that the probe is stable on 600
#: correlated inputs, small enough that it still fits the mapping.
RIDGE_ALPHA = 10.0


def _sample(rng: np.random.Generator, x: np.ndarray, n: int) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), size=n, replace=False)]


def covariate_drift(x_a: np.ndarray, x_b: np.ndarray) -> float:
    """Mean per-feature 1-Wasserstein distance between two environments' inputs.

    Computed on the last timestep of each window, on already-standardised features, so
    the distance is in units of source-environment standard deviations and comparable
    across features and streams.
    """
    a = x_a[:, -1, :]
    b = x_b[:, -1, :]
    return float(np.mean([wasserstein_distance(a[:, k], b[:, k]) for k in range(a.shape[1])]))


def label_drift(actions_a: np.ndarray, actions_b: np.ndarray, n_actions: int = 3) -> float:
    """Total-variation distance between the two action-label distributions.

    A cheap marginal check: if the *action mix* changes, some of the apparent concept
    drift is really prior shift, which is worth separating.
    """
    pa = np.bincount(actions_a, minlength=n_actions) / max(len(actions_a), 1)
    pb = np.bincount(actions_b, minlength=n_actions) / max(len(actions_b), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def concept_drift(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> float:
    """How much the input-to-target mapping itself changes, on shared inputs.

    Fits a ridge probe on each environment, then compares their predictions **on the
    same inputs**. Evaluating on common support is what separates this from covariate
    drift: if the two probes agree everywhere they are shown the same X, the mapping did
    not change and only the input distribution moved.

    Symmetrised over both environments' inputs, and scaled by the variance of the
    targets so the number is a fraction of explainable signal rather than an
    unit-dependent MSE.
    """
    from sklearn.linear_model import Ridge

    fa = x_a.reshape(len(x_a), -1)
    fb = x_b.reshape(len(x_b), -1)

    probe_a = Ridge(alpha=alpha).fit(fa, y_a)
    probe_b = Ridge(alpha=alpha).fit(fb, y_b)

    disagreement = 0.5 * (
        np.mean((probe_a.predict(fb) - probe_b.predict(fb)) ** 2)
        + np.mean((probe_a.predict(fa) - probe_b.predict(fa)) ** 2)
    )
    scale = 0.5 * (np.var(y_a) + np.var(y_b)) + 1e-12
    return float(disagreement / scale)


def transfer_gap(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> float:
    """Loss increase when the probe fitted on A is applied to B, relative to B's own.

    The operational version of concept drift: how much worse a model does simply by
    having been fitted somewhere else. Unlike :func:`concept_drift` it is affected by
    both kinds of shift, which is exactly why both are reported.
    """
    from sklearn.linear_model import Ridge

    fa = x_a.reshape(len(x_a), -1)
    fb = x_b.reshape(len(x_b), -1)
    probe_a = Ridge(alpha=alpha).fit(fa, y_a)
    probe_b = Ridge(alpha=alpha).fit(fb, y_b)

    own = np.mean((probe_b.predict(fb) - y_b) ** 2)
    foreign = np.mean((probe_a.predict(fb) - y_b) ** 2)
    return float((foreign - own) / (own + 1e-12))


def estimate_drift_rate(
    stream,
    processed_dir: str | Path,
    normaliser: Normaliser,
    window: int = 32,
    stride: int = 8,
    sample: int = DEFAULT_SAMPLE,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per-transition drift between consecutive environments of a stream.

    Returns one record per transition with covariate, concept, label and transfer-gap
    measures. This is the empirical stand-in for the drift rate delta in
    :mod:`nestedric.theory.bound`, and the per-transition granularity is what
    docs/THEORY.md requires -- an aggregate cannot be disaggregated afterwards.
    """
    processed_dir = Path(processed_dir)
    rng = np.random.default_rng(seed)

    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load(env):
        if env.env_id not in cache:
            ws = build_windows(env, processed_dir, normaliser, env.train_traces, window, stride)
            idx = (
                rng.choice(len(ws.x), size=min(sample, len(ws.x)), replace=False)
                if len(ws.x)
                else np.empty(0, dtype=int)
            )
            cache[env.env_id] = (ws.x[idx], ws.y[idx], ws.actions[idx])
        return cache[env.env_id]

    records: list[dict[str, Any]] = []
    envs = list(stream)
    for i in range(len(envs) - 1):
        a, b = envs[i], envs[i + 1]
        xa, ya, aa = load(a)
        xb, yb, ab = load(b)
        if not len(xa) or not len(xb):
            continue
        records.append(
            {
                "stream": stream.name,
                "transition": f"{a.env_id}->{b.env_id}",
                "from": a.env_id,
                "to": b.env_id,
                "covariate_drift": covariate_drift(xa, xb),
                "concept_drift": concept_drift(xa, ya, xb, yb),
                "label_drift": label_drift(aa, ab),
                "transfer_gap": transfer_gap(xa, ya, xb, yb),
                "n_a": int(len(xa)),
                "n_b": int(len(xb)),
            }
        )
    return records


def inject_drift(stream, magnitude: float, kind: str = "traffic_scale", seed: int = 0):
    """Return a copy of *stream* with synthetic drift of controlled magnitude."""
    raise NotImplementedError("Day 10")
