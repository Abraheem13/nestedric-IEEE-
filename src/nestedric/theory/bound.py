"""Frequency-separation forgetting bound.

**Statement (Theorem 1).** For a two-level nested learner whose fast level updates every
tau_f steps and slow level every tau_s steps, with separation ratio rho = tau_s / tau_f,
under concept drift of rate delta between consecutive environments, the backward-transfer
degradation obeys

    |BWT| <= C_r * delta / rho  +  C_a * delta**2 * rho  +  C_n / n_eff

The three terms are the decomposition the proof follows (docs/THEORY.md):

* **Retention**, ``C_r * delta / rho``. The slow level is overwritten once per tau_s
  steps, so the drift it absorbs per environment falls as the separation grows. This is
  the term that motivates separation at all, and it is the one the original statement
  captured alone.
* **Adaptation**, ``C_a * delta**2 * rho``. A slow level that rarely updates is stale,
  and staleness costs more when the environment has moved further -- hence delta squared.
  This term *grows* with rho and was missing from the Day 0 sketch, which is why that
  sketch predicted "more separation is always better".
* **Estimation**, ``C_n / n_eff``, from finite samples per environment; independent of
  rho and delta.

**Proposition 2.** Minimising over rho gives

    rho* = sqrt(C_r / (C_a * delta))

which decreases in delta: high drift favours *less* separation. A fixed rho is therefore
optimal at exactly one drift rate and harmful far from it.

**Why this form and not the Day 0 one.** The Day 10 sweep measured a fixed rho = 32
against rho = 1 under injected drift and found separation helping at delta = 0.25 and
0.50 and *hurting* at delta = 1.00. A bound monotone in rho cannot produce that
reversal; this one does, and predicts it in the right direction. The functional form was
chosen to be consistent with a measurement that already existed, which is a weaker claim
than deriving it first -- and the paper must say so rather than presenting the fit as a
prediction.

At rho = 1 the bound reduces to ``C_r * delta + C_a * delta**2 + C_n / n_eff``, the
single-timescale case, which is the degeneracy check ``tests/test_bound.py`` enforces.
"""

from __future__ import annotations

import numpy as np

#: Default constants. Fitted on the drift sweep by :func:`fit_constants`; these are
#: placeholders so the shape can be exercised before a fit exists.
DEFAULT_CONSTS: dict[str, float] = {"C_r": 1.0, "C_a": 1.0, "C_n": 1.0}


def forgetting_bound(
    drift_rate: float | np.ndarray,
    ratio: float | np.ndarray,
    n_eff: float | np.ndarray,
    consts: dict | None = None,
) -> np.ndarray:
    """Evaluate the analytical upper bound on |BWT|.

    Vectorised over any broadcastable combination of *drift_rate*, *ratio* and *n_eff*,
    so the Day 13 figure can evaluate a surface in one call.
    """
    c = {**DEFAULT_CONSTS, **(consts or {})}
    delta = np.asarray(drift_rate, dtype="float64")
    rho = np.asarray(ratio, dtype="float64")
    n = np.asarray(n_eff, dtype="float64")

    if np.any(rho < 1):
        raise ValueError("separation ratio must be >= 1; rho = 1 is the degenerate case")

    retention = c["C_r"] * delta / rho
    adaptation = c["C_a"] * delta**2 * rho
    estimation = c["C_n"] / np.maximum(n, 1.0)
    return retention + adaptation + estimation


def optimal_ratio(drift_rate: float, consts: dict | None = None) -> float:
    """Risk-optimal separation ratio ``rho*`` for a given drift rate (Proposition 2).

    ``rho* = sqrt(C_r / (C_a * delta))``, decreasing in delta. Clipped at 1 because a
    ratio below one inverts the hierarchy rather than degenerating it: at drift high
    enough that rho* would fall below 1, the prediction is simply "do not separate".
    """
    c = {**DEFAULT_CONSTS, **(consts or {})}
    if drift_rate <= 0:
        return float("inf")  # no drift: separation costs nothing, so any rho will do
    return float(max(1.0, np.sqrt(c["C_r"] / (c["C_a"] * drift_rate))))


def fit_constants(
    drift: np.ndarray,
    ratios: np.ndarray,
    measured_bwt: np.ndarray,
    n_eff: np.ndarray | float = 1e4,
) -> dict[str, float]:
    """Least-squares fit of ``(C_r, C_a, C_n)`` to measured |BWT|.

    The bound is an upper bound, so fitting it to the mean of the measurements would
    make it an approximation rather than a bound. The fit is therefore constrained to
    lie above every observation: constants are scaled up until no measurement violates
    them, and the reported slack is what the paper quotes as tightness.
    """
    delta = np.asarray(drift, dtype="float64")
    rho = np.asarray(ratios, dtype="float64")
    y = np.abs(np.asarray(measured_bwt, dtype="float64"))
    n = np.broadcast_to(np.asarray(n_eff, dtype="float64"), y.shape)

    design = np.stack([delta / rho, delta**2 * rho, 1.0 / np.maximum(n, 1.0)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    coeffs = np.maximum(coeffs, 0.0)  # negative constants have no interpretation

    predicted = design @ coeffs
    with np.errstate(divide="ignore", invalid="ignore"):
        shortfall = np.max(np.where(predicted > 0, y / predicted, 1.0))
    if np.isfinite(shortfall) and shortfall > 1:
        coeffs = coeffs * shortfall  # lift until it bounds every observation

    return {"C_r": float(coeffs[0]), "C_a": float(coeffs[1]), "C_n": float(coeffs[2])}


def check_bound(
    measured_bwt: np.ndarray,
    ratios: np.ndarray,
    drift: np.ndarray,
    consts: dict | None = None,
    n_eff: np.ndarray | float = 1e4,
) -> dict:
    """Verify empirically that measured |BWT| lies below the bound, and report slack.

    Slack is the ratio of bound to measurement: 1.0 is tight, large values mean the
    bound is true but uninformative. A bound that holds only because it is loose is
    worth little, so the paper reports the distribution rather than the fact of holding.
    """
    y = np.abs(np.asarray(measured_bwt, dtype="float64"))
    bound = forgetting_bound(drift, ratios, n_eff, consts)
    bound = np.broadcast_to(bound, y.shape)

    violations = y > bound
    with np.errstate(divide="ignore", invalid="ignore"):
        slack = np.where(y > 0, bound / y, np.inf)

    return {
        "holds": bool(not violations.any()),
        "n_violations": int(violations.sum()),
        "n_points": int(y.size),
        "worst_violation": float(np.max(y - bound)) if violations.any() else 0.0,
        "median_slack": float(np.median(slack[np.isfinite(slack)])),
        "max_slack": float(np.max(slack[np.isfinite(slack)])),
    }


def crossover_drift(consts: dict | None = None, ratio: float = 32.0) -> float:
    """Drift rate at which a *fixed* ``ratio`` stops beating no separation at all.

    Separation at ``rho`` beats ``rho = 1`` when

        C_r * delta / rho + C_a * delta**2 * rho  <  C_r * delta + C_a * delta**2

    which rearranges to ``C_a * delta < C_r / rho``, hence

        delta_cross = C_r / (C_a * rho)

    This is the quantity the Day 10 sweep can falsify: it measured separation at
    rho = 32 helping at delta = 0.25 and 0.50 and hurting at delta = 1.00, so a fitted
    bound must place delta_cross between 0.5 and 1.0.

    Distinct from :func:`ratio_is_optimal_at`, which asks when *rho* is the best ratio
    rather than merely better than none. The two differ by a factor of rho, and
    conflating them is easy: an earlier draft of this module did exactly that.
    """
    c = {**DEFAULT_CONSTS, **(consts or {})}
    return float(c["C_r"] / (c["C_a"] * ratio))


def ratio_is_optimal_at(consts: dict | None = None, ratio: float = 32.0) -> float:
    """Drift rate at which ``ratio`` is exactly the risk-optimal separation.

    Inverting Proposition 2: ``delta = C_r / (C_a * rho**2)``. Below this the configured
    ratio is smaller than optimal, above it larger -- but it keeps beating rho = 1 until
    :func:`crossover_drift`, a factor of rho further on.
    """
    c = {**DEFAULT_CONSTS, **(consts or {})}
    return float(c["C_r"] / (c["C_a"] * ratio**2))
