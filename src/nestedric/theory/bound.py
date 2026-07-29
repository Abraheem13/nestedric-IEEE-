"""Frequency-separation forgetting bound.\n\nInformal statement (to be sharpened on Day 11):\n  For a two-level nested learner whose slow level updates every tau_s steps and\n  fast level every tau_f steps, under drift rate delta between consecutive\n  environments, the backward-transfer degradation obeys\n      |BWT| <= C1 * delta * f(tau_s / tau_f) + C2 / n_eff\n  with f decreasing in the separation ratio, recovering naive fine-tuning as the\n  degenerate single-timescale case tau_s = tau_f.\n\nA companion proposition characterises the risk-optimal ratio (tau_s/tau_f)* as a\nfunction of delta -- the stability/plasticity trade-off.

Status: STUB -- implemented on Day 11 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import numpy as np


def forgetting_bound(drift_rate: float, ratio: float, n_eff: float, consts: dict) -> float:
    """Evaluate the analytical upper bound on |BWT|."""
    raise NotImplementedError("Day 11")


def optimal_ratio(drift_rate: float, consts: dict) -> float:
    """Risk-optimal separation ratio (tau_s / tau_f)* for a given drift rate."""
    raise NotImplementedError("Day 11")


def check_bound(measured_bwt: np.ndarray, ratios: np.ndarray, drift: np.ndarray, consts: dict) -> dict:
    """Verify empirically that measured |BWT| lies below the bound, and report slack."""
    raise NotImplementedError("Day 11")
