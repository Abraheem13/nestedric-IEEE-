# Theory: the frequency-separation forgetting bound

Working notes. The proof is written on Day 11; this file holds the statement, the
assumptions, and the proof strategy so that the empirical design on Days 8-10 collects
exactly the quantities the proof needs.

## Setup

A two-level nested learner. The fast level (parameters phi) updates every tau_f
optimiser steps; the slow level (parameters psi) updates every tau_s steps, with
separation ratio rho = tau_s / tau_f >= 1. Environments arrive sequentially; the
distribution shift between consecutive environments has drift rate delta, measured
empirically by `data/drift.estimate_drift_rate`.

Note rho = 1 is exactly the single-timescale case, which reduces to naive fine-tuning
with a memory. This degeneracy check is the analogue of the kernel operator collapsing
to the source mean in the path-loss work, and it must be verified numerically in
`tests/test_bound.py`.

## Target statement (Theorem 1, to be sharpened)

Under assumptions A1-A4 below, the backward-transfer degradation satisfies

    |BWT| <= C1 * delta * f(rho) + C2 / n_eff + O(higher order)

with f strictly decreasing on [1, rho_max], f(1) = 1, so that the single-timescale case
is recovered exactly and any separation strictly improves the bound.

## Companion statement (Proposition 2)

The risk-optimal separation ratio rho* solves a bias-variance trade-off between
stability (large rho: the slow level retains) and plasticity (small rho: the fast level
adapts), giving rho* as an explicit decreasing function of delta. Prediction: **high
drift favours smaller separation**, which is directly testable in the Day-10 sweep.

## Assumptions to state honestly

- A1: bounded gradients / Lipschitz loss.
- A2: drift between consecutive environments bounded by delta in a stated metric.
- A3: the slow level's update is a contraction toward the retained solution.
- A4: within-environment sampling is exchangeable at the trace level (not i.i.d. at the
  sample level — environments are spatially and temporally correlated, and pretending
  otherwise is exactly the error the statistical protocol is designed to avoid).

## Proof strategy

1. Decompose the post-shift loss into a retention term and an adaptation term.
2. Bound the retention term by the slow level's drift accumulated over tau_s steps.
3. Bound the adaptation term by the fast level's convergence within one environment.
4. Combine; observe that increasing rho shrinks (2) while inflating (3), giving f and
   the optimal rho*.
5. Verify against `theory/simulate.py` (toy quadratic drift), then against the real
   Day-10 sweep.

## What the experiments must therefore log

- the realised separation ratio for every run
- the estimated drift rate for every environment transition
- effective sample size per environment
- per-environment BWT, not just the aggregate

If these are not logged from Day 8, the theorem cannot be checked without re-running.
