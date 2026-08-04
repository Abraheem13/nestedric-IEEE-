"""Continuum Memory System: the properties the paper's claim depends on.

Each test here corresponds to something a reviewer could reasonably doubt: that the
levels really do update at different frequencies, that reads are differentiable, that
slow levels retain what fast levels lose, and that the whole stack fits the same byte
budget as replay.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nestedric.models.cms import (  # noqa: E402
    AssociativeMemoryBlock,
    ContinuumMemory,
    capacity_for_budget,
)

DIM = 16


def _kv(n=4, dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g), torch.randn(n, dim, generator=g)


def test_block_updates_exactly_every_tau_steps():
    """The Day 6 gate asserts this; the memory has to honour it first."""
    block = AssociativeMemoryBlock(dim=DIM, capacity=8, update_period=4)
    due = [s for s in range(20) if block.due(s)]
    assert due == [0, 4, 8, 12, 16]


def test_levels_fire_at_their_own_periods():
    cms = ContinuumMemory(dim=DIM, periods=(1, 4, 16), capacity=8)
    k, v = _kv()
    fired = {s: cms.write(s, k, v) for s in range(17)}
    assert fired[1] == [0]
    assert fired[4] == [0, 1]
    assert fired[16] == [0, 1, 2]
    assert [int(b.writes.item()) for b in cms.blocks] == [17, 5, 2]


def test_separation_ratios_are_reported():
    cms = ContinuumMemory(dim=DIM, periods=(1, 32), capacity=8)
    assert cms.separation_ratios == (32.0,)
    assert ContinuumMemory(dim=DIM, periods=(1, 4, 16), capacity=4).separation_ratios == (4.0, 4.0)


def test_equal_periods_are_allowed_because_rho_one_is_the_theorem_control():
    """rho = 1 is the degenerate single-timescale case Theorem 1 must recover.

    The Day 10 ratio sweep starts at periods (1, 1); rejecting it would exclude the
    control the bound is checked against.
    """
    cms = ContinuumMemory(dim=DIM, periods=(1, 1), capacity=8)
    assert cms.separation_ratios == (1.0,)
    k, v = _kv()
    assert cms.write(0, k, v) == [0, 1]  # both levels fire together, as they must


def test_decreasing_periods_are_rejected():
    """An inverted hierarchy, where the 'slow' level updates more often, is an error."""
    with pytest.raises(ValueError, match="non-decreasing"):
        ContinuumMemory(dim=DIM, periods=(8, 2), capacity=8)


def test_read_is_differentiable_with_respect_to_the_query():
    cms = ContinuumMemory(dim=DIM, periods=(1, 8), capacity=8)
    k, v = _kv()
    cms.write(0, k, v)

    q = torch.randn(3, DIM, requires_grad=True)
    out = cms.read(q)
    out.sum().backward()

    assert q.grad is not None
    assert torch.isfinite(q.grad).all()
    assert q.grad.abs().sum() > 0


def test_empty_block_reads_zero_rather_than_noise():
    """Attending over empty slots would inject a constant into the encoder."""
    block = AssociativeMemoryBlock(dim=DIM, capacity=8, update_period=1)
    out = block.read(torch.randn(3, DIM))
    assert torch.equal(out, torch.zeros(3, DIM))


def test_written_content_is_recoverable():
    block = AssociativeMemoryBlock(dim=DIM, capacity=8, update_period=1)
    k, v = _kv(n=3)
    block.write(k, v)
    out = block.read(k * 10.0)  # sharpen attention toward the stored keys
    assert torch.isfinite(out).all()
    # each query should be closest to its own stored value
    for i in range(3):
        d_self = torch.norm(out[i] - v[i])
        d_other = min(torch.norm(out[i] - v[j]) for j in range(3) if j != i)
        assert d_self <= d_other


def test_slow_level_retains_what_the_fast_level_overwrites():
    """The mechanism the whole paper rests on.

    Write environment A, then flood the memory with environment B. The fast level is
    written every step and should be dominated by B; the slow level is written rarely
    and should still resemble A.
    """
    torch.manual_seed(0)
    cms = ContinuumMemory(dim=DIM, periods=(1, 64), capacity=16, write_rate=0.5)

    a_keys, a_values = _kv(n=8, seed=1)
    cms.write(0, a_keys, a_values)  # both levels fire at step 0
    fast_after_a = cms.blocks[0].values.clone()
    slow_after_a = cms.blocks[1].values.clone()

    b_keys, b_values = _kv(n=8, seed=2)
    for step in range(1, 64):  # only the fast level is due
        cms.write(step, b_keys, b_values)

    fast_drift = torch.norm(cms.blocks[0].values - fast_after_a)
    slow_drift = torch.norm(cms.blocks[1].values - slow_after_a)

    assert slow_drift < fast_drift
    assert slow_drift == 0.0  # the slow level was never due in this window


def test_capacity_is_derived_from_a_shared_byte_budget():
    budget = 4_000_000
    cap = capacity_for_budget(budget, dim=128, value_dim=128, n_levels=2)
    cms = ContinuumMemory(dim=128, periods=(1, 32), capacity=cap, value_dim=128)
    assert cms.state_bytes() <= budget
    assert cms.state_bytes() > 0.9 * budget  # and actually uses the budget it is given


def test_more_levels_do_not_buy_more_bytes():
    """Design rule 2: an ablation over n_levels must not confound depth with capacity."""
    budget = 2_000_000
    sizes = []
    for periods in ((1, 32), (1, 8, 64), (1, 4, 16, 64)):
        cap = capacity_for_budget(budget, 64, 64, len(periods))
        sizes.append(ContinuumMemory(dim=64, periods=periods, capacity=cap).state_bytes())
    assert max(sizes) <= budget
    assert max(sizes) / min(sizes) < 1.1


def test_summary_reports_what_the_theory_needs():
    cms = ContinuumMemory(dim=DIM, periods=(1, 32), capacity=8)
    k, v = _kv()
    cms.write(0, k, v)
    s = cms.summary()
    assert s["separation_ratios"] == [32.0]
    assert s["writes"] == [1, 1]
    assert 0.0 < s["occupancy"][0] <= 1.0
    assert s["state_bytes"] > 0
