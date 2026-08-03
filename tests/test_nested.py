"""Nested levels and the level-scheduled optimiser.

The Day 6 gate is that level-1 updates fire exactly every tau_s steps. That is asserted
here rather than eyeballed in a log, along with the property it exists to produce: slow
parameters must actually move less than fast ones.
"""

import pytest

torch = pytest.importorskip("torch")

from nestedric.models.backbone import build_backbone  # noqa: E402
from nestedric.models.deep_optimizer import DeepMomentum, LevelScheduledOptimizer  # noqa: E402
from nestedric.models.nested import NestedRIC  # noqa: E402

MODEL_CFG = {
    "encoder": {"type": "gru", "hidden": 16, "n_layers": 2, "dropout": 0.0},
    "heads": {"prediction": {"out_dim": 2}, "policy": {"n_actions": 3}},
}
BUDGET = 200_000


def _model(n_levels=2, periods=(1, 8), self_modifying=True):
    return NestedRIC(
        build_backbone(MODEL_CFG, in_dim=19),
        n_levels=n_levels,
        periods=periods,
        memory_budget_bytes=BUDGET,
        self_modifying=self_modifying,
    )


def _batch(n=4, window=6, in_dim=19):
    g = torch.Generator().manual_seed(0)
    return (
        torch.randn(n, window, in_dim, generator=g),
        torch.randn(n, 2, generator=g),
        torch.randint(0, 3, (n,), generator=g),
    )


# ------------------------------------------------------------------ the Day 6 gate


def test_slow_level_fires_exactly_every_tau_s_steps():
    model = _model(periods=(1, 8))
    fired = [model.due_levels(s) for s in range(25)]
    slow_steps = [s for s, levels in enumerate(fired) if 1 in levels]
    assert slow_steps == [0, 8, 16, 24]
    assert all(0 in levels for levels in fired)  # fast level fires every step


def test_three_levels_fire_on_their_own_schedules():
    model = _model(n_levels=3, periods=(1, 4, 16))
    assert model.due_levels(4) == [0, 1]
    assert model.due_levels(16) == [0, 1, 2]
    assert model.due_levels(5) == [0]


def test_optimizer_steps_only_due_levels():
    model = _model(periods=(1, 8))
    opt = LevelScheduledOptimizer(
        dict(model.named_parameters()), model.parameter_levels, model.periods, lr=0.1
    )
    assert opt.due_levels(3) == [0]
    assert opt.due_levels(8) == [0, 1]


def test_slow_parameters_move_less_than_fast_ones():
    """The mechanism, not just the schedule: separation must change what moves."""
    torch.manual_seed(0)
    model = _model(periods=(1, 16))
    named = dict(model.named_parameters())
    opt = LevelScheduledOptimizer(named, model.parameter_levels, model.periods, lr=0.05)

    fast_names = [n for n in model.parameter_levels[0] if n in named]
    slow_names = [n for n in model.parameter_levels[1] if n in named]
    assert fast_names and slow_names

    before = {n: p.detach().clone() for n, p in named.items()}
    x, y, a = _batch()
    for step in range(1, 16):  # never a multiple of 16, so the slow level never fires
        opt.zero_grad()
        pred, logits = model(x)
        loss = torch.nn.functional.mse_loss(pred, y) + torch.nn.functional.cross_entropy(logits, a)
        loss.backward()
        opt.step(step)

    fast_move = sum(float((named[n] - before[n]).detach().abs().sum()) for n in fast_names)
    slow_move = sum(float((named[n] - before[n]).detach().abs().sum()) for n in slow_names)
    assert fast_move > 0
    assert slow_move == pytest.approx(0.0, abs=1e-9)


def test_ungated_gradients_do_not_accumulate_on_idle_levels():
    """A slow level must not take one huge step carrying tau_s batches of gradient."""
    model = _model(periods=(1, 4))
    named = dict(model.named_parameters())
    opt = LevelScheduledOptimizer(named, model.parameter_levels, model.periods, lr=0.01)

    x, y, a = _batch()
    for step in (1, 2, 3):
        opt.zero_grad()
        pred, logits = model(x)
        (torch.nn.functional.mse_loss(pred, y)).backward()
        opt.step(step)
        for name in model.parameter_levels[1]:
            if name in named:
                assert named[name].grad is None


def test_single_level_is_the_degenerate_case():
    """n_levels=1 must reduce to ordinary training -- the theorem recovers it at rho=1."""
    model = _model(n_levels=1, periods=(1,))
    assert model.due_levels(7) == [0]
    assert model.parameter_levels[0]
    assert model.memory.separation_ratios == ()


# ------------------------------------------------------------- self-modification


def test_lr_gain_is_bounded():
    """An unbounded gain lets the model score on retention by refusing to learn."""
    model = _model()
    model.memory.blocks[-1].write(torch.randn(4, 16) * 50, torch.randn(4, 16) * 50)
    gain = float(model.fast_lr_gain())
    assert 0.5 <= gain <= 1.5


def test_gain_is_exactly_one_when_self_modification_is_off():
    model = _model(self_modifying=False)
    model.memory.blocks[-1].write(torch.randn(4, 16), torch.randn(4, 16))
    assert float(model.fast_lr_gain()) == 1.0


def test_gain_is_one_before_the_slow_memory_is_written():
    assert float(_model().fast_lr_gain()) == 1.0


def test_gain_scales_only_the_fast_level():
    model = _model(periods=(1, 4))
    named = dict(model.named_parameters())
    opt = LevelScheduledOptimizer(named, model.parameter_levels, model.periods, lr=0.1)
    base_slow = opt.optimizers[1].param_groups[0]["lr"]
    opt.step(4, fast_gain=1.4)
    assert opt.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.14)
    assert opt.optimizers[1].param_groups[0]["lr"] == pytest.approx(base_slow)


# ---------------------------------------------------------------- deep optimizer


def test_deep_momentum_depth_one_is_plain_momentum():
    p = torch.nn.Parameter(torch.zeros(3))
    opt = DeepMomentum([p], lr=0.1, memory_depth=1)
    p.grad = torch.ones(3)
    opt.step()
    assert torch.allclose(p.detach(), torch.full((3,), -0.01), atol=1e-6)


def test_deep_momentum_keeps_one_state_per_depth():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = DeepMomentum([p], lr=0.1, memory_depth=3)
    p.grad = torch.ones(4)
    opt.step()
    assert len(opt.state[p]["memories"]) == 3
    assert opt.state_bytes() == 3 * 4 * 4


def test_deep_momentum_rejects_zero_depth():
    with pytest.raises(ValueError, match="memory_depth"):
        DeepMomentum([torch.nn.Parameter(torch.zeros(1))], memory_depth=0)


# ------------------------------------------------------------------- accounting


def test_forward_shapes_match_the_backbone():
    model = _model()
    x, _, _ = _batch()
    pred, logits = model(x)
    assert pred.shape == (4, 2)
    assert logits.shape == (4, 3)


def test_memory_respects_the_shared_budget():
    for n_levels, periods in ((1, (1,)), (2, (1, 8)), (3, (1, 4, 16))):
        model = _model(n_levels=n_levels, periods=periods)
        assert model.state_bytes() <= BUDGET


def test_summary_carries_what_the_theory_needs():
    model = _model(periods=(1, 8))
    s = model.summary()
    assert s["periods"] == [1, 8]
    assert s["separation_ratios"] == [8.0]
    assert "fast_lr_gain" in s and "occupancy" in s
