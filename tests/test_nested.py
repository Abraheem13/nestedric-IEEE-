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
    gain = float(model.fast_lr_gain().detach())
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


# ------------------------------------------------------------ NestedRIC as a Method


def _method(**overrides):
    from nestedric.methods import build_method

    cfg = {
        "n_levels": 2,
        "periods": [1, 8],
        "self_modifying": True,
        "memory": {"budget_mb": 0.2},
        "deep_optimizer": {"enabled": True, "memory_depth": 2},
        "optimizer": {"type": "adam", "lr": 0.01},
    }
    cfg.update(overrides)
    return build_method("nestedric", build_backbone(MODEL_CFG, in_dim=19), cfg)


def test_nestedric_takes_a_step_and_logs_its_schedule():
    method = _method()
    logs = method.observe(_batch(), step=0)
    assert logs["levels_fired"] == 2  # step 0: everything is due
    assert logs["levels_written"] == 2
    logs = method.observe(_batch(), step=1)
    assert logs["levels_fired"] == 1  # only the fast level


def test_nestedric_shares_the_backbone_capacity():
    """Design rule 1: the trunk must be the baselines' trunk."""
    from nestedric.methods import build_method

    baseline = build_backbone(MODEL_CFG, in_dim=19).n_parameters()
    backbone = build_backbone(MODEL_CFG, in_dim=19)
    build_method("nestedric", backbone, {"memory": {"budget_mb": 0.2}})
    assert backbone.n_parameters() == baseline


def test_nestedric_declares_memory_and_optimiser_bytes():
    method = _method()
    method.observe(_batch(), step=0)
    fp = method.footprint()
    assert fp["memory_bytes"] > 0
    assert fp["optimizer_bytes"] > 0
    assert fp["total_bytes"] == fp["param_bytes"] + fp["memory_bytes"] + fp["optimizer_bytes"]


def test_memory_persists_across_environments():
    """Carrying structure between environments is the point; only the counter resets."""
    method = _method()
    method.observe(_batch(), step=0)
    writes_before = [int(b.writes.item()) for b in method.model.memory.blocks]
    method.begin_environment(None, 1)
    assert [int(b.writes.item()) for b in method.model.memory.blocks] == writes_before
    assert method._step_in_env == 0


def test_realised_ratio_is_reported_not_assumed():
    method = _method(periods=[1, 8])
    for step in range(16):
        method.observe(_batch(), step=step)
    summary = method.state_summary()
    assert summary["realised_ratios"] == [8.0]
    assert summary["separation_ratios"] == [8.0]


def test_single_level_config_runs_and_reports_no_separation():
    method = _method(n_levels=1, periods=[1])
    logs = method.observe(_batch(), step=0)
    assert logs["levels_fired"] == 1
    assert method.state_summary()["separation_ratios"] == []


def test_self_modification_can_be_switched_off():
    method = _method(self_modifying=False)
    logs = method.observe(_batch(), step=0)
    assert logs["lr_gain"] == 1.0


def test_deep_optimizer_can_be_switched_off():
    from nestedric.models.deep_optimizer import DeepMomentum

    method = _method(deep_optimizer={"enabled": False})
    assert not any(isinstance(o, DeepMomentum) for o in method.optimizer.optimizers.values())


def test_nestedric_is_byte_comparable_with_replay():
    """The comparison that decides the paper has to be at equal bytes."""
    from nestedric.methods import build_method

    nested = _method(memory={"budget_mb": 1.0})
    nested.observe(_batch(), step=0)

    replay = build_method("replay", build_backbone(MODEL_CFG, in_dim=19), {"memory_budget_mb": 1.0})
    replay.observe(_batch(), step=0)

    # NestedRIC also carries optimiser memory, which replay does not; the memory
    # itself is what must match the budget.
    assert nested.model.state_bytes() <= 1.05e6
    assert replay.extra_state_bytes() <= 1.05e6


def test_self_modification_head_actually_receives_gradient():
    """The self-modification head must receive gradient.

    Used only as a learning-rate multiplier the gain is detached, so the modulator would
    never train, and the ablation would compare a fixed random constant against nothing.
    """
    model = _model(periods=(1, 4), self_modifying=True)
    model.memory.blocks[-1].write(torch.randn(4, 16), torch.randn(4, 16))

    x, y, _ = _batch()
    pred, _ = model(x)
    torch.nn.functional.mse_loss(pred, y).backward()

    grads = [p.grad for p in model.modulator.parameters() if p.grad is not None]
    assert grads, "modulator received no gradient at all"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_no_gain_path_when_self_modification_is_off():
    model = _model(self_modifying=False)
    model.memory.blocks[-1].write(torch.randn(4, 16), torch.randn(4, 16))
    x, y, _ = _batch()
    pred, _ = model(x)
    torch.nn.functional.mse_loss(pred, y).backward()
    assert all(
        p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.modulator.parameters()
    )


def test_slow_levels_accumulate_gradients_rather_than_discarding_them():
    """Frequency separation must be about timescale, not about training budget.

    The first implementation zeroed gradients for levels that were not due, so a
    period-32 level saw one gradient in 32. That confounds rho with the amount of
    training, which would have made the Day 10 ratio sweep meaningless: larger rho
    would lower |BWT| simply by learning less.
    """
    import torch

    from nestedric.models.deep_optimizer import LevelScheduledOptimizer

    fast = torch.nn.Parameter(torch.zeros(3))
    slow = torch.nn.Parameter(torch.zeros(3))
    opt = LevelScheduledOptimizer(
        named_parameters={"fast": fast, "slow": slow},
        parameter_levels={0: ["fast"], 1: ["slow"]},
        periods=(1, 4),
        lr=0.1,
        deep=False,
    )

    # A constant gradient on every step. After four steps the slow level has fired
    # once, on the average of four identical gradients -- so it should have moved by
    # one ordinary step, not by a quarter of one and not by four.
    for step in range(1, 5):
        fast.grad = torch.ones(3)
        slow.grad = torch.ones(3)
        opt.step(step)

    assert torch.allclose(slow.detach(), torch.full((3,), -0.1), atol=1e-6)


def test_accumulated_gradient_is_the_mean_not_the_sum():
    """A level integrating k gradients acts on their average, not their total.

    Checked on the gradient presented to the optimiser at firing time rather than on
    the displacement, because Adam normalises step size and would hide the difference.
    """
    import torch

    from nestedric.models.deep_optimizer import LevelScheduledOptimizer

    slow = torch.nn.Parameter(torch.zeros(2))
    opt = LevelScheduledOptimizer(
        named_parameters={"slow": slow},
        parameter_levels={0: ["slow"]},
        periods=(4,),
        lr=1.0,
        deep=False,
    )
    for step, g in zip(range(1, 5), (1.0, 2.0, 3.0, 4.0), strict=True):
        slow.grad = torch.full((2,), g)
        fired = opt.step(step)

    assert fired == [0]  # period 4 fires at step 4
    # mean of 1,2,3,4 is 2.5; their sum would be 10
    assert torch.allclose(slow.grad, torch.full((2,), 2.5), atol=1e-6)


def test_accumulators_are_reported_in_the_footprint():
    """Accumulators are real memory this method spends and a baseline does not."""
    import torch

    from nestedric.models.deep_optimizer import LevelScheduledOptimizer

    slow = torch.nn.Parameter(torch.zeros(100))
    opt = LevelScheduledOptimizer(
        named_parameters={"slow": slow},
        parameter_levels={0: ["slow"]},
        periods=(8,),
        lr=0.1,
        deep=False,
    )
    slow.grad = torch.ones(100)
    opt.step(1)  # not due at step 1 with period 8: accumulator retained
    assert opt.state_bytes() >= 100 * 4


def test_default_level_assignment_leaves_the_backbone_fast():
    """The frequency separation belongs to the memory, not to the network weights.

    Under the old 'depth' default, 27% of parameters -- including the first GRU layer,
    which reads the raw KPIs -- took one Adam step per 32. Adam is scale-invariant, so
    that is a deficit in the number of steps, and it made NestedRIC underfit rather than
    retain: cross-dataset avg_perf -0.0862 against finetune's -0.0698.
    """
    from nestedric.models.backbone import build_backbone
    from nestedric.models.nested import NestedRIC

    cfg = {
        "encoder": {"type": "gru", "hidden": 32, "n_layers": 2, "dropout": 0.0},
        "heads": {"prediction": {"out_dim": 2}, "policy": {"n_actions": 3}},
    }
    model = NestedRIC(
        backbone=build_backbone(cfg, in_dim=19),
        n_levels=2,
        periods=(1, 32),
        memory_budget_bytes=100_000,
    )
    params = dict(model.named_parameters())
    total = sum(p.numel() for p in params.values())
    slow = sum(params[n].numel() for n in model.parameter_levels[1])

    assert model.level_assignment == "memory"
    assert slow / total < 0.05, "the backbone must not be trained at the slow period"
    assert all(
        n.startswith(("memory", "modulator")) for n in model.parameter_levels[1]
    ), model.parameter_levels[1]


def test_depth_assignment_is_still_available_as_an_ablation():
    from nestedric.models.backbone import build_backbone
    from nestedric.models.nested import NestedRIC

    cfg = {
        "encoder": {"type": "gru", "hidden": 32, "n_layers": 2, "dropout": 0.0},
        "heads": {"prediction": {"out_dim": 2}, "policy": {"n_actions": 3}},
    }
    model = NestedRIC(
        backbone=build_backbone(cfg, in_dim=19),
        n_levels=2,
        periods=(1, 32),
        memory_budget_bytes=100_000,
        level_assignment="depth",
    )
    slow = model.parameter_levels[1]
    assert any(n.startswith("backbone.encoder.rnn") for n in slow)
