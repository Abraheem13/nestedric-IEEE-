"""Smoke tests: every implemented method takes a step and reports its footprint."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nestedric.methods import build_method, load_all  # noqa: E402
from nestedric.models.backbone import build_backbone  # noqa: E402

IMPLEMENTED = ["finetune", "joint", "ewc", "si", "replay", "agem", "lwf", "bilevel", "titans"]
PENDING = ["nestedric"]  # Days 5-7

MODEL_CFG = {
    "encoder": {"type": "gru", "hidden": 16, "n_layers": 1, "dropout": 0.0},
    "heads": {"prediction": {"out_dim": 2}, "policy": {"n_actions": 3}},
}


def _batch(n: int = 8, window: int = 6, in_dim: int = 19):
    rng = np.random.default_rng(0)
    return (
        torch.from_numpy(rng.standard_normal((n, window, in_dim)).astype("float32")),
        torch.from_numpy(rng.standard_normal((n, 2)).astype("float32")),
        torch.from_numpy(rng.integers(0, 3, size=n)),
    )


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_method_takes_a_step(name):
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method(name, model, {"optimizer": {"type": "adam", "lr": 1e-3}})
    before = [p.detach().clone() for p in model.parameters()]

    logs = method.observe(_batch(), step=0)

    assert np.isfinite(logs["loss"])
    after = list(model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(after, before, strict=True))


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_method_reports_footprint_in_bytes(name):
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method(name, model, {})
    fp = method.footprint()
    assert fp["params"] > 0
    assert fp["total_bytes"] == fp["param_bytes"] + fp["extra_state_bytes"]


@pytest.mark.parametrize("name", IMPLEMENTED)
def test_every_method_shares_the_same_backbone_size(name):
    """Design rule 1: differences must come from the learning rule, not capacity."""
    model = build_backbone(MODEL_CFG, in_dim=19)
    reference = build_backbone(MODEL_CFG, in_dim=19).n_parameters()
    build_method(name, model, {})
    assert model.n_parameters() == reference


def test_memory_methods_declare_their_bytes():
    """Design rule 2: NestedRIC must not win by storing more, so all state is declared."""
    for name, cfg in (
        ("replay", {"buffer_size": 32}),
        ("agem", {"buffer_size": 32}),
        ("titans", {"memory_capacity": 16}),
        ("lwf", {}),
        ("bilevel", {}),
    ):
        model = build_backbone(MODEL_CFG, in_dim=19)
        method = build_method(name, model, cfg)
        method.observe(_batch(), step=0)
        method.end_environment(None, 0)
        assert method.extra_state_bytes() > 0, f"{name} reports no method state"


def test_agem_projects_only_when_gradients_conflict():
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method("agem", model, {"buffer_size": 64, "ref_batch_size": 8})
    first = method.observe(_batch(), step=0)
    assert first["projected"] == 0.0  # empty buffer: nothing to conflict with
    second = method.observe(_batch(n=8), step=1)
    assert second["projected"] in (0.0, 1.0)


def test_titans_writes_to_memory_every_step_by_default():
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method("titans", model, {"memory_capacity": 16})
    assert method.update_period == 1  # the point of the ablation
    method.observe(_batch(), step=0)
    assert method.filled > 0


def test_replay_buffer_is_counted_in_bytes():
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method("replay", model, {"buffer_size": 32, "replay_ratio": 0.5})
    assert method.extra_state_bytes() == 0
    method.observe(_batch(), step=0)
    assert method.extra_state_bytes() > 0


def test_ewc_penalty_is_zero_before_any_environment_finishes():
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method("ewc", model, {"lambda_ewc": 100.0})
    logs = method.observe(_batch(), step=0)
    assert logs["ewc_penalty"] == 0.0


def test_joint_declares_it_wants_the_union_of_environments():
    model = build_backbone(MODEL_CFG, in_dim=19)
    method = build_method("joint", model, {})
    assert getattr(method, "wants_joint_data", False) is True


@pytest.mark.parametrize("name", PENDING)
def test_pending_methods_are_registered_but_unimplemented(name):
    assert name in load_all()
    model = build_backbone(MODEL_CFG, in_dim=19)
    with pytest.raises(NotImplementedError):
        build_method(name, model, {})


def test_memory_methods_are_byte_matched():
    """Design rule 2: replay, A-GEM and Titans must hold the same bytes, not the same
    nominal 'capacity'. Configured as 5,000 windows vs 512 slots they differed by 45x.
    """
    budgets = {}
    for name in ("replay", "agem", "titans"):
        model = build_backbone(MODEL_CFG, in_dim=19)
        method = build_method(name, model, {"memory_budget_mb": 1.0})
        for step in range(3):
            method.observe(_batch(n=4), step=step)
        budgets[name] = method.extra_state_bytes()

    assert all(b > 0 for b in budgets.values()), budgets
    largest, smallest = max(budgets.values()), min(budgets.values())
    # Discretisation into whole slots means they cannot match exactly; 10% is the
    # tolerance that matters, against the 4,500% gap this test was written for.
    assert largest / smallest < 1.1, budgets
    assert largest <= 1.05e6, budgets
