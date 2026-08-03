"""The Method protocol, plus the shared training step every baseline inherits.

A uniform interface lets the continual loop in ``engine/trainer.py`` run without a
single method-specific branch, which is what keeps the comparison honest: a method
customises behaviour by overriding hooks, never by the trainer special-casing it.

:class:`SgdMethod` holds the parts that must be identical across methods -- the loss,
the optimiser construction, the forward pass -- so no baseline can win by quietly
optimising a different objective or a different schedule from its neighbours.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class Method(Protocol):
    """Uniform interface so the engine treats all methods identically."""

    def begin_environment(self, env, env_index: int) -> None:
        """Hook called before training on a new environment."""

    def observe(self, batch, step: int) -> dict:
        """One optimisation step. Returns a dict of scalars to log."""

    def end_environment(self, env, env_index: int) -> None:
        """Hook called after finishing an environment (e.g. Fisher/importance update)."""

    def predict(self, batch):
        """Inference used by the evaluator."""

    def footprint(self) -> dict:
        """Parameter count, memory bytes and per-step latency -- for the RIC budget check."""


#: Shared memory budget for every method that stores anything (design rule 2). Methods
#: size their own structures to fit it, rather than each config naming a capacity in its
#: own units -- 5,000 replayed windows and a 512-slot key-value memory sound comparable
#: and differ by a factor of 45 in bytes.
DEFAULT_MEMORY_BUDGET_MB = 4.0


def memory_budget_bytes(cfg: dict) -> int:
    """Bytes this method may spend on stored state, from the shared budget."""
    return int(float(cfg.get("memory_budget_mb", DEFAULT_MEMORY_BUDGET_MB)) * 1e6)


def capacity_from_budget(budget_bytes: int, bytes_per_item: int, minimum: int = 1) -> int:
    """How many items of *bytes_per_item* fit the budget."""
    return max(minimum, int(budget_bytes // max(bytes_per_item, 1)))


def build_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
    """Construct the optimiser named in a method config.

    Shared, so "the optimiser" is one decision recorded in one place rather than ten
    independent ones that drift apart over a fortnight.
    """
    spec = cfg.get("optimizer", {}) or {}
    kind = str(spec.get("type", "adam")).lower()
    lr = float(spec.get("lr", 1e-3))
    weight_decay = float(spec.get("weight_decay", 0.0))
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer {kind!r}")


class SgdMethod:
    """Plain supervised learning on the current environment; base for every baseline.

    The loss combines both benchmark tasks: standardised MSE on the KPI forecast and
    cross-entropy on the derived allocation action, weighted by ``policy_weight``. Both
    heads train on every batch for every method.
    """

    def __init__(self, model: nn.Module, cfg: dict, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.policy_weight = float(cfg.get("policy_weight", 0.5))
        self.optimizer = build_optimizer(self.model.parameters(), cfg)
        self.env_index = 0

    # ------------------------------------------------------------------ hooks
    def begin_environment(self, env, env_index: int) -> None:
        self.env_index = env_index
        self.model.train()

    def end_environment(self, env, env_index: int) -> None:
        return None

    # ------------------------------------------------------------------- core
    def losses(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass and combined loss. Shared by every method."""
        x, y, a = (t.to(self.device) for t in batch)
        pred, logits = self.model(x)
        mse = F.mse_loss(pred, y)
        ce = F.cross_entropy(logits, a)
        total = mse + self.policy_weight * ce
        logs = {
            "loss": float(total.detach()),
            "mse": float(mse.detach()),
            "ce": float(ce.detach()),
        }
        return total, logs

    def observe(self, batch, step: int) -> dict:
        total, logs = self.losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return logs

    @torch.no_grad()
    def predict(self, batch):
        """Return ``(prediction, action_logits)`` with the model in eval mode."""
        was_training = self.model.training
        self.model.eval()
        out = self.model(batch[0].to(self.device))
        self.model.train(was_training)
        return out

    # -------------------------------------------------------------- reporting
    def extra_state_bytes(self) -> int:
        """Bytes of method-specific state beyond the shared backbone.

        Design rule 2: memory budgets are matched *in bytes*, so every method must be
        able to say what it stores. Zero by default; anything keeping a buffer, a
        Fisher diagonal or a memory bank overrides this.
        """
        return 0

    def optimizer_state_bytes(self) -> int:
        """Bytes held by the optimiser itself.

        Reported for every method, not just the ones that volunteer it. Adam keeps two
        states per parameter and says nothing; DeepMomentum keeps `memory_depth` and
        was reporting them. Counting one and not the other made NestedRIC look like it
        stored 45% more than replay when the memories were identical -- an artefact of
        the accounting, which is exactly the kind of thing that turns into a wrong
        column in a paper.
        """
        opt = getattr(self, "optimizer", None)
        if opt is None or not hasattr(opt, "param_groups"):
            return 0
        total = 0
        for group in opt.param_groups:
            for p in group["params"]:
                for value in opt.state.get(p, {}).values():
                    if hasattr(value, "numel"):
                        total += value.numel() * value.element_size()
        return total

    def footprint(self) -> dict:
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        memory_bytes = self.extra_state_bytes()
        optimizer_bytes = self.optimizer_state_bytes()
        return {
            "params": sum(p.numel() for p in self.model.parameters()),
            "param_bytes": param_bytes,
            # The byte-matched quantity (design rule 2): what the method chooses to
            # remember. Optimiser state is a consequence of the optimiser, not of the
            # continual-learning strategy, so it is reported beside it rather than
            # inside it.
            "memory_bytes": memory_bytes,
            "optimizer_bytes": optimizer_bytes,
            "extra_state_bytes": memory_bytes,
            "total_bytes": param_bytes + memory_bytes + optimizer_bytes,
        }

    def state_summary(self) -> dict[str, Any]:
        """Method-specific scalars worth logging per environment."""
        return {}
