"""NestedRIC as a Method: wires models.nested.NestedRIC into the continual engine.

Every ablation axis in docs/PLAN.md Day 10 is a config key here -- ``n_levels``,
``periods``, ``self_modifying``, ``deep_optimizer.enabled``, ``memory_budget_mb`` -- so
the sweep varies one thing at a time through the same code path the headline runs use.
An ablation that takes a different branch is not an ablation of the method.

The engine sees an ordinary Method. NestedRIC needs no trainer support beyond the
``observe(batch, step)`` signature every baseline already has, because the step index it
needs for the period schedule is passed to all of them.

**What must be true for this to be a contribution.** Day 4 measured replay taking
cross-dataset |BWT| from 0.0266 to 0.0053 at 4 MB. NestedRIC gets the same 4 MB. Beating
finetune here is worth nothing; the comparison that matters is against replay at equal
bytes, and against ``titans`` (same memory, no frequency separation) and ``bilevel``
(frequency separation, no memory), which between them isolate the two halves of the
claim. If NestedRIC lands with titans, the memory is doing the work and the tiering is
decoration -- and the paper says that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nestedric.methods import register
from nestedric.methods.base import SgdMethod, memory_budget_bytes
from nestedric.models.deep_optimizer import LevelScheduledOptimizer
from nestedric.models.nested import NestedRIC


@register("nestedric")
class NestedRICMethod(SgdMethod):
    """Proposed method. Ablation axes are all exposed through ``cfg``."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        # Deliberately not calling SgdMethod.__init__: it builds a single flat
        # optimiser over every parameter, which is precisely what this method replaces.
        self.cfg = cfg
        self.device = device
        self.policy_weight = float(cfg.get("policy_weight", 0.5))

        n_levels = int(cfg.get("n_levels", 2))
        periods = tuple(cfg.get("periods", (1, 32)))[:n_levels]
        memory_cfg = cfg.get("memory", {}) or {}
        # The budget may be stated under memory.budget_mb or at the top level; both
        # resolve to the same shared allowance replay and A-GEM are held to.
        budget = memory_budget_bytes(
            {"memory_budget_mb": memory_cfg["budget_mb"]} if "budget_mb" in memory_cfg else cfg
        )

        self.model = NestedRIC(
            backbone=model,
            n_levels=n_levels,
            periods=periods,
            memory_budget_bytes=budget,
            self_modifying=bool(cfg.get("self_modifying", True)),
            write_rate=float(memory_cfg.get("write_rate", 0.1)),
        ).to(device)

        deep_cfg = cfg.get("deep_optimizer", {}) or {}
        optimiser_cfg = cfg.get("optimizer", {}) or {}
        self.optimizer = LevelScheduledOptimizer(
            named_parameters=dict(self.model.named_parameters()),
            parameter_levels=self.model.parameter_levels,
            periods=self.model.periods,
            lr=float(optimiser_cfg.get("lr", 1e-3)),
            weight_decay=float(optimiser_cfg.get("weight_decay", 0.0)),
            deep=bool(deep_cfg.get("enabled", True)),
            memory_depth=int(deep_cfg.get("memory_depth", 2)),
        )

        self.env_index = 0
        self._step_in_env = 0
        self._fired: dict[int, int] = {}
        self._gains: list[float] = []

    # ------------------------------------------------------------------ hooks
    def begin_environment(self, env, env_index: int) -> None:
        """Reset the within-environment step counter, not the memory.

        The counter restarts so every environment begins with all levels due at step 0,
        which makes the schedule identical across environments and keeps the realised
        separation ratio equal to the configured one. The memory deliberately persists:
        carrying structure across environments is the entire point.
        """
        self.env_index = env_index
        self._step_in_env = 0
        self.model.train()

    def end_environment(self, env, env_index: int) -> None:
        return None

    # ------------------------------------------------------------------- core
    def losses(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Identical objective to every baseline; only the update rule differs."""
        x, y, a = (t.to(self.device) for t in batch)
        pred, logits = self.model(x)
        mse = F.mse_loss(pred, y)
        ce = F.cross_entropy(logits, a)
        total = mse + self.policy_weight * ce
        return total, {
            "loss": float(total.detach()),
            "mse": float(mse.detach()),
            "ce": float(ce.detach()),
        }

    def observe(self, batch, step: int) -> dict:
        total, logs = self.losses(batch)

        self.optimizer.zero_grad()
        total.backward()

        gain = float(self.model.fast_lr_gain().detach())
        fired = self.optimizer.step(self._step_in_env, fast_gain=gain)

        # The memory is written on the same schedule that governs the parameters, so
        # one period controls both. Writing on a different cadence would make the
        # "separation ratio" ambiguous and the theorem untestable.
        x = batch[0].to(self.device)
        written = self.model.write_memory(self._step_in_env, x, batch[1])

        self._step_in_env += 1
        for level in fired:
            self._fired[level] = self._fired.get(level, 0) + 1
        self._gains.append(gain)

        logs["levels_fired"] = len(fired)
        logs["levels_written"] = len(written)
        logs["lr_gain"] = gain
        return logs

    @torch.no_grad()
    def predict(self, batch):
        """Inference with the memory read in place -- as it would run in a RIC."""
        was_training = self.model.training
        self.model.eval()
        out = self.model(batch[0].to(self.device))
        self.model.train(was_training)
        return out

    # -------------------------------------------------------------- reporting
    def extra_state_bytes(self) -> int:
        """The continuum memory: the quantity byte-matched against replay and titans."""
        return self.model.state_bytes()

    def optimizer_state_bytes(self) -> int:
        """Level-scheduled optimiser state, reported beside the memory, not inside it."""
        return self.optimizer.state_bytes()

    def state_summary(self) -> dict:
        """Per-environment state, including the quantities docs/THEORY.md needs."""
        summary = self.model.summary()
        summary["level_step_counts"] = dict(self._fired)
        summary["mean_lr_gain"] = sum(self._gains) / len(self._gains) if self._gains else 1.0
        # The realised ratio, not the configured one: if a level never fired because an
        # environment was shorter than its period, the theory needs to know.
        fired = [self._fired.get(i, 0) for i in range(self.model.n_levels)]
        summary["realised_ratios"] = [
            (fired[i] / fired[i + 1]) if fired[i + 1] else float("inf")
            for i in range(len(fired) - 1)
        ]
        self._gains.clear()
        return summary
