"""Two-timescale bilevel CL baseline, after Sun et al., IEEE TSP 70:1900-1917, 2022.

The closest published idea to ours: an inner loop adapts fast, an outer loop updates a
slow copy of the parameters at a lower frequency. It is the honest comparison point for
NestedRIC, because it isolates what frequency separation alone buys before the continuum
memory, the deep optimizer and self-modification are added.

The slow parameters are an exponential moving average of the fast ones, updated every
`outer_period` steps; the fast parameters are pulled back toward them, which is the
stabilising force. Unlike NestedRIC the slow level has no memory of its own and does not
modify the fast level's update rule.
"""

from __future__ import annotations

import torch

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("bilevel")
class Bilevel(SgdMethod):
    """Two-timescale bilevel CL baseline (Sun et al., IEEE TSP 2022)."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.inner_steps = int(cfg.get("inner_steps", 1))
        self.outer_lr = float(cfg.get("outer_lr", 1e-4))
        self.outer_period = int(cfg.get("outer_period", 32))
        self.pull = float(cfg.get("pull", 0.01))
        self._slow = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._steps = 0

    def observe(self, batch, step: int) -> dict:
        logs = super().observe(batch, step)
        self._steps += 1

        if self._steps % self.outer_period == 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name not in self._slow:
                        continue
                    # Outer update: slow level tracks the fast level slowly.
                    self._slow[name] += self.outer_lr * (param.detach() - self._slow[name])
                    # Inner pull-back: the stabilising direction.
                    param.add_(self.pull * (self._slow[name] - param.detach()))
            logs["outer_update"] = 1.0
        return logs

    def state_summary(self) -> dict:
        return {"outer_updates": self._steps // max(self.outer_period, 1)}

    def extra_state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._slow.values())
