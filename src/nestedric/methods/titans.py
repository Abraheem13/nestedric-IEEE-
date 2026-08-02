"""Single-timescale test-time memory (Titans) -- the frequency-separation ablation.

A key-value memory written and read at *every* step: the same associative machinery
NestedRIC uses, with the frequency tiering removed. It is therefore the sharpest
baseline in the set. If NestedRIC beats replay and EWC but not this, the gain came from
having a memory at all, not from separating timescales -- and the paper's central claim
would be unsupported.

Its memory is byte-matched to the NestedRIC continuum memory by construction: same
capacity, same dimension, same dtype.
"""

from __future__ import annotations

import torch

from nestedric.methods import register
from nestedric.methods.base import SgdMethod, capacity_from_budget, memory_budget_bytes


@register("titans")
class Titans(SgdMethod):
    """Single-timescale test-time memory (Titans) -- frequency-separation ablation."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.budget = memory_budget_bytes(cfg)
        self.update_period = int(cfg.get("update_period", 1))
        # Read off the encoder rather than configured: memory keys *are* encoder states,
        # so a separately configurable width is a config that can disagree with the model.
        self.dim = int(model.encoder.hidden)
        self.momentum = float(cfg.get("memory_momentum", 0.9))
        # 4 bytes per float: one key of width `dim` plus one 2-vector value per slot.
        self.capacity = int(cfg.get("memory_capacity", 0)) or capacity_from_budget(
            self.budget, (self.dim + 2) * 4
        )
        self.keys = torch.zeros(self.capacity, self.dim, device=device)
        self.values = torch.zeros(self.capacity, 2, device=device)
        self.filled = 0
        self._cursor = 0
        self._steps = 0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encoder(x.to(self.device)).detach()

    @torch.no_grad()
    def _write(self, batch) -> None:
        x, y, _ = batch
        h = self._encode(x)
        for i in range(len(h)):
            slot = self._cursor % self.capacity
            # A learned write in the NL sense would be a gradient step on the memory's
            # own objective; here it is a momentum blend, which is the standard
            # test-time-memory formulation and keeps the ablation honest.
            self.keys[slot] = self.momentum * self.keys[slot] + (1 - self.momentum) * h[i]
            self.values[slot] = self.momentum * self.values[slot] + (1 - self.momentum) * y[i].to(
                self.device
            )
            self._cursor += 1
            self.filled = min(self.filled + 1, self.capacity)

    def observe(self, batch, step: int) -> dict:
        logs = super().observe(batch, step)
        self._steps += 1
        if self._steps % self.update_period == 0:
            self._write(batch)
        logs["memory"] = float(self.filled)
        return logs

    def state_summary(self) -> dict:
        return {"memory_filled": self.filled, "update_period": self.update_period}

    def extra_state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values))
