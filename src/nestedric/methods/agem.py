"""Averaged Gradient Episodic Memory (Chaudhry et al., 2019).

Rather than adding a penalty, A-GEM constrains the update direction: if the proposed
gradient would increase loss on a reference batch drawn from memory, it is projected
onto the half-space that does not. One inequality constraint against the average
reference gradient, which is what makes it cheap enough to be plausible in a RIC.

Inherits the reservoir buffer from Replay so the two share a byte budget and differ
only in how they use it -- a comparison of mechanism rather than of memory size.
"""

from __future__ import annotations

import torch

from nestedric.methods import register
from nestedric.methods.replay import Replay


@register("agem")
class Agem(Replay):
    """Averaged Gradient Episodic Memory (Chaudhry et al., 2019)."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.ref_batch_size = int(cfg.get("ref_batch_size", 256))
        self.replay_ratio = 0.0  # A-GEM projects gradients; it does not mix batches

    def _flat_grad(self) -> torch.Tensor:
        parts = [p.grad.detach().reshape(-1) for p in self.model.parameters() if p.grad is not None]
        return torch.cat(parts) if parts else torch.zeros(0, device=self.device)

    def _assign_grad(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.model.parameters():
            if p.grad is None:
                continue
            n = p.grad.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
            offset += n

    def observe(self, batch, step: int) -> dict:
        if self._x is None:
            self._init_storage(batch)

        total, logs = self.losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        g = self._flat_grad()

        projected = 0.0
        if self._size > 0:
            ref = self._sample(self.ref_batch_size)
            ref_loss, _ = self.losses(ref)
            self.optimizer.zero_grad(set_to_none=True)
            ref_loss.backward()
            g_ref = self._flat_grad()

            dot = torch.dot(g, g_ref)
            if float(dot) < 0:
                # The update would raise loss on remembered data: remove the offending
                # component instead of abandoning the step.
                g = g - (dot / (g_ref.dot(g_ref) + 1e-12)) * g_ref
                projected = 1.0
            self._assign_grad(g)

        self.optimizer.step()
        self._reservoir_add(batch)
        logs["projected"] = projected
        logs["buffer"] = self._size
        return logs
