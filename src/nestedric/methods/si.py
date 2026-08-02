"""Synaptic Intelligence (Zenke et al., 2017).

Accumulates a per-parameter importance online, from the path integral of each
parameter's contribution to the loss decrease, and penalises later movement of the
parameters that carried the most. Unlike EWC it needs no extra pass at the end of an
environment, which makes it the cheaper of the two regularisation baselines -- a point
the near-RT footprint table should make.
"""

from __future__ import annotations

import torch

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("si")
class Si(SgdMethod):
    """Synaptic Intelligence (Zenke et al., 2017)."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.c = float(cfg.get("c", 0.1))
        self.xi = float(cfg.get("xi", 1e-3))
        self._omega: dict[str, torch.Tensor] = {}
        self._w: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}
        self._prev: dict[str, torch.Tensor] = {}
        self._reset_path()

    def _reset_path(self) -> None:
        self._w = {
            n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad
        }
        self._prev = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }

    def penalty(self) -> torch.Tensor:
        if not self._omega:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._omega:
                total = total + (self._omega[name] * (param - self._anchor[name]) ** 2).sum()
        return total

    def observe(self, batch, step: int) -> dict:
        total, logs = self.losses(batch)
        pen = self.penalty()
        loss = total + self.c * pen
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            n: p.grad.detach().clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }
        self.optimizer.step()

        # Path integral: -grad . delta accumulates the loss decrease this parameter caused.
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in grads:
                    delta = param.detach() - self._prev[name]
                    self._w[name] -= grads[name] * delta
                    self._prev[name] = param.detach().clone()

        logs["si_penalty"] = float(pen.detach())
        return logs

    def end_environment(self, env, env_index: int) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self._w:
                    continue
                anchor = self._anchor.get(name, torch.zeros_like(param))
                delta = param.detach() - anchor if self._anchor else param.detach() * 0
                denom = delta**2 + self.xi
                contribution = torch.clamp(self._w[name] / denom, min=0.0)
                self._omega[name] = self._omega.get(name, torch.zeros_like(param)) + contribution
            self._anchor = {
                n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
            }
        self._reset_path()

    def extra_state_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for d in (self._omega, self._w, self._anchor, self._prev)
            for t in d.values()
        )
