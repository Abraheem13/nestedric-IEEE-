"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

Penalises movement of parameters that mattered to earlier environments, weighted by the
diagonal of the Fisher information. The penalty is accumulated across environments
rather than kept per environment: storing one Fisher per environment would grow the
method's byte footprint with stream length, which the byte-matched comparison would
then have to account for.
"""

from __future__ import annotations

import torch

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("ewc")
class Ewc(SgdMethod):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.lambda_ewc = float(cfg.get("lambda_ewc", 100.0))
        self.fisher_samples = int(cfg.get("fisher_samples", 2000))
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchor: dict[str, torch.Tensor] = {}
        self._loader = None

    def begin_environment(self, env, env_index: int) -> None:
        super().begin_environment(env, env_index)
        self._loader = getattr(env, "_train_loader", None)

    def penalty(self) -> torch.Tensor:
        if not self._fisher:
            return torch.zeros((), device=self.device)
        total = torch.zeros((), device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._fisher:
                total = total + (self._fisher[name] * (param - self._anchor[name]) ** 2).sum()
        return total

    def observe(self, batch, step: int) -> dict:
        total, logs = self.losses(batch)
        pen = self.penalty()
        loss = total + 0.5 * self.lambda_ewc * pen
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        logs["ewc_penalty"] = float(pen.detach())
        return logs

    def end_environment(self, env, env_index: int) -> None:
        """Accumulate the Fisher diagonal over a sample of the finished environment."""
        loader = getattr(env, "_train_loader", None)
        if loader is None:
            return

        fisher = {
            n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad
        }
        seen = 0
        self.model.train()
        for batch in loader:
            if seen >= self.fisher_samples:
                break
            total, _ = self.losses(batch)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2 * len(batch[0])
            seen += len(batch[0])
        self.optimizer.zero_grad(set_to_none=True)

        if seen == 0:
            return
        for name in fisher:
            fisher[name] /= seen
            self._fisher[name] = self._fisher.get(name, 0) + fisher[name]
        self._anchor = {
            n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad
        }

    def extra_state_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for d in (self._fisher, self._anchor) for t in d.values()
        )
