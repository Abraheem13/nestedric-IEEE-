"""Learning without Forgetting distillation (Li & Hoiem, 2017).

Keeps a frozen copy of the model as it stood at the end of the previous environment and
distils its outputs on the *current* environment's inputs. No stored data, which makes
it the cheapest baseline in bytes -- and the natural comparison for whether NestedRIC's
memory earns its footprint.

Distillation covers both heads: KL on the action logits at temperature T, and MSE on the
regression output. Distilling only the classifier, as the original image-classification
formulation does, would leave the forecasting head entirely unprotected and understate
the baseline.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("lwf")
class Lwf(SgdMethod):
    """Learning without Forgetting distillation (Li & Hoiem, 2017)."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.alpha = float(cfg.get("alpha_distill", 1.0))
        self.temperature = float(cfg.get("temperature", 2.0))
        self._teacher = None

    def observe(self, batch, step: int) -> dict:
        total, logs = self.losses(batch)

        distill = torch.zeros((), device=self.device)
        if self._teacher is not None:
            x = batch[0].to(self.device)
            with torch.no_grad():
                t_pred, t_logits = self._teacher(x)
            s_pred, s_logits = self.model(x)
            tau = self.temperature
            distill = F.kl_div(
                F.log_softmax(s_logits / tau, dim=1),
                F.log_softmax(t_logits / tau, dim=1),
                reduction="batchmean",
                log_target=True,
            ) * (tau**2)
            distill = distill + F.mse_loss(s_pred, t_pred)

        loss = total + self.alpha * distill
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        logs["distill"] = float(distill.detach())
        return logs

    def end_environment(self, env, env_index: int) -> None:
        self._teacher = copy.deepcopy(self.model).eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

    def extra_state_bytes(self) -> int:
        if self._teacher is None:
            return 0
        return sum(p.numel() * p.element_size() for p in self._teacher.parameters())
