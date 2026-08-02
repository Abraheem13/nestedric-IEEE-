"""Shared KPI encoder and heads, used identically by every method.

Design rule 1: one backbone for all methods, so any performance difference comes from
the learning rule rather than model capacity. A reviewer checks this first, so there is
a single construction path -- :func:`build_backbone`, driven by
``configs/model/base.yaml`` -- and every method receives the model already built. No
method may add parameters to the shared trunk; methods needing extra state (EWC's
Fisher, replay's buffer, NestedRIC's memory) hold it outside the backbone and declare
it in ``footprint()``, where the byte cost is visible and comparable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KPIEncoder(nn.Module):
    """Sequence encoder over a window of canonical KPI vectors.

    A GRU rather than a transformer: with a 32-step window and a ~10 ms near-RT budget,
    attention buys little and costs latency. The choice is ablated on Day 10.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(batch, window, in_dim)`` into ``(batch, hidden)``."""
        out, _ = self.rnn(x)
        return self.norm(out[:, -1])


class PredictionHead(nn.Module):
    """Regression head for throughput / buffer KPI forecasting."""

    def __init__(self, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class PolicyHead(nn.Module):
    """Discrete head selecting an allocation action (the xApp control task)."""

    def __init__(self, hidden: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_actions))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class Backbone(nn.Module):
    """Encoder plus both heads: the complete shared model.

    Both outputs are returned on every forward pass; the trainer weights the two losses.
    A method never sees a different architecture from any other method.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 2,
        n_actions: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = KPIEncoder(in_dim, hidden, n_layers, dropout)
        self.prediction = PredictionHead(hidden, out_dim)
        self.policy = PolicyHead(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction, action_logits)`` for a batch of windows."""
        h = self.encoder(x)
        return self.prediction(h), self.policy(h)

    def n_parameters(self) -> int:
        """Trainable parameter count, reported in the near-RT footprint."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_backbone(cfg: dict, in_dim: int) -> Backbone:
    """Construct the shared backbone from ``configs/model/base.yaml``.

    *in_dim* is passed rather than configured: it is a property of the feature set
    (18 KPI features plus the missingness channel), and a config that could disagree
    with the data is a config that eventually will.
    """
    encoder = cfg.get("encoder", {})
    heads = cfg.get("heads", {})
    kind = encoder.get("type", "gru")
    if kind != "gru":
        raise NotImplementedError(f"encoder type {kind!r} is a Day 10 ablation")

    return Backbone(
        in_dim=in_dim,
        hidden=int(encoder.get("hidden", 128)),
        n_layers=int(encoder.get("n_layers", 2)),
        dropout=float(encoder.get("dropout", 0.1)),
        out_dim=int(heads.get("prediction", {}).get("out_dim", 2)),
        n_actions=int(heads.get("policy", {}).get("n_actions", 3)),
    )
