"""Shared KPI encoder/heads used identically by every method (fair comparison).

Status: STUB -- implemented on Day 3 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

import torch.nn as nn


class KPIEncoder(nn.Module):
    """Sequence encoder over a window of canonical KPI vectors."""

    def __init__(self, in_dim: int, hidden: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("Day 3")


class PredictionHead(nn.Module):
    """Regression head for throughput / latency KPI forecasting."""

    def __init__(self, hidden: int, out_dim: int) -> None:
        super().__init__()
        raise NotImplementedError("Day 3")


class PolicyHead(nn.Module):
    """Discrete head selecting a scheduling / slicing action (xApp control task)."""

    def __init__(self, hidden: int, n_actions: int) -> None:
        super().__init__()
        raise NotImplementedError("Day 3")
