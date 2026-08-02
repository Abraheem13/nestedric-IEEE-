"""Reservoir experience replay.

Keeps a fixed-size sample of past windows and mixes them into each batch. Reservoir
sampling gives every window seen so far an equal chance of being in the buffer without
knowing the stream length in advance, which matters here because environments differ in
size by up to a factor of two.

The buffer size is matched *in bytes* against the NestedRIC continuum memory
(design rule 2), so the proposed method cannot win simply by storing more.
"""

from __future__ import annotations

import numpy as np
import torch

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("replay")
class Replay(SgdMethod):
    """Reservoir experience replay."""

    def __init__(self, model, cfg, device: str = "cpu") -> None:
        super().__init__(model, cfg, device)
        self.capacity = int(cfg.get("buffer_size", 5000))
        self.replay_ratio = float(cfg.get("replay_ratio", 0.5))
        self.rng = np.random.default_rng(int(cfg.get("seed", 0)))
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._a: torch.Tensor | None = None
        self._n_seen = 0
        self._size = 0

    def _init_storage(self, batch) -> None:
        x, y, a = batch
        self._x = torch.zeros((self.capacity, *x.shape[1:]), dtype=x.dtype)
        self._y = torch.zeros((self.capacity, *y.shape[1:]), dtype=y.dtype)
        self._a = torch.zeros((self.capacity, *a.shape[1:]), dtype=a.dtype)

    def _reservoir_add(self, batch) -> None:
        """Standard reservoir sampling over the stream of windows."""
        x, y, a = batch
        for i in range(len(x)):
            if self._size < self.capacity:
                slot = self._size
                self._size += 1
            else:
                j = int(self.rng.integers(0, self._n_seen + 1))
                slot = j if j < self.capacity else -1
            if slot >= 0:
                self._x[slot], self._y[slot], self._a[slot] = x[i], y[i], a[i]
            self._n_seen += 1

    def _sample(self, n: int):
        idx = self.rng.choice(self._size, size=min(n, self._size), replace=False)
        idx = torch.from_numpy(np.asarray(idx, dtype="int64"))
        return self._x[idx], self._y[idx], self._a[idx]

    def observe(self, batch, step: int) -> dict:
        if self._x is None:
            self._init_storage(batch)

        if self._size > 0 and self.replay_ratio > 0:
            n_replay = max(1, int(len(batch[0]) * self.replay_ratio))
            past = self._sample(n_replay)
            mixed = tuple(torch.cat([cur, old]) for cur, old in zip(batch, past, strict=True))
        else:
            mixed = batch

        logs = super().observe(mixed, step)
        self._reservoir_add(batch)
        logs["buffer"] = self._size
        return logs

    def extra_state_bytes(self) -> int:
        if self._x is None:
            return 0
        return sum(t.numel() * t.element_size() for t in (self._x, self._y, self._a))
