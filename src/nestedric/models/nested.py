"""The NestedRIC learner.

Maps the O-RAN control loops onto nested optimisation levels::

    level 0 (fast, tau_f)  <->  near-RT RIC   (10 ms - 1 s)
    level 1 (slow, tau_s)  <->  non-RT RIC    (> 1 s)
    level 2 (optional)     <->  SMO / policy horizon

Each level owns a context flow and a memory block; the slow level supplies the update
rule for the fast level (self-modification).

**What is actually nested.** Three things are tiered to the same period schedule, and it
matters that they are the same schedule rather than three coincidentally similar ones:

1. *Memory* -- the continuum memory block at level i is written every tau_i steps.
2. *Parameters* -- each level owns a slice of the model. The fast level owns the heads
   and the last encoder layer, which must track the current environment; the slow levels
   own the earlier encoder layers, which should hold what is invariant across
   environments. A slow parameter group takes an optimiser step only when its level is
   due, so it moves ~tau_s times less often than the fast group.
3. *The update rule itself* -- with self-modification on, the slow level emits a gain
   that scales the fast level's effective learning rate.

Point 2 is the one that does the work, and it is also where the honest risk sits. Freezing
early layers for tau_s steps is a strong form of stability; if NestedRIC beats the
baselines only because early layers move less, then the result is "partial freezing
helps", not "frequency separation helps". The ablation that separates those is
period_ratio at fixed n_levels: freezing is monotone in the ratio, whereas Proposition 2
predicts an interior optimum. Day 10 measures it, and if the curve is monotone the
paper should say the mechanism is closer to structured freezing than to nesting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nestedric.models.cms import ContinuumMemory, capacity_for_budget


class NestedRIC(nn.Module):
    """Multi-timescale nested learner for O-RAN control.

    Parameters
    ----------
    backbone
        The shared model, identical to every baseline's (design rule 1). NestedRIC adds
        no capacity to it; the memory is buffers and the only new parameters are the
        level gates and the self-modification head, both reported in ``footprint()``.
    n_levels
        Number of nested optimisation levels (ablated over {1, 2, 3}).
    periods
        Update period per level, in optimiser steps. Strictly increasing.
    memory_budget_bytes
        Shared with replay and A-GEM, split across levels.
    self_modifying
        If True, the slow level parameterises the fast level's update rule.

    """

    def __init__(
        self,
        backbone: nn.Module,
        n_levels: int = 2,
        periods: tuple[int, ...] = (1, 32),
        memory_budget_bytes: int = 4_000_000,
        self_modifying: bool = True,
        write_rate: float = 0.1,
    ) -> None:
        super().__init__()
        periods = tuple(int(p) for p in periods)[:n_levels]
        if len(periods) != n_levels:
            raise ValueError(f"need {n_levels} periods, got {periods}")

        self.backbone = backbone
        self.n_levels = n_levels
        self.periods = periods
        self.self_modifying = self_modifying

        hidden = int(backbone.encoder.hidden)
        capacity = capacity_for_budget(memory_budget_bytes, hidden, hidden, n_levels)
        self.memory = ContinuumMemory(
            dim=hidden,
            periods=periods,
            capacity=capacity,
            value_dim=hidden,
            write_rate=write_rate,
        )

        # Fuses the memory read back into the encoder state. Small and counted.
        self.fuse = nn.Linear(hidden * 2, hidden)

        # Self-modification: the slow level reads its own memory summary and emits a
        # scalar gain on the fast level's learning rate. This is the Hope mechanism in
        # its most restrained form -- a full learned update rule is a Day 7 extension,
        # and starting with one interpretable scalar means an ablation can attribute
        # any gain to it rather than to extra capacity.
        self.modulator = nn.Sequential(
            nn.Linear(hidden, hidden // 4), nn.GELU(), nn.Linear(hidden // 4, 1)
        )

        self.parameter_levels = self._assign_levels()

    def _assign_levels(self) -> dict[int, list[str]]:
        """Assign parameter groups to levels: later layers are faster.

        The heads and the final encoder layer track the current environment and belong
        to level 0. Earlier encoder layers hold representations that should survive a
        change of allocation regime, so they are updated at the slower periods. With one
        level everything is fast, which is exactly the degenerate case the theorem must
        recover.
        """
        groups: dict[int, list[str]] = {i: [] for i in range(self.n_levels)}
        n_rnn_layers = self.backbone.encoder.rnn.num_layers

        for name, _ in self.named_parameters():
            if name.startswith("backbone.encoder.rnn"):
                # torch names GRU weights ..._l0, ..._l1: layer 0 is closest to input.
                layer = 0
                for suffix in range(n_rnn_layers):
                    if name.endswith(f"_l{suffix}"):
                        layer = suffix
                depth_from_output = n_rnn_layers - 1 - layer
                level = min(depth_from_output, self.n_levels - 1)
            elif name.startswith("memory") or name.startswith("modulator"):
                level = self.n_levels - 1  # memory gates and the modulator are slow
            else:
                level = 0  # heads, layer norm, fusion: fast
            groups[level].append(name)
        return groups

    def due_levels(self, step: int) -> list[int]:
        """Levels scheduled to update at *step*.

        Level 0 fires every step; level i fires every ``periods[i]`` steps. Asserted in
        tests -- the Day 6 gate is that level-1 updates fire exactly every tau_s steps.
        """
        return [i for i, p in enumerate(self.periods) if step % p == 0]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder state fused with the memory read."""
        h = self.backbone.encoder(x)
        recalled = self.memory.read(h)
        return self.fuse(torch.cat([h, recalled], dim=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(prediction, action_logits)``, memory-augmented."""
        h = self.encode(x)
        return self.backbone.prediction(h), self.backbone.policy(h)

    def fast_lr_gain(self) -> torch.Tensor:
        """Multiplier on the fast level's learning rate, emitted by the slow level.

        Centred on 1.0 and bounded to [0.5, 1.5] through a tanh: an unbounded gain would
        let the model reduce its own plasticity to zero and score well on retention by
        refusing to learn, which is a degenerate optimum rather than a result.
        """
        if not self.self_modifying:
            return torch.ones((), device=self.memory.level_gates.device)
        slow = self.memory.blocks[-1]
        if slow.is_empty:
            return torch.ones((), device=self.memory.level_gates.device)
        summary = slow.values[slow.usage > 0].mean(dim=0, keepdim=True)
        return 1.0 + 0.5 * torch.tanh(self.modulator(summary).squeeze())

    @torch.no_grad()
    def write_memory(self, step: int, x: torch.Tensor, targets: torch.Tensor) -> list[int]:
        """Write the current batch into every memory level due at *step*."""
        h = self.backbone.encoder(x)
        return self.memory.write(step, h, h)

    def summary(self) -> dict:
        """State worth logging per environment, including what the theory needs."""
        return {
            "n_levels": self.n_levels,
            "periods": list(self.periods),
            **self.memory.summary(),
            "self_modifying": self.self_modifying,
            "fast_lr_gain": float(self.fast_lr_gain()),
        }

    def state_bytes(self) -> int:
        """Memory bytes, for the byte-matched comparison."""
        return self.memory.state_bytes()
