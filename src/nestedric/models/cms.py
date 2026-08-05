"""Continuum Memory System (CMS).

A stack of associative-memory blocks, each with its own update period. Block i is
refreshed every tau_i steps with tau_0 < tau_1 < ... < tau_{L-1}; the frequency
separation ratio tau_{i+1}/tau_i is the central knob of this paper.

**What this has to do that replay does not.** Day 4 measured replay removing 80% of
cross-dataset forgetting (-0.0266 -> -0.0053) at a 4 MB budget, with no timescale
separation whatsoever. A memory that merely stores past windows is replay with extra
steps and will land in the same place. The claim here is different in kind: the blocks
store *compressed structure* rather than samples, and the slow blocks are written rarely
enough that what survives in them is what stayed true across environments -- the
invariant part of the allocation regime -- while the fast block tracks the current one.

Two consequences follow, and both are testable rather than decorative:

* A slow block written every tau_s steps sees an average over roughly tau_s steps of
  experience. If environments change on a comparable timescale, that average is a
  mixture and retains nothing; if tau_s is much longer, it retains the common structure.
  This is where the risk-optimal ratio in Proposition 2 comes from.
* Reading is a soft-attention lookup over all blocks at once, so the fast block can be
  overwritten completely without destroying what the slow blocks hold. That asymmetry is
  the mechanism that is supposed to produce lower |BWT| than a single-timescale memory
  of the same size -- which is precisely what the `titans` baseline controls for.

Byte budget is shared with replay and A-GEM (design rule 2): capacity is derived from
``memory_budget_mb``, split across levels, so NestedRIC cannot win by storing more.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociativeMemoryBlock(nn.Module):
    """A key-value associative memory refreshed every ``update_period`` steps.

    Writes are a gated blend rather than an overwrite: a slot moves toward the new
    content by ``write_rate``, so a block written rarely accumulates an average over
    many steps instead of taking whichever batch happened to arrive last. That is what
    makes a slow block *slow* in a useful sense -- a rare hard overwrite would be a
    high-variance snapshot, not a stable summary.

    Slot selection is by key similarity, so related states land in the same slot and
    the memory compresses rather than merely buffering.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        update_period: int,
        value_dim: int | None = None,
        write_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if update_period < 1:
            raise ValueError(f"update_period must be >= 1, got {update_period}")
        self.dim = dim
        self.capacity = capacity
        self.update_period = update_period
        self.value_dim = value_dim or dim
        self.write_rate = write_rate
        self.temperature = temperature

        # Buffers, not parameters: the memory is written by its own update rule, not by
        # backpropagation through the task loss. Registering them as buffers keeps them
        # in the state dict and on the right device without joining the optimiser.
        self.register_buffer("keys", torch.zeros(capacity, dim))
        self.register_buffer("values", torch.zeros(capacity, self.value_dim))
        self.register_buffer("usage", torch.zeros(capacity))
        self.register_buffer("writes", torch.zeros((), dtype=torch.long))

    @property
    def is_empty(self) -> bool:
        """True until the block has been written at least once."""
        return bool(self.writes.item() == 0)

    def due(self, step: int) -> bool:
        """Whether this block is scheduled to be written at *step*."""
        return step % self.update_period == 0

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Blend a batch of key-value pairs into the nearest slots.

        Empty slots are claimed first, so a cold memory fills before it starts
        compressing; afterwards each pair moves its most similar slot toward itself.
        """
        keys = keys.detach()
        values = values.detach()
        n = len(keys)
        if n == 0:
            return

        free = (self.usage == 0).nonzero(as_tuple=True)[0]
        n_free = min(len(free), n)

        # Fill empty slots first: a cold memory should populate before it compresses.
        if n_free:
            slots = free[:n_free]
            self.keys[slots] = keys[:n_free]
            self.values[slots] = values[:n_free]
            self.usage[slots] += 1.0

        remaining = keys[n_free:]
        if len(remaining):
            # One similarity matrix for the whole batch instead of a Python loop over
            # items. The loop cost ~1.2M iterations per run, each scoring a 4,000-slot
            # memory, and made NestedRIC 13x slower than replay -- an implementation
            # artefact that would otherwise appear in the paper's runtime column as
            # though it were a property of the method.
            rest_values = values[n_free:]
            occupied = self.usage > 0
            candidates = occupied.nonzero(as_tuple=True)[0]
            if len(candidates) == 0:
                candidates = torch.arange(self.capacity, device=keys.device)

            sims = F.normalize(remaining, dim=1) @ F.normalize(self.keys[candidates], dim=1).t()
            chosen = candidates[sims.argmax(dim=1)]

            # Several items can select the same slot. index_add accumulates their pulls,
            # and dividing by the hit count makes the result their mean -- which is what
            # the sequential loop converged to anyway, without depending on batch order.
            deltas_k = remaining - self.keys[chosen]
            deltas_v = rest_values - self.values[chosen]

            hits = torch.zeros(self.capacity, device=keys.device)
            hits.index_add_(0, chosen, torch.ones(len(chosen), device=keys.device))

            sum_k = torch.zeros_like(self.keys)
            sum_v = torch.zeros_like(self.values)
            sum_k.index_add_(0, chosen, deltas_k)
            sum_v.index_add_(0, chosen, deltas_v)

            touched = hits > 0
            scale = self.write_rate / hits[touched].unsqueeze(1)
            self.keys[touched] += scale * sum_k[touched]
            self.values[touched] += scale * sum_v[touched]
            self.usage += hits

        self.writes += 1

    def read(self, queries: torch.Tensor) -> torch.Tensor:
        """Soft-attention lookup, differentiable with respect to *queries*.

        Returns zeros for an unwritten block rather than attending over empty slots,
        which would inject a meaningless constant into the encoder's representation.
        """
        if self.is_empty:
            return queries.new_zeros(len(queries), self.value_dim)

        occupied = self.usage > 0
        keys = self.keys[occupied]
        values = self.values[occupied]

        scores = (queries @ keys.t()) / (self.temperature * (self.dim**0.5))
        weights = torch.softmax(scores, dim=1)
        return weights @ values

    def state_bytes(self) -> int:
        """Bytes held by this block, for the byte-matched comparison."""
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values, self.usage))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, capacity={self.capacity}, period={self.update_period}"


class ContinuumMemory(nn.Module):
    """Frequency-tiered stack of :class:`AssociativeMemoryBlock`.

    Parameters
    ----------
    dim
        Key width; the encoder's hidden size. Read off the model, never configured
        separately -- a config that can disagree with the model eventually will.
    periods
        Update periods ``(tau_0, ..., tau_{L-1})`` in optimiser steps. In the RIC
        mapping these are anchored to the near-RT (10 ms - 1 s) and non-RT (> 1 s)
        control loops.

        Must be non-decreasing. Equal periods are explicitly allowed because
        ``rho = 1`` is the degenerate single-timescale case Theorem 1 has to recover,
        and the Day 10 ratio sweep starts there: forbidding it would exclude the
        control the theory is checked against. A *decreasing* stack, where the
        nominally slow level updates more often than the fast one, is rejected -- that
        is an inverted hierarchy, not a degenerate one.
    capacity
        Slots per block. With a shared byte budget this is derived, not chosen.

    """

    def __init__(
        self,
        dim: int,
        periods: tuple[int, ...],
        capacity: int,
        value_dim: int | None = None,
        write_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if not periods:
            raise ValueError("a continuum memory needs at least one level")
        if any(b < a for a, b in zip(periods, periods[1:], strict=False)):
            raise ValueError(f"periods must be non-decreasing, got {periods}")

        self.periods = tuple(int(p) for p in periods)
        self.dim = dim
        self.value_dim = value_dim or dim
        self.blocks = nn.ModuleList(
            [
                AssociativeMemoryBlock(
                    dim=dim,
                    capacity=capacity,
                    update_period=p,
                    value_dim=value_dim,
                    # Slower levels blend more gently: a block written every 32 steps
                    # should move less per write than one written every step, or its
                    # "slowness" is only in how often it is disturbed, not in how much.
                    write_rate=write_rate / (1 + i),
                )
                for i, p in enumerate(self.periods)
            ]
        )
        # One scalar gate per level, learned with the task loss: the model decides how
        # much to trust each timescale rather than the ratio being hard-coded.
        self.level_gates = nn.Parameter(torch.zeros(len(self.periods)))

    @property
    def separation_ratios(self) -> tuple[float, ...]:
        """``(tau_1/tau_0, tau_2/tau_1, ...)`` -- reported with every run.

        docs/THEORY.md requires the *realised* ratio per run, not the configured one,
        because the bound is stated in it.
        """
        return tuple(b / a for a, b in zip(self.periods, self.periods[1:], strict=False))

    def write(self, step: int, keys: torch.Tensor, values: torch.Tensor) -> list[int]:
        """Write to every block due at *step*; returns the levels that fired."""
        fired = []
        for level, block in enumerate(self.blocks):
            if block.due(step):
                block.write(keys, values)
                fired.append(level)
        return fired

    def read(self, queries: torch.Tensor) -> torch.Tensor:
        """Gated sum of every level's lookup."""
        gates = torch.softmax(self.level_gates, dim=0)
        out = queries.new_zeros(len(queries), self.value_dim)
        for gate, block in zip(gates, self.blocks, strict=True):
            out = out + gate * block.read(queries)
        return out

    def state_bytes(self) -> int:
        """Total bytes across all levels, for the byte-matched comparison."""
        return sum(block.state_bytes() for block in self.blocks)

    def summary(self) -> dict:
        """Per-level occupancy and write counts, logged per environment."""
        return {
            "periods": list(self.periods),
            "separation_ratios": list(self.separation_ratios),
            "writes": [int(b.writes.item()) for b in self.blocks],
            "occupancy": [float((b.usage > 0).float().mean().item()) for b in self.blocks],
            "state_bytes": self.state_bytes(),
        }


def capacity_for_budget(budget_bytes: int, dim: int, value_dim: int, n_levels: int) -> int:
    """Slots per level that fit *budget_bytes* across the whole stack.

    Shared with replay and A-GEM so the comparison is at equal bytes, not equal
    nominal capacity -- 5,000 replayed windows and a 512-slot memory read as comparable
    and differ by 45x, which is how design rule 2 was being violated until Day 4.
    """
    per_slot = (dim + value_dim + 1) * 4  # keys + values + usage, float32
    return max(1, int(budget_bytes // (per_slot * max(n_levels, 1))))
