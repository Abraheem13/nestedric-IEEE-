"""Deep optimisers: gradient-based updates re-read as associative memory.

The Nested Learning reinterpretation is that momentum is already a memory -- an
exponential average compressing the gradient history into one state per parameter, and
Adam adds a second. Seen that way, "deeper memory" means keeping several averages at
different decay rates and stepping along their combination, which is the optimiser-level
analogue of the continuum memory's frequency tiering.

:class:`LevelScheduledOptimizer` is what makes the nesting mechanical rather than
nominal. Parameters assigned to level i take a step only when level i is due, so a slow
group moves roughly tau_s times less often than the fast one. Without it, "levels" would
be a label on parameters that all update together and the separation ratio would have no
effect at all -- which is exactly how a paper ends up with a flat ablation curve and no
explanation for it.
"""

from __future__ import annotations

import torch


class DeepMomentum(torch.optim.Optimizer):
    """Momentum as a learned associative memory over the gradient stream.

    Keeps ``memory_depth`` exponential averages per parameter at geometrically spaced
    decay rates and steps along their mean. Depth 1 is ordinary momentum -- the
    degeneracy the ablation should recover, and a cheap negative result to state if
    depth buys nothing.
    """

    def __init__(self, params, lr: float = 1e-3, memory_depth: int = 2, base_beta: float = 0.9):
        if memory_depth < 1:
            raise ValueError(f"memory_depth must be >= 1, got {memory_depth}")
        # Depth k uses beta = 1 - (1 - base_beta) / 2**k, so deeper states forget more
        # slowly and the stack spans timescales instead of repeating one.
        betas = tuple(1.0 - (1.0 - base_beta) / (2**k) for k in range(memory_depth))
        super().__init__(params, {"lr": lr, "betas": betas, "memory_depth": memory_depth})

    @torch.no_grad()
    def step(self, closure=None):
        """Take one step along the mean of the gradient-memory states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "memories" not in state:
                    state["memories"] = [torch.zeros_like(p) for _ in betas]

                update = torch.zeros_like(p)
                for memory, beta in zip(state["memories"], betas, strict=True):
                    memory.mul_(beta).add_(p.grad, alpha=1.0 - beta)
                    update.add_(memory)
                p.add_(update, alpha=-lr / len(betas))

        return loss

    def state_bytes(self) -> int:
        """Bytes of optimiser memory, reported in the footprint."""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                for memory in self.state.get(p, {}).get("memories", []):
                    total += memory.numel() * memory.element_size()
        return total


class LevelScheduledOptimizer:
    """Wraps optimisers so each parameter level steps only when that level is due.

    Also carries the scalar gain through which self-modification acts: the slow level
    emits it, and level 0's step size is scaled by it.

    Gradients on levels that are not due are cleared rather than accumulated. Letting
    them accumulate would mean a slow level eventually taking one enormous step carrying
    tau_s batches of gradient -- large-batch training on a delay, which is a different
    algorithm from the one the theorem describes.
    """

    def __init__(
        self,
        named_parameters: dict[str, torch.nn.Parameter],
        parameter_levels: dict[int, list[str]],
        periods: tuple[int, ...],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        deep: bool = False,
        memory_depth: int = 2,
    ) -> None:
        self.periods = tuple(periods)
        self.levels = sorted(parameter_levels)
        self.optimizers: dict[int, torch.optim.Optimizer] = {}
        self.level_params: dict[int, list[torch.nn.Parameter]] = {}

        for level in self.levels:
            params = [
                named_parameters[name]
                for name in parameter_levels[level]
                if name in named_parameters and named_parameters[name].requires_grad
            ]
            self.level_params[level] = params
            if not params:
                continue
            self.optimizers[level] = (
                DeepMomentum(params, lr=lr, memory_depth=memory_depth)
                if deep
                else torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            )
        self._accum: dict = {}
        self._accum_counts: dict[int, int] = {}
        self._base_lrs = {
            level: [g["lr"] for g in opt.param_groups] for level, opt in self.optimizers.items()
        }

    def due_levels(self, step: int) -> list[int]:
        """Levels scheduled at *step*."""
        return [i for i, p in enumerate(self.periods) if step % p == 0]

    def zero_grad(self) -> None:
        """Clear gradients on every level."""
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

    def step(self, step: int, fast_gain: float = 1.0) -> list[int]:
        """Accumulate gradients per level and step every level that is due.

        Gradients arriving between a level's firings are **accumulated and averaged**,
        not discarded. This is the part that makes frequency separation a statement
        about timescale rather than about training budget.

        The first implementation set ``p.grad = None`` for levels that were not due, so
        a level with period 32 saw one gradient in 32 and that one was a single-batch
        estimate. It therefore received ~1/32 of the total gradient signal, and the
        Day 7 result reflected it: NestedRIC forgot less than finetune (-0.0093 against
        -0.0266) while fitting worse than finetune (-0.0861 against -0.0698), which is
        what underfitting looks like. Worse, it would have invalidated the Day 10 ratio
        sweep outright -- larger rho would have meant less training, so |BWT| would fall
        with rho for reasons having nothing to do with the theorem.

        With accumulation, every level sees every gradient; they differ only in how
        often they act on them and over how long they integrate. That is the quantity
        Theorem 1 is about.

        *fast_gain* scales level 0 only. The gain is emitted by the slow level, so
        scaling the slow level with it would be a feedback loop on itself.
        """
        due = self.due_levels(step)

        for level, params in self.level_params.items():
            for p in params:
                if p.grad is None:
                    continue
                buffer = self._accum.get(p)
                if buffer is None:
                    self._accum[p] = p.grad.detach().clone()
                else:
                    buffer.add_(p.grad.detach())
            self._accum_counts[level] = self._accum_counts.get(level, 0) + 1

        for level in due:
            opt = self.optimizers.get(level)
            if opt is None:
                continue
            count = max(self._accum_counts.get(level, 1), 1)
            for p in self.level_params[level]:
                buffer = self._accum.get(p)
                # The mean, not the sum: a level that integrates 32 gradients should
                # take one step of ordinary size along their average, not a step 32
                # times too large.
                p.grad = (buffer / count) if buffer is not None else None
            if level == 0 and fast_gain != 1.0:
                for group, base in zip(opt.param_groups, self._base_lrs[level], strict=True):
                    group["lr"] = base * fast_gain
            opt.step()

            for p in self.level_params[level]:
                self._accum.pop(p, None)
            self._accum_counts[level] = 0

        for level, params in self.level_params.items():
            if level not in due:
                for p in params:
                    p.grad = None
        return due

    def state_bytes(self) -> int:
        """Optimiser state bytes across levels, for the footprint.

        Includes the per-level gradient accumulators, which are real memory this
        method spends and a baseline does not.
        """
        total = sum(b.numel() * b.element_size() for b in self._accum.values())
        for opt in self.optimizers.values():
            if isinstance(opt, DeepMomentum):
                total += opt.state_bytes()
                continue
            for group in opt.param_groups:
                for p in group["params"]:
                    st = opt.state.get(p, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            total += st[key].numel() * st[key].element_size()
        return total
