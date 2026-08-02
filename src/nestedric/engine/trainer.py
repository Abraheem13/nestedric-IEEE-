"""The continual-learning loop shared by every method (no method-specific branches).

The loop iterates the stream, calls the Method hooks, and drives the evaluator. Its
only concession to a particular method is the ``wants_joint_data`` flag on the oracle,
which is not a learning rule but a different data regime -- and is declared by the
method rather than tested for by name.
"""

from __future__ import annotations

import itertools
import time

import numpy as np


class Divergence(RuntimeError):
    """Training loss went non-finite or implausibly large."""


#: A standardised-MSE loss above this is not slow learning, it is divergence. With
#: targets standardised to roughly unit variance, predicting the mean scores ~1.0.
MAX_PLAUSIBLE_LOSS = 100.0


class ContinualTrainer:
    """Iterate the stream, call the Method hooks, drive the ContinualEvaluator."""

    def __init__(self, method, stream, evaluator, train_loaders: dict, cfg: dict) -> None:
        self.method = method
        self.stream = stream
        self.evaluator = evaluator
        self.train_loaders = train_loaders
        self.cfg = cfg
        self.epochs = int(cfg.get("epochs_per_env", 3))
        self.history: list[dict] = []

    def _joint_batches(self):
        """Interleave batches from every environment, for the oracle only."""
        iters = [iter(loader) for loader in self.train_loaders.values()]
        while iters:
            for it in list(iters):
                try:
                    yield next(it)
                except StopIteration:
                    iters.remove(it)

    def run(self) -> dict:
        """Execute the full stream and return the results record."""
        joint = bool(getattr(self.method, "wants_joint_data", False))
        step = 0

        for i, env in enumerate(self.stream):
            self.method.begin_environment(env, i)
            loader = self.train_loaders[env.env_id]
            # EWC needs a second pass over this environment at its end; handing the
            # loader to the environment keeps that out of the trainer's signature.
            env._train_loader = loader

            t0 = time.time()
            losses: list[float] = []
            for _epoch in range(self.epochs):
                batches = self._joint_batches() if joint else loader
                for batch in batches:
                    logs = self.method.observe(batch, step)
                    loss = logs["loss"]
                    if not np.isfinite(loss) or loss > MAX_PLAUSIBLE_LOSS:
                        # Fail here rather than write a results.json full of numbers
                        # that then have to be recognised as nonsense by eye. A
                        # diverged run once reported BWT = +5.90 and was summarised as
                        # "FORGETS" before anyone noticed.
                        raise Divergence(
                            f"loss {loss:.4g} at step {step} on environment "
                            f"{env.env_id!r} (limit {MAX_PLAUSIBLE_LOSS}). Targets are "
                            "standardised, so predicting the mean scores about 1.0. "
                            "Check scaling before adjusting the learning rate."
                        )
                    losses.append(loss)
                    step += 1

            self.method.end_environment(env, i)
            row = self.evaluator.evaluate_all(self.method, i)
            self.history.append(
                {
                    "env_index": i,
                    "env_id": env.env_id,
                    "train_seconds": time.time() - t0,
                    "mean_train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "scores": row,
                    "state": self.method.state_summary(),
                }
            )
            if joint:
                # The oracle sees everything at once; one pass over the union is the
                # whole of its training, and repeating it per environment would give it
                # T times the compute of every other method.
                for j in range(i + 1, len(self.stream)):
                    self.evaluator.R[j] = self.evaluator.R[i]
                    self.evaluator.R_mse[j] = self.evaluator.R_mse[i]
                    self.evaluator.R_acc[j] = self.evaluator.R_acc[i]
                break

        results = self.evaluator.finalise()
        results["history"] = self.history
        results["footprint"] = self.method.footprint()
        results["steps"] = step
        return results
