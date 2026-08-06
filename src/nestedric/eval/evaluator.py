"""Builds the T x T evaluation matrix and the canonical results record.

After finishing environment *i*, the learner is scored on **every** environment in the
stream -- those already seen (retention) and those not yet seen (forward transfer).
That full matrix, rather than a running average, is what makes BWT, forgetting and FWT
computable after the fact, and what a later reader needs to recompute a metric we did
not think to record.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


#: Backward transfer above this is divergence, not retention. The run that motivated
#: the guard scored +5.90; genuine positive transfer on a null stream is ~+0.001.
IMPLAUSIBLE_BWT = 0.5


class ContinualEvaluator:
    """Evaluates on every seen (and unseen) environment after each environment."""

    def __init__(self, stream, loaders: dict, device: str = "cpu") -> None:
        self.stream = stream
        self.loaders = loaders  # env_id -> eval DataLoader
        self.device = device
        self.T = len(stream)
        self.R = np.full((self.T, self.T), np.nan, dtype="float64")
        self.R_mse = np.full((self.T, self.T), np.nan, dtype="float64")
        self.R_acc = np.full((self.T, self.T), np.nan, dtype="float64")

    @torch.no_grad()
    def score(self, method, env) -> dict:
        """Mean loss and action accuracy on one environment's eval split."""
        loader = self.loaders[env.env_id]
        se, n, correct = 0.0, 0, 0
        for batch in loader:
            x, y, a = batch
            pred, logits = method.predict((x,))
            y = y.to(pred.device)
            se += float(F.mse_loss(pred, y, reduction="sum"))
            correct += int((logits.argmax(dim=1).cpu() == a).sum())
            n += len(x)
        if n == 0:
            return {"mse": float("nan"), "accuracy": float("nan"), "n": 0}
        return {"mse": se / (n * pred.shape[1]), "accuracy": correct / n, "n": n}

    def evaluate_all(self, method, after_env_index: int) -> dict:
        """Score every environment in the stream; fill row *after_env_index* of R."""
        row = {}
        for j, env in enumerate(self.stream):
            scores = self.score(method, env)
            self.R_mse[after_env_index, j] = scores["mse"]
            self.R_acc[after_env_index, j] = scores["accuracy"]
            # Performance, higher-is-better: negated error. Converted once, here, so no
            # downstream metric has to remember the sign convention.
            self.R[after_env_index, j] = -scores["mse"]
            row[env.env_id] = scores
        return row

    def sanity(self) -> dict:
        """Checks a reader would otherwise have to perform by eye.

        A results file should say whether it is trustworthy. These flags are written
        into every run so a broken sweep announces itself instead of being caught by
        someone squinting at a summary table.
        """
        from nestedric.eval import metrics as M

        bwt = M.backward_transfer(self.R)
        finite = bool(np.isfinite(self.R).all())

        # Positive backward transfer is only suspicious when it is LARGE. A threshold of
        # 0.01 was wrong: on a stream with no forgetting, small positive BWT is the
        # correct answer -- finetune scores +0.0012 on radio-shift -- and the guard
        # silently excluded nestedric from the two null streams in the main benchmark,
        # which are precisely the streams where it looks best.
        #
        # The failure this exists to catch was BWT = +5.90 from a diverged run, three
        # orders of magnitude away. IMPLAUSIBLE_BWT is set where that is caught and
        # ordinary positive transfer is not.
        implausible = bool(bwt > IMPLAUSIBLE_BWT)
        bad_performance = bool(np.nanmin(self.R) < -50.0)
        return {
            "all_finite": finite,
            "positive_bwt": bool(bwt > 0.0),
            "implausible_bwt": implausible,
            "implausible_performance": bad_performance,
            "trustworthy": bool(finite and not implausible and not bad_performance),
        }

    def finalise(self) -> dict:
        """All metrics plus the raw matrices for the results artefact."""
        from nestedric.eval import metrics as M

        return {
            "sanity": self.sanity(),
            "R": self.R.tolist(),
            "R_mse": self.R_mse.tolist(),
            "R_accuracy": self.R_acc.tolist(),
            "env_ids": [e.env_id for e in self.stream],
            "average_performance": M.average_performance(self.R),
            "bwt": M.backward_transfer(self.R),
            "per_environment_bwt": M.per_environment_bwt(self.R).tolist(),
            "forgetting": M.forgetting_measure(self.R),
        }
