"""Windowing, normalisation and torch DataLoaders over an :class:`Environment`.

Turning an environment into training examples involves three choices that each affect
whether a reported number means anything:

**Windows never cross a trace boundary.** A window spanning two traces would splice
two different UEs into one input sequence. Windows are built per trace and concatenated.

**Normalisation constants are fitted on source environments only** (design rule 5).
Fitting per environment would erase exactly the covariate shift the benchmark exists to
measure -- every environment would arrive pre-whitened and look identical to the model.
Fitting globally is worse: it leaks statistics of future environments into the past,
which in a continual-learning paper is the mistake reviewers look for first.

**Missingness is a channel, not a fill value.** After standardisation, missing entries
become 0 (the fitted mean) and a companion channel records what fraction of features
were missing at that timestep. Silently imputing without that channel tells the model a
masked ``ratio_granted_req`` is an average one, which is a fabricated observation --
the same error the Day 1 sanitisation policy exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nestedric.data import colosseum as C
from nestedric.data.schema import FEATURE_COLUMNS, LOG1P_COLUMNS, TARGET_COLUMNS
from nestedric.data.stream import Environment


@dataclass
class Normaliser:
    """Per-feature mean and standard deviation, fitted once on source environments.

    Held in float64 and applied in float32. Standard deviations are floored rather than
    allowed to reach zero: a feature constant within the source environments but varying
    later would otherwise divide by zero and produce infinities on exactly the
    environments the paper cares about.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]
    n_rows_fitted: int = 0
    source_env_ids: tuple[str, ...] = ()

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise a ``(n, n_features)`` block."""
        return ((x - self.mean) / self.std).astype("float32")

    def to_dict(self) -> dict:
        """Serialisable form, written into every run directory for reproducibility."""
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_rows_fitted": self.n_rows_fitted,
            "source_env_ids": list(self.source_env_ids),
        }


def apply_log1p(block: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    """log1p the heavy-tailed non-negative KPIs, in place on a copy.

    Applied before standardisation and before the constants are fitted, so the fitted
    mean and std describe the transformed quantity -- fitting on raw values and then
    transforming would make the constants meaningless.
    """
    out = np.array(block, dtype="float64", copy=True)
    for i, name in enumerate(columns):
        if name in LOG1P_COLUMNS:
            col = out[:, i]
            with np.errstate(invalid="ignore"):
                out[:, i] = np.log1p(np.clip(col, 0.0, None))
    return out


def fit_normaliser(
    envs: list[Environment],
    processed_dir: str | Path,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
    std_floor: float = 1e-6,
) -> Normaliser:
    """Fit standardisation constants on *envs*, which must be the source environments.

    Uses training traces only. Evaluation traces of the source environments are held
    out of the fit as well: they are part of what the model is scored on, and letting
    their statistics into the constants is a small leak that is free to avoid.
    """
    if not envs:
        raise ValueError("normalisation needs at least one source environment")

    processed_dir = Path(processed_dir)
    n = np.zeros(len(columns), dtype="float64")
    s1 = np.zeros(len(columns), dtype="float64")
    s2 = np.zeros(len(columns), dtype="float64")
    total = 0

    for env in envs:
        frame = _load_env_frame(env, processed_dir, list(columns), env.train_traces)
        block = apply_log1p(frame.to_numpy(dtype="float64", na_value=np.nan), columns)
        mask = ~np.isnan(block)
        n += mask.sum(axis=0)
        s1 += np.nansum(block, axis=0)
        s2 += np.nansum(block**2, axis=0)
        total += len(block)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / np.maximum(n, 1), 0.0)
        var = np.where(n > 0, s2 / np.maximum(n, 1) - mean**2, 1.0)
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std < std_floor, 1.0, std)

    return Normaliser(
        mean=mean,
        std=std,
        columns=tuple(columns),
        n_rows_fitted=total,
        source_env_ids=tuple(e.env_id for e in envs),
    )


def _load_env_frame(
    env: Environment,
    processed_dir: Path,
    columns: list[str],
    traces: list[str],
) -> pd.DataFrame:
    """Load the rows of *traces* within *env*, in trace then timestamp order."""
    needed = list(dict.fromkeys([*columns, "trace_id", "timestamp_ms", *env.row_filter]))
    df = C.load_shards(processed_dir, env.dataset, env.shards, columns=needed)

    for key, want in env.row_filter.items():
        wanted = set(want) if isinstance(want, (list, tuple, set)) else {want}
        df = df[df[key].isin(wanted)]

    df = df[df["trace_id"].isin(set(traces))]
    df = df.sort_values(["trace_id", "timestamp_ms"], kind="stable")
    return df[columns] if columns else df


#: Relative change in granted PRBs below which the action is "hold". The control task
#: is a derived label, not a logged one: no action is recorded in the traces, and the
#: obvious candidate (``sched_policy``) is constant within every environment we cut, so
#: predicting it would be a lookup rather than control. See docs/BENCHMARK.md.
ACTION_DEADBAND = 0.05

#: Control-task classes, in label order.
ACTIONS: tuple[str, ...] = ("decrease", "hold", "increase")


def derive_actions(granted: np.ndarray, deadband: float = ACTION_DEADBAND) -> np.ndarray:
    """Label each step by the next change in granted PRBs: decrease / hold / increase.

    This is the xApp-shaped task: given a window of KPIs, choose how the allocation
    should move. It is a *constructed* label and the paper must say so -- the traces
    log outcomes, not decisions. The deadband keeps sensor noise out of the two active
    classes, which would otherwise dominate a signal that is mostly flat.
    """
    current = granted[:-1]
    nxt = granted[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(current > 0, (nxt - current) / np.maximum(current, 1e-9), 0.0)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    labels = np.ones(len(rel), dtype="int64")  # hold
    labels[rel > deadband] = 2  # increase
    labels[rel < -deadband] = 0  # decrease
    return labels


@dataclass
class WindowSet:
    """Model-ready arrays for one environment split, plus the trace each window came from."""

    x: np.ndarray  # (n, window, n_features + 1)
    y: np.ndarray  # (n, n_targets) standardised regression targets
    actions: np.ndarray  # (n,) control-task class labels
    trace_index: np.ndarray  # (n,) which trace each window came from

    def __len__(self) -> int:
        return len(self.x)


def build_windows(
    env: Environment,
    processed_dir: str | Path,
    normaliser: Normaliser,
    traces: list[str],
    window: int = 32,
    stride: int = 8,
    horizon: int = 1,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> WindowSet:
    """Build ``(X, y, trace_index)`` for one environment split.

    ``X`` has shape ``(n_windows, window, n_features + 1)``: the standardised features
    plus a trailing channel giving the fraction of features missing at that timestep.
    ``y`` holds the *standardised* targets ``horizon`` steps after the window ends, so
    losses are comparable across KPIs of wildly different scale (throughput in Mbps,
    buffer in bytes). ``trace_index`` records which trace each window came from, which
    the fold-level statistics need.

    Windows never span two traces.
    """
    processed_dir = Path(processed_dir)
    cols = list(feature_columns)
    frame = _load_env_frame(env, processed_dir, [*cols, "trace_id"], traces)

    target_pos = [cols.index(c) for c in target_columns]
    granted_pos = cols.index("sum_granted_prbs")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    tidx: list[np.ndarray] = []

    for order, (_trace, part) in enumerate(frame.groupby("trace_id", observed=True, sort=True)):
        raw_block = part[cols].to_numpy(dtype="float64", na_value=np.nan)
        if len(raw_block) < window + horizon + 1:
            continue
        block = apply_log1p(raw_block, cols)

        missing = np.isnan(block)
        standardised = normaliser.transform(np.where(missing, normaliser.mean, block))
        missing_rate = missing.mean(axis=1, keepdims=True).astype("float32")
        channels = np.concatenate([standardised, missing_rate], axis=1)

        # Actions come from the RAW counter, before log1p: a relative change is only
        # meaningful in the original units.
        raw_granted = np.nan_to_num(raw_block[:, granted_pos], nan=0.0)
        actions = derive_actions(raw_granted)

        last = len(block) - window - horizon
        starts = np.arange(0, min(last, len(actions) - window - horizon + 1), stride)
        if not len(starts):
            continue
        idx = starts[:, None] + np.arange(window)[None, :]
        end = starts + window + horizon - 1
        xs.append(channels[idx])
        ys.append(standardised[end][:, target_pos])
        acts.append(actions[end])
        tidx.append(np.full(len(starts), order, dtype="int32"))
        del block, standardised, channels

    if not xs:
        return WindowSet(
            x=np.empty((0, window, len(cols) + 1), dtype="float32"),
            y=np.empty((0, len(target_columns)), dtype="float32"),
            actions=np.empty((0,), dtype="int64"),
            trace_index=np.empty((0,), dtype="int32"),
        )

    return WindowSet(
        x=np.concatenate(xs).astype("float32"),
        y=np.concatenate(ys).astype("float32"),
        actions=np.concatenate(acts),
        trace_index=np.concatenate(tidx),
    )


def make_dataloaders(
    env: Environment,
    cfg: dict,
    normaliser: Normaliser,
    processed_dir: str | Path = "data/processed",
    seed: int = 0,
):
    """Return ``(train_loader, eval_loader)`` for one environment.

    Imports torch lazily so the data-side tooling and its tests run without it.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    window = int(cfg.get("window", 32))
    stride = int(cfg.get("stride", 8))
    horizon = int(cfg.get("horizon", 1))
    batch_size = int(cfg.get("batch_size", 256))

    loaders = []
    for traces, shuffle in ((env.train_traces, True), (env.eval_traces, False)):
        ws = build_windows(env, processed_dir, normaliser, traces, window, stride, horizon)
        dataset = TensorDataset(
            torch.from_numpy(ws.x), torch.from_numpy(ws.y), torch.from_numpy(ws.actions)
        )
        generator = torch.Generator().manual_seed(seed)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator if shuffle else None,
                drop_last=False,
                num_workers=0,  # 4 vCPU total; workers cost more than they return here
            )
        )
    return loaders[0], loaders[1]
