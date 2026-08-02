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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nestedric.data import colosseum as C
from nestedric.data.schema import FEATURE_COLUMNS, TARGET_COLUMNS
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
        block = frame.to_numpy(dtype="float64", na_value=np.nan)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    tidx: list[np.ndarray] = []

    for order, (trace, part) in enumerate(frame.groupby("trace_id", observed=True, sort=True)):
        block = part[cols].to_numpy(dtype="float64", na_value=np.nan)
        if len(block) < window + horizon:
            continue

        missing = np.isnan(block)
        standardised = normaliser.transform(np.where(missing, normaliser.mean, block))
        missing_rate = missing.mean(axis=1, keepdims=True).astype("float32")
        channels = np.concatenate([standardised, missing_rate], axis=1)

        starts = np.arange(0, len(block) - window - horizon + 1, stride)
        if not len(starts):
            continue
        idx = starts[:, None] + np.arange(window)[None, :]
        xs.append(channels[idx])
        ys.append(standardised[starts + window + horizon - 1][:, target_pos])
        tidx.append(np.full(len(starts), order, dtype="int32"))
        del block, standardised, channels

    if not xs:
        n_features = len(cols) + 1
        return (
            np.empty((0, window, n_features), dtype="float32"),
            np.empty((0, len(target_columns)), dtype="float32"),
            np.empty((0,), dtype="int32"),
        )

    return (
        np.concatenate(xs).astype("float32"),
        np.concatenate(ys).astype("float32"),
        np.concatenate(tidx),
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
        x, y, _ = build_windows(env, processed_dir, normaliser, traces, window, stride, horizon)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
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
