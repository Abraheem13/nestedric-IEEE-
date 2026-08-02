"""Environment-stream construction.

An O-RAN-CL *stream* is an ordered list of environments; each environment is a set of
traces sharing a context signature (slice assignment, mobility, distance, scheduling
policy, RBG allocation). The learner sees them sequentially and is evaluated on every
environment seen so far, producing the T x T matrix the metrics are computed from.

Three invariants hold by construction, and each is asserted rather than assumed:

1. **The trace is the atom.** Train/eval splits divide traces, never rows. Consecutive
   rows of one trace are 250 ms apart and heavily autocorrelated, so a row-level split
   would put near-duplicate samples on both sides and report a generalisation gap that
   does not exist.
2. **No trace appears in two environments.** Otherwise BWT would measure re-fitting of
   data the model already had, not retention.
3. **Streams are a deterministic function of (config, seed).** Same inputs, same
   environments, same splits, on any machine -- see :mod:`nestedric.utils.seeding`.

Environments are selected from the shard manifest written by preparation, so building
a stream reads a CSV and a handful of one-column parquet reads. The KPI data itself is
opened later, by the loaders, one environment at a time.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nestedric.data import colosseum as C
from nestedric.utils.seeding import seeded_rng

#: Context axes selectable at manifest level (no data is opened to filter on these).
SHARD_AXES: tuple[str, ...] = (
    "dataset",
    "scenario",
    "slice_assignment",
    "mobility",
    "distance",
    "tr_config",
)

#: Context axes that live in the rows themselves and are applied after loading.
ROW_AXES: tuple[str, ...] = ("sched_policy", "slice_id")


class StreamError(ValueError):
    """Raised when a stream config cannot be satisfied by the prepared corpus."""


@dataclass
class Environment:
    """One task in the continual stream.

    Holds *references* to data (shards, trace ids, row filters), never the data. A
    stream of twelve environments therefore costs kilobytes, and the 16 GB node only
    ever materialises the environment currently being trained or evaluated.
    """

    env_id: str
    dataset: str
    context: dict[str, Any]
    shards: list[str]
    train_traces: list[str]
    eval_traces: list[str]
    row_filter: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_traces(self) -> int:
        """Total traces in this environment, both splits."""
        return len(self.train_traces) + len(self.eval_traces)

    @property
    def n_rows(self) -> int:
        """Row count recorded at build time (0 if the builder did not count)."""
        return int(self.meta.get("n_rows", 0))

    def describe(self) -> str:
        """One-line summary for the CLI environment table."""
        ctx = ", ".join(f"{k}={v}" for k, v in sorted(self.context.items()))
        return (
            f"{self.env_id:<28} {self.dataset:<8} {self.n_rows:>9,} rows  "
            f"{len(self.train_traces):>4}/{len(self.eval_traces):<4} traces  {ctx}"
        )


@dataclass
class EnvironmentStream:
    """An ordered sequence of :class:`Environment` objects plus stream-level metadata."""

    name: str
    family: str
    environments: list[Environment]
    seed: int = 0
    drift_schedule: dict = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Environment]:
        return iter(self.environments)

    def __len__(self) -> int:
        return len(self.environments)

    def __getitem__(self, i: int) -> Environment:
        return self.environments[i]

    def table(self) -> str:
        """The environment table printed by ``nestedric stream``."""
        head = f"{self.name} [{self.family}] -- {len(self)} environments, seed {self.seed}"
        lines = [head, "-" * len(head)]
        lines += [f"{i:>2}. {env.describe()}" for i, env in enumerate(self.environments)]
        total = sum(e.n_rows for e in self.environments)
        lines.append("-" * len(head))
        lines.append(f"    total {total:,} rows across {sum(e.n_traces for e in self)} traces")
        return "\n".join(lines)


# --------------------------------------------------------------------- selection


def _match_shards(manifest: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Rows of *manifest* matching every shard-level filter."""
    sel = manifest
    for key, want in filters.items():
        if key not in sel.columns:
            raise StreamError(f"unknown shard axis {key!r}; available: {sorted(sel.columns)}")
        wanted = set(want) if isinstance(want, (list, tuple, set)) else {want}
        sel = sel[sel[key].astype(str).isin({str(w) for w in wanted})]
    return sel


def _split_filters(spec: dict[str, Any]) -> tuple[dict, dict]:
    """Partition an environment spec into shard-level and row-level filters."""
    shard_f = {k: v for k, v in spec.items() if k in SHARD_AXES}
    row_f = {k: v for k, v in spec.items() if k in ROW_AXES}
    unknown = set(spec) - set(SHARD_AXES) - set(ROW_AXES) - {"env_id"}
    if unknown:
        raise StreamError(
            f"unknown environment axes {sorted(unknown)}; "
            f"shard axes {SHARD_AXES}, row axes {ROW_AXES}"
        )
    return shard_f, row_f


def _enumerate_traces(
    processed_dir: Path,
    dataset: str,
    shards: list[str],
    row_filter: dict[str, Any],
    quality_column: str | None = None,
    max_missing: float = 1.0,
) -> pd.Series:
    """Row count per trace for a shard selection, after row-level filtering.

    Reads only the columns needed to identify, filter and quality-screen traces --
    never the KPI block. Returns a Series indexed by ``trace_id`` so a caller that
    subsamples traces still knows the exact row count of what it kept.

    When *quality_column* is given, traces whose missing fraction in that column
    exceeds *max_missing* are dropped. This is the response to a "concentrated in
    traces" verdict from ``scripts/diagnose_quality.py``: on COMMAG the worst decile
    of traces carries 86% of all missing ``sum_requested_prbs``, with a median trace
    at 0.0%, so removing a few hundred traces is far cheaper than losing a feature or
    masking across the whole corpus.
    """
    cols = ["trace_id", *row_filter]
    if quality_column:
        cols.append(quality_column)
    df = C.load_shards(processed_dir, dataset, shards, columns=list(dict.fromkeys(cols)))

    for key, want in row_filter.items():
        wanted = set(want) if isinstance(want, (list, tuple, set)) else {want}
        df = df[df[key].isin(wanted)]

    if quality_column and max_missing < 1.0:
        missing = df.groupby("trace_id", observed=True)[quality_column].apply(
            lambda s: s.isna().mean()
        )
        keep = set(missing[missing <= max_missing].index)
        df = df[df["trace_id"].isin(keep)]

    return df.groupby("trace_id", observed=True).size().sort_index()


def _split_traces(
    traces: Sequence[str], eval_fraction: float, seed: int, env_id: str
) -> tuple[list[str], list[str]]:
    """Deterministically hold out whole traces for evaluation.

    Seeded from ``(seed, env_id)`` rather than a running counter, so adding or removing
    an environment elsewhere in the stream cannot shift this environment's split.
    """
    if not traces:
        return [], []
    rng = seeded_rng(seed, env_id)
    order = list(traces)
    rng.shuffle(order)
    n_eval = max(1, round(len(order) * eval_fraction)) if eval_fraction > 0 else 0
    n_eval = min(n_eval, len(order) - 1) if len(order) > 1 else 0
    return sorted(order[n_eval:]), sorted(order[:n_eval])


# ------------------------------------------------------------------ enumeration


def _auto_environments(manifest: pd.DataFrame, axes: Sequence[str], limit: int) -> list[dict]:
    """Derive environment specs by grouping the manifest on *axes*.

    Within a dataset, groups are ordered by row count, largest first, so truncating to
    ``n_environments`` keeps the cells with the most data rather than an arbitrary
    alphabetical prefix.

    Across datasets, selection is **round-robin, not global rank**. Ranking globally
    silently produced a nine-environment "cross-dataset" stream containing only
    ColO-RAN: its cells span three scheduling policies and hold ~1.4M rows each against
    COMMAG's ~400k, so they took every slot. The stream would have measured
    within-testbed shift while being reported as cross-testbed -- a wrong number that
    looks entirely reasonable in a table.
    """
    bad = [a for a in axes if a not in manifest.columns]
    if bad:
        raise StreamError(f"cannot group on {bad}; manifest has {sorted(manifest.columns)}")

    ranked: dict[str, list[dict]] = {}
    for dataset, part in manifest.groupby("dataset", observed=True):
        grouped = (
            part.groupby(list(axes), observed=True)["n_rows"].sum().sort_values(ascending=False)
        )
        cells = []
        for key, _ in grouped.items():
            values = key if isinstance(key, tuple) else (key,)
            # The dataset is always part of the spec, even when it is not a grouping
            # axis. Both testbeds contain a 'rome_static_medium' scenario, so grouping
            # on scenario alone yields the *same* spec from each dataset; each then
            # matches both, and two environments claim one set of traces. The
            # disjointness guard catches it, but only after a stream has been built.
            cells.append({"dataset": str(dataset), **dict(zip(axes, values, strict=True))})
        ranked[str(dataset)] = cells

    specs: list[dict] = []
    for depth in range(max((len(v) for v in ranked.values()), default=0)):
        for dataset in sorted(ranked):
            if depth < len(ranked[dataset]) and len(specs) < limit:
                specs.append(ranked[dataset][depth])
        if len(specs) >= limit:
            break
    return specs


#: Manifest column summarising each row-level axis, as a '|'-joined list of values.
_ROW_AXIS_MANIFEST_COLUMN: dict[str, str] = {
    "sched_policy": "sched_policies",
    "slice_id": "slice_ids",
}


def _row_axis_values(manifest: pd.DataFrame, axis: str) -> list[int]:
    """Distinct values of a row-level axis, read from the manifest summary column."""
    column = _ROW_AXIS_MANIFEST_COLUMN[axis]
    values = {
        int(float(v))
        for cell in manifest[column].dropna()
        for v in str(cell).split("|")
        if v not in ("", "nan")
    }
    return sorted(values)


def _expand_row_axis(specs: list[dict], axis: str, values: Sequence[Any]) -> list[dict]:
    """Cross every spec with each value of a row-level axis (e.g. scheduling policy)."""
    return [{**spec, axis: value} for spec in specs for value in values]


# ----------------------------------------------------------------------- public


def build_stream(cfg: dict, processed_dir: str | Path = "data/processed") -> EnvironmentStream:
    """Materialise a stream from a ``configs/stream/*.yaml`` config.

    The config states *what* varies between environments; this function resolves that
    against the prepared corpus and fixes the splits. Supported keys:

    ``environments``
        Explicit list of context filters, one per environment. Most legible, and what
        ``radio_shift.yaml`` uses -- the environments are a scientific choice, so
        writing them out beats inferring them.
    ``context_axes``
        Alternative to the above: group the manifest on these axes and take the
        ``n_environments`` largest cells.
    ``order``
        ``fixed`` keeps config order; ``random`` permutes under the stream seed;
        ``cyclic`` repeats the sequence ``repeat`` times so retention is measured on
        environments that genuinely recur.
    ``env_min_samples``
        Environments with fewer rows are dropped, with a warning. An environment too
        small to fit a window model is noise in the T x T matrix.
    ``eval_fraction``
        Share of *traces* (never rows) held out for evaluation.

    Parameters
    ----------
    cfg
        Parsed stream config.
    processed_dir
        Directory holding ``<dataset>.manifest.csv`` and the shard subdirectories.

    Returns
    -------
    EnvironmentStream
        Ordered environments with disjoint trace sets and fixed train/eval splits.

    """
    processed_dir = Path(processed_dir)
    name = cfg.get("name", "unnamed")
    family = cfg.get("family", name)
    seed = int(cfg.get("seed", 0))
    sources = cfg.get("source") or ["coloran"]
    if isinstance(sources, str):
        sources = [sources]
    eval_fraction = float(cfg.get("eval_fraction", 0.2))
    min_samples = int(cfg.get("env_min_samples", 0))
    n_environments = int(cfg.get("n_environments", 0)) or None
    max_traces = int(cfg.get("max_traces_per_env", 0)) or None
    quality_column = cfg.get("trace_quality_column")
    max_missing = float(cfg.get("trace_max_missing", 1.0))

    manifests = {}
    for ds in sources:
        try:
            manifests[ds] = C.read_manifest(processed_dir, ds)
        except FileNotFoundError as exc:
            raise StreamError(f"stream {name!r} needs dataset {ds!r}: {exc}") from exc
    manifest = pd.concat(manifests.values(), ignore_index=True)

    specs: list[dict]
    if cfg.get("environments"):
        specs = [dict(s) for s in cfg["environments"]]
    elif cfg.get("context_axes"):
        axes = [a for a in cfg["context_axes"] if a in SHARD_AXES]
        row_axes = [a for a in cfg["context_axes"] if a in ROW_AXES]
        specs = _auto_environments(manifest, axes or ["scenario"], n_environments or 8)
        for axis in row_axes:
            specs = _expand_row_axis(specs, axis, _row_axis_values(manifest, axis))
    else:
        raise StreamError(
            f"stream {name!r} defines neither 'environments' nor 'context_axes'; "
            "one of them must say what changes between environments"
        )

    environments: list[Environment] = []
    claimed: dict[str, str] = {}
    dropped: list[tuple[str, int]] = []

    for i, spec in enumerate(specs):
        base_id = spec.pop("env_id", None) or f"env{i:02d}"
        shard_f, row_f = _split_filters(spec)

        for ds, ds_manifest in manifests.items():
            sel = _match_shards(ds_manifest, shard_f)
            if sel.empty:
                continue
            # One spec can match both datasets (both have a 'rome_static_medium'
            # scenario). Those are different environments and must not share an id:
            # the T x T matrix is keyed by environment, so a duplicate id silently
            # overwrites a column of results.
            env_id = base_id if len(manifests) == 1 else f"{base_id}_{ds}"
            shards = sorted(sel["shard"].tolist())
            counts = _enumerate_traces(
                processed_dir, ds, shards, row_f, quality_column, max_missing
            )

            if max_traces and len(counts) > max_traces:
                # Environments here run to millions of rows; the campaign is 300 runs
                # on one L4. Subsampling *traces* (never rows) keeps the trace as the
                # atom and keeps environments comparable in size, which the bound's
                # per-environment effective sample size depends on.
                rng = seeded_rng(seed, name, env_id, "subsample")
                keep = rng.choice(counts.index.to_numpy(), size=max_traces, replace=False)
                counts = counts.loc[sorted(keep)]

            traces = counts.index.tolist()
            n_rows = int(counts.sum())

            if n_rows < min_samples:
                dropped.append((env_id, n_rows, f"below env_min_samples={min_samples}"))
                continue

            overlap = [t for t in traces if t in claimed]
            if overlap:
                raise StreamError(
                    f"environment {env_id!r} reuses {len(overlap)} traces already in "
                    f"{claimed[overlap[0]]!r} (e.g. {overlap[0]!r}). Environments must be "
                    "disjoint or backward transfer measures re-fitting, not retention."
                )
            for t in traces:
                claimed[t] = env_id

            train, evl = _split_traces(traces, eval_fraction, seed, f"{name}:{env_id}")
            if eval_fraction > 0 and (not train or not evl):
                # One trace cannot be both trained on and evaluated on. Such an
                # environment would contribute an empty column to the T x T matrix,
                # which every downstream metric would then average over as if real.
                dropped.append(
                    (env_id, n_rows, f"only {len(traces)} trace(s); cannot split train/eval")
                )
                for t in traces:
                    claimed.pop(t, None)
                continue

            environments.append(
                Environment(
                    env_id=env_id,
                    dataset=ds,
                    context={**shard_f, **row_f},
                    shards=shards,
                    train_traces=train,
                    eval_traces=evl,
                    row_filter=row_f,
                    meta={"n_rows": n_rows, "spec_index": i},
                )
            )

    if not environments:
        reasons = "; ".join(f"{eid} ({rows:,} rows): {why}" for eid, rows, why in dropped[:6])
        raise StreamError(
            f"stream {name!r} produced no environments. Dropped {len(dropped)}: {reasons}"
        )

    order = cfg.get("order", "fixed")
    if order == "random":
        rng = seeded_rng(seed, name, "order")
        rng.shuffle(environments)
    elif order == "cyclic":
        environments = environments * int(cfg.get("repeat", 2))

    if n_environments and order != "cyclic":
        environments = environments[:n_environments]

    return EnvironmentStream(
        name=name,
        family=family,
        environments=environments,
        seed=seed,
        meta={
            "sources": list(sources),
            "eval_fraction": eval_fraction,
            "dropped_small": dropped,
            "processed_dir": str(processed_dir),
        },
    )
