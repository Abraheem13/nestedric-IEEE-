"""Adapter for the two Colosseum O-RAN slice-metrics datasets.

Both datasets export the same 31-column slice-metrics CSV, so one adapter serves both;
they differ only in directory layout and in which context axes vary.

ColO-RAN (Polese et al., IEEE Trans. Mobile Computing 22(10):5787-5800, 2022)
    rome_static_medium/sched{0,1,2}/tr{0..27}/exp{1..5}/bs{1..7}/slices_bsN/<IMSI>_metrics.csv
    7 BS, 42 UE, 10 MHz / 50 PRB, static, medium distance.
    Varies: scheduling policy, RBG allocation (tr), repetition, BS, slice.

COMMAG (Bonati et al., IEEE Communications Magazine 59(10):21-27, 2021)
    slice_{mixed,traffic}/rome_{static,slow}_{close,medium,far}/tr{0..17}/exp{1..6}/bs{1..4}/slices_bsN/<IMSI>_metrics.csv
    4 BS, 40 UE, 3 MHz / 15 PRB.
    Additionally varies mobility (static/slow), distance (close/medium/far) and slice
    assignment (mixed/traffic) -- the axes producing real radio-condition shift.

Licence note: both datasets are GPL-3.0. This repository is Apache-2.0 and does NOT
vendor or redistribute the data or any derived parquet; it downloads and prepares
locally. See docs/DATASETS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from nestedric.data.schema import (
    ALL_COLUMNS,
    KPI_COLUMNS,
    RAW_TO_CANONICAL,
    KPISchema,
    sanitise,
)

COLORAN_REPO = "https://github.com/wineslab/colosseum-oran-coloran-dataset"
COMMAG_REPO = "https://github.com/wineslab/colosseum-oran-commag-dataset"
LICENCE = "GPL-3.0"

#: ColO-RAN:  .../rome_static_medium/sched0/tr7/exp3/bs5/slices_bs5/1010123456049_metrics.csv
_COLORAN_RE = re.compile(
    r"(?P<scenario>rome_[a-z]+_[a-z]+)/sched(?P<sched>\d+)/(?P<tr>tr\d+)/"
    r"(?P<exp>exp\d+)/bs(?P<bs>\d+)/slices_bs\d+/(?P<imsi>\d+)_metrics\.csv$"
)

#: COMMAG:  .../slice_traffic/rome_static_close/tr0/exp1/bs1/slices_bs1/1010123456007_metrics.csv
_COMMAG_RE = re.compile(
    r"slice_(?P<assignment>mixed|traffic)/(?P<scenario>rome_(?P<mobility>static|slow)_"
    r"(?P<distance>close|medium|far))/(?P<tr>tr\d+)/(?P<exp>exp\d+)/bs(?P<bs>\d+)/"
    r"slices_bs\d+/(?P<imsi>\d+)_metrics\.csv$"
)


@dataclass(frozen=True)
class TraceMeta:
    """Context recovered from a file path. Environments are defined over these."""

    dataset: str
    scenario: str
    slice_assignment: str
    mobility: str
    distance: str
    tr_config: str
    exp_id: str
    bs_id: int
    imsi: int

    @property
    def trace_id(self) -> str:
        """Stable unique identifier for this trace, used as the split key."""
        return (
            f"{self.dataset}:{self.scenario}:{self.slice_assignment}:"
            f"{self.tr_config}:{self.exp_id}:bs{self.bs_id}:{self.imsi}"
        )

    @property
    def shard_key(self) -> str:
        """Grouping key for one parquet shard.

        Preparation writes one shard per ``(scenario, slice_assignment, tr_config)``
        rather than one file per dataset. Two reasons, in order of importance:

        1. The full corpus does not fit in memory on the target node (4 vCPU / 16 GB).
           Concatenating ~35.5M rows x 35 columns needs roughly 5 GB for the frame plus
           a full copy during ``pd.concat``. Sharding caps peak usage at one shard:
           ~1.3M rows for ColO-RAN, ~275k for COMMAG.
        2. Environments are defined over exactly these context axes, so the stream
           builder can select environments by reading the manifest and then opening
           only the shards it needs -- never the whole corpus.

        Note that ColO-RAN's three scheduling policies land in the *same* shard: the
        policy lives in a path component not carried on TraceMeta (it is read from the
        ``scheduling_policy`` column instead), and sched-shift environments are cut
        within a shard rather than across shards.
        """
        return f"{self.slice_assignment}__{self.scenario}__{self.tr_config}"

    def with_(self, **kw) -> TraceMeta:
        """Return a copy with fields replaced (used by tests)."""
        return replace(self, **kw)


def parse_path(path: Path, dataset: str) -> TraceMeta | None:
    """Recover trace context from a metrics-file path, or ``None`` if it does not match.

    ColO-RAN carries its scheduling policy in the directory name; COMMAG carries it in
    the ``scheduling_policy`` column instead, so it is read from the file in both cases
    for consistency (verified identical to the directory for ColO-RAN).
    """
    posix = path.as_posix()

    if dataset == "coloran":
        m = _COLORAN_RE.search(posix)
        if not m:
            return None
        scenario = m.group("scenario")
        _, mobility, distance = scenario.split("_", 2)
        return TraceMeta(
            dataset="coloran",
            scenario=scenario,
            slice_assignment="traffic",
            mobility=mobility,
            distance=distance,
            tr_config=m.group("tr"),
            exp_id=m.group("exp"),
            bs_id=int(m.group("bs")),
            imsi=int(m.group("imsi")),
        )

    if dataset == "commag":
        m = _COMMAG_RE.search(posix)
        if not m:
            return None
        return TraceMeta(
            dataset="commag",
            scenario=m.group("scenario"),
            slice_assignment=m.group("assignment"),
            mobility=m.group("mobility"),
            distance=m.group("distance"),
            tr_config=m.group("tr"),
            exp_id=m.group("exp"),
            bs_id=int(m.group("bs")),
            imsi=int(m.group("imsi")),
        )

    raise ValueError(f"unknown dataset: {dataset!r}")


def iter_metric_files(root: Path) -> list[Path]:
    """All slice-metrics CSVs under *root*."""
    return sorted(root.rglob("*_metrics.csv"))


def read_metrics_csv(path: Path) -> pd.DataFrame:
    """Read one slice-metrics CSV, dropping the logger's unnamed spacer columns."""
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    return df


def recompute_ratio(df: pd.DataFrame) -> pd.Series:
    """Derive ``ratio_granted_req`` from already-sanitised PRB counters.

    Missing where nothing was requested (idle UE: 19% of COMMAG rows, 42% of the
    ColO-RAN sample) or where either counter was masked. Dividing by a clipped
    denominator instead fabricates ratios in the thousands -- the first implementation
    did exactly that and produced a maximum of 11,424.
    """
    req = pd.to_numeric(df["sum_requested_prbs"], errors="coerce")
    gr = pd.to_numeric(df["sum_granted_prbs"], errors="coerce")
    return (gr / req).where(req > 0).astype("float32")


def to_canonical(
    raw: pd.DataFrame, meta: TraceMeta, report: bool = False
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map one raw trace onto the canonical schema, sanitising corrupt logger values.

    Order matters: values are masked BEFORE ``ratio_granted_req`` is derived, so the
    corruption in the raw PRB counters cannot propagate into a feature that would then
    look superficially reasonable.
    """
    keep = {k: v for k, v in RAW_TO_CANONICAL.items() if k in raw.columns}
    df = raw[list(keep)].rename(columns=keep).copy()

    df["ratio_granted_req"] = pd.NA  # derived after sanitisation

    df["dataset"] = meta.dataset
    df["trace_id"] = meta.trace_id
    df["bs_id"] = meta.bs_id
    df["imsi"] = meta.imsi
    df["scenario"] = meta.scenario
    df["slice_assignment"] = meta.slice_assignment
    df["mobility"] = meta.mobility
    df["distance"] = meta.distance
    df["tr_config"] = meta.tr_config
    df["exp_id"] = meta.exp_id

    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[list(ALL_COLUMNS)]
    df = df.sort_values("timestamp_ms", kind="stable").reset_index(drop=True)

    df["slice_id"] = pd.to_numeric(df["slice_id"], errors="coerce").astype("Int8")
    df["sched_policy"] = pd.to_numeric(df["sched_policy"], errors="coerce").astype("Int8")
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
    for col in KPI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    df, masked = sanitise(df, report=report)
    df["ratio_granted_req"] = recompute_ratio(df)
    df, masked2 = sanitise(df, report=report)  # bound the derived ratio too
    for k, v in masked2.items():
        masked[k] = masked.get(k, 0) + v

    return df, masked


def group_by_shard(
    files: list[Path], dataset: str
) -> tuple[dict[str, list[tuple[Path, TraceMeta]]], int]:
    """Bucket metrics files by :attr:`TraceMeta.shard_key`.

    Returns the buckets and the number of files whose path did not parse. Parsing every
    path up front is cheap (no file is opened) and means the number of shards is known
    before any data is read.
    """
    buckets: dict[str, list[tuple[Path, TraceMeta]]] = {}
    unparsed = 0
    for path in files:
        meta = parse_path(path, dataset)
        if meta is None:
            unparsed += 1
            continue
        buckets.setdefault(meta.shard_key, []).append((path, meta))
    return buckets, unparsed


def prepare_shard(
    items: list[tuple[Path, TraceMeta]],
    out: Path,
    min_rows: int = 100,
    validate: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Convert one shard's traces to canonical form and write a single parquet file.

    Returns the concatenated frame, the shard's sanitisation counts, and the number of
    traces skipped for being shorter than *min_rows*. The frame is returned so the
    caller can compute manifest statistics before dropping it -- it is the caller's job
    to let it go out of scope, which is what keeps peak memory to one shard.
    """
    frames: list[pd.DataFrame] = []
    masked_total: dict[str, int] = {}
    skipped_short = 0

    for path, meta in items:
        raw = read_metrics_csv(path)
        if len(raw) < min_rows:
            skipped_short += 1
            continue
        frame, masked = to_canonical(raw, meta, report=verbose)
        frames.append(frame)
        for k, v in masked.items():
            masked_total[k] = masked_total.get(k, 0) + v

    if not frames:
        return pd.DataFrame(columns=list(ALL_COLUMNS)), masked_total, skipped_short

    df = pd.concat(frames, ignore_index=True)
    del frames

    if validate:
        KPISchema(strict=True).validate(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")

    return df, masked_total, skipped_short


def _shard_summary(df: pd.DataFrame, shard: str, path: Path) -> dict:
    """One manifest row: enough context to choose environments without opening data."""
    return {
        "shard": shard,
        "path": path.name,
        "dataset": df["dataset"].iloc[0],
        "scenario": df["scenario"].iloc[0],
        "slice_assignment": df["slice_assignment"].iloc[0],
        "mobility": df["mobility"].iloc[0],
        "distance": df["distance"].iloc[0],
        "tr_config": df["tr_config"].iloc[0],
        "n_rows": len(df),
        "n_traces": df["trace_id"].nunique(),
        "sched_policies": "|".join(str(v) for v in sorted(df["sched_policy"].dropna().unique())),
        "slice_ids": "|".join(str(v) for v in sorted(df["slice_id"].dropna().unique())),
        "exp_ids": "|".join(sorted(df["exp_id"].dropna().unique())),
        "bs_ids": "|".join(str(v) for v in sorted(df["bs_id"].dropna().unique())),
        "t_start_ms": int(df["timestamp_ms"].min()),
        "t_end_ms": int(df["timestamp_ms"].max()),
    }


def prepare(
    root: Path,
    out_dir: Path,
    dataset: str,
    min_rows: int = 100,
    limit: int | None = None,
    validate: bool = True,
    verbose: bool = False,
) -> tuple[Path, dict[str, int]]:
    """Convert a dataset tree into per-scenario canonical parquet shards.

    One shard per ``(scenario, slice_assignment, tr_config)``; see
    :attr:`TraceMeta.shard_key` for why the corpus is not written as a single file.
    Peak memory is one shard, so the full 35.5M-row ColO-RAN corpus prepares inside a
    16 GB node.

    Parameters
    ----------
    root
        Dataset root (the cloned repository directory).
    out_dir
        Destination directory. Shards go to ``out_dir/<dataset>/<shard>.parquet``.
    dataset
        ``'coloran'`` or ``'commag'``.
    min_rows
        Traces shorter than this are dropped: a trace with a handful of samples cannot
        support a windowed model and only adds noise to the environment statistics.
    limit
        Process at most this many files (for smoke runs).
    validate
        Run the schema validator on every shard before writing it.
    verbose
        Print per-trace sanitisation counts. Off by default: on the full corpus this
        is tens of thousands of lines.

    Returns
    -------
    (manifest_path, report)
        Path to ``out_dir/<dataset>.manifest.csv`` and the aggregated sanitisation
        counts. The manifest lists every shard with its context and row count, so the
        stream builder never has to open data to decide what an environment contains.
        The sanitisation report is written to ``out_dir/<dataset>.sanitisation.csv``
        because the masked fractions are a paper artefact, not a debugging aid.

    """
    files = iter_metric_files(root)
    if limit is not None:
        files = files[:limit]

    buckets, skipped_unparsed = group_by_shard(files, dataset)
    if not buckets:
        raise RuntimeError(f"no parseable metrics files under {root} ({len(files)} seen)")

    shard_dir = out_dir / dataset
    shard_dir.mkdir(parents=True, exist_ok=True)

    total: dict[str, int] = {}
    rows: list[dict] = []
    n_rows_seen = 0
    skipped_short = 0

    for i, (shard, items) in enumerate(sorted(buckets.items()), start=1):
        path = shard_dir / f"{shard}.parquet"
        df, masked, short = prepare_shard(items, path, min_rows, validate, verbose)
        skipped_short += short
        for k, v in masked.items():
            total[k] = total.get(k, 0) + v

        if len(df):
            rows.append(_shard_summary(df, shard, path))
            n_rows_seen += len(df)
            print(
                f"  [{i:>3}/{len(buckets)}] {shard:<48} "
                f"{len(df):>9,} rows  {df.trace_id.nunique():>5} traces"
            )
        else:
            print(f"  [{i:>3}/{len(buckets)}] {shard:<48} EMPTY (all traces < {min_rows} rows)")
        del df  # peak memory is one shard, and only one shard

    if not rows:
        raise RuntimeError(
            f"every trace under {root} was skipped "
            f"(unparsed={skipped_unparsed}, short={skipped_short})"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / f"{dataset}.manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    rep = pd.DataFrame(
        [
            {"column": c, "masked": n, "fraction": n / max(n_rows_seen, 1)}
            for c, n in sorted(total.items(), key=lambda kv: -kv[1])
        ],
        columns=["column", "masked", "fraction"],
    )
    rep.to_csv(out_dir / f"{dataset}.sanitisation.csv", index=False)

    print(
        f"\n  {dataset}: {n_rows_seen:,} rows in {len(rows)} shards "
        f"(skipped: {skipped_unparsed} unparsed paths, {skipped_short} short traces)"
    )
    return manifest_path, total


def read_manifest(out_dir: Path, dataset: str) -> pd.DataFrame:
    """Load the shard manifest written by :func:`prepare`."""
    path = out_dir / f"{dataset}.manifest.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run scripts/prepare_data.py --dataset {dataset}"
        )
    return pd.read_csv(path)


def load_shards(
    out_dir: Path,
    dataset: str,
    shards: list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read selected shards into one frame.

    Always pass *shards* (and ideally *columns*) in production paths: loading the whole
    corpus is exactly what the sharding exists to avoid. The unrestricted form is for
    small datasets and tests.
    """
    shard_dir = out_dir / dataset
    names = shards if shards is not None else [p.stem for p in sorted(shard_dir.glob("*.parquet"))]
    frames = [pd.read_parquet(shard_dir / f"{s}.parquet", columns=columns) for s in names]
    if not frames:
        raise FileNotFoundError(f"no shards found under {shard_dir}")
    return pd.concat(frames, ignore_index=True)
