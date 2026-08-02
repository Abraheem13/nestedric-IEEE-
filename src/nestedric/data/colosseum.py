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


def to_canonical(raw: pd.DataFrame, meta: TraceMeta) -> tuple[pd.DataFrame, dict[str, int]]:
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

    df, masked = sanitise(df, report=True)
    df["ratio_granted_req"] = recompute_ratio(df)
    df, masked2 = sanitise(df, report=True)  # bound the derived ratio too
    for k, v in masked2.items():
        masked[k] = masked.get(k, 0) + v

    return df, masked


def prepare(
    root: Path,
    out: Path,
    dataset: str,
    min_rows: int = 100,
    limit: int | None = None,
    validate: bool = True,
) -> tuple[Path, dict[str, int]]:
    """Convert a whole dataset tree into one canonical parquet file.

    Parameters
    ----------
    root
        Dataset root (the cloned repository directory).
    out
        Destination parquet path.
    dataset
        ``'coloran'`` or ``'commag'``.
    min_rows
        Traces shorter than this are dropped: a trace with a handful of samples cannot
        support a windowed model and only adds noise to the environment statistics.
    limit
        Process at most this many files (for smoke runs).
    validate
        Run the schema validator on the concatenated frame before writing.

    Returns
    -------
    (path, report)
        The written parquet path and the aggregated sanitisation report. The report is
        also written next to the parquet as ``<name>.sanitisation.csv``, because the
        masked fractions are a paper artefact, not a debugging aid.

    """
    files = iter_metric_files(root)
    if limit is not None:
        files = files[:limit]

    frames: list[pd.DataFrame] = []
    total: dict[str, int] = {}
    n_rows_seen = 0
    skipped_unparsed = 0
    skipped_short = 0

    for path in files:
        meta = parse_path(path, dataset)
        if meta is None:
            skipped_unparsed += 1
            continue
        raw = read_metrics_csv(path)
        if len(raw) < min_rows:
            skipped_short += 1
            continue
        frame, masked = to_canonical(raw, meta)
        frames.append(frame)
        n_rows_seen += len(frame)
        for k, v in masked.items():
            total[k] = total.get(k, 0) + v

    if not frames:
        raise RuntimeError(
            f"no usable traces under {root} "
            f"(unparsed={skipped_unparsed}, short={skipped_short})"
        )

    df = pd.concat(frames, ignore_index=True)

    if validate:
        KPISchema(strict=True).validate(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    rep = pd.DataFrame(
        [
            {"column": c, "masked": n, "fraction": n / max(n_rows_seen, 1)}
            for c, n in sorted(total.items(), key=lambda kv: -kv[1])
        ],
        columns=["column", "masked", "fraction"],
    )
    rep.to_csv(out.with_suffix(".sanitisation.csv"), index=False)

    return out, total
