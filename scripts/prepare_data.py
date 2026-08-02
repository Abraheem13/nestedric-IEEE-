#!/usr/bin/env python3
"""Day 1-2: prepare raw Colosseum datasets into canonical parquet shards, then profile.

Standalone so it runs before the CLI exists (Day 2).

    python scripts/prepare_data.py --dataset coloran --limit 500   # quick check
    python scripts/prepare_data.py --dataset all                   # full run

Preparation writes one parquet shard per (scenario, slice_assignment, tr_config) plus a
manifest, rather than one file per dataset: the full corpus does not fit in memory on a
16 GB node, and the stream builder wants to select environments without opening data.

The profile is computed by *streaming* over shards and combining per-shard aggregates,
for the same reason -- a single ``describe()`` over 35.5M rows would undo the sharding.
Counts, means, minima and maxima come out identical to the one-shot computation; the
standard deviation is accumulated from sums of squares.

The profile is the input to the Day 2 stream design and to the Day 4 "is there any
forgetting at all" gate, so read it, don't just run it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data import colosseum as C  # noqa: E402
from nestedric.data.schema import KPI_COLUMNS  # noqa: E402

ROOTS = {
    "coloran": "colosseum-oran-coloran-dataset",
    "commag": "colosseum-oran-commag-dataset",
}


class RunningStats:
    """Streaming per-column count/mean/std/min/max, combined across shards.

    Sums of squares are numerically adequate here because the values are sanitised KPIs
    with bounded physical ranges; the catastrophic-cancellation case (huge mean, tiny
    variance) does not arise once dl_mcs = 2.4e8 has been masked.
    """

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns
        self.n: dict[str, int] = dict.fromkeys(columns, 0)
        self.missing: dict[str, int] = dict.fromkeys(columns, 0)
        self.total = 0
        self.s1: dict[str, float] = dict.fromkeys(columns, 0.0)
        self.s2: dict[str, float] = dict.fromkeys(columns, 0.0)
        self.lo: dict[str, float] = dict.fromkeys(columns, math.inf)
        self.hi: dict[str, float] = dict.fromkeys(columns, -math.inf)

    def update(self, df: pd.DataFrame) -> None:
        """Fold one shard into the running aggregates."""
        self.total += len(df)
        for c in self.columns:
            if c not in df.columns:
                continue
            v = pd.to_numeric(df[c], errors="coerce").dropna().astype("float64")
            self.missing[c] += len(df) - len(v)
            if v.empty:
                continue
            self.n[c] += len(v)
            self.s1[c] += float(v.sum())
            self.s2[c] += float((v**2).sum())
            self.lo[c] = min(self.lo[c], float(v.min()))
            self.hi[c] = max(self.hi[c], float(v.max()))

    def frame(self) -> pd.DataFrame:
        """Combined statistics, one row per column."""
        rows = []
        for c in self.columns:
            n = self.n[c]
            mean = self.s1[c] / n if n else math.nan
            var = (self.s2[c] / n - mean**2) if n else math.nan
            rows.append(
                {
                    "column": c,
                    "n": n,
                    "missing_frac": self.missing[c] / self.total if self.total else math.nan,
                    "mean": mean,
                    "std": math.sqrt(max(var, 0.0)) if n else math.nan,
                    "min": self.lo[c] if n else math.nan,
                    "max": self.hi[c] if n else math.nan,
                }
            )
        return pd.DataFrame(rows).set_index("column")


def profile(out_dir: Path, dataset: str) -> None:
    """Print the facts that decide how environments are cut."""
    manifest = C.read_manifest(out_dir, dataset)
    print(f"\n{'=' * 72}\nPROFILE: {dataset}\n{'=' * 72}")
    print(f"rows       : {manifest.n_rows.sum():,}")
    print(f"traces     : {manifest.n_traces.sum():,}")
    print(f"shards     : {len(manifest)}")
    span = (manifest.t_end_ms.max() - manifest.t_start_ms.min()) / 1000
    print(f"time span  : {span:,.0f} s")

    print("\n--- context cardinality (candidate environment axes) ---")
    for col in ("scenario", "slice_assignment", "mobility", "distance", "tr_config"):
        vals = sorted(manifest[col].dropna().astype(str).unique().tolist())
        print(f"  {col:18s} n={len(vals):3d}  {vals[:8]}{' ...' if len(vals) > 8 else ''}")
    for col, key in (("sched_policy", "sched_policies"), ("slice_id", "slice_ids")):
        seen = sorted({v for s in manifest[key].dropna() for v in str(s).split("|")})
        print(f"  {col:18s} n={len(seen):3d}  {seen}")

    print("\n--- rows per shard ---")
    print(
        f"  min {manifest.n_rows.min():,}  median {int(manifest.n_rows.median()):,}  "
        f"max {manifest.n_rows.max():,}"
    )
    print("  five smallest shards (watch these when cutting environments):")
    print(manifest.nsmallest(5, "n_rows")[["shard", "n_rows", "n_traces"]].to_string(index=False))

    stats = RunningStats(KPI_COLUMNS)
    per_slice: list[pd.DataFrame] = []
    cols = [*KPI_COLUMNS, "slice_id"]

    for shard in manifest.shard:
        df = C.load_shards(out_dir, dataset, [shard], columns=cols)
        stats.update(df)
        per_slice.append(
            df.groupby("slice_id", observed=True)
            .dl_thpt_mbps.agg(["count", "sum", "max"])
            .reset_index()
        )
        del df

    summary = stats.frame()
    print("\n--- KPI summary (streamed over shards) ---")
    print(summary.round(3).to_string())

    miss = summary.missing_frac[summary.missing_frac > 0].sort_values(ascending=False)
    print("\n--- KPI missingness ---")
    print(miss.round(5).to_string() if len(miss) else "  none")

    print("\n--- per-slice downlink throughput (slice semantics check) ---")
    ps = (
        pd.concat(per_slice)
        .groupby("slice_id", observed=True)
        .agg(count=("count", "sum"), total=("sum", "sum"), max=("max", "max"))
    )
    ps["mean"] = ps.total / ps["count"]
    print(ps[["count", "mean", "max"]].round(4).to_string())

    const = [
        c
        for c in KPI_COLUMNS
        if summary.loc[c, "n"] and summary.loc[c, "min"] == summary.loc[c, "max"]
    ]
    if const:
        print(f"\n!! constant KPI columns (drop these as features): {const}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coloran", "commag", "all"], default="all")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None, help="max files (smoke run)")
    ap.add_argument("--min-rows", type=int, default=100)
    ap.add_argument("--profile-only", action="store_true", help="skip preparation")
    args = ap.parse_args()

    names = ["coloran", "commag"] if args.dataset == "all" else [args.dataset]

    for name in names:
        if not args.profile_only:
            root = args.raw_dir / ROOTS[name]
            if not root.exists():
                print(f"!! {root} missing -- run scripts/download_data.sh first")
                continue

            print(f"\n==> preparing {name} from {root}")
            t0 = time.time()
            manifest_path, report = C.prepare(
                root,
                args.out_dir,
                dataset=name,
                min_rows=args.min_rows,
                limit=args.limit,
            )
            dt = time.time() - t0

            shard_bytes = sum(p.stat().st_size for p in (args.out_dir / name).glob("*.parquet"))
            print(f"    wrote {shard_bytes / 1e6:.1f} MB of shards in {dt:.1f}s")
            print(f"    manifest: {manifest_path}")

            print("\n--- sanitisation: values masked as physically impossible ---")
            n_rows = int(C.read_manifest(args.out_dir, name).n_rows.sum())
            if not report:
                print("  none")
            else:
                for col, cnt in sorted(report.items(), key=lambda kv: -kv[1]):
                    print(f"  {col:26s} {cnt:>12,}  ({cnt / max(n_rows, 1):7.4%})")
                print(f"  (report written to {args.out_dir / f'{name}.sanitisation.csv'})")

        profile(args.out_dir, name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
