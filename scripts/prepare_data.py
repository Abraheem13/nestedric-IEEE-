#!/usr/bin/env python3
"""Day 1: prepare raw Colosseum datasets into canonical parquet, then profile them.

Standalone so it runs before the CLI exists (Day 2).

    python scripts/prepare_data.py --dataset coloran --limit 500   # quick check
    python scripts/prepare_data.py --dataset all                   # full run

The profile it prints is the input to the Day 2 stream design and to the Day 4
"is there any forgetting at all" gate, so read it, don't just run it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data import colosseum as C  # noqa: E402
from nestedric.data.schema import (  # noqa: E402
    CONTEXT_COLUMNS,
    KPI_COLUMNS,
)

ROOTS = {
    "coloran": "colosseum-oran-coloran-dataset",
    "commag": "colosseum-oran-commag-dataset",
}


def profile(df: pd.DataFrame, name: str) -> None:
    """Print the facts that decide how environments are cut."""
    print(f"\n{'=' * 72}\nPROFILE: {name}\n{'=' * 72}")
    print(f"rows       : {len(df):,}")
    print(f"traces     : {df.trace_id.nunique():,}")
    print(f"columns    : {len(df.columns)}")

    span = (df.timestamp_ms.max() - df.timestamp_ms.min()) / 1000
    print(f"time span  : {span:,.0f} s")

    print("\n--- context cardinality (candidate environment axes) ---")
    for col in CONTEXT_COLUMNS:
        vals = df[col].dropna().unique()
        shown = sorted(vals.tolist())[:8]
        print(f"  {col:18s} n={len(vals):3d}  {shown}{' ...' if len(vals) > 8 else ''}")

    print("\n--- rows per candidate environment (sched_policy x slice_id) ---")
    grp = df.groupby(["sched_policy", "slice_id"], observed=True).size()
    print(grp.to_string())

    print("\n--- KPI missingness ---")
    miss = df[list(KPI_COLUMNS)].isna().mean().sort_values(ascending=False)
    nonzero = miss[miss > 0]
    print(nonzero.round(4).to_string() if len(nonzero) else "  none")

    print("\n--- KPI summary ---")
    desc = df[list(KPI_COLUMNS)].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.round(3).to_string())

    print("\n--- per-slice downlink throughput (slice semantics check) ---")
    print(
        df.groupby("slice_id", observed=True)
        .dl_thpt_mbps.agg(["count", "mean", "std", "max"])
        .round(4)
        .to_string()
    )

    const = [c for c in KPI_COLUMNS if df[c].nunique(dropna=True) <= 1]
    if const:
        print(f"\n!! constant KPI columns (drop these as features): {const}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coloran", "commag", "all"], default="all")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--limit", type=int, default=None, help="max files (smoke run)")
    ap.add_argument("--min-rows", type=int, default=100)
    args = ap.parse_args()

    names = ["coloran", "commag"] if args.dataset == "all" else [args.dataset]

    for name in names:
        root = args.raw_dir / ROOTS[name]
        if not root.exists():
            print(f"!! {root} missing -- run scripts/download_data.sh first")
            continue

        out = args.out_dir / f"{name}.parquet"
        print(f"\n==> preparing {name} from {root}")
        t0 = time.time()
        _, report = C.prepare(root, out, dataset=name, min_rows=args.min_rows, limit=args.limit)
        dt = time.time() - t0

        df = pd.read_parquet(out)
        size_mb = out.stat().st_size / 1e6
        print(f"    wrote {out} ({size_mb:.1f} MB) in {dt:.1f}s")

        print("\n--- sanitisation: values masked as physically impossible ---")
        n = len(df)
        if not report:
            print("  none")
        else:
            for col, cnt in sorted(report.items(), key=lambda kv: -kv[1]):
                print(f"  {col:22s} {cnt:>12,}  ({cnt / max(n, 1):7.4%})")
            print(f"  (report written to {out.with_suffix('.sanitisation.csv')})")

        profile(df, name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
