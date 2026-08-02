#!/usr/bin/env python3
"""Diagnose how missingness is distributed in the prepared parquet files.

Sanitisation runs inside ``to_canonical``, so corruption in a prepared parquet has
already become missingness. The question this script answers is therefore not "which
values are invalid" but "is the missingness diffuse or concentrated", because the two
demand different fixes:

    diffuse across traces   -> mask, keep the feature
    concentrated in traces  -> drop those traces
    concentrated by dataset -> the feature leaks dataset identity; DROP THE FEATURE

The third case is the one to worry about. Day 1 showed ``sum_requested_prbs`` masked at
2-14% per trace on COMMAG and effectively 0% on ColO-RAN. If that asymmetry survives,
a model can infer the testbed from the missingness pattern, and in the cross-dataset
stream that is leakage rather than signal.

    python scripts/diagnose_quality.py
    python scripts/diagnose_quality.py --column sum_requested_prbs --by scenario
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data.quality import (  # noqa: E402
    concentration,
    constant_columns,
    missingness,
    missingness_by_group,
)
from nestedric.data.schema import FEATURE_COLUMNS, KPI_COLUMNS  # noqa: E402


def report(df: pd.DataFrame, name: str, column: str, by: str) -> None:
    bar = "=" * 72
    head = f"QUALITY: {name}  ({len(df):,} rows, {df.trace_id.nunique():,} traces)"
    print(f"\n{bar}\n{head}\n{bar}")

    print("\n--- missingness per feature ---")
    miss = missingness(df, KPI_COLUMNS)
    print(miss.round(5).to_string() if len(miss) else "  none")

    const = constant_columns(df, KPI_COLUMNS)
    print(f"\n--- constant columns ---\n  {const if const else 'none'}")

    if column not in df.columns:
        print(f"\n!! column {column!r} not present, skipping concentration analysis")
        return

    print(f"\n--- missingness of {column!r} by {by!r} ---")
    grp = missingness_by_group(df, (column,), by)
    print(grp.round(5).to_string())

    print(f"\n--- concentration of {column!r} across traces ---")
    con = concentration(df, column, by="trace_id")
    share = con.attrs.get("top_decile_share")
    if share is None:
        print("  no missing values")
        return
    print(f"  traces               : {con.attrs['n_groups']:,}")
    print(f"  total missing        : {con.attrs['total_missing']:,}")
    print(f"  worst-decile share   : {share:.1%}")
    print(
        f"  per-trace fraction   : median {con.missing_fraction.median():.4%}, "
        f"p90 {con.missing_fraction.quantile(0.9):.4%}, "
        f"max {con.missing_fraction.max():.4%}"
    )
    verdict = (
        "CONCENTRATED -- consider dropping the worst traces"
        if share > 0.5
        else "DIFFUSE -- masking is the right response"
    )
    print(f"  verdict              : {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coloran", "commag", "all"], default="all")
    ap.add_argument("--dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--column", default="sum_requested_prbs")
    ap.add_argument("--by", default="dataset")
    ap.add_argument("--sample", type=int, default=None, help="rows to sample (memory)")
    args = ap.parse_args()

    names = ["coloran", "commag"] if args.dataset == "all" else [args.dataset]
    frames = []

    for name in names:
        path = args.dir / f"{name}.parquet"
        if not path.exists():
            print(f"!! {path} missing -- run scripts/prepare_data.py first")
            continue
        df = pd.read_parquet(path)
        if args.sample and len(df) > args.sample:
            df = df.sample(args.sample, random_state=0)
        report(df, name, args.column, "scenario" if args.by == "dataset" else args.by)
        frames.append(df)

    if len(frames) > 1:
        print(f"\n{'=' * 72}\nCROSS-DATASET LEAKAGE CHECK\n{'=' * 72}")
        both = pd.concat(frames, ignore_index=True)
        grp = missingness_by_group(both, FEATURE_COLUMNS, "dataset")
        spread = (grp.max() - grp.min()).sort_values(ascending=False)
        print("\nmissingness gap between datasets, per feature (descending):")
        print(spread[spread > 0].round(5).to_string())
        risky = spread[spread > 0.05]
        if len(risky):
            print(
                "\n!! features whose missingness differs by >5pp between datasets:\n"
                f"   {list(risky.index)}\n"
                "   These let a model identify the testbed from missingness alone.\n"
                "   Exclude them from the cross-dataset stream, or impute identically."
            )
        else:
            print("\n  no feature exceeds a 5pp gap -- no obvious leakage channel")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
