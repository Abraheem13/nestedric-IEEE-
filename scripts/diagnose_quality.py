#!/usr/bin/env python3
"""Diagnose how missingness is distributed across the prepared parquet shards.

Sanitisation runs inside ``to_canonical``, so corruption in a prepared shard has
already become missingness. The question this script answers is therefore not "which
values are invalid" but "is the missingness diffuse or concentrated", because the two
demand different fixes:

    diffuse across traces   -> mask, keep the feature
    concentrated in traces  -> drop those traces
    concentrated by dataset -> the feature leaks dataset identity; DROP THE FEATURE

The third case is the one to worry about. Day 1 showed ``sum_requested_prbs`` masked at
2-14% per trace on COMMAG and effectively 0% on ColO-RAN. If that asymmetry survives,
a model can infer the testbed from the missingness pattern, and in the cross-dataset
stream that is leakage rather than signal. This script is the Day 2 gate: run it, read
the verdict, and decide the feature set BEFORE cutting environments.

Everything streams shard by shard -- the corpus does not fit in 16 GB, and a diagnostic
that only runs on a machine bigger than the training node is not much of a diagnostic.

    python scripts/diagnose_quality.py
    python scripts/diagnose_quality.py --column sum_requested_prbs --by scenario
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data import colosseum as C  # noqa: E402
from nestedric.data.schema import CONTEXT_COLUMNS, FEATURE_COLUMNS, KPI_COLUMNS  # noqa: E402

#: A feature whose missingness differs by more than this between datasets lets a model
#: identify the testbed from the missingness pattern alone.
LEAKAGE_THRESHOLD_PP = 0.05


def scan(out_dir: Path, dataset: str, column: str, by: str) -> dict:
    """Stream every shard once, accumulating what the three verdicts need."""
    manifest = C.read_manifest(out_dir, dataset)
    group_col = by if by in CONTEXT_COLUMNS else "scenario"
    cols = list(dict.fromkeys([*KPI_COLUMNS, "trace_id", group_col]))

    missing = pd.Series(0, index=list(KPI_COLUMNS), dtype="int64")
    nunique: dict[str, set] = {c: set() for c in KPI_COLUMNS}
    per_trace: list[pd.Series] = []
    per_group: list[pd.DataFrame] = []
    total = 0

    for shard in manifest.shard:
        df = C.load_shards(out_dir, dataset, [shard], columns=cols)
        total += len(df)
        missing += df[list(KPI_COLUMNS)].isna().sum()

        for c in KPI_COLUMNS:
            if len(nunique[c]) <= 2:  # only need to know "constant or not"
                nunique[c].update(df[c].dropna().unique()[:3].tolist())

        if column in df.columns:
            per_trace.append(
                df.groupby("trace_id", observed=True)[column].apply(lambda s: s.isna().mean())
            )
            per_group.append(
                df.assign(_miss=df[column].isna())
                .groupby(group_col, observed=True)
                .agg(n_missing=("_miss", "sum"), n=("_miss", "size"))
            )
        del df

    return {
        "total": total,
        "missing": missing,
        "constant": [c for c, vals in nunique.items() if len(vals) <= 1],
        "per_trace": pd.concat(per_trace) if per_trace else pd.Series(dtype="float64"),
        "per_group": (pd.concat(per_group).groupby(level=0).sum() if per_group else pd.DataFrame()),
    }


def report(res: dict, dataset: str, column: str, by: str) -> None:
    """Print the per-dataset verdict."""
    bar = "=" * 72
    print(f"\n{bar}\nQUALITY: {dataset}  ({res['total']:,} rows)\n{bar}")

    frac = (res["missing"] / max(res["total"], 1)).sort_values(ascending=False)
    frac = frac[frac > 0]
    print("\n--- missingness per feature ---")
    print(frac.round(5).to_string() if len(frac) else "  none")

    print(f"\n--- constant columns ---\n  {res['constant'] or 'none'}")

    per_group = res["per_group"]
    if len(per_group):
        print(f"\n--- missingness of {column!r} by {by!r} ---")
        out = per_group.assign(fraction=per_group.n_missing / per_group.n)
        print(out.round(5).to_string())

    per_trace = res["per_trace"]
    if not len(per_trace) or per_trace.sum() == 0:
        print(f"\n--- concentration of {column!r} ---\n  no missing values")
        return

    print(f"\n--- concentration of {column!r} across traces ---")
    top_n = max(1, len(per_trace) // 10)
    share = per_trace.nlargest(top_n).sum() / per_trace.sum()
    print(f"  traces             : {len(per_trace):,}")
    print(f"  worst-decile share : {share:.1%}")
    print(
        f"  per-trace fraction : median {per_trace.median():.4%}, "
        f"p90 {per_trace.quantile(0.9):.4%}, max {per_trace.max():.4%}"
    )
    verdict = (
        "CONCENTRATED -- consider dropping the worst traces"
        if share > 0.5
        else "DIFFUSE -- masking is the right response"
    )
    print(f"  verdict            : {verdict}")


def leakage_check(results: dict[str, dict]) -> None:
    """Compare per-dataset missingness; flag features that encode dataset identity."""
    print(f"\n{'=' * 72}\nCROSS-DATASET LEAKAGE CHECK\n{'=' * 72}")

    fracs = pd.DataFrame(
        {name: res["missing"] / max(res["total"], 1) for name, res in results.items()}
    ).loc[list(FEATURE_COLUMNS)]

    fracs["gap"] = fracs.max(axis=1) - fracs.min(axis=1)
    shown = fracs[fracs.gap > 0].sort_values("gap", ascending=False)

    print("\nmissingness by dataset, per feature (features with any gap):")
    print(shown.round(5).to_string() if len(shown) else "  no feature differs")

    risky = fracs[fracs.gap > LEAKAGE_THRESHOLD_PP]
    if len(risky):
        print(
            f"\n!! features whose missingness differs by >{LEAKAGE_THRESHOLD_PP:.0%} "
            f"between datasets:\n   {list(risky.index)}\n"
            "   These let a model identify the testbed from missingness alone.\n"
            "   Exclude them from the cross-dataset stream, or impute identically."
        )
    else:
        print(
            f"\n  no feature exceeds a {LEAKAGE_THRESHOLD_PP:.0%} gap "
            "-- no obvious leakage channel"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coloran", "commag", "all"], default="all")
    ap.add_argument("--dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--column", default="sum_requested_prbs")
    ap.add_argument(
        "--by",
        default="scenario",
        help="context axis for the per-group breakdown (scenario, tr_config, ...)",
    )
    args = ap.parse_args()

    names = ["coloran", "commag"] if args.dataset == "all" else [args.dataset]
    results: dict[str, dict] = {}

    for name in names:
        try:
            res = scan(args.dir, name, args.column, args.by)
        except FileNotFoundError as exc:
            print(f"!! {exc}")
            continue
        report(res, name, args.column, args.by)
        results[name] = res

    if len(results) > 1:
        leakage_check(results)
    elif results:
        print("\n  (only one dataset present -- the leakage check needs both)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
