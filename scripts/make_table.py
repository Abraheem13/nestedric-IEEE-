#!/usr/bin/env python3
"""Build the paper's results table: fold-level CIs, effect sizes, corrected p-values.

A table of means over seeds is not a result. This computes, per stream, every method's
difference from a reference (replay by default -- the strongest baseline measured, and
therefore the comparison that matters) with a paired bootstrap CI over folds, a paired
permutation p-value, Cohen's d_z, and a Holm-Bonferroni decision across methods.

    python scripts/make_table.py --dir results/runs/main --reference replay

Writes results/tables/main.csv and prints the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from nestedric.eval.evaluator import IMPLAUSIBLE_BWT  # noqa: E402
from nestedric.eval.metrics import backward_transfer  # noqa: E402
from nestedric.utils.stats import bootstrap_ci, compare_methods  # noqa: E402


def _trustworthy(record: dict) -> bool:
    """Recompute trustworthiness from the stored R matrix.

    The flag inside results.json was written when the run executed, so a later
    correction to the criterion cannot reach it. That is how nestedric stayed missing
    from two streams after the threshold was fixed: the runs were fine, the stored
    verdict was stale, and regenerating the table changed nothing.

    Recomputing from R makes the criterion a property of the analysis rather than a
    fossil of whatever the code believed on the day the run happened.
    """
    R = np.asarray(record.get("R", []), dtype="float64")
    if R.ndim != 2 or R.size == 0:
        return bool(record.get("sanity", {}).get("trustworthy", True))
    if not np.isfinite(R).all():
        return False
    return bool(backward_transfer(R) <= IMPLAUSIBLE_BWT and np.nanmin(R) >= -50.0)


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.rglob("results.json")):
        r = json.loads(path.read_text())
        fp = r.get("footprint", {})
        rows.append(
            {
                "stream": r.get("stream"),
                "method": r.get("method"),
                "seed": r.get("seed"),
                "avg_perf": r.get("average_performance"),
                "bwt": r.get("bwt"),
                "forgetting": r.get("forgetting"),
                "wall_s": r.get("wall_seconds"),
                "p50_ms": fp.get("latency_p50_ms"),
                "p99_ms": fp.get("latency_p99_ms"),
                "near_rt": r.get("near_rt_feasible"),
                "memory_mb": fp.get("extra_state_bytes", 0) / 1e6,
                "trustworthy": _trustworthy(r),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/runs/main"))
    ap.add_argument("--reference", default="replay")
    ap.add_argument(
        "--metric",
        default="bwt",
        choices=["bwt", "avg_perf", "forgetting", "all"],
        help="'all' writes every metric to one table, so a single run answers "
        "both 'does it retain' and 'does it learn' -- which have to be read "
        "together: low forgetting is trivial to obtain by not learning.",
    )
    ap.add_argument("--out", type=Path, default=Path("results/tables/main.csv"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    metrics = ["bwt", "avg_perf", "forgetting"] if args.metric == "all" else [args.metric]

    df = load(args.dir)
    if df.empty:
        print(f"!! no results under {args.dir}")
        return 1

    # A method missing from a stream is a fact about the pipeline, not about the
    # method, and it must be visible rather than inferable from a short column.
    counts = df.groupby("stream").method.nunique()
    if counts.nunique() > 1:
        full = int(counts.max())
        for stream, n in counts.items():
            if n < full:
                missing = set(df[df.stream == counts.idxmax()].method) - set(
                    df[df.stream == stream].method
                )
                print(f"!! {stream}: {n}/{full} methods present, missing {sorted(missing)}")

    untrustworthy = df[~df.trustworthy]
    if len(untrustworthy):
        print(f"!! {len(untrustworthy)} run(s) flagged untrustworthy; excluding them:")
        print(untrustworthy[["stream", "method", "seed"]].to_string(index=False))
        df = df[df.trustworthy]

    pd.set_option("display.width", 200)
    out_rows = []

    for metric, (stream, part) in (
        (m, sp) for m in metrics for sp in df.groupby("stream", observed=True)
    ):
        methods = sorted(part.method.unique())
        if args.reference not in methods:
            print(f"\n!! {stream}: reference {args.reference!r} absent, skipping comparison")
            continue

        # Fold = (stream, seed). Methods are aligned on seed so the pairing is real.
        seeds = sorted(set.intersection(*(set(part[part.method == m].seed) for m in methods)))
        if len(seeds) < 2:
            print(f"\n!! {stream}: only {len(seeds)} shared seed(s); CIs need more folds")

        scores = {
            m: np.array(
                [float(part[(part.method == m) & (part.seed == s)][metric].iloc[0]) for s in seeds]
            )
            for m in methods
        }

        print(
            f"\n{'=' * 100}\n{stream}  --  {metric} vs {args.reference}  "
            f"({len(seeds)} folds)\n{'=' * 100}"
        )

        comparisons = compare_methods(scores, reference=args.reference, alpha=args.alpha)
        table = []
        for method in methods:
            own = bootstrap_ci(scores[method])
            row = {
                "method": method,
                metric: own.format(),
                "memory_mb": round(part[part.method == method].memory_mb.mean(), 3),
                "p99_ms": round(part[part.method == method].p99_ms.mean(), 2),
                "near_rt": bool(part[part.method == method].near_rt.all()),
            }
            comp = comparisons.get(method)
            if comp:
                row["vs_ref"] = f"{comp['mean_difference']:+.4f}"
                row["ci"] = f"[{comp['ci_low']:+.4f}, {comp['ci_high']:+.4f}]"
                row["d_z"] = round(comp["effect_size"], 2)
                row["p"] = round(comp["p_value"], 4)
                row["holm"] = "*" if comp.get("significant_holm") else ""
            table.append(row)
            out_rows.append({"stream": stream, "metric": metric, **row})

        print(pd.DataFrame(table).to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(
        "\nCIs are paired bootstrap over folds; p-values are paired permutation, "
        f"Holm-corrected across methods within a stream at alpha={args.alpha}.\n"
        "'*' marks differences that survive correction. The fold, not the sample, is\n"
        "the unit of analysis throughout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
