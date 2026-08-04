#!/usr/bin/env python3
"""Summarise the Day 10 ablations, one table per axis.

The ratio sweep is the theorem's empirical content: Theorem 1 says |BWT| decreases in
the separation ratio rho with f(1) = 1, and Proposition 2 predicts an interior optimum
rather than "larger is always better". Both are checkable here, and the sweep is printed
with its rho column so the shape is visible rather than inferred.

Every cell is compared against the default configuration (periods [1, 32], two levels,
self-modification on, deep optimizer on, memory-only level assignment) using the
fold-level machinery in utils.stats -- paired over seeds, because each cell saw the same
environments and splits.

    python scripts/report_ablation.py --dir results/runs/ablation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.utils.stats import (
    bootstrap_ci,
    paired_bootstrap_ci,
    paired_permutation_p,
)  # noqa: E402

#: The cell every other cell is compared against.
DEFAULT_CELLS = {
    "periods": "[1,32]",
    "n_levels": "2",
    "self_modifying": "True",
    "deep_optimizer_enabled": "True",
    "level_assignment": "memory",
}


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.rglob("results.json")):
        r = json.loads(path.read_text())
        # .../<stream>/<axis>=<value>/seed<N>/results.json
        cell = path.parent.parent.name
        axis, _, value = cell.partition("=")
        fp = r.get("footprint", {})
        rows.append(
            {
                "stream": r.get("stream"),
                "axis": axis,
                "value": value,
                "seed": r.get("seed"),
                "avg_perf": r.get("average_performance"),
                "bwt": r.get("bwt"),
                "forgetting": r.get("forgetting"),
                "memory_mb": fp.get("extra_state_bytes", 0) / 1e6,
                "p99_ms": fp.get("latency_p99_ms"),
                "wall_s": r.get("wall_seconds"),
                "trustworthy": r.get("sanity", {}).get("trustworthy", True),
            }
        )
    return pd.DataFrame(rows)


def _rho(value: str) -> float:
    """Separation ratio from a periods value like '[1,32]'."""
    nums = [float(n) for n in re.findall(r"\d+", value)]
    return nums[1] / nums[0] if len(nums) >= 2 and nums[0] else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/runs/ablation"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/ablation.csv"))
    args = ap.parse_args()

    df = load(args.dir)
    if df.empty:
        print(f"!! no results under {args.dir}")
        return 1

    bad = df[~df.trustworthy]
    if len(bad):
        print(f"!! {len(bad)} untrustworthy run(s) excluded:")
        print(bad[["axis", "value", "seed"]].to_string(index=False))
        df = df[df.trustworthy]

    pd.set_option("display.width", 200)
    out_rows = []

    for axis, part in df.groupby("axis", observed=True):
        print(f"\n{'=' * 92}\n{axis}\n{'=' * 92}")
        default_value = DEFAULT_CELLS.get(axis)
        ref = None
        if default_value is not None:
            ref_rows = part[part.value == default_value].sort_values("seed")
            ref = ref_rows.bwt.to_numpy() if len(ref_rows) else None

        table = []
        values = sorted(part.value.unique(), key=lambda v: (_rho(v) if axis == "periods" else v))
        for value in values:
            cell = part[part.value == value].sort_values("seed")
            bwt = cell.bwt.to_numpy()
            ci = bootstrap_ci(bwt)
            row = {
                "value": value,
                "n": len(bwt),
                "bwt": ci.format(),
                "avg_perf": round(float(cell.avg_perf.mean()), 4),
                "memory_mb": round(float(cell.memory_mb.mean()), 2),
                "p99_ms": round(float(cell.p99_ms.mean()), 2),
                "wall_s": int(cell.wall_s.mean()),
            }
            if axis == "periods":
                row = {"rho": _rho(value), **row}
            if ref is not None and len(bwt) == len(ref) and value != default_value:
                diff = paired_bootstrap_ci(bwt, ref)
                row["vs_default"] = f"{diff.estimate:+.4f}"
                row["p"] = round(paired_permutation_p(bwt, ref), 4)
            table.append(row)
            out_rows.append({"axis": axis, **row})

        print(pd.DataFrame(table).to_string(index=False))

        if axis == "periods":
            sweep = pd.DataFrame(
                [
                    {"rho": _rho(v), "abs_bwt": abs(float(part[part.value == v].bwt.mean()))}
                    for v in values
                ]
            ).sort_values("rho")
            best = sweep.loc[sweep.abs_bwt.idxmin()]
            monotone = bool((sweep.abs_bwt.diff().dropna() <= 1e-6).all())
            print(
                f"\n  |BWT| is minimised at rho = {best.rho:g} ({best.abs_bwt:.4f}).\n"
                f"  monotonically decreasing in rho: {monotone}\n"
                "  Theorem 1 predicts a decrease in rho with f(1) = 1; Proposition 2\n"
                "  predicts an interior optimum. An interior minimum supports Prop 2;\n"
                "  a monotone decrease supports Thm 1 but leaves Prop 2 unevidenced;\n"
                "  a flat or rising curve contradicts both."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
