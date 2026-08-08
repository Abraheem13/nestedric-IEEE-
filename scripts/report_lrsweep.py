#!/usr/bin/env python3
"""Pick each method's learning rate from the validation stream.

Every method inherits lr = 1e-3 from finetune.yaml. NestedRIC turned out to be badly
under-trained at that value, so the same question applies to the baselines, and tuning
only the proposed method would bias the comparison toward it.

Selection rule, fixed before reading: **best average performance**, not best BWT. A
learning rate so small that the model barely moves produces excellent BWT and a useless
model; selecting on retention would systematically choose under-training for every
method. Divergent cells are reported, not silently dropped -- where a method's stability
ends is part of the result.

    python scripts/report_lrsweep.py --dir results/runs/tune_baselines
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.utils.stats import bootstrap_ci  # noqa: E402


def load(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, failures = [], []
    for path in run_dir.rglob("results.json"):
        r = json.loads(path.read_text())
        lr = float(re.search(r"lr([\d.eE+-]+)", path.parent.parent.name).group(1))
        rows.append(
            {
                "method": r.get("method"),
                "lr": lr,
                "seed": r.get("seed"),
                "avg_perf": r.get("average_performance"),
                "bwt": r.get("bwt"),
            }
        )
    for path in run_dir.rglob("diverged.json"):
        d = json.loads(path.read_text())
        failures.append({"method": d.get("method"), "lr": d.get("lr"), "seed": d.get("seed")})
    return pd.DataFrame(rows), pd.DataFrame(failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/runs/tune_baselines"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/lrsweep.csv"))
    args = ap.parse_args()

    df, failures = load(args.dir)
    if df.empty:
        print(f"!! no results under {args.dir}")
        return 1

    pd.set_option("display.width", 200)

    if len(failures):
        print(f"{'=' * 78}\nDIVERGED CELLS ({len(failures)})\n{'=' * 78}")
        print(failures.groupby(["method", "lr"], observed=True).size().rename("seeds").to_string())
        print("\n  A learning rate that diverges is excluded from selection for that")
        print("  method, and the boundary is reported: it is a property worth stating.")

    print(f"\n{'=' * 78}\nVALIDATION SWEEP\n{'=' * 78}")
    chosen = []
    for method, part in df.groupby("method", observed=True):
        table = []
        for lr, cell in part.groupby("lr", observed=True):
            ap_ci = bootstrap_ci(cell.avg_perf.to_numpy())
            bwt_ci = bootstrap_ci(cell.bwt.to_numpy())
            table.append(
                {
                    "lr": lr,
                    "n": len(cell),
                    "avg_perf": ap_ci.format(),
                    "avg_perf_mean": ap_ci.estimate,
                    "bwt": bwt_ci.format(),
                }
            )
        t = pd.DataFrame(table).sort_values("lr")
        best = t.loc[t.avg_perf_mean.idxmax()]
        print(f"\n--- {method}  (selected lr = {best.lr:g}) ---")
        print(t.drop(columns=["avg_perf_mean"]).to_string(index=False))
        chosen.append(
            {
                "method": method,
                "selected_lr": best.lr,
                "avg_perf": best.avg_perf_mean,
                "n_lrs_tried": len(t),
            }
        )

    summary = pd.DataFrame(chosen).sort_values("method")
    print(f"\n{'=' * 78}\nSELECTED LEARNING RATES\n{'=' * 78}")
    print(summary.to_string(index=False))
    print("\nApply these to configs/method/*.yaml before the reported benchmark.")
    print("Selection is on average performance: choosing on BWT would pick the")
    print("learning rate at which each method learns least.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
