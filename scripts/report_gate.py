#!/usr/bin/env python3
"""Summarise the Day 4 gate: is there forgetting on these traces?

Reads every results.json under a run directory and prints one row per
(stream, method, seed), then the question the gate actually asks -- finetune's |BWT|
per stream, averaged over seeds, against the threshold below which the paper needs
reframing rather than more experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: |BWT| below this is indistinguishable from run-to-run noise at these seed counts,
#: and would mean the public traces do not exhibit the phenomenon under study.
FORGETTING_THRESHOLD = 0.01


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
                "steps": r.get("steps"),
                "wall_s": r.get("wall_seconds"),
                "p50_ms": fp.get("latency_p50_ms"),
                "p99_ms": fp.get("latency_p99_ms"),
                "near_rt": r.get("near_rt_feasible"),
                "extra_mb": fp.get("extra_state_bytes", 0) / 1e6,
                "path": str(path.parent),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/runs/gate"))
    args = ap.parse_args()

    df = load(args.dir)
    if df.empty:
        print(f"!! no results under {args.dir}")
        return 1

    pd.set_option("display.width", 160)
    print("\n=== every run ===")
    print(df.drop(columns=["path"]).round(4).to_string(index=False))

    print("\n=== mean over seeds ===")
    agg = (
        df.groupby(["stream", "method"], observed=True)[["avg_perf", "bwt", "forgetting"]]
        .agg(["mean", "std"])
        .round(4)
    )
    print(agg.to_string())

    print("\n=== THE GATE: does finetune forget? ===")
    ft = df[df.method == "finetune"]
    if ft.empty:
        print("  no finetune runs found")
        return 1

    verdicts = []
    for stream, part in ft.groupby("stream", observed=True):
        mean_bwt = float(part.bwt.mean())
        verdict = "FORGETS" if abs(mean_bwt) >= FORGETTING_THRESHOLD else "negligible"
        verdicts.append(verdict)
        print(
            f"  {stream:<16} BWT={mean_bwt:+.4f} "
            f"(seeds: {', '.join(f'{v:+.4f}' for v in part.bwt)})  -> {verdict}"
        )

    if all(v == "negligible" for v in verdicts):
        print(
            "\n  GATE FAILS: no stream shows forgetting above "
            f"{FORGETTING_THRESHOLD}. Per docs/PLAN.md, pull drift injection forward\n"
            "  and reframe: 'how much drift is required before multi-timescale nesting pays'."
        )
    else:
        print("\n  GATE PASSES on at least one stream: there is a phenomenon to improve on.")

    print("\n=== oracle sanity: joint should beat finetune ===")
    piv = df.pivot_table(index="stream", columns="method", values="avg_perf", aggfunc="mean")
    if {"joint", "finetune"} <= set(piv.columns):
        piv["joint_minus_finetune"] = piv["joint"] - piv["finetune"]
        print(piv.round(4).to_string())
        if (piv.joint_minus_finetune <= 0).any():
            print("\n  !! joint does not beat finetune somewhere -- suspect the loop, not the data")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
