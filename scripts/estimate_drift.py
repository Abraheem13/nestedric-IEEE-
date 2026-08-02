#!/usr/bin/env python3
"""Measure what kind of drift each stream actually contains.

The Day 4 gate found forgetting on sched-shift (BWT -0.0443) and slice-shift (-0.0202)
but effectively none on radio-shift (-0.0006) -- the reverse of the Day 1 prediction,
which promoted COMMAG's radio-condition axes precisely because they were expected to
forget hardest.

This script tests the explanation: that covariate drift (P(X) moves) does not cause
forgetting, while concept drift (P(Y|X) moves) does. It prints per-transition drift for
each stream and, at the end, the correlation between each drift measure and the
finetune |BWT| measured by the gate.

    python scripts/estimate_drift.py --gate-dir results/runs/gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data.drift import estimate_drift_rate  # noqa: E402
from nestedric.data.loaders import fit_normaliser  # noqa: E402
from nestedric.data.stream import build_stream  # noqa: E402
from nestedric.utils.config import load_config  # noqa: E402

STREAMS = ["radio_shift", "sched_shift", "slice_shift"]


def gate_bwt(gate_dir: Path) -> dict[str, float]:
    """finetune |BWT| per stream from the gate runs, for the correlation at the end."""
    out: dict[str, list[float]] = {}
    for path in gate_dir.rglob("results.json"):
        r = json.loads(path.read_text())
        if r.get("method") == "finetune":
            out.setdefault(r["stream"], []).append(abs(r["bwt"]))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--gate-dir", type=Path, default=Path("results/runs/gate"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/drift.csv"))
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()

    rows = []
    for name in STREAMS:
        cfg = load_config(f"configs/stream/{name}.yaml")
        stream = build_stream(cfg, args.processed)
        norm = fit_normaliser([stream[0]], args.processed)
        records = estimate_drift_rate(stream, args.processed, norm, sample=args.sample)
        rows.extend(records)

        print(f"\n{'=' * 78}\n{stream.name}\n{'=' * 78}")
        df = pd.DataFrame(records)
        print(
            df[["transition", "covariate_drift", "concept_drift", "label_drift", "transfer_gap"]]
            .round(4)
            .to_string(index=False)
        )
        print(
            f"  mean: covariate={df.covariate_drift.mean():.4f}  "
            f"concept={df.concept_drift.mean():.4f}  "
            f"label={df.label_drift.mean():.4f}  "
            f"transfer_gap={df.transfer_gap.mean():.4f}"
        )

    all_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out, index=False)

    print(f"\n{'=' * 78}\nDRIFT TYPE vs MEASURED FORGETTING\n{'=' * 78}")
    bwt = gate_bwt(args.gate_dir)
    if not bwt:
        print(f"  no gate results under {args.gate_dir}; skipping the comparison")
        return 0

    summary = all_df.groupby("stream", observed=True)[
        ["covariate_drift", "concept_drift", "label_drift", "transfer_gap"]
    ].mean()
    summary["finetune_abs_bwt"] = pd.Series(bwt)
    summary = summary.dropna()
    print(summary.round(4).to_string())

    if len(summary) >= 3:
        print("\ncorrelation with |BWT| across streams (n=3, indicative only):")
        for col in ("covariate_drift", "concept_drift", "label_drift", "transfer_gap"):
            r = summary[col].corr(summary.finetune_abs_bwt)
            print(f"  {col:18s} r = {r:+.3f}")
        print(
            "\nIf concept drift tracks |BWT| and covariate drift does not, then the delta\n"
            "in Theorem 1 must be defined as concept drift. Stating it as generic\n"
            "'distribution shift' would leave the theorem contradicted by radio-shift,\n"
            "a counterexample sitting inside our own benchmark."
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
