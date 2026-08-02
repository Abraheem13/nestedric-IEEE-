#!/usr/bin/env python3
"""Why does a stream diverge? Inspect the standardised inputs and targets per environment.

Normalisation constants are fitted on the source environment only, which is the correct
protocol -- but it has a failure mode. A feature with a small spread in the source
environment and a large one later is divided by a small number and arrives at the model
as a huge value. The std floor in fit_normaliser only guards the exactly-degenerate case
(std < 1e-6); a source std of 1e-3 against later values a hundred times larger passes
the floor and still produces inputs in the tens of thousands.

sched-shift-commag produced joint avg_perf = -37.97 (std 52.7 across two seeds) and
finetune BWT = +5.90. Those are divergence, not learning. This script says which feature
is responsible.

    python scripts/diagnose_scaling.py --stream configs/stream/sched_shift_commag.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nestedric.data.loaders import build_windows, fit_normaliser  # noqa: E402
from nestedric.data.schema import FEATURE_COLUMNS, TARGET_COLUMNS  # noqa: E402
from nestedric.data.stream import build_stream  # noqa: E402
from nestedric.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    stream = build_stream(load_config(args.stream), args.processed)
    norm = fit_normaliser([stream[0]], args.processed)

    print(f"\nnormaliser fitted on {norm.source_env_ids} ({norm.n_rows_fitted:,} rows)")
    small = pd.DataFrame({"feature": norm.columns, "mean": norm.mean, "std": norm.std})
    print("\n--- fitted constants, smallest std first (the risky end) ---")
    print(small.nsmallest(6, "std").round(6).to_string(index=False))

    print("\n--- standardised inputs per environment ---")
    rows = []
    for env in stream:
        ws = build_windows(env, args.processed, norm, env.train_traces[:20])
        if not len(ws):
            continue
        x = ws.x[: args.sample, :, :-1]
        worst = int(np.argmax(np.abs(x).max(axis=(0, 1))))
        rows.append(
            {
                "env": env.env_id,
                "x_absmax": float(np.abs(x).max()),
                "x_p99": float(np.percentile(np.abs(x), 99)),
                "worst_feature": FEATURE_COLUMNS[worst],
                "y_absmax": float(np.abs(ws.y).max()),
                "y_p99": float(np.percentile(np.abs(ws.y), 99)),
                "n": len(ws),
            }
        )
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    print("\n--- per-feature worst standardised value across environments ---")
    peaks = np.zeros(len(FEATURE_COLUMNS))
    for env in stream:
        ws = build_windows(env, args.processed, norm, env.train_traces[:20])
        if len(ws):
            peaks = np.maximum(peaks, np.abs(ws.x[:, :, :-1]).max(axis=(0, 1)))
    worst = pd.DataFrame({"feature": FEATURE_COLUMNS, "absmax": peaks}).nlargest(8, "absmax")
    print(worst.round(2).to_string(index=False))

    print(f"\ntargets are {TARGET_COLUMNS}; standardised with the same constants.")
    if df.x_absmax.max() > 50 or df.y_absmax.max() > 50:
        print(
            "\n!! Inputs or targets exceed 50 source-sigma. The model is being asked to\n"
            "   fit values it cannot represent stably, and the loss will explode. Fix the\n"
            "   scaling -- do not paper over it with a smaller learning rate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
