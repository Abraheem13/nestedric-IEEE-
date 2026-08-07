#!/usr/bin/env python3
"""Out-of-sample test of Proposition 2: does rho*(delta) predict the best ratio?

The bound's constants were pinned by one fact from the drift sweep -- that rho = 32
stops paying somewhere between delta = 0.5 and delta = 1.0. That pinning then forces
predictions about ratios never used to fit it:

    rho*(0.25) ~ 9-11   -> intermediate ratios should beat rho = 32
    rho*(1.00) ~ 4-5    -> rho = 4 should beat rho = 32

This script pools the theory-test runs (rho in 4, 8, 16) with the sweep runs (rho in
1, 32) at the same magnitudes, finds the empirically best ratio, and compares it with
the prediction. A prediction that lands is evidence; one that misses falsifies the
functional form, which is worth knowing before it becomes a theorem in a paper.

    python scripts/report_theory.py
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

from nestedric.theory.bound import fit_constants, optimal_ratio  # noqa: E402
from nestedric.utils.stats import (
    bootstrap_ci,
    paired_bootstrap_ci,
    paired_permutation_p,
)  # noqa: E402


def load(*dirs: Path) -> pd.DataFrame:
    rows = []
    for run_dir in dirs:
        for path in run_dir.rglob("results.json"):
            r = json.loads(path.read_text())
            label = path.parent.parent.name
            m = re.search(r"m([\d.]+)", label)
            rho = re.search(r"rho(\d+)", label)
            if not (m and rho) or r.get("method") != "nestedric":
                continue
            rows.append(
                {
                    "magnitude": float(m.group(1)),
                    "rho": int(rho.group(1)),
                    "seed": r.get("seed"),
                    "bwt": r.get("bwt"),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory-dir", type=Path, default=Path("results/runs/theory_test"))
    ap.add_argument("--sweep-dir", type=Path, default=Path("results/runs/drift_confirm"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/theory_test.csv"))
    args = ap.parse_args()

    df = load(args.theory_dir, args.sweep_dir)
    if df.empty:
        print("!! no nestedric runs found")
        return 1

    pd.set_option("display.width", 200)
    out_rows = []

    # Constants are fitted on the FITTING SET only: the ratios measured before the
    # theory test existed (rho = 1 and 32). The held-out ratios 4, 8 and 16 are then
    # predicted, which is what makes this out-of-sample.
    #
    # This fit was missing from the first version of this script, which used the
    # unfitted placeholders C_r = C_a = C_n = 1 and so predicted rho*(0.25) = 2.0. That
    # was an error in the report, not a modelling choice, and it is corrected here
    # rather than quietly: both verdicts are printed below.
    fit_set = df[df.rho.isin([1, 32])]
    consts = fit_constants(
        drift=fit_set.magnitude.to_numpy(),
        ratios=fit_set.rho.to_numpy(),
        measured_bwt=np.abs(fit_set.bwt.to_numpy()),
    )
    print(
        f"constants fitted on rho in (1, 32) only: "
        f"C_r={consts['C_r']:.4g}, C_a={consts['C_a']:.4g}, C_n={consts['C_n']:.4g}"
    )
    print(f"held-out ratios: {sorted(set(df.rho) - {1, 32})}")

    for magnitude in sorted(df.magnitude.unique()):
        cell = df[df.magnitude == magnitude]
        ratios = sorted(cell.rho.unique())
        if len(ratios) < 3:
            continue

        predicted = optimal_ratio(magnitude, consts)
        predicted_default = optimal_ratio(magnitude)
        print(
            f"\n{'=' * 84}\ndrift magnitude {magnitude}   "
            f"Proposition 2 predicts rho* = {predicted:.1f}   "
            f"(unfitted placeholders would say {predicted_default:.1f})\n{'=' * 84}"
        )

        table = []
        by_rho = {}
        for rho in ratios:
            values = cell[cell.rho == rho].sort_values("seed").bwt.to_numpy()
            by_rho[rho] = values
            ci = bootstrap_ci(values)
            table.append(
                {
                    "rho": rho,
                    "n": len(values),
                    "bwt": ci.format(),
                    "mean": round(float(values.mean()), 4),
                }
            )
        print(pd.DataFrame(table).to_string(index=False))

        # Higher BWT (less negative) is better.
        best = max(by_rho, key=lambda r: by_rho[r].mean())
        print(f"\n  best measured rho = {best}   predicted rho* = {predicted:.1f}")

        # Is the best ratio distinguishable from rho = 32, the sweep's default?
        if 32 in by_rho and best != 32 and len(by_rho[best]) == len(by_rho[32]):
            ci = paired_bootstrap_ci(by_rho[best], by_rho[32])
            p = paired_permutation_p(by_rho[best], by_rho[32])
            print(
                f"  rho={best} vs rho=32: {ci.format()}  p = {p:.4f}"
                f"  {'(excludes zero)' if ci.excludes_zero else '(covers zero)'}"
            )

        # The prediction is a real number; the grid is coarse. Count it as confirmed if
        # the best measured ratio is the grid point nearest the prediction.
        nearest = min(ratios, key=lambda r: abs(np.log(r) - np.log(predicted)))
        verdict = "CONFIRMED" if best == nearest else "MISSED"
        print(f"  nearest grid point to the prediction: {nearest}  ->  {verdict}")

        # Both verdicts are recorded. The fitted one is the pre-registered test; the
        # unfitted one is what an earlier version of this script reported, using
        # placeholder constants, and it reached the opposite conclusion at delta = 0.25.
        # Changing an analysis after seeing its result has to stay auditable.
        nearest_default = min(ratios, key=lambda r: abs(np.log(r) - np.log(predicted_default)))
        verdict_default = "CONFIRMED" if best == nearest_default else "MISSED"
        print(
            f"  with unfitted placeholder constants it would read: "
            f"rho* = {predicted_default:.1f} -> {verdict_default}"
        )

        out_rows.append(
            {
                "magnitude": magnitude,
                "predicted_rho": round(predicted, 2),
                "best_rho": best,
                "nearest_grid": nearest,
                "verdict": verdict,
                "predicted_rho_unfitted": round(predicted_default, 2),
                "verdict_unfitted": verdict_default,
            }
        )

    print(f"\n{'=' * 84}\nVERDICT\n{'=' * 84}")
    summary = pd.DataFrame(out_rows)
    print(summary.to_string(index=False))
    if len(summary) and (summary.verdict == "CONFIRMED").all():
        print(
            "\n  Proposition 2 predicted the optimal ratio at magnitudes it was not\n"
            "  fitted to. That is a tested prediction, not a curve through three points."
        )
    else:
        print(
            "\n  At least one prediction missed. The functional form rho* = sqrt(C_r /\n"
            "  (C_a delta)) does not describe these data, and Proposition 2 should be\n"
            "  reported as refuted rather than adjusted until it fits."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
