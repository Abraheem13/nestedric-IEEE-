#!/usr/bin/env python3
"""Read the drift sweep: does frequency separation pay once drift is large enough?

Day 10 measured no effect of rho on real traces. This sweep adds concept drift under
our control and asks whether a threshold exists. It reports in a fixed order, and the
order matters:

1. **The manipulation check.** finetune's |BWT| must grow with drift magnitude. If it
   does not, the injection is not reaching the model and nothing below means anything --
   so this script refuses to interpret rho until that check passes.
2. **The rho comparison at each magnitude**, paired across seeds.
3. **The threshold**, if any: the smallest magnitude at which rho = 32 beats rho = 1 by
   more than the seed noise.

    python scripts/report_drift.py --dir results/runs/drift_sweep
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

from nestedric.utils.stats import paired_bootstrap_ci, paired_permutation_p  # noqa: E402


def load(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.rglob("results.json")):
        r = json.loads(path.read_text())
        label = path.parent.parent.name  # e.g. m1.0_rho32 or m1.0
        magnitude = float(re.search(r"m([\d.]+)", label).group(1))
        rho_match = re.search(r"rho(\d+)", label)
        rows.append(
            {
                "method": r.get("method"),
                "rho": int(rho_match.group(1)) if rho_match else None,
                "magnitude": magnitude,
                "seed": r.get("seed"),
                "bwt": r.get("bwt"),
                "avg_perf": r.get("average_performance"),
                "trustworthy": r.get("sanity", {}).get("trustworthy", True),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/runs/drift_sweep"))
    ap.add_argument("--out", type=Path, default=Path("results/tables/drift_sweep.csv"))
    args = ap.parse_args()

    df = load(args.dir)
    if df.empty:
        print(f"!! no results under {args.dir}")
        return 1
    df = df[df.trustworthy]

    pd.set_option("display.width", 200)

    print(
        f"{'=' * 88}\n1. MANIPULATION CHECK: "
        f"does injected drift actually cause forgetting?\n{'=' * 88}"
    )
    ft = (
        df[df.method == "finetune"]
        .groupby("magnitude", observed=True)
        .bwt.agg(["mean", "std", "count"])
    )
    print(ft.round(4).to_string())

    magnitudes = sorted(ft.index)
    worsens = ft["mean"].iloc[-1] < ft["mean"].iloc[0] - 0.005
    if not worsens:
        print(
            "\n  !! finetune's BWT does not worsen with magnitude. The injection is not\n"
            "     reaching the model, or the drift is too small to matter at these levels.\n"
            "     Nothing below can be interpreted until this is fixed."
        )
        return 1
    print(
        f"\n  OK: finetune BWT {ft['mean'].iloc[0]:+.4f} at magnitude {magnitudes[0]} "
        f"-> {ft['mean'].iloc[-1]:+.4f} at {magnitudes[-1]}. The manipulation works."
    )

    print(
        f"\n{'=' * 88}\n2. DOES SEPARATION PAY? "
        f"rho = 32 against rho = 1 at each magnitude\n{'=' * 88}"
    )
    rows = []
    for magnitude in magnitudes:
        cell = df[(df.method == "nestedric") & (df.magnitude == magnitude)]
        fast = cell[cell.rho == 1].sort_values("seed").bwt.to_numpy()
        slow = cell[cell.rho == 32].sort_values("seed").bwt.to_numpy()
        if len(fast) != len(slow) or not len(fast):
            continue
        ci = paired_bootstrap_ci(slow, fast)
        baseline = df[(df.method == "replay") & (df.magnitude == magnitude)].bwt.mean()
        rows.append(
            {
                "magnitude": magnitude,
                "rho=1": round(float(fast.mean()), 4),
                "rho=32": round(float(slow.mean()), 4),
                "difference": f"{ci.estimate:+.4f}",
                "ci": f"[{ci.low:+.4f}, {ci.high:+.4f}]",
                "p": round(paired_permutation_p(slow, fast), 4),
                "separation_helps": ci.low > 0,  # higher BWT is better (less negative)
                "replay": round(float(baseline), 4) if np.isfinite(baseline) else None,
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # A paired permutation test on n folds has 2**n sign patterns, so the smallest
    # attainable p is 2/(2**n + 1). With three seeds that is 0.22, and every p in the
    # table above sits at that floor -- which is a statement about the test's resolution,
    # not about the effects. Say so, loudly, next to the numbers it disqualifies.
    n_folds = int(df[df.method == "nestedric"].groupby(["magnitude", "rho"]).size().min())
    p_floor = 2 / (2**n_folds + 1)
    print(f"\n  n_folds = {n_folds}; smallest attainable permutation p = {p_floor:.4f}")

    # Four magnitudes is four comparisons. Holm requires the smallest p to clear
    # alpha/m, and if the permutation floor is above that threshold no result can
    # survive correction whatever the effects are -- a fact about the design, not the
    # data, and one a reviewer will notice before we do.
    from nestedric.utils.stats import holm_bonferroni

    raw = {str(r["magnitude"]): float(r["p"]) for r in rows}
    decisions = holm_bonferroni(raw, alpha=0.05)
    m = len(raw)
    print(f"  Holm across {m} magnitudes: strictest threshold = {0.05 / m:.4f}")
    for magnitude, survived in decisions.items():
        print(
            f"    magnitude {magnitude:<6} p = {raw[magnitude]:.4f}  "
            f"{'survives' if survived else 'does NOT survive'} correction"
        )
    if p_floor > 0.05 / m:
        needed = 1
        while 2 / (2**needed + 1) > 0.05 / m:
            needed += 1
        print(
            f"\n  !! The floor ({p_floor:.4f}) exceeds the Holm threshold ({0.05 / m:.4f}).\n"
            f"     No result can survive correction at {n_folds} folds regardless of effect\n"
            f"     size. {needed} seeds would be required."
        )
    if p_floor > 0.05:
        print(
            "  !! No p-value below 0.05 is REACHABLE at this seed count. The CIs above\n"
            "     are suggestive but nothing here is statistically established. Five seeds\n"
            "     reach 0.061, seven reach 0.015. Re-run before any claim of significance."
        )

    print(f"\n{'=' * 88}\n3. THRESHOLD\n{'=' * 88}")
    helping = table[table.separation_helps]
    if len(helping):
        first = helping.magnitude.min()
        print(
            f"  Separation first pays at drift magnitude {first}.\n"
            "  Real O-RAN traces correspond to magnitude 0 in this sweep, so the paper\n"
            f"  reports delta* = {first} and states how far the public corpora sit below it."
        )
        harmful = table[(table.magnitude > first) & table.difference.str.startswith("-")]
        if len(harmful):
            print(
                f"\n  It REVERSES at magnitude {harmful.magnitude.min()}: separation then\n"
                "  costs more than it buys. The benefit is a window, not a direction, and\n"
                "  reporting only its lower edge would misdescribe the result."
            )
        print(
            "\n  Note: NestedRIC is compared here against itself at a different ratio.\n"
            "  Check the replay column before drawing any conclusion about competitiveness."
        )
    else:
        print(
            "  No magnitude tested shows a benefit from separation whose CI excludes zero.\n"
            "  Frequency separation does not reduce forgetting even under drift we control,\n"
            "  up to the largest magnitude swept. That is a stronger negative result than\n"
            "  Day 10 alone: it rules out 'the traces do not drift enough' as the\n"
            "  explanation, at least for this family of synthetic concept shifts."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(
        "\nNote for the paper: the injected shift is a random linear perturbation of the\n"
        "input-to-target mapping. A threshold found here characterises that family, not\n"
        "every real shift, and every table using it must be labelled synthetic."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
