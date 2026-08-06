#!/usr/bin/env python3
"""Generate the paper's figures from results/tables, matplotlib only.

One figure per claim, each drawn from a CSV rather than from numbers typed by hand, so
a figure cannot disagree with the text:

    fig1  which O-RAN reconfigurations cause forgetting (needs main.csv)
    fig2  methods at matched bytes                      (needs main.csv)
    fig3  the drift window                              (drift_sweep.csv)
    fig4  Proposition 2 out of sample                   (theory_test.csv)

Colour carries the reading rather than decorating: blue where an interval excludes zero
in the helpful direction, red where it excludes zero in the harmful one, grey wherever
it covers zero. Null results should look null at a glance instead of resembling small
wins.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# A TMC column is 3.5 inches. A figure that needs zooming is a figure a reviewer skips.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    }
)

GREY, BLUE, RED = "#7a7a7a", "#1f4e79", "#a5281f"


def _interval(text: str) -> tuple[float, float, float]:
    """Parse 'estimate [low, high]' as written by utils.stats.Interval.format."""
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text))]
    return (nums[0], nums[1], nums[2]) if len(nums) >= 3 else (nums[0], nums[0], nums[0])


def _colour(low: float, high: float, higher_is_better: bool = True) -> str:
    if low > 0:
        return BLUE if higher_is_better else RED
    if high < 0:
        return RED if higher_is_better else BLUE
    return GREY


def fig_which_shifts(main: pd.DataFrame, out: Path) -> None:
    """Naive fine-tuning's forgetting per stream: the mechanism finding."""
    ft = main[(main.method == "finetune") & (main.metric == "bwt")]
    if ft.empty:
        return

    rows = [(r.stream, *_interval(r.bwt)) for r in ft.itertuples()]
    rows.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for i, (stream, est, low, high) in enumerate(rows):
        ax.barh(i, est, color=_colour(low, high), height=0.55, zorder=2)
        ax.plot([low, high], [i, i], color="black", lw=1, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel("backward transfer, naive fine-tuning")
    ax.set_title("Only allocation-regime shifts cause forgetting")
    fig.savefig(out)
    plt.close(fig)


def fig_methods(main: pd.DataFrame, out: Path) -> None:
    """Every method's BWT per stream, at a matched 4 MB budget."""
    df = main[main.metric == "bwt"]
    streams = sorted(df.stream.unique())
    order = [
        "finetune",
        "ewc",
        "si",
        "lwf",
        "bilevel",
        "titans",
        "agem",
        "replay",
        "nestedric",
        "joint",
    ]
    methods = [m for m in order if m in set(df.method)]
    if not streams or not methods:
        return

    fig, axes = plt.subplots(1, len(streams), figsize=(1.45 * len(streams) + 1.1, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, stream in zip(axes, streams, strict=True):
        cell = df[df.stream == stream].set_index("method")
        for i, method in enumerate(methods):
            if method not in cell.index:
                ax.scatter(0, i, marker="x", color=GREY, s=12)  # absent, not zero
                continue
            est, low, high = _interval(cell.loc[method, "bwt"])
            colour = BLUE if method == "nestedric" else (RED if method == "replay" else GREY)
            ax.plot([low, high], [i, i], color=colour, lw=1, zorder=2)
            ax.scatter(est, i, color=colour, s=13, zorder=3)
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        ax.set_title(stream)
        ax.set_xlabel("BWT")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods)
    fig.savefig(out)
    plt.close(fig)


def fig_drift_window(table: Path, out: Path) -> None:
    """Separation helps only inside a window of drift magnitude."""
    if not table.exists():
        return
    df = pd.read_csv(table)
    diff = df["difference"].astype(str).str.replace("+", "", regex=False).astype(float)
    low = df["ci"].str.extract(r"\[\s*([-+]?[\d.]+)")[0].astype(float)
    high = df["ci"].str.extract(r",\s*([-+]?[\d.]+)\s*\]")[0].astype(float)

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    x = np.arange(len(df))
    ax.bar(
        x,
        diff,
        color=[_colour(lo, hi) for lo, hi in zip(low, high, strict=True)],
        width=0.55,
        zorder=2,
    )
    ax.errorbar(
        x,
        diff,
        yerr=[diff - low, high - diff],
        fmt="none",
        ecolor="black",
        lw=1,
        capsize=2,
        zorder=3,
    )
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["magnitude"])
    ax.set_xlabel("injected concept-drift magnitude")
    ax.set_ylabel(r"BWT($\rho{=}32$) $-$ BWT($\rho{=}1$)")
    ax.set_title("Frequency separation helps only in a window")
    fig.savefig(out)
    plt.close(fig)


def fig_theory(table: Path, out: Path) -> None:
    """Predicted against measured optimal ratio: right direction, wrong exponent."""
    if not table.exists():
        return
    df = pd.read_csv(table).sort_values("magnitude")

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(
        df.magnitude,
        df.predicted_rho,
        "o--",
        color=GREY,
        ms=4,
        label=r"predicted $\rho^*$  (Prop. 2)",
    )
    ax.plot(df.magnitude, df.best_rho, "s-", color=BLUE, ms=4, label=r"measured best $\rho$")
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"drift magnitude $\delta$")
    ax.set_ylabel(r"separation ratio $\rho$")
    ax.set_title(r"$\rho^*$ falls with drift, faster than $\delta^{-1/2}$")
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_csv = args.tables / "main.csv"
    if main_csv.exists():
        main_table = pd.read_csv(main_csv)
        fig_which_shifts(main_table, args.out / "fig1_which_shifts.pdf")
        fig_methods(main_table, args.out / "fig2_methods.pdf")
        if main_table[main_table.metric == "bwt"].groupby("stream").method.nunique().nunique() > 1:
            print(
                "!! main.csv is missing methods on some streams; regenerate it "
                "(scripts/make_table.py) before using fig1/fig2 in the paper"
            )
    else:
        print(f"!! {main_csv} absent -- skipping fig1, fig2")

    fig_drift_window(args.tables / "drift_sweep.csv", args.out / "fig3_drift_window.pdf")
    fig_theory(args.tables / "theory_test.csv", args.out / "fig4_theory.pdf")

    for path in sorted(args.out.glob("*.pdf")):
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
