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
#: The strongest baseline gets its own colour. Red would read as "bad", which is the
#: opposite of what replay is in this benchmark.
ORANGE = "#c8730b"


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
    for i, (_stream, est, low, high) in enumerate(rows):
        ax.barh(i, est, color=_colour(low, high), height=0.55, zorder=2)
        ax.plot([low, high], [i, i], color="black", lw=1, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel("backward transfer, naive fine-tuning")
    ax.set_title("Which shifts cause forgetting", pad=6)
    fig.savefig(out)
    plt.close(fig)


def fig_methods(main: pd.DataFrame, out: Path) -> None:
    """Every method's BWT per stream, at a matched 4 MB budget.

    Panels share one x range. With free axes, NestedRIC's +0.053 on radio-shift stretched
    that panel to +/-0.06 while every other method collapsed onto the zero line, so a
    glance across the row read as NestedRIC dominating two streams -- a claim the data do
    not settle, since large positive BWT on a stream where nothing forgets can equally be
    a model recovering from a poor start. Values outside the shared range are drawn as
    arrows at the boundary and labelled, which states the magnitude without letting one
    outlier set the scale for nine other methods.
    """
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

    parsed = {
        (r.stream, r.method): _interval(r.bwt) for r in df.itertuples() if r.method in methods
    }
    # Range set by everything except the proposed method, then padded: the scale should
    # be legible for the field, not for one outlier.
    others = [v[0] for (s_, m_), v in parsed.items() if m_ != "nestedric"]
    lo, hi = min(others), max(others)
    pad = 0.25 * (hi - lo)
    xlim = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        1, len(streams), figsize=(1.5 * len(streams) + 1.2, 2.7), sharey=True, sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, stream in zip(axes, streams, strict=True):
        for i, method in enumerate(methods):
            key = (stream, method)
            if key not in parsed:
                ax.scatter(0, i, marker="x", color=GREY, s=12)  # absent, not zero
                continue
            est, low, high = parsed[key]
            colour = BLUE if method == "nestedric" else (ORANGE if method == "replay" else GREY)
            if est > xlim[1]:
                ax.annotate(
                    f"{est:+.3f}",
                    xy=(xlim[1], i),
                    xytext=(-2, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=5.5,
                    color=colour,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=0.8),
                )
                continue
            ax.plot([max(low, xlim[0]), min(high, xlim[1])], [i, i], color=colour, lw=1, zorder=2)
            ax.scatter(est, i, color=colour, s=13, zorder=3)
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        ax.set_title(stream)
        ax.set_xlabel("BWT")
        ax.set_xlim(*xlim)
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods)
    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", ls="-", ms=4, label="NestedRIC (ours)"),
        plt.Line2D([], [], color=ORANGE, marker="o", ls="-", ms=4, label="replay"),
        plt.Line2D([], [], color=GREY, marker="o", ls="-", ms=4, label="other baselines"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06)
    )
    fig.savefig(out, bbox_inches="tight")
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
    # Log-log: a power law is then a straight line, and the two slopes are directly
    # comparable as exponents -- which is the entire claim. On a linear x axis the
    # difference between delta^(-1/2) and something steeper is not readable.
    ax.set_yscale("log", base=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"drift magnitude $\delta$  (log scale)")
    ax.set_ylabel(r"separation ratio $\rho$")
    ax.set_title(r"$\rho^*$ falls with drift, faster than $\delta^{-1/2}$")
    ax.text(
        0.02,
        0.02,
        f"n = {len(df)} magnitudes",
        transform=ax.transAxes,
        fontsize=6,
        color=GREY,
        va="bottom",
    )
    ax.legend(frameon=False)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=Path("results/tables"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument(
        "--format",
        default="pdf",
        choices=["pdf", "png"],
        help="pdf for the paper, png for on-screen review",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ext = args.format

    main_csv = args.tables / "main.csv"
    if main_csv.exists():
        main_table = pd.read_csv(main_csv)
        fig_which_shifts(main_table, args.out / f"fig1_which_shifts.{ext}")
        fig_methods(main_table, args.out / f"fig2_methods.{ext}")
        if main_table[main_table.metric == "bwt"].groupby("stream").method.nunique().nunique() > 1:
            print(
                "!! main.csv is missing methods on some streams; regenerate it "
                "(scripts/make_table.py) before using fig1/fig2 in the paper"
            )
    else:
        print(f"!! {main_csv} absent -- skipping fig1, fig2")

    fig_drift_window(args.tables / "drift_sweep.csv", args.out / f"fig3_drift_window.{ext}")
    fig_theory(args.tables / "theory_test.csv", args.out / f"fig4_theory.{ext}")

    for path in sorted(args.out.glob(f"*.{ext}")):
        print(f"  {path}  ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
