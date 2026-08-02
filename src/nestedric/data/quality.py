"""Data-quality diagnostics over the prepared parquet files.

This module deliberately owns NO range definitions. The physically admissible ranges
live in :mod:`nestedric.data.schema` and are re-exported here, because three
independent copies of the same table (as briefly existed during Day 1) can drift apart
with no way to tell which is authoritative.

Because :func:`nestedric.data.schema.sanitise` runs inside ``to_canonical``, corruption
in a prepared parquet has already become missingness. So the diagnostic question is not
"which values are out of range" (none are) but "how is missingness distributed" -- and
in particular whether it is spread thinly across all traces or concentrated in a few.
The two cases call for different responses:

    thin and uniform      -> mask, keep the column as a feature
    concentrated by trace -> drop those traces
    concentrated by group -> the column leaks group identity; drop the FEATURE

The third case is the dangerous one. On COMMAG, ``sum_requested_prbs`` is masked at
2-14% per trace while ColO-RAN shows none at all. A model given that feature can infer
which testbed it is looking at from the missingness pattern alone, which in the
cross-dataset stream is leakage wearing a feature's clothes.
"""

from __future__ import annotations

import pandas as pd

from nestedric.data.schema import SENTINEL_VALUES, VALID_RANGES

#: Re-exported for callers that want the ranges without importing the whole schema.
PHYSICAL_RANGES = VALID_RANGES

__all__ = [
    "PHYSICAL_RANGES",
    "SENTINEL_VALUES",
    "violation_mask",
    "missingness",
    "missingness_by_group",
    "concentration",
    "constant_columns",
]


def violation_mask(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Boolean mask of values falling outside the closed interval ``[lo, hi]``.

    Missing values are not violations; they are reported separately.
    """
    vals = pd.to_numeric(series, errors="coerce")
    return ((vals < lo) | (vals > hi)).fillna(False)


def missingness(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Fraction missing per column, descending, zero-entries dropped."""
    frac = df[list(columns)].isna().mean().sort_values(ascending=False)
    return frac[frac > 0]


def missingness_by_group(
    df: pd.DataFrame, columns: tuple[str, ...], by: str | list[str]
) -> pd.DataFrame:
    """Fraction missing per column within each group of *by*.

    Use ``by='dataset'`` to test for the leakage case above, and ``by='scenario'`` or
    ``by='trace_id'`` to test whether corruption tracks a particular experimental
    condition.
    """
    return df.groupby(by, observed=True)[list(columns)].apply(lambda g: g.isna().mean())


def concentration(df: pd.DataFrame, column: str, by: str = "trace_id") -> pd.DataFrame:
    """Distribution of a column's missingness across groups.

    Returns per-group missing fraction plus the share of all missing values contributed
    by the worst 10% of groups. A high share means the problem is concentrated and the
    right fix is dropping those groups; a low share means it is diffuse and the right
    fix is masking.
    """
    per = df.groupby(by, observed=True)[column].apply(lambda s: s.isna().mean())
    counts = df.groupby(by, observed=True)[column].apply(lambda s: int(s.isna().sum()))
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame({"missing_fraction": per, "missing_count": counts})

    top_n = max(1, len(counts) // 10)
    top_share = counts.nlargest(top_n).sum() / total

    out = pd.DataFrame({"missing_fraction": per, "missing_count": counts})
    out.attrs["top_decile_share"] = float(top_share)
    out.attrs["n_groups"] = int(len(counts))
    out.attrs["total_missing"] = total
    return out


def constant_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    """Columns with at most one distinct non-missing value.

    Verified constant across the full corpus: ``ul_rssi`` and ``tx_errors_dl_pct``.
    """
    return [c for c in columns if c in df.columns and df[c].nunique(dropna=True) <= 1]
