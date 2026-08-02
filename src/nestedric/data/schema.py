"""Canonical KPI schema for the O-RAN-CL benchmark.

Single source of truth for column names, dtypes and units. Every dataset adapter emits
a frame conforming to this schema so the continual-learning engine is dataset-agnostic.

Verified against the real files on Day 1. Both layouts end in
``.../bs{N}/slices_bs{N}/<IMSI>_metrics.csv``:

  * ColO-RAN  ``rome_static_medium/sched{0,1,2}/tr{0..27}/exp{1..5}/bs{1..7}/...``
  * COMMAG    ``slice_{mixed,traffic}/rome_{static,slow}_{close,medium,far}/``
              ``tr{0..17}/exp{1..6}/bs{1..4}/...``

Both datasets export the *same* 31-column slice-metrics format, including four unnamed
spacer columns produced by the original logger. That coincidence is what makes the
cross-dataset stream clean rather than a harmonisation exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------- raw
#: Exact header of the slice-metrics CSVs, in file order. Empty strings are the
#: logger's spacer columns; pandas names them ``Unnamed: 4`` etc.
RAW_SLICE_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "num_ues",
    "IMSI",
    "RNTI",
    "",  # spacer
    "slicing_enabled",
    "slice_id",
    "slice_prb",
    "power_multiplier",
    "scheduling_policy",
    "",  # spacer
    "dl_mcs",
    "dl_n_samples",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "tx_pkts downlink",
    "tx_errors downlink (%)",
    "dl_cqi",
    "",  # spacer
    "ul_mcs",
    "ul_n_samples",
    "ul_buffer [bytes]",
    "rx_brate uplink [Mbps]",
    "rx_pkts uplink",
    "rx_errors uplink (%)",
    "ul_rssi",
    "ul_sinr",
    "phr",
    "",  # spacer
    "sum_requested_prbs",
    "sum_granted_prbs",
    "",  # spacer
    "dl_pmi",
    "dl_ri",
    "ul_n",
    "ul_turbo_iters",
)

#: Native logging period of the slice metrics, in milliseconds (verified: 250 ms).
NATIVE_PERIOD_MS = 250

# --------------------------------------------------------------------- canonical
#: Identity / indexing columns.
INDEX_COLUMNS: tuple[str, ...] = (
    "dataset",  # 'coloran' | 'commag'
    "trace_id",  # unique per (scenario, tr, exp, bs, imsi)
    "bs_id",
    "imsi",
    "slice_id",
    "timestamp_ms",
)

#: KPIs used as model inputs and targets. Names are the raw ones, lightly normalised.
KPI_COLUMNS: tuple[str, ...] = (
    "dl_thpt_mbps",  # tx_brate downlink [Mbps]
    "ul_thpt_mbps",  # rx_brate uplink [Mbps]
    "dl_buffer_bytes",  # dl_buffer [bytes]
    "ul_buffer_bytes",  # ul_buffer [bytes]
    "dl_mcs",
    "ul_mcs",
    "dl_cqi",
    "ul_sinr",
    "ul_rssi",
    "phr",
    "dl_n_samples",
    "ul_n_samples",
    "tx_pkts_dl",
    "rx_pkts_ul",
    "tx_errors_dl_pct",
    "rx_errors_ul_pct",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "ratio_granted_req",  # derived: granted / max(requested, 1)
    "num_ues",
    "ul_turbo_iters",
)

#: Context columns describing the operating regime. Environment boundaries are defined
#: by (subsets of) these, which is why they are carried through preparation.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "scenario",  # e.g. 'rome_static_medium'
    "slice_assignment",  # 'traffic' | 'mixed' (coloran is always 'traffic')
    "mobility",  # 'static' | 'slow'
    "distance",  # 'close' | 'medium' | 'far'
    "sched_policy",  # 0 = RR, 1 = WF, 2 = PF
    "slice_prb",  # RBG allocation for this slice
    "tr_config",  # training configuration index, e.g. 'tr7'
    "exp_id",  # repetition, e.g. 'exp3'
)

ALL_COLUMNS: tuple[str, ...] = INDEX_COLUMNS + KPI_COLUMNS + CONTEXT_COLUMNS

#: Raw header -> canonical name. Anything not listed is dropped.
RAW_TO_CANONICAL: dict[str, str] = {
    "Timestamp": "timestamp_ms",
    "IMSI": "imsi",
    "slice_id": "slice_id",
    "slice_prb": "slice_prb",
    "scheduling_policy": "sched_policy",
    "num_ues": "num_ues",
    "tx_brate downlink [Mbps]": "dl_thpt_mbps",
    "rx_brate uplink [Mbps]": "ul_thpt_mbps",
    "dl_buffer [bytes]": "dl_buffer_bytes",
    "ul_buffer [bytes]": "ul_buffer_bytes",
    "dl_mcs": "dl_mcs",
    "ul_mcs": "ul_mcs",
    "dl_cqi": "dl_cqi",
    "ul_sinr": "ul_sinr",
    "ul_rssi": "ul_rssi",
    "phr": "phr",
    "dl_n_samples": "dl_n_samples",
    "ul_n_samples": "ul_n_samples",
    "tx_pkts downlink": "tx_pkts_dl",
    "rx_pkts uplink": "rx_pkts_ul",
    "tx_errors downlink (%)": "tx_errors_dl_pct",
    "rx_errors uplink (%)": "rx_errors_ul_pct",
    "sum_requested_prbs": "sum_requested_prbs",
    "sum_granted_prbs": "sum_granted_prbs",
    "ul_turbo_iters": "ul_turbo_iters",
}

KPI_UNITS: dict[str, str] = {
    "dl_thpt_mbps": "Mbit/s",
    "ul_thpt_mbps": "Mbit/s",
    "dl_buffer_bytes": "bytes",
    "ul_buffer_bytes": "bytes",
    "dl_mcs": "MCS index",
    "ul_mcs": "MCS index",
    "dl_cqi": "CQI index",
    "ul_sinr": "dB",
    "ul_rssi": "dBm",
    "phr": "dB",
    "dl_n_samples": "count",
    "ul_n_samples": "count",
    "tx_pkts_dl": "packets",
    "rx_pkts_ul": "packets",
    "tx_errors_dl_pct": "percent",
    "rx_errors_ul_pct": "percent",
    "sum_requested_prbs": "PRBs",
    "sum_granted_prbs": "PRBs",
    "ratio_granted_req": "ratio",
    "num_ues": "count",
    "ul_turbo_iters": "iterations",
}

#: Default prediction targets (KPI-forecasting task).
TARGET_COLUMNS: tuple[str, ...] = ("dl_thpt_mbps", "dl_buffer_bytes")

#: Scheduling-policy codes, per both dataset READMEs.
SCHED_POLICIES: dict[int, str] = {0: "round_robin", 1: "waterfilling", 2: "proportional_fair"}

#: Slice semantics under the 'traffic' assignment (ColO-RAN, COMMAG slice_traffic).
SLICE_TRAFFIC_CLASS: dict[int, str] = {0: "eMBB", 1: "MTC", 2: "URLLC"}


# ------------------------------------------------------------------- sanitising
#: Logger sentinels observed in the real corpus (Day 1 profiling). These are NOT data.
#: ``dl_buffer_bytes`` carries INT32_MIN where the buffer report is unavailable.
SENTINEL_VALUES: dict[str, tuple[float, ...]] = {
    "dl_buffer_bytes": (-2147483648.0,),
    "ul_buffer_bytes": (-2147483648.0,),
}

#: Physically admissible ranges. Values outside these are logger corruption and are
#: masked to missing rather than clipped, so the model never sees a fabricated value.
#: Verified against the corpus on Day 1: dl_mcs reached 2.42e8 and
#: sum_requested_prbs reached -501 before masking.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "dl_mcs": (0.0, 31.0),  # 5-bit MCS index; 29-31 signal redundancy versions
    "ul_mcs": (0.0, 31.0),  # UL-SCH: 0-28 carry data, 29-31 are RV signalling
    "dl_cqi": (0.0, 15.0),
    "ul_sinr": (-50.0, 60.0),  # dB
    "phr": (-30.0, 40.0),  # dB
    "dl_thpt_mbps": (0.0, 1000.0),
    "ul_thpt_mbps": (0.0, 1000.0),
    "dl_buffer_bytes": (0.0, 1e9),
    "ul_buffer_bytes": (0.0, 1e9),
    "sum_requested_prbs": (0.0, 20000.0),
    "sum_granted_prbs": (0.0, 20000.0),
    "ratio_granted_req": (0.0, 50.0),
    "tx_errors_dl_pct": (0.0, 100.0),
    "rx_errors_ul_pct": (0.0, 100.0),
    "num_ues": (0.0, 200.0),
    "ul_turbo_iters": (0.0, 20.0),
    "dl_n_samples": (0.0, 1e5),
    "ul_n_samples": (0.0, 1e5),
    "tx_pkts_dl": (0.0, 1e6),
    "rx_pkts_ul": (0.0, 1e6),
}

#: Columns observed constant across the whole corpus (Day 1). Excluded from features:
#: a constant column contributes nothing and wastes normalisation capacity.
CONSTANT_COLUMNS: tuple[str, ...] = ("ul_rssi", "tx_errors_dl_pct")

#: KPI columns actually used as model features.
FEATURE_COLUMNS: tuple[str, ...] = tuple(c for c in KPI_COLUMNS if c not in CONSTANT_COLUMNS)


def sanitise(df: pd.DataFrame, report: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mask logger sentinels and out-of-range values to missing.

    Masking, not clipping: a clipped sentinel is still a fabricated observation, and
    the model cannot tell it from a real one. Missingness is honest and is handled
    explicitly by the loader.

    Returns the sanitised frame and a per-column count of masked cells, which is
    reported in the paper so the reader knows how much of the corpus was unusable.
    """
    df = df.copy()
    masked: dict[str, int] = {}

    for col, sentinels in SENTINEL_VALUES.items():
        if col not in df.columns:
            continue
        hit = df[col].isin(sentinels)
        if hit.any():
            df.loc[hit, col] = pd.NA
            masked[f"{col}:sentinel"] = int(hit.sum())

    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = ((vals < lo) | (vals > hi)) & vals.notna()
        if bad.any():
            df.loc[bad, col] = pd.NA
            masked[f"{col}:range"] = int(bad.sum())

    if report and masked:
        total = len(df)
        for k, v in sorted(masked.items(), key=lambda kv: -kv[1]):
            print(f"    masked {k:38s} {v:>10,} ({100 * v / total:.4f}%)")

    return df, masked


class SchemaError(ValueError):
    """Raised when a dataset adapter emits a non-conforming frame."""


@dataclass(frozen=True)
class KPISchema:
    """Validation helper for the canonical schema."""

    strict: bool = True

    def validate(self, df: pd.DataFrame) -> None:
        """Raise :class:`SchemaError` if *df* violates the canonical schema."""
        missing = [c for c in ALL_COLUMNS if c not in df.columns]
        if missing:
            raise SchemaError(f"missing canonical columns: {missing}")

        if self.strict:
            extra = [c for c in df.columns if c not in ALL_COLUMNS]
            if extra:
                raise SchemaError(f"unexpected columns: {extra}")

        if df["timestamp_ms"].isna().any():
            raise SchemaError("timestamp_ms contains NaN")

        bad_slice = set(df["slice_id"].dropna().unique()) - {0, 1, 2}
        if bad_slice:
            raise SchemaError(f"unexpected slice_id values: {sorted(bad_slice)}")

        bad_sched = set(df["sched_policy"].dropna().unique()) - set(SCHED_POLICIES)
        if bad_sched:
            raise SchemaError(f"unexpected sched_policy values: {sorted(bad_sched)}")

        # Range checks over EVERY column with a declared physical range. The Day 1
        # profile passed a validator that only checked throughput, while
        # dl_buffer_bytes carried INT32_MIN and dl_mcs reached 2.4e8. Never again.
        for col, (lo, hi) in VALID_RANGES.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            if ((vals < lo) | (vals > hi)).any():
                bad = vals[(vals < lo) | (vals > hi)]
                raise SchemaError(
                    f"{col} has {len(bad)} values outside [{lo}, {hi}] "
                    f"(min={bad.min()}, max={bad.max()}); run sanitise() first"
                )

        for col, sentinels in SENTINEL_VALUES.items():
            if col in df.columns and df[col].isin(sentinels).any():
                raise SchemaError(f"{col} still contains logger sentinels; run sanitise()")

    def undocumented_units(self) -> list[str]:
        """KPI columns lacking a unit entry -- guards against silent unit drift."""
        return [c for c in KPI_COLUMNS if c not in KPI_UNITS]
