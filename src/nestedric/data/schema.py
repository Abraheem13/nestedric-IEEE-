"""Canonical KPI schema.\n\nEvery dataset adapter must emit a pandas DataFrame conforming to this schema so\nthat the continual-learning engine is dataset-agnostic. This file is the single\nsource of truth for column names, dtypes and units.

Status: STUB -- implemented on Day 1 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Identity / indexing columns.
INDEX_COLUMNS = ("dataset", "trace_id", "bs_id", "ue_id", "slice_id", "timestamp_ms")

#: Downlink / uplink KPIs harmonised across datasets (units in KPI_UNITS).
KPI_COLUMNS = (
    "dl_thpt_mbps",
    "ul_thpt_mbps",
    "dl_buffer_bytes",
    "ul_buffer_bytes",
    "dl_prb_used",
    "ul_prb_used",
    "dl_mcs",
    "ul_mcs",
    "cqi",
    "sinr_db",
    "ratio_granted_req",
    "tx_pkts",
    "tx_errors_pct",
)

#: Context columns describing the operating regime (used to define environments).
CONTEXT_COLUMNS = ("slice_type", "traffic_profile", "sched_policy", "mobility_regime", "n_ue")

KPI_UNITS: dict[str, str] = {}  # TODO(Day 1): fill per column.


@dataclass(frozen=True)
class KPISchema:
    """Validation helper for the canonical schema."""

    def validate(self, df) -> None:
        """Raise ``SchemaError`` if *df* violates the canonical schema."""
        raise NotImplementedError("Day 1")


class SchemaError(ValueError):
    """Raised when a dataset adapter emits a non-conforming frame."""
