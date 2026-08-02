"""Schema and adapter tests against the real Colosseum file format.

These use synthetic frames matching the verified 31-column header, so they run without
the 16 GB of raw data. The path-parsing tests use real path strings taken from the
actual repositories.
"""

from pathlib import Path

import pandas as pd
import pytest

from nestedric.data import colosseum as C
from nestedric.data.schema import (
    ALL_COLUMNS,
    KPI_UNITS,
    RAW_SLICE_COLUMNS,
    KPISchema,
    SchemaError,
)

CORAN_PATH = Path(
    "data/raw/colosseum-oran-coloran-dataset/rome_static_medium/sched0/tr7/exp3/"
    "bs5/slices_bs5/1010123456049_metrics.csv"
)
COMMAG_PATH = Path(
    "data/raw/colosseum-oran-commag-dataset/slice_traffic/rome_static_close/tr0/exp1/"
    "bs1/slices_bs1/1010123456007_metrics.csv"
)


def _fake_raw(n: int = 20) -> pd.DataFrame:
    """A frame with the real header, spacer columns already dropped."""
    cols = [c for c in RAW_SLICE_COLUMNS if c]
    data = {c: [1.0] * n for c in cols}
    data["Timestamp"] = [1618103191351 + 250 * i for i in range(n)]
    data["IMSI"] = [1010123456049] * n
    data["slice_id"] = [1] * n
    data["scheduling_policy"] = [0] * n
    data["num_ues"] = [9] * n
    data["sum_requested_prbs"] = [7] * (n // 2) + [0] * (n - n // 2)
    data["sum_granted_prbs"] = [18] * n
    data["tx_brate downlink [Mbps]"] = [0.0055] * n
    data["rx_brate uplink [Mbps]"] = [0.1] * n
    return pd.DataFrame(data)


def _meta() -> C.TraceMeta:
    return C.TraceMeta(
        dataset="coloran",
        scenario="rome_static_medium",
        slice_assignment="traffic",
        mobility="static",
        distance="medium",
        tr_config="tr7",
        exp_id="exp3",
        bs_id=5,
        imsi=1010123456049,
    )


def test_coloran_path_parses():
    m = C.parse_path(CORAN_PATH, "coloran")
    assert m is not None
    assert m.dataset == "coloran"
    assert m.scenario == "rome_static_medium"
    assert (m.mobility, m.distance) == ("static", "medium")
    assert (m.tr_config, m.exp_id, m.bs_id) == ("tr7", "exp3", 5)
    assert m.imsi == 1010123456049


def test_commag_path_parses_extra_axes():
    m = C.parse_path(COMMAG_PATH, "commag")
    assert m is not None
    assert m.slice_assignment == "traffic"
    assert (m.mobility, m.distance) == ("static", "close")
    assert m.bs_id == 1


def test_wrong_dataset_regex_returns_none():
    assert C.parse_path(COMMAG_PATH, "coloran") is None


def test_trace_id_is_unique_per_context():
    a = _meta()
    b = C.TraceMeta(**{**a.__dict__, "exp_id": "exp4"})
    assert a.trace_id != b.trace_id


def test_to_canonical_emits_exact_schema():
    df, _ = C.to_canonical(_fake_raw(), _meta())
    assert list(df.columns) == list(ALL_COLUMNS)
    KPISchema(strict=True).validate(df)


def test_ratio_is_missing_when_nothing_requested():
    """Idle rows must be NaN, never a fabricated ratio from a clipped denominator."""
    df, _ = C.to_canonical(_fake_raw(20), _meta())
    idle = df.sum_requested_prbs == 0
    assert idle.sum() > 0
    assert df.loc[idle, "ratio_granted_req"].isna().all()
    assert df.loc[~idle, "ratio_granted_req"].notna().all()


def test_validator_rejects_bad_slice_id():
    df, _ = C.to_canonical(_fake_raw(), _meta())
    df.loc[0, "slice_id"] = 7
    with pytest.raises(SchemaError, match="slice_id"):
        KPISchema().validate(df)


def test_validator_rejects_negative_throughput():
    df, _ = C.to_canonical(_fake_raw(), _meta())
    df.loc[0, "dl_thpt_mbps"] = -1.0
    with pytest.raises(SchemaError, match="dl_thpt_mbps"):
        KPISchema().validate(df)


def test_all_kpi_units_documented():
    assert KPISchema().undocumented_units() == []
    assert set(KPI_UNITS) >= set(KPISchema().undocumented_units())


def test_timestamps_sorted_within_trace():
    raw = _fake_raw()
    raw = raw.iloc[::-1].reset_index(drop=True)
    df, _ = C.to_canonical(raw, _meta())
    assert df.timestamp_ms.is_monotonic_increasing


# --------------------------------------------------------------------- Day 1.5
# Regression tests for the corruptions found by profiling the full corpus. The
# original validator passed data containing INT32_MIN buffers and dl_mcs = 2.4e8.

from nestedric.data.schema import (  # noqa: E402
    CONSTANT_COLUMNS,
    FEATURE_COLUMNS,
    SENTINEL_VALUES,
    VALID_RANGES,
    sanitise,
)


def test_int32_min_buffer_sentinel_is_masked():
    """dl_buffer_bytes carries INT32_MIN where the buffer report is unavailable."""
    df = pd.DataFrame({"dl_buffer_bytes": [100.0, -2147483648.0, 5000.0]})
    out, masked = sanitise(df)
    assert out.dl_buffer_bytes.isna().sum() == 1
    assert masked["dl_buffer_bytes:sentinel"] == 1
    assert out.dl_buffer_bytes.dropna().tolist() == [100.0, 5000.0]


def test_out_of_range_mcs_is_masked_not_clipped():
    """A clipped sentinel is still a fabricated observation. Mask it."""
    df = pd.DataFrame({"dl_mcs": [14.0, 2.4238e8, 20.0]})
    out, masked = sanitise(df)
    assert out.dl_mcs.isna().sum() == 1
    assert masked["dl_mcs:range"] == 1
    assert 28.0 not in out.dl_mcs.values  # not clipped to the bound


def test_negative_prb_requests_are_masked():
    df = pd.DataFrame({"sum_requested_prbs": [36.0, -501.0, 7.0]})
    out, _ = sanitise(df)
    assert out.sum_requested_prbs.isna().sum() == 1


def test_validator_now_rejects_sentinels():
    df, _ = C.to_canonical(_fake_raw(), _meta())
    df.loc[0, "dl_buffer_bytes"] = -2147483648.0
    with pytest.raises(SchemaError):
        KPISchema().validate(df)


def test_validator_now_rejects_out_of_range_mcs():
    df, _ = C.to_canonical(_fake_raw(), _meta())
    df.loc[0, "dl_mcs"] = 2.4238e8
    with pytest.raises(SchemaError, match="dl_mcs"):
        KPISchema().validate(df)


def test_to_canonical_output_passes_validator():
    """Sanitisation runs inside to_canonical, so its output is always valid."""
    raw = _fake_raw()
    raw.loc[0, "dl_buffer [bytes]"] = -2147483648.0
    raw.loc[1, "dl_mcs"] = 2.4238e8
    df, _ = C.to_canonical(raw, _meta())
    KPISchema(strict=True).validate(df)
    assert df.dl_buffer_bytes.isna().sum() >= 1


def test_constant_columns_excluded_from_features():
    assert set(CONSTANT_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert "ul_rssi" not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 18  # 21 KPIs - 2 constant - 1 leaky


def test_leaky_columns_excluded_from_features():
    """Day 2 gate: sum_requested_prbs is 0% missing on ColO-RAN, 10.2% on COMMAG."""
    from nestedric.data.schema import KPI_COLUMNS, LEAKY_COLUMNS

    assert set(LEAKY_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert "sum_requested_prbs" not in FEATURE_COLUMNS
    # The derived ratio measured a 2.3pp gap, inside threshold, so it is kept.
    assert "ratio_granted_req" in FEATURE_COLUMNS
    # The dropped counter stays in the parquet: excluded as a feature, not deleted.
    assert "sum_requested_prbs" in KPI_COLUMNS


def test_every_ranged_column_is_a_known_kpi():
    from nestedric.data.schema import KPI_COLUMNS as K

    assert set(VALID_RANGES) <= set(K)
    assert set(SENTINEL_VALUES) <= set(K)
