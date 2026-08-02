"""End-to-end preparation tests: real directory layout, real CSV header, real parquet.

The Day 1 tests all worked on in-memory frames, so nothing verified that preparation
actually walks a dataset tree and writes files. These write a miniature dataset to a
tmp_path, including the logger's unnamed spacer columns, and prepare it for real.
"""

from pathlib import Path

import pandas as pd
import pytest

from nestedric.data import colosseum as C
from nestedric.data.schema import ALL_COLUMNS, RAW_SLICE_COLUMNS

IMSI_A = 1010123456049
IMSI_B = 1010123456050


def _write_metrics_csv(path: Path, n: int, imsi: int, slice_id: int = 1, t0: int = 1618103191351):
    """Write one slice-metrics CSV with the verified 31-column header, spacers included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, list] = {}
    for i, col in enumerate(RAW_SLICE_COLUMNS):
        name = col if col else f"__spacer{i}"
        values[name] = [1.0] * n
    values["Timestamp"] = [t0 + 250 * i for i in range(n)]
    values["IMSI"] = [imsi] * n
    values["slice_id"] = [slice_id] * n
    values["scheduling_policy"] = [0] * n
    values["num_ues"] = [9] * n
    values["sum_requested_prbs"] = [7] * n
    values["sum_granted_prbs"] = [18] * n
    values["tx_brate downlink [Mbps]"] = [0.0055] * n

    df = pd.DataFrame(values)
    # Spacer columns are unnamed in the real files; blank the header so pandas reads
    # them back as 'Unnamed: N', which is what read_metrics_csv drops.
    df.columns = ["" if c.startswith("__spacer") else c for c in df.columns]
    df.to_csv(path, index=False)


@pytest.fixture
def coloran_root(tmp_path: Path) -> Path:
    """Two shards (tr0, tr7), two traces each, plus one trace too short to keep."""
    root = tmp_path / "colosseum-oran-coloran-dataset"
    base = root / "rome_static_medium"
    for tr in ("tr0", "tr7"):
        d = base / "sched0" / tr / "exp1" / "bs5" / "slices_bs5"
        _write_metrics_csv(d / f"{IMSI_A}_metrics.csv", 200, IMSI_A)
        _write_metrics_csv(d / f"{IMSI_B}_metrics.csv", 150, IMSI_B, slice_id=2)
    short = base / "sched0" / "tr9" / "exp1" / "bs5" / "slices_bs5"
    _write_metrics_csv(short / f"{IMSI_A}_metrics.csv", 10, IMSI_A)
    return root


def test_shard_key_groups_by_scenario_and_tr():
    a = C.parse_path(
        Path("x/rome_static_medium/sched0/tr7/exp3/bs5/slices_bs5/1010123456049_metrics.csv"),
        "coloran",
    )
    b = C.parse_path(
        Path("x/rome_static_medium/sched2/tr7/exp1/bs1/slices_bs1/1010123456050_metrics.csv"),
        "coloran",
    )
    c = C.parse_path(
        Path("x/rome_static_medium/sched0/tr8/exp3/bs5/slices_bs5/1010123456049_metrics.csv"),
        "coloran",
    )
    # Scheduling policy does not split shards; tr config does.
    assert a.shard_key == b.shard_key
    assert a.shard_key != c.shard_key


def test_prepare_writes_one_shard_per_tr_config(coloran_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    manifest_path, report = C.prepare(coloran_root, out, dataset="coloran", min_rows=100)

    shards = sorted(p.stem for p in (out / "coloran").glob("*.parquet"))
    assert shards == ["traffic__rome_static_medium__tr0", "traffic__rome_static_medium__tr7"]

    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 2
    assert manifest.n_rows.sum() == 4 * 175  # (200 + 150) rows x 2 shards
    assert manifest.n_traces.sum() == 4
    assert set(manifest.tr_config) == {"tr0", "tr7"}
    assert report == {}  # the synthetic values are all in range


def test_prepare_drops_traces_below_min_rows(coloran_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    C.prepare(coloran_root, out, dataset="coloran", min_rows=100)
    manifest = pd.read_csv(manifest_path := out / "coloran.manifest.csv")
    assert manifest_path.exists()
    # tr9 holds only a 10-row trace, so its shard is written off entirely.
    assert "tr9" not in set(manifest.tr_config)


def test_shards_roundtrip_through_load_shards(coloran_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    C.prepare(coloran_root, out, dataset="coloran", min_rows=100)

    df = C.load_shards(out, "coloran")
    assert list(df.columns) == list(ALL_COLUMNS)
    assert len(df) == 700
    assert df.trace_id.nunique() == 4

    one = C.load_shards(out, "coloran", ["traffic__rome_static_medium__tr0"])
    assert len(one) == 350
    assert set(one.tr_config) == {"tr0"}

    cols = C.load_shards(out, "coloran", ["traffic__rome_static_medium__tr0"], columns=["dl_mcs"])
    assert list(cols.columns) == ["dl_mcs"]


def test_manifest_carries_context_needed_to_pick_environments(coloran_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    C.prepare(coloran_root, out, dataset="coloran", min_rows=100)
    m = pd.read_csv(out / "coloran.manifest.csv")

    for col in ("scenario", "slice_assignment", "mobility", "distance", "tr_config"):
        assert m[col].notna().all()
    assert set(str(m.slice_ids.iloc[0]).split("|")) == {"1", "2"}
    assert str(m.sched_policies.iloc[0]) == "0"
    assert (m.t_end_ms > m.t_start_ms).all()


def test_sanitisation_report_is_written(tmp_path: Path):
    root = tmp_path / "colosseum-oran-coloran-dataset"
    d = root / "rome_static_medium" / "sched0" / "tr0" / "exp1" / "bs5" / "slices_bs5"
    _write_metrics_csv(d / f"{IMSI_A}_metrics.csv", 200, IMSI_A)

    raw = pd.read_csv(d / f"{IMSI_A}_metrics.csv")
    raw.loc[0, "dl_mcs"] = 2.4238e8
    raw.loc[1, "dl_buffer [bytes]"] = -2147483648.0
    raw.to_csv(d / f"{IMSI_A}_metrics.csv", index=False)

    out = tmp_path / "processed"
    _, report = C.prepare(root, out, dataset="coloran", min_rows=100)

    assert report["dl_mcs:range"] == 1
    assert report["dl_buffer_bytes:sentinel"] == 1
    rep = pd.read_csv(out / "coloran.sanitisation.csv")
    assert set(rep.column) == {"dl_mcs:range", "dl_buffer_bytes:sentinel"}
    assert (rep.fraction < 0.01).all()


def test_prepare_raises_when_nothing_parses(tmp_path: Path):
    root = tmp_path / "empty-dataset"
    (root / "junk").mkdir(parents=True)
    (root / "junk" / "notes.txt").write_text("no metrics here")
    with pytest.raises(RuntimeError, match="no parseable metrics files"):
        C.prepare(root, tmp_path / "processed", dataset="coloran")
