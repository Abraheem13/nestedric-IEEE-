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

COMMAG_PATH_STR = Path(
    "data/raw/colosseum-oran-commag-dataset/slice_traffic/rome_static_close/tr0/exp1/"
    "bs1/slices_bs1/1010123456007_metrics.csv"
)


def _write_metrics_csv(
    path: Path,
    n: int,
    imsi: int,
    slice_id: int = 1,
    t0: int = 1618103191351,
    sched: int = 0,
):
    """Write one slice-metrics CSV with the verified 31-column header, spacers included."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, list] = {}
    for i, col in enumerate(RAW_SLICE_COLUMNS):
        name = col if col else f"__spacer{i}"
        values[name] = [1.0] * n
    values["Timestamp"] = [t0 + 250 * i for i in range(n)]
    values["IMSI"] = [imsi] * n
    values["slice_id"] = [slice_id] * n
    values["scheduling_policy"] = [sched] * n
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


def _coloran_meta(sched: int, tr: str = "tr7", imsi: int = IMSI_A):
    return C.parse_path(
        Path(f"x/rome_static_medium/sched{sched}/{tr}/exp3/bs5/slices_bs5/{imsi}_metrics.csv"),
        "coloran",
    )


def test_shard_key_separates_scheduling_policy_and_tr_config():
    assert _coloran_meta(0).shard_key != _coloran_meta(2).shard_key
    assert _coloran_meta(0, "tr7").shard_key != _coloran_meta(0, "tr8").shard_key


def test_trace_id_distinguishes_scheduling_policy():
    """ColO-RAN logs the same UE under sched0/1/2; the split key must tell them apart.

    Without the sched component these are one id, so a sched-shift environment cut by
    policy would put the same UE's data in two environments.
    """
    ids = {_coloran_meta(s).trace_id for s in (0, 1, 2)}
    assert len(ids) == 3


def test_commag_trace_id_has_no_sched_component():
    m = C.parse_path(COMMAG_PATH_STR, "commag")
    assert m.sched_dir is None
    assert "sched" not in m.trace_id


def test_prepare_rejects_colliding_trace_ids(tmp_path: Path, monkeypatch):
    """If a future path change reintroduces a collision, preparation must fail loudly."""
    root = tmp_path / "colosseum-oran-coloran-dataset"
    for sched in (0, 1):
        d = root / "rome_static_medium" / f"sched{sched}" / "tr0" / "exp1" / "bs5" / "slices_bs5"
        _write_metrics_csv(d / f"{IMSI_A}_metrics.csv", 200, IMSI_A)

    # Simulate the old key by dropping the sched component again.
    monkeypatch.setattr(
        C.TraceMeta, "shard_key", property(lambda self: f"{self.scenario}__{self.tr_config}")
    )
    monkeypatch.setattr(
        C.TraceMeta, "trace_id", property(lambda self: f"{self.scenario}:{self.imsi}")
    )
    with pytest.raises(C.TraceIdCollision, match="two files"):
        C.prepare(root, tmp_path / "processed", dataset="coloran", min_rows=100)


def test_prepare_writes_one_shard_per_tr_config(coloran_root: Path, tmp_path: Path):
    out = tmp_path / "processed"
    manifest_path, report = C.prepare(coloran_root, out, dataset="coloran", min_rows=100)

    shards = sorted(p.stem for p in (out / "coloran").glob("*.parquet"))
    assert shards == [
        "traffic__rome_static_medium__tr0__sched0",
        "traffic__rome_static_medium__tr7__sched0",
    ]

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

    one = C.load_shards(out, "coloran", ["traffic__rome_static_medium__tr0__sched0"])
    assert len(one) == 350
    assert set(one.tr_config) == {"tr0"}

    cols = C.load_shards(
        out, "coloran", ["traffic__rome_static_medium__tr0__sched0"], columns=["dl_mcs"]
    )
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


def test_a_crashed_run_does_not_poison_its_directory(tmp_path: Path, monkeypatch):
    """A lock that outlives its process turns one crash into a dead run directory.

    The first ablation sweep died on a ValueError after claiming its directory; every
    later attempt then failed against a pid that had not existed for hours.
    """
    from nestedric.engine import runner

    out = tmp_path / "run"

    def boom(cfg, out_dir):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(runner, "_run_experiment_locked", boom)
    with pytest.raises(ValueError, match="synthetic"):
        runner.run_experiment({}, out)

    assert not (out / ".running").exists(), "lock survived a failed run"
    # And the directory is usable again.
    with pytest.raises(ValueError, match="synthetic"):
        runner.run_experiment({}, out)
