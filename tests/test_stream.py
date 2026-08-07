"""Stream construction invariants: no leakage, deterministic given a seed.

These build real streams from a miniature prepared corpus written to tmp_path, so the
manifest, shard selection and trace-level splitting are exercised together rather than
mocked apart.
"""

from pathlib import Path

import pandas as pd
import pytest

from nestedric.data import colosseum as C
from nestedric.data.stream import StreamError, build_stream
from tests.test_prepare import IMSI_A, IMSI_B, _write_metrics_csv


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A prepared two-dataset corpus: 3 ColO-RAN sched dirs x 2 tr, 2 COMMAG scenarios."""
    raw = tmp_path / "raw"
    for sched in (0, 1, 2):
        for tr in ("tr0", "tr7"):
            d = (
                raw
                / "colosseum-oran-coloran-dataset"
                / "rome_static_medium"
                / f"sched{sched}"
                / tr
                / "exp1"
                / "bs5"
                / "slices_bs5"
            )
            _write_metrics_csv(d / f"{IMSI_A}_metrics.csv", 200, IMSI_A, sched=sched)
            _write_metrics_csv(d / f"{IMSI_B}_metrics.csv", 200, IMSI_B, sched=sched, slice_id=2)

    for scenario in ("rome_static_close", "rome_slow_close"):
        d = (
            raw
            / "colosseum-oran-commag-dataset"
            / "slice_traffic"
            / scenario
            / "tr0"
            / "exp1"
            / "bs1"
            / "slices_bs1"
        )
        for imsi in (IMSI_A, IMSI_B):
            _write_metrics_csv(d / f"{imsi}_metrics.csv", 200, imsi, slice_id=0)

    out = tmp_path / "processed"
    C.prepare(raw / "colosseum-oran-coloran-dataset", out, dataset="coloran", min_rows=100)
    C.prepare(raw / "colosseum-oran-commag-dataset", out, dataset="commag", min_rows=100)
    return out


def _cfg(**kw) -> dict:
    base = {
        "name": "test-stream",
        "family": "sched-shift",
        "source": ["coloran"],
        "seed": 0,
        "eval_fraction": 0.5,
        "env_min_samples": 0,
    }
    base.update(kw)
    return base


def test_train_eval_indices_disjoint(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    assert len(stream) > 0
    for env in stream:
        assert set(env.train_traces).isdisjoint(env.eval_traces)
        assert env.train_traces and env.eval_traces


def test_split_divides_traces_never_rows(corpus: Path):
    """A row-level split would put 250 ms-apart neighbours on both sides."""
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    for env in stream:
        assert all(isinstance(t, str) and ":" in t for t in env.train_traces)


def test_environments_never_share_a_trace(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    seen: set[str] = set()
    for env in stream:
        traces = set(env.train_traces) | set(env.eval_traces)
        assert seen.isdisjoint(traces), "a trace appears in two environments"
        seen |= traces


def test_stream_is_deterministic_given_seed(corpus: Path):
    a = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    b = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    assert [e.env_id for e in a] == [e.env_id for e in b]
    assert [e.train_traces for e in a] == [e.train_traces for e in b]
    assert [e.eval_traces for e in a] == [e.eval_traces for e in b]


def test_different_seed_changes_the_split_but_not_the_environments(corpus: Path):
    a = build_stream(_cfg(context_axes=["sched_policy"], seed=0), corpus)
    b = build_stream(_cfg(context_axes=["sched_policy"], seed=7), corpus)
    assert [e.env_id for e in a] == [e.env_id for e in b]
    assert [set(e.train_traces) | set(e.eval_traces) for e in a] == [
        set(e.train_traces) | set(e.eval_traces) for e in b
    ]


def test_split_is_independent_of_neighbouring_environments(corpus: Path):
    """Adding an environment must not reshuffle another's split, or pairing breaks."""
    full = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    subset = build_stream(_cfg(environments=[{"env_id": "env00", "sched_policy": 0}]), corpus)
    first = next(e for e in full if e.context.get("sched_policy") == 0)
    assert subset[0].eval_traces == first.eval_traces


def test_scheduling_policy_separates_environments(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    assert sorted(e.context["sched_policy"] for e in stream) == [0, 1, 2]


def test_explicit_environments_are_used_verbatim(corpus: Path):
    stream = build_stream(
        _cfg(
            environments=[{"env_id": "rr", "sched_policy": 0}, {"env_id": "pf", "sched_policy": 2}]
        ),
        corpus,
    )
    assert [e.env_id for e in stream] == ["rr", "pf"]


def test_cross_dataset_stream_spans_both_testbeds(corpus: Path):
    stream = build_stream(
        _cfg(source=["coloran", "commag"], context_axes=["dataset", "scenario"]), corpus
    )
    assert {e.dataset for e in stream} == {"coloran", "commag"}


def test_cross_dataset_selection_is_not_dominated_by_the_larger_corpus(corpus: Path):
    """Global rank gave nine ColO-RAN environments and zero COMMAG; round-robin must not.

    ColO-RAN cells span three scheduling policies and are individually much larger than
    COMMAG's, so ranking by size alone fills every slot from one testbed and the stream
    stops being cross-dataset while still being labelled as such.
    """
    stream = build_stream(
        _cfg(
            source=["coloran", "commag"],
            context_axes=["dataset", "scenario", "tr_config"],
            n_environments=4,
        ),
        corpus,
    )
    datasets = [e.dataset for e in stream]
    assert set(datasets) == {"coloran", "commag"}
    assert datasets.count("commag") >= 1
    assert datasets.count("coloran") >= 1


def test_cyclic_order_repeats_environments(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"], order="cyclic", repeat=3), corpus)
    ids = [e.env_id for e in stream]
    assert len(ids) == 9
    assert ids[:3] == ids[3:6] == ids[6:9]


def test_small_environments_are_dropped(corpus: Path):
    stream_cfg = _cfg(context_axes=["sched_policy"], env_min_samples=10_000)
    with pytest.raises(StreamError, match="no environments"):
        build_stream(stream_cfg, corpus)


def test_missing_dataset_raises_clearly(corpus: Path):
    with pytest.raises(StreamError, match="tractor"):
        build_stream(_cfg(source=["tractor"], context_axes=["scenario"]), corpus)


def test_unknown_axis_raises(corpus: Path):
    with pytest.raises(StreamError, match="unknown environment axes"):
        build_stream(_cfg(environments=[{"weather": "sunny"}]), corpus)


def test_config_without_environment_definition_raises(corpus: Path):
    with pytest.raises(StreamError, match="neither"):
        build_stream(_cfg(), corpus)


def test_environment_table_renders(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    text = stream.table()
    assert "test-stream" in text
    assert all(e.env_id in text for e in stream)


def test_row_counts_match_the_data(corpus: Path):
    stream = build_stream(_cfg(context_axes=["sched_policy"]), corpus)
    for env in stream:
        df = C.load_shards(corpus, env.dataset, env.shards, columns=["sched_policy"])
        assert env.n_rows == int((df.sched_policy == env.context["sched_policy"]).sum())


def test_environment_selection_never_reads_kpi_columns(corpus: Path, monkeypatch):
    """Selection must not open KPI columns; whole-corpus reads are what break 16 GB."""
    seen: list[list[str] | None] = []
    original = C.load_shards

    def spy(out_dir, dataset, shards=None, columns=None):
        seen.append(columns)
        return original(out_dir, dataset, shards, columns)

    monkeypatch.setattr(C, "load_shards", spy)
    build_stream(_cfg(context_axes=["sched_policy"]), corpus)

    assert seen, "expected at least one shard read"
    for columns in seen:
        assert columns is not None, "a full-width shard read during stream construction"
        assert set(columns) <= {"trace_id", "sched_policy", "slice_id"}


def test_manifest_row_counts_agree_with_shards(corpus: Path):
    manifest = pd.read_csv(corpus / "coloran.manifest.csv")
    total = sum(
        len(C.load_shards(corpus, "coloran", [s], columns=["trace_id"])) for s in manifest.shard
    )
    assert total == manifest.n_rows.sum()


def test_shared_scenario_names_do_not_collide_across_datasets(corpus: Path):
    """A scenario name shared by both testbeds must still give disjoint environments.

    Both corpora contain a 'rome_static_medium', so grouping on scenario alone produced
    the same spec twice and two environments claimed one set of traces.
    """
    stream = build_stream(
        _cfg(source=["coloran", "commag"], context_axes=["scenario"], eval_fraction=0.25),
        corpus,
    )
    seen: set[str] = set()
    for env in stream:
        traces = set(env.train_traces) | set(env.eval_traces)
        assert seen.isdisjoint(traces)
        seen |= traces
        assert "dataset" in env.context


def test_sched_shift_style_config_builds(corpus: Path):
    """The exact shape of configs/stream/sched_shift.yaml: a row axis, two sources."""
    stream = build_stream(
        _cfg(
            source=["coloran", "commag"],
            context_axes=["sched_policy"],
            n_environments=6,
            eval_fraction=0.25,
        ),
        corpus,
    )
    assert len(stream) >= 2
    ids = [e.env_id for e in stream]
    assert len(ids) == len(set(ids)), f"duplicate environment ids: {ids}"
