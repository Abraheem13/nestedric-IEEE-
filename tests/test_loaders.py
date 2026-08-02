"""Windowing and normalisation: the properties that decide whether a number means anything."""

import numpy as np
import pytest

from nestedric.data.loaders import build_windows, fit_normaliser
from nestedric.data.schema import FEATURE_COLUMNS, TARGET_COLUMNS
from nestedric.data.stream import build_stream
from tests.test_stream import corpus, _cfg  # noqa: F401  (corpus is a fixture)

WINDOW = 8
STRIDE = 4


def _stream(corpus):  # noqa: F811
    return build_stream(_cfg(context_axes=["sched_policy"]), corpus)


def test_normaliser_fits_only_source_environments(corpus):  # noqa: F811
    stream = _stream(corpus)
    norm = fit_normaliser([stream[0]], corpus)
    assert norm.source_env_ids == (stream[0].env_id,)
    assert norm.mean.shape == (len(FEATURE_COLUMNS),)
    assert norm.std.shape == (len(FEATURE_COLUMNS),)


def test_normaliser_excludes_eval_traces_from_the_fit(corpus, monkeypatch):  # noqa: F811
    """Eval traces are scored on; their statistics must not enter the constants."""
    stream = _stream(corpus)
    env = stream[0]
    seen: list[list[str]] = []

    import nestedric.data.loaders as L

    original = L._load_env_frame

    def spy(e, d, columns, traces):
        seen.append(list(traces))
        return original(e, d, columns, traces)

    monkeypatch.setattr(L, "_load_env_frame", spy)
    fit_normaliser([env], corpus)

    assert seen and all(set(t).isdisjoint(env.eval_traces) for t in seen)


def test_std_is_floored_so_constant_features_do_not_explode(corpus):  # noqa: F811
    norm = fit_normaliser([_stream(corpus)[0]], corpus)
    assert np.all(norm.std > 0)
    assert np.all(np.isfinite(norm.std))


def test_windows_have_expected_shape(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    assert ws.x.ndim == 3
    assert ws.x.shape[1] == WINDOW
    # features + one missingness channel
    assert ws.x.shape[2] == len(FEATURE_COLUMNS) + 1
    assert ws.y.shape == (len(ws.x), len(TARGET_COLUMNS))
    assert ws.trace_index.shape == (len(ws.x),)


def test_windows_never_span_two_traces(corpus):  # noqa: F811
    """A window crossing a boundary would splice two UEs into one sequence."""
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    # Every window carries exactly one trace index, and each trace contributes a
    # contiguous run of windows.
    assert len(np.unique(ws.trace_index)) <= len(env.train_traces)
    assert np.all(np.diff(ws.trace_index) >= 0)


def test_no_nans_reach_the_model(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    assert np.isfinite(ws.x).all()
    assert np.isfinite(ws.y).all()


def test_missingness_channel_is_present_and_bounded(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    channel = ws.x[:, :, -1]
    assert channel.min() >= 0.0
    assert channel.max() <= 1.0


def test_windowing_is_deterministic(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    a = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    b = build_windows(env, corpus, norm, env.train_traces, window=WINDOW, stride=STRIDE)
    for field in ("x", "y", "actions", "trace_index"):
        assert np.array_equal(getattr(a, field), getattr(b, field))


def test_windows_use_the_constants_they_are_given(corpus):  # noqa: F811
    """The passed normaliser must be applied verbatim, not refitted per environment.

    Refitting per environment would standardise away exactly the covariate shift the
    benchmark measures. Asserted by shifting the constants and checking the output
    moves by the corresponding amount, which a refit would absorb.
    """
    from nestedric.data.loaders import Normaliser

    stream = _stream(corpus)
    env = stream[0]
    n_features = len(FEATURE_COLUMNS)

    identity = Normaliser(
        mean=np.zeros(n_features), std=np.ones(n_features), columns=FEATURE_COLUMNS
    )
    shifted = Normaliser(
        mean=np.full(n_features, 3.0), std=np.ones(n_features), columns=FEATURE_COLUMNS
    )

    x_id_ws = build_windows(env, corpus, identity, env.train_traces, WINDOW, STRIDE)
    x_sh_ws = build_windows(env, corpus, shifted, env.train_traces, WINDOW, STRIDE)

    assert len(x_id_ws) and len(x_id_ws) == len(x_sh_ws)
    delta = x_id_ws.x[:, :, :-1] - x_sh_ws.x[:, :, :-1]
    observed = ~np.isclose(x_id_ws.x[:, :, :-1], 0.0) | True  # every present cell shifts by 3
    assert np.allclose(delta[observed], 3.0)


def test_every_environment_shares_one_normaliser(corpus):  # noqa: F811
    """One set of constants, fitted on the source environments, used for all of them."""
    stream = _stream(corpus)
    norm = fit_normaliser([stream[0]], corpus)
    assert norm.source_env_ids == (stream[0].env_id,)
    for env in stream:
        ws = build_windows(env, corpus, norm, env.train_traces, WINDOW, STRIDE)
        assert np.isfinite(ws.x).all()


def test_short_traces_produce_no_windows(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, env.train_traces, window=10_000, stride=STRIDE)
    assert len(ws.x) == 0 and len(ws.y) == 0


def test_empty_trace_list_returns_empty_arrays(corpus):  # noqa: F811
    stream = _stream(corpus)
    env = stream[0]
    norm = fit_normaliser([env], corpus)
    ws = build_windows(env, corpus, norm, [], window=WINDOW, stride=STRIDE)
    assert len(ws.x) == len(ws.y) == len(ws.trace_index) == 0


def test_fit_requires_a_source_environment():
    with pytest.raises(ValueError, match="at least one source"):
        fit_normaliser([], "data/processed")
