"""Metric correctness on hand-computed evaluation matrices."""

import numpy as np
import pytest

from nestedric.eval.metrics import (
    adaptation_latency,
    average_performance,
    backward_transfer,
    forgetting_measure,
    per_environment_bwt,
)


def test_bwt_zero_for_identical_rows():
    """A learner that never changes cannot forget."""
    R = np.tile([-1.0, -2.0, -3.0], (3, 1))
    assert backward_transfer(R) == pytest.approx(0.0)


def test_bwt_negative_when_earlier_environments_degrade():
    R = np.array([[-1.0, -9.0], [-4.0, -2.0]])  # env0 went -1 -> -4 after training env1
    assert backward_transfer(R) == pytest.approx(-3.0)


def test_bwt_positive_when_later_training_helps():
    R = np.array([[-4.0, -9.0], [-1.0, -2.0]])
    assert backward_transfer(R) == pytest.approx(3.0)


def test_per_environment_bwt_matches_manual_example():
    R = np.array([[-1.0, -9.0, -9.0], [-2.0, -2.0, -9.0], [-5.0, -4.0, -3.0]])
    assert per_environment_bwt(R).tolist() == pytest.approx([-4.0, -2.0])
    assert backward_transfer(R) == pytest.approx(-3.0)


def test_average_performance_uses_the_final_row():
    R = np.array([[-1.0, -1.0], [-2.0, -4.0]])
    assert average_performance(R) == pytest.approx(-3.0)


def test_forgetting_matches_manual_example():
    R = np.array([[-1.0, -9.0], [-5.0, -2.0]])
    # best ever on env0 is -1, final is -5, so forgetting is 4
    assert forgetting_measure(R) == pytest.approx(4.0)


def test_adaptation_latency_monotonic():
    curve = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])
    assert adaptation_latency(curve, target=-3.0) == 2
    assert adaptation_latency(curve, target=-1.0) == 4
    assert adaptation_latency(curve, target=0.0) == float("inf")


def test_single_environment_has_no_backward_transfer():
    R = np.array([[-1.0]])
    assert backward_transfer(R) == 0.0
    assert forgetting_measure(R) == 0.0


def test_non_square_matrix_raises():
    with pytest.raises(ValueError, match="square"):
        backward_transfer(np.zeros((2, 3)))


def test_small_positive_bwt_is_not_treated_as_divergence():
    """On a stream with no forgetting, slight positive transfer is the right answer.

    A 0.01 threshold excluded nestedric from radio-shift and sched-shift in the main
    benchmark -- the two streams where it performs best -- while finetune itself scored
    +0.0012 on radio-shift. The guard exists to catch +5.90 from a diverged run, which
    is three orders of magnitude away.
    """
    import numpy as np

    from nestedric.eval.evaluator import ContinualEvaluator

    class _Stream:
        environments = [type("E", (), {"env_id": "a"}), type("E", (), {"env_id": "b"})]

        def __iter__(self):
            return iter(self.environments)

        def __len__(self):
            return 2

    ev = ContinualEvaluator(_Stream(), {})
    ev.R = np.array([[-0.070, -0.070], [-0.068, -0.068]])  # BWT = +0.002
    flags = ev.sanity()
    assert flags["positive_bwt"] is True
    assert flags["implausible_bwt"] is False
    assert flags["trustworthy"] is True, "a null stream must not be discarded"


def test_evaluator_sanity_flags_a_diverged_run():
    """A results file should say whether it can be believed."""
    import numpy as np

    from nestedric.eval.evaluator import ContinualEvaluator

    class _Stream:
        environments = [type("E", (), {"env_id": "a"}), type("E", (), {"env_id": "b"})]

        def __iter__(self):
            return iter(self.environments)

        def __len__(self):
            return 2

    ev = ContinualEvaluator(_Stream(), {})
    ev.R = np.array([[-60.0, -60.0], [-0.1, -0.1]])  # the sched-shift-commag shape
    flags = ev.sanity()
    assert flags["implausible_bwt"] is True
    assert flags["implausible_performance"] is True
    assert flags["trustworthy"] is False

    ev.R = np.array([[-0.02, -0.5], [-0.06, -0.34]])  # a healthy run
    assert ev.sanity()["trustworthy"] is True


def test_divergence_limit_scales_with_target_variance():
    """A fixed limit assumed unit-variance targets, which injected drift breaks.

    At drift magnitude 4 the perturbation carries variance ~16, so a mean predictor
    scores ~17 and an early transient of 112 is ordinary. The absolute limit of 100
    killed a sweep twelve cells from the end for that reason.
    """
    import torch

    from nestedric.engine.trainer import DIVERGENCE_MULTIPLE, MIN_DIVERGENCE_LIMIT, _target_scale

    wide = [(None, torch.randn(64, 2) * 4.0, None)]
    tight = [(None, torch.randn(64, 2) * 1.0, None)]

    wide_scale = _target_scale(wide)
    tight_scale = _target_scale(tight)
    assert wide_scale > 5 * tight_scale

    wide_limit = max(MIN_DIVERGENCE_LIMIT, DIVERGENCE_MULTIPLE * wide_scale)
    assert wide_limit > 112.0, "the drift cell that crashed the sweep must now pass"


def test_divergence_limit_has_a_floor():
    """A near-constant-target environment must not make every nonzero loss a failure."""
    import torch

    from nestedric.engine.trainer import DIVERGENCE_MULTIPLE, MIN_DIVERGENCE_LIMIT, _target_scale

    scale = _target_scale([(None, torch.zeros(64, 2), None)])
    assert max(MIN_DIVERGENCE_LIMIT, DIVERGENCE_MULTIPLE * scale) >= MIN_DIVERGENCE_LIMIT
