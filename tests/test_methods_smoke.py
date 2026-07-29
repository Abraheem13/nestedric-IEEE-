"""Smoke tests: every registered method runs two tiny environments end to end."""

import pytest


@pytest.mark.parametrize(
    "method",
    ["finetune", "ewc", "si", "replay", "agem", "lwf", "bilevel", "titans", "joint", "nestedric"],
)
def test_method_smoke(method):
    pytest.skip("Day 3-7")
