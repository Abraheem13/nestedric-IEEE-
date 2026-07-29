"""Schema conformance tests -- every adapter must pass these."""

import pytest


@pytest.mark.parametrize("adapter", ["coloran", "commag", "tractor"])
def test_adapter_emits_canonical_schema(adapter):
    pytest.skip("Day 1")


def test_units_are_documented():
    pytest.skip("Day 1")
