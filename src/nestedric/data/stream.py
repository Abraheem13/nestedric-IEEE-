"""Environment-stream construction.

An O-RAN-CL *stream* is an ordered list of environments; each environment is a
contiguous slice of harmonised KPI records sharing a context signature (slice mix,
traffic profile, scheduler, mobility regime, UE count). The learner sees them
sequentially and is evaluated on all previously seen environments.

Status: STUB -- implemented on Day 2 of the 15-day plan (see docs/PLAN.md).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field


@dataclass
class Environment:
    """One task in the continual stream."""

    env_id: str
    dataset: str
    context: dict
    train_idx: Sequence[int]
    eval_idx: Sequence[int]
    meta: dict = field(default_factory=dict)


@dataclass
class EnvironmentStream:
    """An ordered sequence of :class:`Environment` objects plus stream-level metadata."""

    name: str
    environments: list[Environment]
    drift_schedule: dict = field(default_factory=dict)

    def __iter__(self) -> Iterator[Environment]:
        return iter(self.environments)

    def __len__(self) -> int:
        return len(self.environments)


def build_stream(cfg: dict) -> EnvironmentStream:
    """Materialise a stream from a ``configs/stream/*.yaml`` config.

    Supported stream families (see docs/BENCHMARK.md):
      * ``slice-shift``   : slice mix changes between environments (ColO-RAN).
      * ``traffic-shift`` : traffic profile changes (TRACTOR classes).
      * ``sched-shift``   : scheduling policy changes (ColO-RAN/COMMAG).
      * ``cross-dataset`` : environments drawn from different datasets (hardest).
      * ``cyclic``        : environments recur, to measure genuine retention.
    """
    raise NotImplementedError("Day 2")
