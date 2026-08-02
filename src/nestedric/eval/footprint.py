"""Compute/memory accounting against near-RT RIC feasibility.

A method that cannot infer inside the near-RT control loop is not a deployable xApp,
whatever its BWT. Reporting this alongside accuracy is design rule 6, and it is the
metric most likely to constrain the proposed method: memory reads and self-modification
cost latency that regularisation baselines do not pay.

Latency is measured per *window*, batch size 1, because that is the shape of an
inference inside a control loop -- a batched throughput figure would flatter every
method equally and answer a question nobody deploying an xApp is asking.
"""

from __future__ import annotations

import numpy as np
import torch

#: O-RAN near-real-time RIC control loop: 10 ms to 1 s. We report against the tight end.
NEAR_RT_BUDGET_MS = 10.0


@torch.no_grad()
def measure_footprint(
    method,
    batch,
    device: str = "cuda",
    repeats: int = 100,
    warmup: int = 10,
) -> dict:
    """Return params, peak memory (MB), and p50/p99 single-window inference latency (ms)."""
    x = batch[0][:1].to(device)
    base = method.footprint()

    for _ in range(warmup):
        method.predict((x,))

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True) if device.startswith("cuda") else None
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            method.predict((x,))
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            method.predict((x,))
            timings.append((time.perf_counter() - t0) * 1000.0)

    timings = np.asarray(timings, dtype="float64")
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else float("nan")

    return {
        **base,
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p99_ms": float(np.percentile(timings, 99)),
        "latency_mean_ms": float(timings.mean()),
        "peak_memory_mb": float(peak_mb),
        "device": device,
    }


def near_rt_feasible(footprint: dict, budget_ms: float = NEAR_RT_BUDGET_MS) -> bool:
    """Whether the method could actually run inside a near-RT RIC control loop.

    Judged on p99 rather than the median: a control loop that misses its deadline one
    time in a hundred has missed it.
    """
    p99 = footprint.get("latency_p99_ms")
    return p99 is not None and np.isfinite(p99) and p99 <= budget_ms
