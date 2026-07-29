# O-RAN-CL: a continual-learning benchmark for O-RAN control

This is the released artefact. It must be usable by someone who never reads our paper —
that is what makes it a citation engine rather than a supplementary file.

## What the benchmark fixes

1. **Five stream families** (`configs/stream/`), each a deterministic function of a seed:
   - `slice-shift`    — slice mix changes (ColO-RAN). Mildest.
   - `traffic-shift`  — traffic class / application mix changes (TRACTOR).
   - `sched-shift`    — scheduling policy changes (ColO-RAN + COMMAG).
   - `cross-dataset`  — environments drawn from different sources. Hardest.
   - `cyclic`         — environments recur, so retention is measured, not inferred.
2. **Two tasks per environment**
   - *KPI forecasting* (regression): predict downlink throughput and a latency proxy.
   - *Control* (discrete): select a scheduling/slicing action — the xApp-shaped task.
3. **A fixed metric suite** (`src/nestedric/eval/metrics.py`), all computed from the
   T x T matrix R, where R[i, j] is performance on environment j after training through
   environment i:
   - average performance after the final environment
   - backward transfer (BWT)
   - forward transfer (FWT)
   - forgetting measure
   - **adaptation latency** — steps and seconds to recover target performance after a
     shift. This is the operationally meaningful one for a RIC.
   - **near-RT footprint** — params, peak memory, p50/p99 inference latency, and a
     boolean feasibility flag against a 10 ms budget.
4. **A statistical protocol.** The unit of analysis is the stream/fold, never the
   sample. Paired bootstrap CIs over folds, Holm-Bonferroni across methods, effect
   sizes on fold-level differences.
5. **Frozen splits** under `benchmark/oran_cl/splits/`, so numbers are comparable
   across papers.

## Leaderboard protocol

- Hyper-parameters are selected on a designated **validation stream** only. Never on a
  reported stream.
- Memory-based methods are **byte-matched**: replay buffer size and continuum memory
  capacity must be reported in bytes, and compared at equal budget.
- All methods share the backbone in `configs/model/base.yaml`. A submission that
  changes the backbone must report both its own and the standard backbone.
- Five seeds minimum. Report mean and between-fold standard deviation, not just the mean.

## Reference results

Populated on Day 9. Table lives at `results/tables/main.csv` and is mirrored into
`benchmark/oran_cl/REFERENCE.md` at release.
