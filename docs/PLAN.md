# 15-Day Implementation Plan

Hardware: one NVIDIA L4 (24 GB) + MacBook Air M4 (16 GB) for authoring and light runs.
Every day has a **deliverable** and a **gate**. If a gate fails, the contingency in the
same row applies — do not carry a broken foundation forward.

---

## Week 1 — data, harness, baselines

| Day | Work | Deliverable | Gate |
|---|---|---|---|
| **1** | Environment setup. Download ColO-RAN, COMMAG, TRACTOR. Inspect raw schemas. Write `data/schema.py` and the three adapters' `load_raw` + `to_canonical`. | `data/processed/*.parquet` for all three sources; `docs/DATASETS.md` filled with pinned commits, licences, column maps. | All three adapters pass `tests/test_schema.py`. Row counts and KPI ranges sane. |
| **2** | `data/stream.py`: the five stream families. `data/loaders.py`, `utils/config.py`, `utils/seeding.py`, `cli.py`. | `nestedric stream --config configs/stream/*.yaml` prints environment tables. | No train/eval leakage; streams deterministic given seed (`tests/test_stream.py`). |
| **3** | `models/backbone.py`. `engine/trainer.py` + `engine/runner.py`. Methods: `finetune`, `joint`, `ewc`, `si`, `replay`. | `make smoke` completes end to end. | Smoke run produces a valid T×T matrix. `joint` beats `finetune` on average performance. |
| **4** | Methods: `agem`, `lwf`, `bilevel`, `titans`. `eval/metrics.py`, `eval/evaluator.py`, `eval/footprint.py`. | Full metric suite computed for all 9 baselines on one stream. | `tests/test_metrics.py` passes on hand-computed matrices. **Critical gate: is there measurable forgetting on public traces?** If |BWT| for `finetune` is negligible on every stream, jump to Day 10 drift injection now and re-plan. |
| **5** | `models/cms.py`: associative memory block + frequency-tiered stack. | Continuum memory trains standalone on one environment. | Memory read/write is differentiable; byte footprint matches the replay buffer. |
| **6** | `models/nested.py` levels + routing; `models/deep_optimizer.py`. | Two-level NestedRIC runs one environment. | Level-1 updates fire exactly every τ_s steps (assert in test). |
| **7** | Self-modification (slow level parameterises the fast level's update rule). Wire `methods/nestedric.py`. Tune on the validation stream only. | NestedRIC completes a full stream. | Beats `finetune` on at least one stream. If not, freeze self-modification off and proceed with the two-level + CMS variant. |

**Week 1 buffer:** Day 7 evening. If behind, drop `bilevel` and `agem` to Week 2 —
`finetune`, `ewc`, `replay`, `titans` are the four baselines the paper cannot lose.

---

## Week 2 — experiments, theory, write-up

| Day | Work | Deliverable | Gate |
|---|---|---|---|
| **8** | Main benchmark part 1: all methods × slice-shift, traffic-shift, sched-shift × 5 seeds. Log footprint throughout. | `results/runs/main/` populated for 3 streams. | Runs complete without OOM; per-run time on budget. |
| **9** | Main benchmark part 2: cross-dataset + cyclic streams. First full results table. | Complete `results/tables/main.csv`. | NestedRIC's ranking is stable across seeds (not a seed artefact). |
| **10** | Ablations: `n_levels` {1,2,3}; period ratio {1,4,16,32,64,128}; memory capacity; self-modification on/off; deep optimizer on/off. Drift-rate estimation + controlled drift sweep. | `results/runs/ablation/`, `results/runs/drift_sweep/`. | **Key scientific gate:** does performance vary monotonically-ish with the separation ratio? This is the empirical content of the theorem. |
| **11** | Prove the frequency-separation bound. Implement `theory/bound.py` + `theory/simulate.py`. Fit constants; overlay bound on the Day-10 sweep. | Written proof (in `docs/THEORY.md`) + `tests/test_bound.py` passing. | Measured \|BWT\| lies under the bound; bound degenerates to fine-tuning at ratio 1. |
| **12** | Reproducibility: seed sweeps, cluster-robust CIs, Holm–Bonferroni, effect sizes. Freeze `benchmark/oran_cl/splits/`. | `utils/stats.py` complete; significance table. | Every headline claim has a CI and a corrected p-value at fold level. |
| **13** | All figures and tables. Related-work positioning table. | `paper/figures/`, `paper/tables/`. | Figures readable at print size; every number traceable to a run directory. |
| **14** | Manuscript draft: intro, method, theory, experiments, discussion, limitations. | Full draft. | Every claim maps to a figure, table or proposition. |
| **15** | Polish, artefact release prep (README, CITATION, licence audit, split freeze), final proofread. | Submission-ready manuscript + public repo. | A fresh clone reproduces the smoke run and one main-benchmark cell. |

**Day 16 (reserve):** buffer for whichever gate slipped.

---

## Standing rules

- **Log everything in `docs/EXPERIMENT_LOG.md` the day it happens.** Results that
  aren't logged the same day get misremembered.
- **Never tune on a test stream.** Hyper-parameters are selected on a held-out
  validation stream, stated explicitly in the paper.
- **Commit at the end of every day** with the day number in the message.
- **If a result surprises you, suspect the code first.** Budget half a day for a bug
  hunt before believing any large improvement.

## Contingency ladder

1. **No forgetting on public traces** (Day 4 gate fails) → move drift injection forward
   to Day 5; the paper becomes "how much drift is required before multi-timescale
   nesting pays", which is a sharper and still novel question.
2. **NestedRIC ties the baselines** (Day 9) → pivot to the negative-result framing:
   *frequency separation does not help below drift threshold δ\*, and here is the bound
   that explains why*. The benchmark and the theorem still carry the paper.
3. **Bound resists proof** (Day 11) → present the two-level case only, state the
   L-level version as a conjecture with the simulation supporting it, and be explicit
   that it is a conjecture.
4. **Time runs out** (Day 14) → cut the cyclic stream and the three-level ablation
   first; they are the least load-bearing.
