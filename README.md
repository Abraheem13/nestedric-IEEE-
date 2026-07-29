# NestedRIC

**Multi-Timescale Nested Learning for Forgetting-Robust O-RAN Control — A Framework, a Continual-Learning Benchmark, and a Frequency-Separation Bound on Catastrophic Forgetting**

Target venue: *IEEE Transactions on Mobile Computing* (TMC).
Status: **Day 0 — scaffold complete, implementation begins Day 1.**

---

## Thesis

Mapping the O-RAN near-RT and non-RT RIC control loops onto the nested optimisation
levels of the Nested Learning paradigm — with a Continuum Memory System whose update
frequency is tiered to the RIC timescales — yields provably lower catastrophic
forgetting under non-stationary traffic than any single-timescale continual-learning
xApp.

Three contributions:

1. **NestedRIC** — a named framework mapping RIC control loops to nested optimisation
   levels, with a frequency-tiered associative memory and a self-modifying fast level.
2. **O-RAN-CL** — a public continual-learning benchmark over real O-RAN KPI traces,
   with five stream families, a fixed metric suite (BWT / FWT / forgetting /
   adaptation latency / near-RT footprint) and a fold-level statistical protocol.
3. **A frequency-separation forgetting bound** — an upper bound on backward-transfer
   degradation that decreases in the timescale-separation ratio τ_s/τ_f and recovers
   naive fine-tuning as the degenerate single-timescale case, plus a proposition
   characterising the risk-optimal ratio as a function of the drift rate.

## Why this is open

Nested Learning (Behrouz, Razaviyayn, Zhong, Mirrokni, NeurIPS 2025) has, as of July
2026, no wireless / RAN / mobile application. The closest work, FedNL, applies nested
optimisation to federated **LLM training** on generic edge devices — not radio, RIC,
spectrum or mobility. Existing wireless continual learning uses off-the-shelf EWC /
SI / replay / LwF or transfer learning; none uses a true multi-timescale nested
formulation anchored to physical control-loop periods.

See `docs/RELATED.md` for the full positioning table.

## Repository layout

```
nestedric/
├── configs/           # every experiment is a YAML config, nothing hard-coded
│   ├── data/          #   dataset preparation
│   ├── stream/        #   the five O-RAN-CL stream families
│   ├── model/         #   shared backbone (identical across all methods)
│   ├── method/        #   9 baselines + nestedric
│   └── experiment/    #   smoke, main, ablation, drift sweep
├── src/nestedric/
│   ├── data/          # dataset adapters -> one canonical KPI schema; stream builder
│   ├── models/        # backbone, continuum memory, deep optimizer, nested learner
│   ├── methods/       # all CL methods behind a single Method protocol
│   ├── eval/          # T×T evaluation matrix, CL metrics, near-RT footprint
│   ├── theory/        # frequency-separation bound + validation simulator
│   ├── engine/        # continual trainer, runner, registry
│   └── utils/         # config, seeding, logging, cluster-robust statistics
├── benchmark/oran_cl/ # the released benchmark spec + frozen splits
├── scripts/           # download, main benchmark, ablations, figures
├── docs/              # PLAN, DATASETS, BENCHMARK, THEORY, RELATED, EXPERIMENT_LOG
├── tests/             # schema, metrics, stream, bound, method smoke tests
└── paper/             # figures and tables destined for the manuscript
```

## Quickstart

```bash
make setup                  # editable install + pre-commit
bash scripts/gpu_check.sh   # confirm the L4 is visible
make data                   # download + harmonise all datasets
make smoke                  # 5-minute end-to-end sanity run
make main                   # full benchmark (see docs/PLAN.md for wall-clock)
make ablate
make figures
```

## Design rules (non-negotiable)

- **One backbone for all methods.** Any performance difference must come from the
  learning rule, not model capacity. Enforced in `configs/model/base.yaml`.
- **Byte-matched memory.** The replay buffer and the continuum memory are matched in
  bytes, so NestedRIC cannot win by simply storing more.
- **The fold is the unit of analysis.** Environments within a dataset are correlated;
  treating samples as independent inflates significance by orders of magnitude. All
  inference is at stream/fold level with paired bootstrap CIs and Holm–Bonferroni.
- **Near-RT feasibility is a reported metric, not an afterthought.** A method that
  cannot infer inside the near-RT budget is not a deployable xApp, and we say so.
- **Negative results ship.** If frequency separation gives no advantage below a drift
  threshold, that becomes the paper's finding, stated with the bound that explains it.
  The benchmark is released either way.

## Data

All sources are public. See `docs/DATASETS.md` for exact links, licences, pinned
commits and the harmonisation map.

## Licence

Apache-2.0. See `LICENSE`.
