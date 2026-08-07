# When Does Multi-Timescale Learning Help an O-RAN xApp? A Continual-Learning Benchmark and a Negative Result

**Target venue:** IEEE Transactions on Mobile Computing
**Status:** draft. Every number below comes from `results/tables/`; nothing is typed by hand.
**Open item:** see §7.1 — one measurement is outstanding and one claim depends on it.

---

## Abstract (draft)

Machine-learning xApps for the O-RAN near-real-time RIC are trained on one traffic and
radio regime and are expected to degrade when it shifts. The Nested Learning paradigm
suggests a remedy: map the near-RT and non-RT control loops onto nested optimisation
levels with a frequency-tiered associative memory, so that a slow level retains what a
fast level overwrites. We test that proposal on 65 million rows of real O-RAN KPI traces
from two Colosseum testbeds, and report three findings, two of them negative.

First, we release **O-RAN-CL**, a continual-learning benchmark over public O-RAN traces
with five environment streams, frozen trace-level splits, byte-matched memory budgets and
a fold-level statistical protocol.

Second, we find that **forgetting in O-RAN KPI models is caused by changes to the
resource-allocation regime, not by radio conditions or scheduling discipline.** Naive
fine-tuning loses 0.024 in backward transfer when the PRB budget changes across testbeds
and 0.010 when the RBG allocation changes, but 0.002 or less when the scheduling policy
changes and nothing measurable when UE distance and mobility change. This inverts the
intuition that motivated our own experimental design.

Third, **frequency separation does not reduce forgetting on these traces.** Sweeping the
separation ratio from 1 to 128 moves backward transfer by 0.003 with every confidence
interval overlapping; the degenerate single-timescale case performs as well as any
separation. Under synthetic concept drift the mechanism helps only inside a window
(δ = 0.25–0.5) and is harmful outside it on both sides. A byte-matched reservoir buffer
matches or beats the nested learner on every stream at a third of its inference latency.

We also state and test a frequency-separation bound. Its qualitative prediction — that
the risk-optimal ratio decreases with drift — holds, but the predicted δ^(−1/2) scaling
is too shallow by roughly a factor of four, and we report the proposition as refuted in
its stated form.

---

## 1. Introduction

*Motivation from the RIC control-loop timescales; the Nested Learning proposal; the gap
(no wireless application as of July 2026, verified by citation sweep; FedNL is the
nearest adjacent work and applies nested optimisation to federated LLM training, not to
radio, RIC, spectrum or mobility).*

**Contributions.**

1. O-RAN-CL, a public continual-learning benchmark over real O-RAN KPI traces (§3).
2. A measurement of which O-RAN reconfigurations cause catastrophic forgetting and which
   do not, with two clean nulls and a mechanism (§5.1).
3. A negative result on frequency separation, established by a ratio sweep, a drift
   sweep and an out-of-sample test of the accompanying theory (§5.2–5.4).

**A note on framing.** The three contributions were designed as mutual insurance: the
benchmark and the theory were to carry the paper if the empirical result disappointed.
It did. We report the negative result as the primary empirical finding rather than as a
limitation, because "this fashionable mechanism does not help here, and here is the
regime where it would" is more useful to a practitioner than a marginal win would be.

## 2. Related work

*Positioning table from `docs/RELATED.md`. Every row to be verified against the PDF
before submission; re-run the Google Scholar "cited by" sweep on arXiv:2512.24695 and
cite any concurrent wireless application explicitly.*

## 3. O-RAN-CL: the benchmark

**Sources.** Colosseum ColO-RAN (Polese et al., IEEE TMC 22(10):5787–5800, 2022) and
Colosseum COMMAG (Bonati et al., IEEE Commun. Mag. 59(10):21–27, 2021). Both GPL-3.0;
we redistribute code and split indices only, never data or derived parquet.

**Scale.** After preparation: ColO-RAN 35,509,885 rows / 18,241 traces in 84 shards;
COMMAG 29,764,320 rows / 20,623 traces in 104 shards. Both export the same 31-column
slice-metrics header at a 250 ms sampling period, so one adapter serves both and the
cross-dataset stream needs no harmonisation guesswork.

**Data quality (§3.2).** Three corruptions found by profiling and reported as a paper
artefact: `dl_buffer_bytes` carries INT32_MIN where the report is unavailable; `dl_mcs`
reaches 2.42×10⁸; `sum_requested_prbs` reaches −501. All masked, never clipped — a
clipped sentinel is still a fabricated observation. Masked fractions are published per
column. `sum_requested_prbs` is excluded as a *feature* because its missingness differs
by 10.2 percentage points between testbeds (0.00% vs 10.19%) and would let a model
identify the testbed from the missingness pattern alone.

**Streams.** Five families, each a deterministic function of a seed, cut on context axes
recovered from trace paths. Environments hold 240 traces each, split 192/48 at the
**trace** level: rows 250 ms apart within a trace are near-duplicates, so a row-level
split would report a generalisation gap that does not exist.

**Protocol.** One backbone for every method (190,213 parameters). Memory budgets matched
in bytes, not in nominal capacity — configured as "5,000 windows" and "512 slots" these
differ by 45×. The unit of analysis is the fold, never the sample; the design effect from
intra-trace correlation is roughly two orders of magnitude. Paired bootstrap CIs over
folds, paired permutation p-values, Holm–Bonferroni across methods.

## 4. NestedRIC

*Continuum Memory System: a stack of associative-memory blocks with strictly increasing
update periods; gated blending into similarity-selected slots, with slower levels
blending more gently; soft-attention reads across all levels with learned per-level
gates. Level-scheduled optimiser with gradient accumulation between firings. Optional
self-modification: the slow level emits a scalar gain on the fast level's step size.*

**Design constraint.** The frequency separation lives in the memory, not in the backbone
weights: 98.2% of parameters train at level 0, exactly as in every baseline. An earlier
variant placed early encoder layers on slow periods, which meant 27% of parameters —
including the layer reading the raw KPI window — took one Adam step per 32. That
underfits rather than retains, and it confounds "tiered memory" with "partly frozen
network". It is retained as an ablation axis (§5.5), not as the method.

## 5. Experiments

### 5.1 Which O-RAN reconfigurations cause forgetting

Backward transfer under naive fine-tuning, five seeds (Fig. 1):

| stream | what changes | BWT |
|---|---|---|
| cross-dataset | PRB budget: 50 → 15 PRB across testbeds | **−0.0239** |
| slice-shift | RBG allocation (tr config) | **−0.0103** |
| cyclic | environments recur | −0.0087 |
| sched-shift | scheduling discipline (RR/WF/PF) | −0.0003 |
| radio-shift | distance (20/50/100 m), mobility (static/slow) | +0.0017 |

Both sources of forgetting change the **resource-allocation regime**. Neither non-source
does: the scheduler changes *which* UE is served and distance and mobility change channel
quality, but neither rewrites the mapping the model learned.

This contradicts the expectation that motivated our stream design, which promoted COMMAG
to a primary source precisely for its mobility and distance axes. It also survived a
control: an early version of `sched-shift` drew on both testbeds, placing a
ColO-RAN→COMMAG boundary inside it, and reported −0.0428. Within-testbed controls give
−0.0021 (ColO-RAN) and +0.0006 (COMMAG). The apparent scheduling effect was the testbed
switch.

### 5.2 Methods at a matched budget

Backward transfer on the stream with the most forgetting, all memory methods at 4 MB
(Fig. 2):

| method | BWT | memory | p99 latency |
|---|---|---|---|
| lwf | −0.0005 | 0.76 MB | 0.89 ms |
| agem | −0.0014 | 4.00 MB | 1.02 ms |
| replay | −0.0061 | 4.00 MB | 0.90 ms |
| si | −0.0061 | 3.04 MB | 0.90 ms |
| **nestedric** | **−0.0097** | 4.00 MB | 2.12 ms |
| ewc | −0.0179 | 1.52 MB | 0.95 ms |
| titans | −0.0226 | 4.00 MB | 0.90 ms |
| finetune | −0.0239 | 0 | 0.90 ms |
| bilevel | −0.0257 | 0.76 MB | 0.90 ms |

NestedRIC ranks fifth of nine continual-learning methods. Learning without Forgetting
beats it with a fifth of the memory and 42% of the latency. Every method is near-RT
feasible (p99 well inside a 10 ms budget), so the comparison is about retention, not
deployability.

With five seeds the smallest attainable paired-permutation p-value is 0.065, so **no
comparison in this table reaches significance**; the ordering and the intervals are what
the table supports.

### 5.3 The separation ratio has no effect

Sweeping ρ on cross-dataset, three seeds:

| ρ | 1 | 4 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| BWT | −0.0088 | −0.0096 | −0.0099 | −0.0103 | −0.0105 | −0.0073 |

The sweep spans 0.003 across a 128-fold change in ρ; every interval covers every other.
ρ = 1 is the degenerate single-timescale case the theory identifies as worst, and it
performs as well as any separation. `n_levels`, `self_modifying`, `deep_optimizer` and
the level assignment are null on the same data.

### 5.4 Under controlled drift, separation helps only in a window

Injecting concept drift of controlled magnitude (seven seeds; Fig. 3). Manipulation
check: naive fine-tuning's BWT moves from −0.024 to −4.72 across the range, so the
injection reaches the model.

| δ | ρ=1 | ρ=32 | difference | 95% CI | p |
|---|---|---|---|---|---|
| 0.00 | −0.0097 | −0.0120 | −0.0023 | [−0.0028, −0.0018] | 0.016 |
| 0.25 | −0.1855 | −0.1774 | +0.0081 | [+0.0033, +0.0126] | 0.049 |
| 0.50 | −1.0622 | −0.9815 | +0.0806 | [+0.0367, +0.1297] | 0.032 |
| 1.00 | −4.3476 | −4.6374 | −0.2898 | [−0.4682, −0.0995] | 0.048 |

Separation pays between δ = 0.25 and 0.5 and is harmful on both sides — significantly so
at zero drift, the regime the public traces occupy, though by a practically negligible
0.0023. Replay beats the nested learner at every magnitude, and the gap widens with
drift.

*Caveat: the injected shift is a random linear perturbation of the input-to-target
mapping. The window characterises that family, not every real shift, and every table
using it is labelled synthetic.*

### 5.5 The bound, tested out of sample

Constants fitted on ρ ∈ {1, 32} only; ratios 4, 8, 16 held out (Fig. 4):

| δ | predicted ρ* | measured best ρ | |
|---|---|---|---|
| 0.25 | 10.3 | 8 | confirmed |
| 1.00 | 5.2 | 1 | missed |

Proposition 2 is **refuted** by the rule fixed before the numbers were read. Its
qualitative content survives — ρ* decreases with drift — but ρ* = √(C_r/(C_a δ)) predicts
a factor-2 decrease across this range where the data show a factor of 8.

## 6. Discussion

*What a practitioner should take from this: do not reach for multi-timescale machinery
for the shifts real O-RAN testbeds exhibit; a 4 MB reservoir buffer, or distillation at
0.76 MB, is both cheaper and better. If the allocation regime changes and drift is large,
there is a regime where separation pays, and a larger one where it does not.*

## 7. Limitations

- Two testbeds from one lineage; TRACTOR was not incorporated.
- Five seeds put the permutation floor at 0.065, above conventional significance. Effect
  sizes and intervals are reported throughout; none of the main-table comparisons is
  significant, and we do not claim otherwise.
- The control task uses a derived action label (increase/hold/decrease from the next
  change in granted PRBs) because no action is logged in these traces.
- The drift injection is synthetic and of one functional family.

### 7.1 Outstanding measurement

NestedRIC reports large *positive* backward transfer on the two null streams (+0.0535 on
radio-shift, +0.0433 on sched-shift) where every other method sits within ±0.002. This is
either genuine positive transfer or an artefact of a model recovering from a poor start,
and the two are distinguished by its average performance on those streams, which has not
yet been extracted. **Fig. 2 marks these as off-scale rather than as wins, and no claim
rests on them until this is settled.**

Command: `scripts/make_table.py --dir results/runs/main --metric avg_perf`.

## 8. Reproducibility

Code and split indices are Apache-2.0 at the repository; data is not redistributed.
A fresh clone plus `pip install -e ".[dev]"` runs 208 tests to green. Every figure is
generated from a results table by `scripts/make_figures.py`, so a figure cannot disagree
with the text. `docs/EXPERIMENT_LOG.md` records every measurement in order, including
the bugs found and the two occasions on which a reported result was withdrawn.
