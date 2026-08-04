# Experiment log

One entry per day. Written the same day, before closing the laptop. Record what was
run, what the numbers were, and what surprised you — especially the things that turned
out to be bugs.

Template:

## Day N - YYYY-MM-DD

**Planned:**
**Done:**
**Numbers:** (with run directory paths)
**Surprises / suspected bugs:**
**Decisions taken:**
**Tomorrow:**

---

## Day 0 - 2026-07-29

**Done:** Repository scaffold created: directory tree, packaging, configs for 5 stream
families and 10 methods, stubs for all modules with day assignments, 15-day plan,
dataset selection (ColO-RAN, COMMAG, TRACTOR), benchmark spec, theory notes.

**Decisions taken:**
- Three real public sources; synthetic traces only as a labelled contingency.
- One shared backbone across all methods; byte-matched memory budgets.
- Fold-level statistics from the start.

**Tomorrow:** Day 1 - download all three datasets, write the adapters, fill the
harmonisation table in docs/DATASETS.md.

---

## Day 1 - 2026-07-29

**Planned:** Download ColO-RAN, COMMAG, TRACTOR. Inspect raw schemas. Write
`data/schema.py` and the adapters. Fill the harmonisation table.

**Done:**
- Cloned and inspected ColO-RAN (9.0 GB, 39,887 files, 18,329 slice-metrics CSVs) and
  COMMAG (45,159 files, 21,612 slice-metrics CSVs) directly.
- **Both datasets share an identical 31-column slice-metrics header.** Wrote ONE
  adapter (`data/colosseum.py`) serving both, with `coloran.py` / `commag.py` as thin
  wrappers. This was not the plan and is strictly better: the cross-dataset stream
  needs no harmonisation guesswork.
- Wrote the real `data/schema.py` against verified columns: 6 index, 21 KPI,
  8 context columns, with units for every KPI.
- Wrote `scripts/prepare_data.py` (prepare + profile) and the real
  `scripts/download_data.sh` with verified file counts.
- 10 schema/adapter tests written and passing.

**Numbers:** `sched0/tr0` alone (206 files) -> **410,048 canonical rows, 206 traces**.
Full corpus is on the order of 3.6e7 rows per dataset.

Per-slice mean downlink throughput (sched0/tr0): eMBB 0.709, MTC 0.053, URLLC 0.105
Mbps. eMBB ~13x MTC, matching the documented traffic configuration.

**Surprises / suspected bugs:**
1. **Caught a real bug in my own first implementation.** `ratio_granted_req` was
   computed as `granted / max(requested, 1)`, giving values up to 11,424. Cause: PRB
   counters are ACCUMULATED over the 250 ms window (max ~11.4k, consistent with
   50 PRB x 250 subframes), and 42% of rows are idle with `requested == 0`. Fixed to
   leave those rows missing. Regression test added. Had this shipped, the feature
   would have been mostly noise and the model would have silently degraded.
2. Granted legitimately exceeds requested (minimum allocations), so the ratio is not
   bounded by 1 — median 0.96, p75 1.20 on non-idle rows.
3. `dl_pmi`, `dl_ri`, `ul_n` appear constant-zero. The profiler flags constant columns.
4. URLLC mean throughput sits above MTC, opposite to the raw packet rates. Probably
   explained by per-slice RBG allocation under tr0 (2/13/2) and UE counts. **Check on
   Day 2 before using slice_id as a feature.**

**Decisions taken:**
- One adapter for both Colosseum datasets.
- **COMMAG is promoted from secondary to primary.** It varies mobility
  (static/slow), distance (close/medium/far) and slice assignment (mixed/traffic) —
  genuine radio-condition shifts that ColO-RAN holds fixed. These are far better
  candidates for producing real catastrophic forgetting than slice-mix changes, so the
  Day 2 stream design should lead with them.
- **TRACTOR moved off the critical path.** ColO-RAN + COMMAG support all five stream
  families on their own. Add TRACTOR only if the Day 4 gate needs more drift.
- Licence: both datasets are GPL-3.0, repo is Apache-2.0. We do not vendor or
  redistribute data or derived parquet; the artefact is code + split indices.

**Tomorrow (Day 2):** Build `data/stream.py`. Lead with COMMAG mobility/distance as the
environment axis. Verify the URLLC/MTC ordering. Then loaders, config, seeding, CLI.

---

## Day 4 - 2026-08-02

**Planned:** Remaining baselines (agem, lwf, bilevel, titans), metrics, evaluator,
footprint. Critical gate: is there measurable forgetting on these traces at all?

**Done:** All nine baselines implemented and running. Near-RT footprint measured at
batch size 1. Byte-matched memory budgets. Gate run: 3 streams x 4 methods x 2 seeds.

**Numbers:** `results/runs/gate/`, summarised by `scripts/report_gate.py`.

finetune BWT (mean of 2 seeds):

| stream | BWT | seeds | verdict |
|---|---|---|---|
| radio-shift | -0.0006 | -0.0006, -0.0006 | negligible |
| sched-shift | **-0.0443** | -0.0432, -0.0455 | forgets |
| slice-shift | -0.0202 | -0.0270, -0.0133 | forgets (noisy) |

joint - finetune (average performance): radio +0.0005, sched +0.0364, slice +0.0165.
Oracle wins everywhere, so the loop is sound.

Forgetting is remediable: on sched-shift replay cuts |BWT| from 0.0443 to 0.0051 and
EWC to 0.0337 -- the ordering (replay > EWC) matches the CL literature.

Near-RT: every method p50 ~0.83 ms, p99 0.88-1.04 ms against the 10 ms budget. All
feasible. Extra state: replay 4.00 MB, EWC 1.52 MB, finetune/joint 0.

**Surprises / suspected bugs:**

1. **radio-shift does not produce forgetting, and it was supposed to be the best
   stream.** Day 1 promoted COMMAG to primary precisely because mobility, distance and
   slice assignment are "genuine radio-condition shifts... far better candidates for
   producing real catastrophic forgetting than slice-mix changes". Measured, it is the
   *weakest* of the three, by two orders of magnitude. That hypothesis was wrong.

2. The likely reason is a distinction the Day 1 argument did not make: distance and
   mobility change the *marginal* distribution of the KPIs (covariate shift), while the
   scheduling policy changes the *mapping* from KPIs to allocation behaviour (concept
   shift). Forgetting is a property of overwriting a learned mapping, so concept shift
   should dominate -- and does. This is checkable: estimate drift rate per stream. If
   radio-shift shows high input drift and near-zero BWT, then |BWT| is not a function of
   drift magnitude alone, which is a premise the frequency-separation bound rests on.

3. slice-shift's seeds differ by 2x (-0.027 vs -0.013). Two seeds is not enough there;
   the main benchmark's five may still be thin for that stream.

**Decisions taken:**
- **sched-shift becomes the primary evaluation stream**, replacing radio-shift.
- radio-shift is kept and reported. A stream where nothing forgets is evidence the
  benchmark discriminates, and the covariate/concept distinction is a finding.
- Before Day 10, run drift estimation across all three streams to test the explanation
  in (2). If it holds it belongs in the paper; if it fails, the bound needs the
  premise examined.

**Tomorrow (Day 5):** Continuum Memory System.

---

## Day 4 (revised) - 2026-08-03

**This supersedes the Day 4 entry above, which was wrong.** That entry reported
sched-shift as the strongest forgetting stream. It was confounded, and the runs behind
part of it were numerically broken. Both are documented here rather than edited away.

**Three faults found and fixed:**

1. *sched-shift was half a cross-dataset stream.* `source: [coloran, commag]` put a
   ColO-RAN -> COMMAG boundary in the middle. Per-transition transfer gap: 87.6 at that
   boundary against 0.47-4.11 everywhere else. Within-testbed controls give BWT -0.0021
   (ColO-RAN) and +0.0006 (COMMAG). Scheduling policy does not cause forgetting.

2. *A scaling failure diverged every COMMAG-heavy run.* `dl_buffer_bytes` spans 0-1e9;
   standardised against source-environment constants it reached 127.9 sigma with its p99
   also at 127.7 -- the whole distribution had moved. It is also a prediction target, so
   squared errors near 1.6e4 diverged the run (joint avg_perf -37.97, seed std 52.7,
   BWT +5.90). Fixed by log1p on heavy-tailed non-negative KPIs before standardisation.

3. *Nothing objected to any of it.* A diverged run wrote a normal-looking results.json
   and the summary labelled BWT +5.90 as "FORGETS", because the verdict tested |BWT|.
   Now: the trainer raises on non-finite or >100 loss, every result carries a
   trustworthiness block, and only negative BWT counts as forgetting.

**Numbers (all post-fix, 2 seeds each):**

| stream | what changes | finetune BWT | seeds | replay BWT |
|---|---|---|---|---|
| cross-dataset | testbed: 50 PRB -> 15 PRB | **-0.0266** | -0.0261, -0.0270 | -0.0053 |
| slice-shift | RBG allocation (tr config) | **-0.0207** | -0.0250, -0.0164 | -0.0023 |
| cyclic | environments recur | -0.0084 | -0.0085, -0.0082 | -0.0064 |
| sched-shift | scheduling discipline | -0.0021 | -0.0027, -0.0015 | -- |
| radio-shift | distance, mobility | +0.0012 | +0.0019, +0.0005 | -0.0040 |

joint beats finetune on every stream that forgets. Near-RT: all methods p50 ~0.83-0.89
ms against a 10 ms budget.

**The finding.** Forgetting appears when the **resource-allocation regime** changes --
the PRB budget (cross-dataset) or the RBG split (slice-shift). It does not appear when
the scheduling discipline changes or when radio conditions change. Day 1 predicted the
opposite, having promoted COMMAG precisely for its mobility and distance axes.

**The bar for NestedRIC.** Replay already removes 80% of cross-dataset forgetting
(-0.0266 -> -0.0053) at a 4 MB byte budget. NestedRIC must beat that at equal bytes.
This is a much harder target than beating finetune, and it is the honest comparison.

**Note on cyclic:** its low BWT is by construction -- environments recur, so early
environments were revisited recently. "Negligible" is not a null result there; the gate
script's framing does not fit that stream and should not be read as one.

**Tomorrow (Day 5):** Continuum Memory System.

---

## Day 7 - 2026-08-03

**Done:** NestedRIC wired end to end (CMS + nested levels + deep optimizer +
self-modification) and measured against every implemented baseline on the two streams
that forget.

**Numbers:** `results/runs/day7/`, 2 seeds. cross-dataset, all memory methods at 4 MB:

| method | mechanism | BWT | avg perf |
|---|---|---|---|
| finetune | -- | -0.0266 | -0.0698 |
| titans | memory, no tiering | -0.0263 | -0.0695 |
| bilevel | tiering, no memory | -0.0297 | -0.0728 |
| **nestedric** | both | **-0.0052** | -0.0738 |
| replay | reservoir buffer | -0.0053 | -0.0528 |

slice-shift: nestedric BWT +0.0054 (mildly positive backward transfer), replay -0.0023.

**The finding.** Neither component alone helps: titans and bilevel are
indistinguishable from finetune. Together they reduce |BWT| fivefold. The effect is an
interaction, which is the paper's claim -- separation needs something to separate, and
the memory needs the separation.

**What it does not show.** NestedRIC ties replay on retention (-0.0052 vs -0.0053) and
is worse on average performance (-0.0738 vs -0.0528) on both streams. No claim of
beating the state of the art is available from these numbers.

**Surprises / suspected bugs (two wrong diagnoses before the right one):**

1. First measurement had nestedric at avg_perf -0.0861, worse than finetune. I blamed
   gradients discarded between slow-level firings and implemented accumulation. It
   moved the result by 0.0001. Wrong: each level runs Adam, which is scale-invariant,
   so discarding gradients never changed step *size*, only step *count*.
2. Measuring the level assignment took one command and found it: 27% of parameters,
   including the first GRU layer, sat on the slow level and took one Adam step per 32.
   NestedRIC was underfitting its input representation, and its low BWT was partly a
   failure to learn. Default changed to level_assignment="memory" -- the backbone
   trains like every baseline, and only the memory gates and modulator are slow.
   avg_perf moved -0.0862 -> -0.0738 and BWT -0.0095 -> -0.0052.
3. Lesson recorded: measure the mechanism before theorising about it. The one-command
   check would have skipped a whole cycle.

**Decisions taken:**
- Frequency separation lives in the memory, not in backbone weights. "depth" retained
  as an ablation axis rather than a default.
- Gradient accumulation kept regardless: without it the Day 10 ratio sweep would
  confound rho with training budget.
- Day 7 gate in docs/PLAN.md ("beats finetune on one stream") is too weak now that
  replay clears it trivially. The bar is replay at equal bytes.

**Tomorrow:** ablations (the ratio sweep is the theorem's empirical content), then the
main benchmark at five seeds.

---

## Day 10 - 2026-08-03

**Planned:** Ablations. Key gate from docs/PLAN.md: "does performance vary
monotonically-ish with the separation ratio? This is the empirical content of the
theorem."

**Done:** 15 cells x 3 seeds on cross-dataset. `results/runs/ablation/`,
`results/tables/ablation.csv`.

**Numbers -- the ratio sweep:**

| rho | BWT | 95% CI | vs default | p |
|---|---|---|---|---|
| 1 | -0.0088 | [-0.0213, +0.0020] | +0.0015 | 0.50 |
| 4 | -0.0096 | [-0.0206, +0.0004] | +0.0007 | 0.50 |
| 16 | -0.0099 | [-0.0203, -0.0000] | +0.0004 | 0.24 |
| 32 | -0.0103 | [-0.0205, -0.0002] | -- | -- |
| 64 | -0.0105 | [-0.0207, -0.0003] | -0.0002 | 0.24 |
| 128 | -0.0073 | [-0.0162, +0.0007] | +0.0030 | 0.24 |

**THE KEY GATE FAILS.** The sweep spans 0.003 across a 128-fold change in rho, every
interval covers every other, nothing is significant. rho = 1 -- the degenerate
single-timescale case Theorem 1 must recover as the *worst* case -- performs the same as
rho = 32.

Other axes, same story: self_modifying off vs on differs by 0.0002 (p = 0.50);
n_levels 1/2/3 are indistinguishable; deep_optimizer and level_assignment differences do
not survive either.

**Surprises:**

1. **Seed dominates configuration by 20x.** Every cell shows the same pattern: seed 0
   about -0.000, seed 1 about -0.010, seed 2 about -0.021. The seed sets the train/eval
   trace split, so most of the measured variation is which traces landed in the eval
   set. Configuration effects are ~0.001 against seed effects of ~0.02. More seeds
   narrow the CI on a paired difference but cannot manufacture an effect that is not
   there.

2. **The Day 7 interaction reading was wrong.** I attributed NestedRIC's -0.0052 (vs
   titans' -0.0263) to tiering plus memory. But rho = 1 here is two memory blocks with
   no separation whatsoever, and it scores the same as rho = 32. The gap to titans is
   therefore the memory *implementation* -- gated blending into similarity-selected
   slots with soft-attention reads, versus a momentum memory -- not the frequency
   separation. That is an engineering difference, not this paper's thesis.

**Decisions taken:**

- **Contingency 2 fires** (docs/PLAN.md): the paper is reframed around the negative
  result. Frequency separation does not reduce forgetting on real O-RAN KPI traces at
  any ratio from 1 to 128, at matched bytes, on the stream where forgetting is largest.
- Theorem 1 may still be provable, but its predicted effect is below the noise floor at
  the drift levels these public traces exhibit. The honest statement of that is a
  contribution: it tells the field where the mechanism cannot help.
- The title claim "provably lower catastrophic forgetting than any single-timescale
  continual-learning xApp" is not supported and must go. Replay is single-timescale and
  matches NestedRIC; rho = 1 is single-timescale and matches rho = 32.

**What the paper still has, all measured:**
1. O-RAN-CL: the benchmark, five streams, frozen splits, fold-level protocol.
2. Which O-RAN reconfigurations cause forgetting (allocation regime) and which do not
   (scheduling discipline, radio conditions) -- with two clean nulls and a mechanism.
3. A negative result on frequency separation with the ablation that establishes it.
4. The bound, presented as theory with its empirical predictions tested and not
   confirmed at achievable drift, which is where drift injection (Day 10 contingency)
   becomes the natural next experiment rather than a fallback.

**Tomorrow:** drift injection -- does frequency separation help once drift is large
enough? That is the sharper question the plan anticipated, and it is now the paper's
central experiment rather than a contingency.
