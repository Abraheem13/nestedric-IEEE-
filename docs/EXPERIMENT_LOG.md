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
