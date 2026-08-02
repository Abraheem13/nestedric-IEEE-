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
