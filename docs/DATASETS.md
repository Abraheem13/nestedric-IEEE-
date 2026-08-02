# Datasets

Status: **ColO-RAN and COMMAG verified on Day 1** by direct inspection of the real
repositories. TRACTOR still pending manual download.

Selection principle: the benchmark needs **real O-RAN KPI traces** (not simulator
output) from **multiple sources**, so the cross-dataset stream exhibits real covariate
shift rather than a within-testbed reshuffle. All sources are CSV trace replay — no
live RF — which is why 15 days on one L4 is realistic.

---

## 1. Colosseum ColO-RAN — VERIFIED

- Repo: `https://github.com/wineslab/colosseum-oran-coloran-dataset`
- **Licence: GPL-3.0** (see the licence note below — this matters)
- Citation required: Polese, Bonati, D'Oro, Basagni, Melodia, "ColO-RAN: Developing
  Machine Learning-based xApps for Open RAN Closed-loop Control on Programmable
  Experimental Platforms," *IEEE Trans. Mobile Computing*, vol. 22, no. 10,
  pp. 5787–5800, 2022.
  *The source paper is itself a TMC paper — useful for our venue argument.*
- Size: **9.0 GB, 39,887 files**, of which **18,329** are slice-metrics CSVs.

Verified layout:
```
rome_static_medium/sched{0,1,2}/tr{0..27}/exp{1..5}/bs{1..7}/
    bsN.csv                              # 4 cols:  time, nof_ue, dl_brate, ul_brate
    ueNN.csv                             # 21 cols: PHY-level per-UE
    slices_bsN/<IMSI>_metrics.csv        # 31 cols: THE FILE WE USE
```

Setup: 7 BS (nodes 1,8,15,22,29,36,43), 42 UEs, 10 MHz / 50 PRB, static mobility,
medium distance (UEs within 50 m). 3 slices per BS. 3 scheduling policies
(0 = round-robin, 1 = waterfilling, 2 = proportional fair). 28 RBG allocations
(tr0–tr27). Traffic: slice 0 = eMBB (4 Mbps CBR), slice 1 = MTC (30 pkt/s × 125 B),
slice 2 = URLLC (10 pkt/s × 125 B).

Context axes: `sched_policy` (3) × `tr_config` (28) × `exp_id` (5) × `bs_id` (7) ×
`slice_id` (3) = **8,820 addressable cells**. Ample for 8–12 environments.

## 2. Colosseum COMMAG — VERIFIED

- Repo: `https://github.com/wineslab/colosseum-oran-commag-dataset`
- **Licence: GPL-3.0**
- Citation required: Bonati, D'Oro, Polese, Basagni, Melodia, "Intelligence and
  Learning in O-RAN for Data-driven NextG Cellular Networks," *IEEE Communications
  Magazine*, vol. 59, no. 10, pp. 21–27, October 2021.
- Size: **45,159 files**, of which **21,612** are slice-metrics CSVs.

Verified layout:
```
slice_{mixed,traffic}/rome_{static,slow}_{close,medium,far}/tr{0..17}/exp{1..6}/bs{1..4}/
    slices_bsN/<IMSI>_metrics.csv        # identical 31-col format
ml_models/                               # pretrained TF agents (not used)
```

Six scenario combinations exist:
`slice_mixed/rome_slow_close`, `slice_mixed/rome_static_close`,
`slice_traffic/rome_slow_close`, `slice_traffic/rome_static_close`,
`slice_traffic/rome_static_far`, `slice_traffic/rome_static_medium`.

Setup: 4 BS, 40 UEs, 3 MHz / 15 PRB.

**This is why COMMAG matters more than originally planned.** It varies three axes
ColO-RAN holds fixed:
- **mobility**: static vs slow (3 m/s)
- **distance**: close (20 m) / medium (50 m) / far (100 m)
- **slice assignment**: traffic-based vs mixed

Those are genuine *radio-condition* shifts, not just configuration relabelling — much
stronger candidates for producing real catastrophic forgetting than slice-mix changes.
Day 2 stream design should lead with these.

## 3. TRACTOR — PENDING

- Page: `https://genesys-lab.org/tractor`, ~2.83 GB, 17 O-RAN KPIs, ~447 min real 5G.
- Not on GitHub; requires manual download. Record licence + citation on arrival.
- Role: traffic-shift stream. **Not on the critical path** — ColO-RAN + COMMAG alone
  now support all five stream families.

---

## Licence note (important)

Both Colosseum datasets are **GPL-3.0**. This repository is Apache-2.0. That is fine
because we **do not vendor or redistribute the data**, nor any derived parquet:

- `data/` is gitignored, including `*.parquet`.
- The released artefact is **code + split indices** (JSON lists of trace IDs and row
  ranges). Indices are not the data.
- The paper cites both source papers, as their licences require.

Do not commit anything under `data/`. Check `git status` before pushing.

---

## Canonical schema — VERIFIED

Both datasets export the **same 31-column slice-metrics header**, including four
unnamed spacer columns from the original logger. One adapter therefore serves both
(`src/nestedric/data/colosseum.py`), and the cross-dataset stream needs no
harmonisation guesswork.

Raw header, in file order:
```
Timestamp, num_ues, IMSI, RNTI, <spacer>,
slicing_enabled, slice_id, slice_prb, power_multiplier, scheduling_policy, <spacer>,
dl_mcs, dl_n_samples, dl_buffer [bytes], tx_brate downlink [Mbps],
tx_pkts downlink, tx_errors downlink (%), dl_cqi, <spacer>,
ul_mcs, ul_n_samples, ul_buffer [bytes], rx_brate uplink [Mbps],
rx_pkts uplink, rx_errors uplink (%), ul_rssi, ul_sinr, phr, <spacer>,
sum_requested_prbs, sum_granted_prbs, <spacer>,
dl_pmi, dl_ri, ul_n, ul_turbo_iters
```

- **Sampling period: 250 ms**, timestamps are epoch milliseconds.
- Spacer columns arrive as `Unnamed: 4` etc. and are dropped in `read_metrics_csv`.
- `dl_pmi`, `dl_ri`, `ul_n` are constant zero in the samples inspected — the profiler
  flags constant columns automatically; drop them as features.

The full mapping lives in `src/nestedric/data/schema.py::RAW_TO_CANONICAL`.

## Data-quality findings (Day 1)

**PRB counters are accumulated, not instantaneous.** `sum_requested_prbs` and
`sum_granted_prbs` accumulate over the 250 ms window: observed max 11,424, consistent
with 50 PRB × 250 subframes. They are not per-subframe values.

**42% of rows have `sum_requested_prbs == 0`** (idle UE). The derived
`ratio_granted_req` is therefore **left missing on those rows** rather than divided by
a clipped denominator — the first implementation clipped to 1 and produced ratios up to
11,424, which would have silently poisoned the model. Regression-tested in
`tests/test_schema.py::test_ratio_is_missing_when_nothing_requested`.

**Granted legitimately exceeds requested** (the scheduler grants minimum allocations),
so `ratio_granted_req` is *not* bounded by 1. On non-idle rows: median 0.96, 75th
percentile 1.20, max 24.9.

**Slice semantics are clearly present**, which is the key signal that the data can
support the task. On `sched0/tr0`, mean downlink throughput by slice:

| slice | class | mean (Mbps) | max (Mbps) |
|---|---|---|---|
| 0 | eMBB | 0.709 | 2.916 |
| 1 | MTC | 0.053 | 0.153 |
| 2 | URLLC | 0.105 | 0.413 |

eMBB is ~13× MTC, as the traffic configuration implies. URLLC sitting above MTC is
worth a second look on Day 2 — it may reflect differing per-slice RBG allocations and
UE counts under `tr0` (2/13/2 RBGs) rather than anything wrong.

**Scale check:** a single `tr0` directory of `sched0` (206 metrics files) yields
**410,048 canonical rows**. Extrapolating, the full ColO-RAN is on the order of 3.6 × 10⁷
rows and COMMAG comparable. Storage and time are not constraints; environment *design*
is.

## Reproducibility checklist

- [x] Verified layout and file counts for ColO-RAN and COMMAG
- [x] Licences recorded (both GPL-3.0) and required citations captured
- [x] Canonical schema fixed and unit-documented
- [x] Adapter regression-tested against the real header
- [ ] Pinned commit hashes (printed by `scripts/download_data.sh` — paste them here)
- [ ] TRACTOR licence, citation and download date
- [ ] Frozen splits under `benchmark/oran_cl/splits/`
