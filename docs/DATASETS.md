# Datasets

Selection principle: the benchmark needs **real O-RAN KPI traces** (not simulator
output) spanning **multiple, genuinely different sources**, so that the hardest stream
family — cross-dataset — exhibits real covariate shift rather than a synthetic
relabelling. All sources are public and CSV/trace-based, so the whole campaign is
*trace replay*, not live RF: this is why 15 days on one L4 is realistic.

---

## Primary sources

### 1. Colosseum ColO-RAN dataset — **core**
- Repo: `https://github.com/wineslab/colosseum-oran-coloran-dataset`
- Landing page: `https://openrangym.com/datasets/colosseum-coloran-dataset`
- Content: Rome scenario, 7 base stations, 42 UEs, eMBB / URLLC / MTC slices, collected
  on the Colosseum wireless network emulator with a full O-RAN near-RT RIC in the loop.
- Role: **slice-shift** and **sched-shift** streams. This is the dataset the O-RAN ML
  community already recognises, which matters for benchmark adoption.
- Licence: TODO(Day 1) — record exactly, and the required citation.
- Pinned commit: TODO(Day 1).

### 2. Colosseum COMMAG dataset — **core**
- Repo: `https://github.com/wineslab/colosseum-oran-commag-dataset`
- Content: companion trace set from the same testbed lineage, different scheduling and
  slicing configurations.
- Role: second source for **sched-shift**; contributes to **cross-dataset**.
- Licence: TODO(Day 1). Pinned commit: TODO(Day 1).

### 3. TRACTOR — **core**
- Landing page: `https://genesys-lab.org/tractor`
- Content: ~2.83 GB, 17 O-RAN KPIs, ~447 minutes of **real 5G traffic** across distinct
  traffic classes.
- Role: **traffic-shift** stream — the most realistic non-stationarity available
  publicly, and the strongest expected drift signal. Also contributes to
  **cross-dataset**.
- Licence: TODO(Day 1). Version/download date: TODO(Day 1).

## Optional / contingency source

### 4. Locally generated SCOPE / OpenRAN Gym traces
- Only used if the Day-4 gate shows insufficient drift in the public traces, or to
  extend the drift sweep beyond what real traces cover.
- Clearly labelled as synthetic in every table. Never mixed silently with real traces.

---

## Why these three and not others

| Candidate | Verdict | Reason |
|---|---|---|
| ColO-RAN | **in** | Real O-RAN RIC in the loop; community-recognised; slice/scheduler context axes. |
| COMMAG | **in** | Independent configuration space; cheap to add; enables sched-shift with two sources. |
| TRACTOR | **in** | Real 5G traffic, many traffic classes, large enough for 8+ environments. |
| Synthetic ns-3 only | out | No reviewer credit for "forgetting" that we manufactured ourselves. |
| DeepMIMO / Sionna | out | PHY/channel-level, not RIC KPI-level; belongs to the second paper. |
| Proprietary operator KPIs | out | Not public — kills the benchmark artefact, which is the citation engine. |

Using three independent sources is the single most important design choice here: it is
what turns "a continual-learning experiment" into **a benchmark other people can adopt**,
and it makes the cross-dataset stream a genuine covariate-shift test rather than a
within-testbed reshuffle.

---

## Canonical schema

Every adapter emits the schema in `src/nestedric/data/schema.py`. Harmonisation map to
be completed on Day 1:

| Canonical column | ColO-RAN source | COMMAG source | TRACTOR source | Unit |
|---|---|---|---|---|
| `dl_thpt_mbps` | TODO | TODO | TODO | Mbit/s |
| `ul_thpt_mbps` | TODO | TODO | TODO | Mbit/s |
| `dl_buffer_bytes` | TODO | TODO | TODO | bytes |
| `dl_prb_used` | TODO | TODO | TODO | PRBs |
| `dl_mcs` | TODO | TODO | TODO | index |
| `cqi` | TODO | TODO | TODO | index |
| `sinr_db` | TODO | TODO | TODO | dB |
| `ratio_granted_req` | TODO | TODO | TODO | ratio |
| `tx_pkts` | TODO | TODO | TODO | count |
| `tx_errors_pct` | TODO | TODO | TODO | % |

Rules:
- Native sampling rates differ; all sources are resampled to a common period
  (`configs/data/all.yaml: resample_ms`). The resampling choice is reported.
- Columns present in only one source are **dropped**, not imputed — a KPI that exists
  in one testbed and not another would leak dataset identity into the model.
- Normalisation constants are fitted on **source environments only** and applied to
  later environments, exactly as in the leave-one-city-out protocol of the path-loss
  work. Fitting them globally leaks future statistics into the past.

## Reproducibility checklist (complete before release)

- [ ] Pinned commit hash / download date for each source
- [ ] Licence text and required citation recorded for each source
- [ ] SHA256 of each raw archive
- [ ] Row counts, environment counts and date ranges per dataset in the paper
- [ ] Frozen splits committed under `benchmark/oran_cl/splits/`
