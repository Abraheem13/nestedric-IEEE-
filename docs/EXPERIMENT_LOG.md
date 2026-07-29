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
