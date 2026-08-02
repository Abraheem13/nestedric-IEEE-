#!/usr/bin/env bash
# Fetch the public O-RAN KPI datasets used by the O-RAN-CL benchmark.
#
# LICENCE WARNING: both Colosseum datasets are GPL-3.0. This repository is Apache-2.0
# and does NOT vendor or redistribute them, nor any derived parquet. We download and
# prepare locally, and release only code plus split indices. Do not commit anything
# under data/ -- .gitignore already blocks it, but check before pushing.
#
# Disk: ColO-RAN ~9.0 GB (39,887 files), COMMAG ~7 GB (45,159 files).
# Ensure >20 GB free before starting.

set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

echo "Free space:"; df -h . | tail -1; echo

CORAN_DIR="$RAW_DIR/colosseum-oran-coloran-dataset"
if [ -d "$CORAN_DIR" ]; then
  echo "==> [1/3] ColO-RAN already present, skipping"
else
  echo "==> [1/3] Colosseum ColO-RAN dataset (~9.0 GB)"
  git clone --depth 1 https://github.com/wineslab/colosseum-oran-coloran-dataset.git "$CORAN_DIR"
fi
(cd "$CORAN_DIR" && echo "    commit: $(git rev-parse HEAD)")

COMMAG_DIR="$RAW_DIR/colosseum-oran-commag-dataset"
if [ -d "$COMMAG_DIR" ]; then
  echo "==> [2/3] COMMAG already present, skipping"
else
  echo "==> [2/3] Colosseum COMMAG dataset (~7 GB)"
  git clone --depth 1 https://github.com/wineslab/colosseum-oran-commag-dataset.git "$COMMAG_DIR"
fi
(cd "$COMMAG_DIR" && echo "    commit: $(git rev-parse HEAD)")

echo "==> [3/3] TRACTOR 5G KPI traces (~2.83 GB)"
echo "    Manual step: see https://genesys-lab.org/tractor for the download link"
echo "    and licence terms. Extract to $RAW_DIR/tractor"
echo "    Record licence + required citation in docs/DATASETS.md."

echo; echo "==> Verifying structure"
for pair in "coloran:$CORAN_DIR" "commag:$COMMAG_DIR"; do
  name="${pair%%:*}"; dir="${pair#*:}"
  if [ -d "$dir" ]; then
    n=$(find "$dir" -name '*_metrics.csv' | wc -l)
    echo "    $name: $n slice-metrics files"
  fi
done

echo; echo "Expected: coloran 18329, commag 21612"
echo "Next: python scripts/prepare_data.py --dataset all"
