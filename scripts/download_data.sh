#!/usr/bin/env bash
# Fetch every public dataset used by the O-RAN-CL benchmark.
# All sources are public; record the resolved commit hash for reproducibility.
set -euo pipefail

RAW_DIR="${RAW_DIR:-data/raw}"
mkdir -p "$RAW_DIR"

echo "==> [1/3] Colosseum ColO-RAN dataset"
# git clone https://github.com/wineslab/colosseum-oran-coloran-dataset "$RAW_DIR/colosseum-oran-coloran-dataset"
# TODO(Day 1): uncomment, pin commit, verify checksum.

echo "==> [2/3] Colosseum COMMAG dataset"
# git clone https://github.com/wineslab/colosseum-oran-commag-dataset "$RAW_DIR/colosseum-oran-commag-dataset"

echo "==> [3/3] TRACTOR 5G KPI traces (~2.83 GB)"
# See https://genesys-lab.org/tractor for the download link and licence terms.
# TODO(Day 1): record licence + citation requirement in docs/DATASETS.md.

echo
echo "Done. Next: nestedric prepare --config configs/data/all.yaml"
