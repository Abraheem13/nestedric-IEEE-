#!/usr/bin/env bash
set -euo pipefail
python3 -m nestedric.cli figures --config configs/experiment/main.yaml
echo "Figures -> results/figures, tables -> results/tables"
