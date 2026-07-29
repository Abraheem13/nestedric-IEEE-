#!/usr/bin/env bash
# Main benchmark: 10 methods x 5 streams x 5 seeds on a single L4.
# Expected wall-clock: see docs/PLAN.md (Day 8-9).
set -euo pipefail
python3 -m nestedric.cli train --config configs/experiment/main.yaml "$@"
