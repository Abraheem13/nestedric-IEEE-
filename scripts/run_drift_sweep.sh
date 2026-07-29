#!/usr/bin/env bash
# Contingency + bound validation.
set -euo pipefail
python3 -m nestedric.cli train --config configs/experiment/drift_sweep.yaml "$@"
