#!/usr/bin/env bash
set -euo pipefail
python3 -m nestedric.cli ablate --config configs/experiment/ablation.yaml "$@"
