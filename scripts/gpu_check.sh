#!/usr/bin/env bash
# Confirm the L4 is visible and report free memory before a long run.
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
