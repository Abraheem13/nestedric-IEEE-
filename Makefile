.PHONY: setup data test lint fmt smoke main ablate figures clean

PY ?= PYTHONPATH=src python3

setup:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

data:
	bash scripts/download_data.sh
	$(PY) -m nestedric.cli prepare --config configs/data/all.yaml

test:
	pytest

lint:
	ruff check src tests && black --check src tests

fmt:
	ruff check --fix src tests && black src tests

smoke:
	$(PY) -m nestedric.cli train --config configs/experiment/smoke.yaml

main:
	bash scripts/run_main_benchmark.sh

ablate:
	bash scripts/run_ablations.sh

figures:
	$(PY) -m nestedric.cli figures --config configs/experiment/main.yaml

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__

gate:
	$(PY) -m nestedric.cli train --config configs/experiment/gate.yaml
	$(PY) scripts/report_gate.py --dir results/runs/gate
