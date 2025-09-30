.PHONY: setup lint type test format train eval embed demo smoke report clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests

type:
	mypy src tests

format:
	ruff check --fix src tests
	black src tests
	isort src tests

test:
	pytest

train:
	biofm train

eval:
	$(PYTHON) scripts/eval.py

embed:
	biofm embed

smoke:
	@echo "Running 60-second smoke test..."
	timeout 60 $(PYTHON) -c "import time; from biofm.cli import app; from typer.testing import CliRunner; runner = CliRunner(); result = runner.invoke(app, ['train', '--profile', 'toy', '--out-dir', 'outputs/smoke', '--device', 'cpu', '--mixed-precision', 'off']); print('Train result:', result.exit_code); time.sleep(1)"
	@test -f outputs/smoke/checkpoints/last.pt || (echo "Checkpoint not found after smoke test" && exit 1)
	$(PYTHON) scripts/eval.py --output-dir outputs/smoke --profile toy
	@test -f outputs/smoke/reports/metrics.json || (echo "Metrics not found after smoke test" && exit 1)
	@echo "Smoke test completed successfully"

report:
	@if [ -f outputs/reports/metrics.json ]; then \
		echo "Opening metrics report..."; \
		cat outputs/reports/metrics.json | $(PYTHON) -m json.tool; \
	else \
		echo "No metrics report found. Run 'make eval' first."; \
		exit 1; \
	fi

demo:
	biofm report --table

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache */__pycache__
	rm -rf outputs/smoke
