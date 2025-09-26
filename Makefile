.PHONY: setup lint test format train eval embed demo clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests
	isort --check-only src tests
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
	biofm eval

embed:
	biofm embed

demo:
	biofm report --table

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache */__pycache__
