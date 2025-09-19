.PHONY: setup lint test train eval format clean

setup:
	poetry install --with dev
	poetry run pre-commit install

lint:
	poetry run ruff check src tests
	poetry run black --check src tests scripts
	poetry run isort --check-only src tests scripts
	poetry run mypy src scripts

format:
	poetry run ruff check --fix src tests
	poetry run black src tests scripts
	poetry run isort src tests scripts

test:
	poetry run pytest

train:
	poetry run python scripts/train.py

eval:
	poetry run python scripts/eval.py

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
