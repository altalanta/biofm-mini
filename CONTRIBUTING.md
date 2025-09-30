# Contributing

Thanks for your interest in improving biofm-mini! To keep the project healthy:

1. Fork or branch from `main` and ensure Python 3.11 is active.
2. Run `make setup` to install Poetry dependencies and pre-commit hooks.
3. Align with our style by running `make lint` and `make test` locally before opening a pull request.
4. For substantial features, add or update unit tests under `tests/` and describe changes in the PR body.
5. Keep docs (README, MODEL_CARD, DATASET_SHEET) up to date when you introduce new capabilities or datasets.
6. By contributing, you agree to follow the `CODE_OF_CONDUCT.md`.

If you hit environment issues (e.g., missing `scanpy`), note them in your PR so we can update fallbacks.
