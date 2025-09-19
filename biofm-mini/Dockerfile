FROM python:3.11-slim

ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git gcc && \
    pip install --no-cache-dir "poetry==${POETRY_VERSION}" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml README.md ./
RUN poetry install --only main --no-root

COPY . .
RUN poetry install

CMD ["poetry", "run", "python", "scripts/train.py"]
