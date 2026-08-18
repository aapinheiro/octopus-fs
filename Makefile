.PHONY: install test lint typecheck build clean

install:      ## create .venv and install everything
	uv sync --all-extras

test:
	uv run pytest

test-fast:    ## skip the arms that fit models
	uv run pytest -m "not slow"

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy src

build:        ## sdist + wheel into dist/
	uv build

clean:
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
