.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -X dev -m pdb -m src $(ARGS)

clean:
	rm -rf .mypy_cache .pytest_cache dist build
	find src -name '__pycache__' -type d -prune -exec rm -rf {} +

lint:
	uv run flake8 src
	uv run mypy src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src
