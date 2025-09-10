check:
	uv run ruff check
	uv run mypy src tests
	uv run pydoclint src