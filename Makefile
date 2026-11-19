.PHONY: setup lint typecheck test check validate evaluate docker clean

setup:            ## install pinned env + git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src

test:
	OMP_NUM_THREADS=1 uv run pytest --durations=5

check: lint typecheck test   ## everything CI runs

validate:         ## every grader must pass its reference and fail its starter
	uv run python -m bugbench.runner validate

evaluate:         ## grade every saved AI-assistant answer into evals/results.md
	uv run python -m bugbench.evaluate

docker:           ## reproduce inside a clean container
	docker build -t mle-bugbench .
	docker run --rm -v "$(PWD)/results:/app/results" mle-bugbench

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache results .coverage
