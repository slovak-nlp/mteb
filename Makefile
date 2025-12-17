install:
	@echo "--- 🚀 Installing project dependencies ---"
	uv pip install -e ".[image]" --group dev
	pre-commit install

install-for-tests:
	@echo "--- 🚀 Installing project dependencies for test ---"
	@echo "This ensures that the project is not installed in editable mode"
	uv pip install ".[image]" --group dev

lint:
	@echo "--- 🧹 Running linters ---"
	uv run ruff format . 			# running ruff formatting
	uv run ruff check . --fix --exit-non-zero-on-fix  	# running ruff linting # --exit-non-zero-on-fix is used for the pre-commit hook to work

lint-check:
	@echo "--- 🧹 Check is project is linted ---"
	# Required for CI to work, otherwise it will just pass
	uv run ruff format . --check						    # running ruff formatting
	uv run ruff check .    						        # running ruff linting

test:
	@echo "--- 🧪 Running tests ---"
	uv run pytest -n auto -m "not (test_datasets or leaderboard_stability)"


test-with-coverage:
	@echo "--- 🧪 Running tests with coverage ---"
	uv run pytest -n auto --cov-report=term-missing --cov-config=pyproject.toml --cov=mteb

pr:
	@echo "--- 🚀 Running requirements for a PR ---"
	make lint
	make test


build-docs:
	@echo "--- 📚 Building documentation ---"
	# since we do not have a documentation site, this just build tables for the .md files
	uv run docs/create_tasks_table.py
	uv run docs/create_benchmarks_table.py


model-load-test:
	@echo "--- 🚀 Running model load test ---"
	uv pip install ".[pylate,gritlm,xformers,model2vec]" --group dev
	uv run scripts/extract_model_names.py $(BASE_BRANCH) --return_one_model_name_per_file
	uv run tests/test_models/model_loading.py --model_name_file scripts/model_names.txt


dataset-load-test:
	@echo "--- 🚀 Running dataset load test ---"
	uv run pytest -m test_datasets

dataset-load-test-pr:
	@echo "--- 🚀 Running dataset load test for PR ---"
	eval "$$(python -m scripts.extract_datasets $(BASE_BRANCH))" && pytest -m test_datasets

leaderboard-build-test:
	@echo "--- 🚀 Running leaderboard build test ---"
	uv run pytest -n auto -m leaderboard_stability

run-leaderboard:
	@echo "--- 🚀 Running leaderboard locally ---"
	uv run python -m mteb.leaderboard.app

format-citations:
	@echo "--- 🧹 Formatting citations ---"
	uv run scripts/format_citations.py benchmarks
	uv run scripts/format_citations.py tasks


.PHONY: check
check: ## Run code quality tools.
	@echo "--- 🧹 Running code quality tools ---"
	@pre-commit run -a
