SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python
PIP ?= pip
UV ?= uv
RUN_ID ?= $(shell date +%Y%m%d_%H%M%S)
CHECKPOINT ?= base
SUBSET ?= lite

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install python deps + pre-commit + sandbox image
	@if command -v $(UV) >/dev/null 2>&1; then $(UV) pip install -e ".[dev]"; else $(PIP) install -e ".[dev]"; fi
	@$(MAKE) build-sandbox

.PHONY: build-sandbox
build-sandbox: ## Build sandbox docker image
	docker build -f docker/sandbox.Dockerfile -t swe-rl/sandbox:latest .

.PHONY: build-learner
build-learner: ## Build learner docker image (vLLM + TRL + CUDA)
	docker build -f docker/learner.Dockerfile -t swe-rl/learner:latest .

.PHONY: services-up
services-up: ## Start postgres / redis / minio for dev
	docker compose -f docker/compose.dev.yaml up -d

.PHONY: services-down
services-down: ## Stop dev services
	docker compose -f docker/compose.dev.yaml down

.PHONY: migrate
migrate: ## Run DB migrations
	$(PYTHON) -m swe_rl.cli db migrate

.PHONY: smoke-deps
smoke-deps: ## Verify imports + DB + Docker
	$(PYTHON) -m swe_rl.cli smoke deps

.PHONY: smoke
smoke: smoke-deps ## Full smoke: deps + 1 instance gold-patch rollout
	bash scripts/02_smoke_rollout.sh

.PHONY: smoke-rollout
smoke-rollout: ## End-to-end rollout on a single Lite instance
	bash scripts/02_smoke_rollout.sh

.PHONY: warmstart
warmstart: ## SFT warmstart on HumanEvalPack + curated trajectories
	bash scripts/03_warmstart_sft.sh

.PHONY: train
train: ## GRPO training run
	bash scripts/04_grpo_train.sh

.PHONY: eval
eval: ## Evaluate checkpoint on SWE-bench Lite
	bash scripts/05_eval_lite.sh $(CHECKPOINT) $(SUBSET)

.PHONY: leaderboard
leaderboard: ## Package predictions for leaderboard
	bash scripts/06_leaderboard_submit.sh $(RUN_ID)

.PHONY: repro
repro: ## Reproduce a recorded trajectory
	$(PYTHON) -m swe_rl.cli repro --run-id $(RUN_ID)

.PHONY: dashboard
dashboard: ## Start read-only local control plane on 127.0.0.1:8088
	$(PYTHON) -m swe_rl.cli dashboard

.PHONY: test
test: ## Unit tests
	$(PYTHON) -m pytest tests/unit -v

.PHONY: test-integration
test-integration: ## Integration tests (require Docker)
	$(PYTHON) -m pytest tests/integration -v -m integration

.PHONY: test-e2e
test-e2e: ## E2E rollout test
	$(PYTHON) -m pytest tests/e2e -v -m e2e

.PHONY: lint
lint: ## Ruff + mypy
	ruff check src tests
	ruff format --check src tests
	mypy src

.PHONY: fmt
fmt: ## Auto-format
	ruff format src tests
	ruff check --fix src tests

.PHONY: clean
clean: ## Remove build / cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
