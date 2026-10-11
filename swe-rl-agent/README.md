# swe-rl-agent

**Application directory for SWE-RL Console.**

For the complete introduction, UI tour, architecture diagram, workflows and
troubleshooting, see the [repository README](../README.md). For the complete
source/function/API inventory, see [Everything_about_project.md](../Everything_about_project.md).

## Bring your own API key

Use **Model connections** in the dashboard to select local vLLM or a major cloud
provider, supply your own key and model, fetch model IDs, test a completion, and
save the active connection. The GUI repair assistant and new agent runs use it.
No API key is included. See [model connections](docs/model-connections.md) for
provider coverage, local secret storage, environment settings, and training limits.

## Operational console

The enterprise UI launches and manages repair runs, evaluations and guarded training.
It includes live execution events, experiment comparison, patch/test inspection,
reward/cost analysis, checkpoints, replay review, dataset slicing and artifact downloads.
See [control center](docs/control-center.md) for every screen, implementation coverage,
API/storage details and environment requirements.

```bash
# From swe-rl-agent/
python3 -m venv .venv-dashboard
.venv-dashboard/bin/pip install -r requirements-dashboard.txt
PYTHONPATH=src .venv-dashboard/bin/python -m uvicorn swe_rl.dashboard.app:app --host 127.0.0.1 --port 8088
```

Open http://127.0.0.1:8088. Workload execution needs the full package, Docker and
dataset access; GPU training needs the explicit opt-in and compatible hardware.
The lightweight console alone does not install model frameworks.

**Code-Repair Agent with Execution-Grounded RL on SWE-Bench.**

Prototype for training and evaluating a code-repair agent on SWE-bench Lite/Verified. Binary reward is computed from actual test outcomes inside an ephemeral Docker sandbox.

```
issue
  -> repo retrieval
  -> ReAct planner LLM
  -> {bash, file_read/write/edit, grep, repo_map, test_run, finish}
  -> sandboxed Docker container (4GB / 2CPU / network verified off post-install)
  -> reward = (FAIL_TO_PASS pass) AND (PASS_TO_PASS still pass)
  -> trajectory recorder
  -> optional SFT / DPO / TRL GRPO training paths
```

## Quickstart

```bash
# 0. install
make setup

# 1. dev services (postgres, redis, minio)
make services-up
make migrate

# 2. smoke: build sandbox + apply a gold patch end-to-end
make smoke-deps
make smoke

# 3. start an OpenAI-compatible vLLM endpoint on a prepared GPU host
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --api-key local-dev

# 4. SFT warmstart -> GRPO
make warmstart
make train

# 5. eval & leaderboard
make eval CHECKPOINT=outputs/grpo SUBSET=lite
make leaderboard RUN_ID=$(date +%Y%m%d_%H%M%S)

# optional local control center
make dashboard  # http://127.0.0.1:8088
```

## Results

> Run `make eval` to generate reports under `outputs/reports/`. This table is not updated automatically.

| Run | Base model | Train compute | resolved@1 (Lite) | resolved@1 (Verified) | $/instance | W&B |
|-----|-----------|---------------|-------------------|------------------------|------------|-----|
| _pending_ | Qwen2.5-Coder-14B | — | — | — | — | — |

## Layout

```
src/swe_rl/
  data/         # SWE-bench and HumanEvalPack loaders, instance schema
  dashboard/    # operational jobs, live events, evidence and experiments UI
  sandbox/      # Docker runner, repo setup, test executor, safety
  agent/        # ReAct loop, tools, prompts, trajectory
  reward/       # exec_reward (binary), shaped_reward, optional PRM
  rollout/      # async worker, Ray pool, replay buffer
  train/        # SFT, DPO, GRPO, curriculum, data assembly
  serve/        # vLLM bootstrap + LoRA hot-swap sync
  eval/         # SWE-bench harness wrapper + leaderboard report
  observability/ # structlog, sentry, prometheus, w&b
docker/         # sandbox & learner Dockerfiles, dev compose
configs/        # Hydra: model/, algo/, reward/, eval/
infra/          # Terraform: VPC, RDS, Redis, ECS, ECR, S3, Secrets
scripts/        # 00_bootstrap → 06_leaderboard_submit
tests/          # unit / integration / e2e
docs/           # architecture, reward design, GRPO notes, runbook
```

## Engineering invariants

- **Execution reward comes from real test outcomes.** Unit tests may mock external services; those tests are not benchmark evidence.
- **Recorded provenance.** Rollouts record seed, generation parameters, model/checkpoint identifiers, repository commit, and sandbox digest. `make repro` replays recorded assistant messages and compares selected execution invariants; exact environment/model replay is not guaranteed.
- **Isolation.** Containers are ephemeral, resource-capped, network-disabled post-install.
- **Cost guardrail.** `MAX_DOLLARS_PER_RUN` stops an individual ReAct rollout after incurred usage exceeds its configured cap. It is not a distributed pool-wide budget.

## Current limitations

- Training is intentionally guarded. `make warmstart` and `make train` refuse to download models or train unless `SWE_RL_ALLOW_HEAVY_TRAINING=YES` is explicitly set on a prepared GPU host.
- The Ray pool remains a separate basic batch executor with retry/budget scaffolding. The console queue uses local subprocess workers; a distributed durable Ray lifecycle is not implemented.
- Local job lifecycle uses durable SQLite and subprocess workers. Trajectories use JSONL/Parquet with bounded PostgreSQL/MinIO hooks and an optional PostgreSQL run index. Redis is not the local queue broker.
- The GRPO reward path evaluates each generated unified diff against its matching instance. Full TRL/vLLM/GPU training remains environment-dependent and is not validated on macOS.
- Exact deterministic model replay cannot be guaranteed by seed alone.
