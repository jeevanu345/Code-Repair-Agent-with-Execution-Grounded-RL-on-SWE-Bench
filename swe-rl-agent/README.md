# swe-rl-agent

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

# optional read-only local dashboard
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
  dashboard/    # read-only local status and trajectory UI
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

- **Reward must come from real test execution.** No mocks for sandbox or test runner.
- **Recorded provenance.** Rollouts record seed, generation parameters, model/checkpoint identifiers, repository commit, and sandbox digest. Exact replay is not yet implemented; `make repro` currently inspects a record.
- **Isolation.** Containers are ephemeral, resource-capped, network-disabled post-install.
- **Cost guardrail.** `MAX_DOLLARS_PER_RUN` prevents an individual ReAct rollout from accepting a token update above its cap. It is not a distributed pool-wide budget.

## Current limitations

- Training is intentionally guarded. `make warmstart` and `make train` refuse to download models or train unless `SWE_RL_ALLOW_HEAVY_TRAINING=YES` is explicitly set on a prepared GPU host.
- The Ray pool is a basic batch executor; retry, cancellation, idempotency, backpressure, and distributed budget coordination are not implemented.
- PostgreSQL models and local Redis/MinIO services exist, but rollout persistence and queues currently use local JSONL/Parquet rather than those services.
- The GRPO reward path evaluates each generated unified diff against its matching instance. Full TRL/vLLM/GPU training remains environment-dependent and is not validated on macOS.
- Exact deterministic model replay cannot be guaranteed by seed alone.
