# swe-rl-agent

**Code-Repair Agent with Execution-Grounded RL on SWE-Bench.**

Trains an LLM agent to resolve real GitHub issues from SWE-bench Lite/Verified using GRPO. Reward comes from running the project's actual test suite inside an ephemeral Docker sandbox.

```
issue
  -> repo retrieval
  -> ReAct planner LLM
  -> {bash, file_read/write/edit, grep, repo_map, test_run, finish}
  -> sandboxed Docker container (4GB / 2CPU / network-disabled post-install)
  -> reward = (FAIL_TO_PASS pass) AND (PASS_TO_PASS still pass)
  -> trajectory recorder
  -> GRPO update (TRL) with vLLM rollout server
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

# 3. start vLLM on a GPU box
python -m swe_rl.serve.vllm_server  # or use scripts

# 4. SFT warmstart -> GRPO
make warmstart
make train

# 5. eval & leaderboard
make eval CHECKPOINT=outputs/grpo SUBSET=lite
make leaderboard RUN_ID=$(date +%Y%m%d_%H%M%S)
```

## Results

> Run `make eval` to populate. The README's results table is overwritten by `eval/report.py`.

| Run | Base model | Train compute | resolved@1 (Lite) | resolved@1 (Verified) | $/instance | W&B |
|-----|-----------|---------------|-------------------|------------------------|------------|-----|
| _pending_ | Qwen2.5-Coder-14B | — | — | — | — | — |

## Layout

```
src/swe_rl/
  data/         # SWE-bench loaders, instance schema
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
- **Determinism.** Every rollout records seed, image digest, model sha, repo commit. `make repro RUN_ID=...` re-runs.
- **Isolation.** Containers are ephemeral, resource-capped, network-disabled post-install.
- **Cost guardrails.** `MAX_DOLLARS_PER_RUN` halts the rollout pool when projected cost exceeds the cap.

