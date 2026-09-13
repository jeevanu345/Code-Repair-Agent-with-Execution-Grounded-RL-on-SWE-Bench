# Runbook

Operational guide. Append a new entry per phase change or incident.

## Routine ops

### Add a new tool
1. Create `src/swe_rl/agent/tools/<name>.py` with a `Args(BaseModel)` and a `Tool` subclass.
2. Register it in `src/swe_rl/agent/tools/__init__.py::default_toolset`.
3. Add a unit test under `tests/unit/test_tools_<name>.py`.
4. Update `docs/architecture.md` if the tool changes the trust boundary.

### Swap base model
1. Add `configs/model/<name>.yaml`.
2. `MODEL_NAME=<HF id> make services-up`.
3. Update vLLM `--max-model-len` if needed.
4. Re-run `make smoke`.

### Run a remote eval
1. Push images: `make build-sandbox && docker push <ECR>/sandbox:latest`; same for learner.
2. `terraform apply` brings up cluster; learner job runs `scripts/05_eval_lite.sh` inside ECS.
3. Predictions land in S3 `swe-rl-trajectories-<account>/predictions/<run-id>.json`.

### Debug a stuck rollout
- `docker ps --filter label=project=swe-rl-agent` → find container.
- `docker logs <id>` → tail the install log.
- Check Postgres `trajectories` for the row with `instance_id` to see the last recorded step.
- Each exec is killed at its timeout and rollout cleanup removes the container. There is no independent orphan-container reaper.

### Recover from a corrupted replay buffer
- `pyarrow` will refuse to read the corrupted shard. Move it aside:
  `mv outputs/replay/shard_<ts>.parquet outputs/replay/quarantine/`
- The buffer is append-only; remaining shards are intact.
- Re-run rollouts to backfill if the corrupted shard held important data.

### Cost cap hit
- The current rollout stops when its local cost meter exceeds the cap. There is no atomic pool-wide budget; stop workers before changing the cap.
- Investigate which instance's tool calls were expensive: `select instance_id, sum(tokens_out) from trajectories group by 1 order by 2 desc;`

## Phase log

### Phase A — Foundations (initial)
- pyproject, Makefile, sandbox/learner Dockerfiles, dev compose
- observability bootstrap (structlog + sentry + prometheus)
- Postgres schema (instances, trajectories, rewards, eval_runs, model_checkpoints)

### Phase B — Sandbox
- DockerRunner with cap-drop, no-new-privileges, resource caps, network detach
- repo_setup with SWE-bench specs lookup + generic fallback
- TestExecutor with json-report + verbose fallback parsing

### Phase C — Agent
- Tools: bash, file_read/write/edit, grep, repo_map, test_run, finish
- ReAct loop with hard caps (50 steps, 6 test runs, 200k ctx)
- Trajectory recorder with deterministic seed + image digest

### Phase D — Reward
- Binary exec_reward: pure function of (TestResults, F2P, P2P)
- Shaped reward with size + lint + no-patch + timeout penalties

### Phase E — Rollout
- Worker function with try/finally container teardown
- Ray actor pool (one DockerRunner per actor)
- Parquet replay buffer with shard rotation

### Phase F — Train
- SFT warmstart on HumanEvalPack (LoRA)
- Offline DPO over (best, worst) rollout pairs
- TRL GRPOTrainer wrapper with execution-grounded reward callable
- vLLM LoRA hot-swap via `/v1/load_lora_adapter`

### Phase G — Eval
- generate_predictions writes leaderboard-format JSON
- run_official_harness invokes `swebench.harness.run_evaluation`
- report.py renders `leaderboard.md` + `summary.json`

### Phase H — Deploy
- Terraform: VPC, RDS, Redis, S3, ECR, ECS cluster, Secrets
- ECS task definition for privileged rollout workers
- GitHub Actions CI: lint + unit (always), integration (push), nightly eval
