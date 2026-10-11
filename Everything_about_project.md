# Complete Project Overview: Code-Repair Agent with Execution-Grounded RL on SWE-Bench

> **Repository studied:** the Git repository root, with the main implementation in [`swe-rl-agent/`](swe-rl-agent/). This guide describes the checked-in code and configuration, not just the aspirational architecture in the READMEs. It was prepared from the repository contents on 2026-10-02. The console/control-plane upgrade and terminal unit/UI checks are included. Actual Docker grading, GPU training and paid provider calls were not certified on this host.

## 1. What this project is

This project is a research/prototype system for training and evaluating an LLM-based software repair agent against real SWE-bench GitHub issues. It normalizes SWE-bench rows, checks out the target repository at the issue's base commit in a disposable Docker container, lets a model inspect and edit the code through a bounded ReAct tool loop, runs the issue's fail-to-pass and pass-to-pass tests, records the patch and trajectory, and can use rewards for SFT/DPO/GRPO experiments.

Its defining idea is **execution-grounded reward**: the primary success signal is whether the target tests actually pass in the prepared repository. The repository also contains a local web dashboard, provider configuration UI, evaluation/report helpers, developer scripts, and starter AWS/Terraform infrastructure.

### Intended data path

```text
Hugging Face SWE-bench row
  -> SWEBenchInstance validation
  -> ephemeral Docker container and target-repository setup
  -> LLM ReAct loop + sandbox tools (or supplied candidate patch)
  -> capture unified diff
  -> apply hidden test patch, run FAIL_TO_PASS and PASS_TO_PASS
  -> execution and shaped reward
  -> JSONL trajectory; best-effort MinIO/PostgreSQL persistence
  -> optional Parquet replay buffer / SFT, DPO, or GRPO training
  -> prediction JSON and official SWE-bench harness/report
```

The local control center connects launch, durable lifecycle, execution events, evidence and experiments. It remains a research system with environment-dependent workload execution. See [implemented scope and limitations](#12-implemented-scope-and-known-limitations).

## 2. Repository map

The workspace root is mostly a wrapper; the Python application is nested under `swe-rl-agent/`.

```text
.
├── README.md                         # short repository-level entry point
├── Everything_about_project.md       # this guide
└── swe-rl-agent/                     # application/project root
    ├── README.md                     # detailed project, control center, quickstart
    ├── pyproject.toml                # package metadata and dependencies
    ├── Makefile                      # common developer commands
    ├── .env.example                  # environment variable template
    ├── requirements-dashboard.txt    # lightweight dashboard dependencies
    ├── src/swe_rl/                   # installable Python package
    ├── configs/                      # YAML model/algo/reward/eval defaults
    ├── docker/                       # sandbox, learner, local services
    ├── infra/                        # AWS Terraform starter configuration
    ├── migrations/                   # PostgreSQL schema SQL
    ├── scripts/                      # bootstrap, smoke, train, eval, packaging
    ├── docs/                         # architecture, reward, operations, GUI docs
    ├── tests/                        # unit, integration, end-to-end, UI checks
    └── .github/workflows/ci.yml      # legacy nested workflow (not discovered at repo root)
```

### Python package map

```text
src/swe_rl/
├── agent/       # model clients, providers, ReAct loop, tools, prompts, trajectory
├── dashboard/   # FastAPI local workbench and bundled static UI
├── data/        # SWE-bench/HumanEvalPack loading and canonical schema
├── eval/        # prediction generation, official harness invocation, reports
├── observability/ # structured logging, Prometheus metrics, Sentry
├── reward/      # execution reward, dense shaping, optional PRM hook
├── rollout/    # single-instance worker, Ray pool, Parquet buffer, replay check
├── sandbox/    # Docker lifecycle, repository setup, safety, test execution
├── serve/      # vLLM process helper and LoRA adapter sync helper
├── train/      # data assembly, SFT, DPO, GRPO, simple curriculum utilities
├── utils/      # cost meter, git/diff helpers, retry helper
├── cli.py      # Typer command-line entry point
├── db.py       # SQLAlchemy schema and persistence helpers
└── settings.py # environment-backed process settings
```

## 3. Main execution flow

### 3.1 Load and normalize an instance

`data/swebench_loader.py` lazily calls Hugging Face `datasets.load_dataset` for `princeton-nlp/SWE-bench_Lite`, `SWE-bench_Verified`, or full SWE-bench. Rows are converted by `data/instance_schema.py` into a strict Pydantic `SWEBenchInstance`: issue id, repository, base commit, problem statement, gold patch, test patch, version, test lists, and metadata. Uppercase `FAIL_TO_PASS` and `PASS_TO_PASS` fields are normalized; unknown fields are retained in `extra`.

`load_one()` searches the chosen dataset or Lite, Verified, and full sequentially. This is convenient but can require loading/scanning datasets repeatedly.

### 3.2 Prepare an isolated repository

`rollout/worker.py::run_rollout` creates a `DockerRunner` and always tears its container down in `finally`. `sandbox/docker_runner.py` starts the configured image with memory/CPU/PID caps, drops Linux capabilities, enables `no-new-privileges`, and initially uses bridge networking so it can fetch repositories and install dependencies. It wraps commands with a timeout and caps captured output.

`sandbox/repo_setup.py::setup_repo` clones `https://github.com/<repo>.git`, checks out the SWE-bench base commit, uses official SWE-bench per-repository setup specs where found, otherwise tries a generic pip editable install, and records setup output. It snapshots a baseline commit so the final diff represents the agent's edits. The repository setup currently does not apply the test patch to the agent-visible checkout. The worker applies the test patch later, after patch capture, before the final test run; this keeps those tests hidden from the agent.

When configured, network is detached after setup. Actual isolation is Docker-based. It is not a VM/firecracker boundary; evaluate the trust model before processing untrusted workloads.

### 3.3 Generate a repair

`agent/react_loop.py::run_react` builds system/issue/tool context, calls the common `LLMClient`, parses the model's tool-call JSON, executes registered tools, adds observations to the conversation, and stops at a finish call or configured step/context limits. Tool calls and messages are appended to a `Trajectory`.

The tool registry in `agent/tools/__init__.py` includes:

| Tool | Purpose |
|---|---|
| `bash` | Run a command in the container after a safety check. |
| `file_read` | Read a repository file. |
| `file_write` | Write a repository file. |
| `file_edit` | Replace a specified text occurrence in a file. |
| `grep` | Search repository text. |
| `repo_map` | Produce a compact repository file map. |
| `test_run` | Spend one of the bounded test runs on selected tests. |
| `finish` | End the ReAct loop. The worker still captures/evaluates the final patch. |
| `repository_graph` | Bounded Python AST imports/symbols/static calls and related-test recommendations. |
| `semantic_code_search` | Symbol/docstring/source relevance with concept-term evidence; no embeddings. |
| `git_history` | Recent commits/patches affecting a relative path. |
| `symbol_edit` | Replace one unique Python symbol after validating the resulting AST. |

Tool argument models use Pydantic. File paths are resolved under the repository directory, and output is truncated to reduce prompt growth. The LLM client supports local vLLM and multiple cloud/OpenAI-compatible provider protocols through `agent/providers.py`; provider list/test and saved-connection UI are implemented in the dashboard.

### 3.4 Produce an execution-grounded reward

The worker captures a unified diff, applies the hidden SWE-bench test patch, and calls `sandbox/test_executor.py` for the instance's F2P and P2P tests. The executor attempts JSON test-report parsing and a verbose-output fallback, and returns per-test outcomes, timeout state, and counts.

`reward/exec_reward.py::compute_exec_reward` supplies the primary binary signal: success requires the requested F2P tests to pass and the P2P regression tests to remain passing; setup/patch/agent failures and timeouts are treated as failures. `reward/shaped_reward.py` computes an auxiliary dense score from pass rates and patch characteristics (including edit-size/empty-patch/timeout penalties and optional PRM contribution). `reward/prm.py` contains an optional judge hook, disabled by default. The rollout stores a configured weighted `reward` and preserves binary execution separately in component details/metadata. Default rollout weights are execution=1, shaped=0, PRM=0. GRPO uses selected weights and optimizer-step shaped decay; execution success is never inferred from a positive blended reward.

### 3.5 Record outputs

`agent/trajectory.py` represents messages, tool calls, provenance, reward details, and final patch, and writes/reads JSONL records. The worker writes a local JSONL trajectory, then makes best-effort MinIO upload and PostgreSQL persistence attempts. Failures are logged without throwing away the local record. The worker also appends uniquely named Parquet replay shards and hashed sidecars for patches/test logs. Local evidence remains available if bounded remote persistence fails.

### 3.6 Train and evaluate

Training code is present for HumanEvalPack SFT, offline DPO pair assembly/training, and a TRL GRPO wrapper with an execution-reward callback. Heavy train CLI commands are guarded by `SWE_RL_ALLOW_HEAVY_TRAINING`; they require local model weights and a suitable ML/GPU environment. The CLI and console expose SFT, DPO and GRPO; reviewed exports provide SFT examples and same-instance DPO preferences. Trainer callbacks provide live metrics, control boundaries and checkpoint events.

Evaluation runs the agent for each requested instance and writes SWE-bench-style prediction JSON. It can invoke the official SWE-bench harness and render markdown/JSON summary reports. The checked-in script defaults and CLI currently run a capped evaluation, not a complete leaderboard-sized benchmark unless explicitly configured.

## 4. Functional areas and source files

### Agent and model access

- `agent/llm_client.py` — normalized model response and HTTP model calls, token counts, model listing and chat completion handling.
- `agent/providers.py` — provider registry, active connection resolution, profile serialization, secure local file writes, clearing profiles, and public settings that omit the key.
- `agent/react_loop.py` — prompt assembly, tool schema formatting, tool-call extraction, turn/step limits, and agent loop.
- `agent/trajectory.py` — message/tool-call records and JSONL serialization.
- `agent/prompts/system.md`, `user_intro.md`, `reflection.md` — prompt templates loaded by the agent.
- `agent/tools/*.py` — tool contracts/implementations described above.

Provider credentials are not checked into the project. The dashboard profile defaults to `.llm-profile.json` in the app working directory and is gitignored. The profile contains OS-keyring references rather than plaintext keys. Approved OS credential backends are required; unavailable backends cause an error instead of plaintext fallback. Cloud inference affects generation, not local sandbox execution. Cloud keys do not provide remote access to train weights through TRL.

### Sandbox and execution

- `sandbox/docker_runner.py` — image lookup, lifecycle, bounded command execution, file transfer, diff capture, teardown.
- `sandbox/repo_setup.py` — clone/checkout, SWE-bench or generic setup, baseline snapshot.
- `sandbox/safety.py` — blocks a limited set of unsafe command patterns before shell execution; this is defense-in-depth, not a complete shell security policy.
- `sandbox/test_executor.py` — runs requested test selectors and parses results.

### Reward and training

- `reward/exec_reward.py` — terminal binary test reward.
- `reward/shaped_reward.py` — dense auxiliary score and configuration.
- `reward/prm.py` — optional process-reward scoring integration.
- `train/data_assembly.py` — converts trajectories to SFT examples or preference pairs; also formats HumanEvalPack.
- `train/sft_warmstart.py` — TRL/Transformers SFT setup, with PEFT/LoRA options in code/configuration.
- `train/dpo_trainer.py` — DPO trainer implementation.
- `train/grpo_trainer.py` — prompt construction, execution reward callback, TRL GRPOTrainer wrapper.
- `train/curriculum.py` — heuristic difficulty and failure-conditioned remediation/practice/retention staging.

### Rollouts and evaluation

- `rollout/worker.py` — authoritative single-rollout orchestration and cleanup.
- `rollout/ray_pool.py` — Ray actor pool, retry/budget/idempotency scaffolding.
- `rollout/replay_buffer.py` — local Parquet shard writer/reader.
- `rollout/repro.py` — replays recorded assistant outputs without a live LLM and compares selected invariants. This is not guaranteed bit-for-bit/environment observation replay.
- `eval/swebench_eval.py` — patch prediction generation and official harness entry point.
- `eval/report.py` — leaderboard-style markdown and summary report output.

### Dashboard and UI

`dashboard/app.py` serves FastAPI and the enterprise console; `control.py` validates launch/lifecycle/selection/review/checkpoint/artifact requests; `jobs.py` persists and schedules local jobs; `job_runner.py` executes isolated worker processes. `analytics.py` computes evidence summaries/comparison, `run_index.py` incrementally discovers trajectories, `model_api.py` handles registry/fallback/prompt versions, and `telemetry.py` probes measured infrastructure.

The supplied enterprise UI foundation is in `static/index.html`, `styles.css`, `app.js`, `connections.css` and `connections.js`. `api.js` centralizes requests/SSE; `control-center.js` and `control-center.css` add operational screens and live interactions. No synthetic demo metrics are mixed into measured views.

#### What is in the UI

| Screen | Content and actions |
|---|---|
| Overview | Actual run/resolution/token/cost/runtime aggregates, outcome and trend charts, recent records and measured service health. Unknown measurements remain unavailable. |
| New Run | One or multiple issue IDs, Lite/Verified/Full, revision, provider/model/checkpoint, strategy, immutable prompt version, sampling/seed, step/test/cost limits, candidate count/selection, CPU/memory/timeout/network limits, reward weights and optional minimizer. |
| Runs | Search/filter/saved views; selection opens the inspector; running jobs expose lifecycle controls and persisted live events. |
| Experiments | Configuration fingerprints, linked runs and selection results; review and rerank candidate patches, including an explicit model critic request. |
| Compare | Two experiments, matched issue cohorts, both/only-A/only-B/neither outcomes, cost/runtime/reward comparison. Unmatched datasets are identified. |
| SWE-bench / Evaluations / Reports | Dataset and instance filters, evaluation launch, queued/running/completed states, parallelism, imported predictions, harness reports and downloadable artifacts. |
| Training Jobs | Guarded SFT/DPO/GRPO launch, model/dataset path or SWE-bench prompts, learning rate/group size/reward mix, progress/loss/reward events and checkpoint result. |
| Checkpoints | Completed training checkpoint provenance; evaluate, configure vLLM model identity, archive/restore and confirmed delete. Configuring a checkpoint does not hot-reload the external server. |
| Replay Buffer | Trajectory review status, approve/reject/annotate, reviewed-only SFT and paired DPO export. |
| Reward Analysis | Execution, shaped and PRM evidence, weighted values and failure distributions. Execution success remains separate from blended optimization reward. |
| Model Connections | Provider/model/endpoint, OS-managed credential save/remove, model discovery, explicit completion test, issue/code suggestion assistant. |
| Model Registry | User-declared tool/JSON/vision/context/pricing capabilities; measured API latency, tokens and rate-limit failures; explicit fallback configuration. |
| Prompts | Built-in strategy profiles plus immutable custom prompt versions and hashes. |
| Infrastructure | Queue and worker state; bounded Docker, PostgreSQL, Redis, MinIO, vLLM and GPU probes. Missing service/hardware produces unavailable values. |
| Artifacts | Paged artifact metadata, size and SHA256, bounded preview and download; paths are restricted to the output root. |
| Dataset Browser | Repository/language/test-count difficulty slices; revision-aware instance selection. Difficulty is a heuristic, not an official label. |
| Settings / API | Theme and local preferences; FastAPI OpenAPI reference at `/api/docs`. |

The enterprise HTML/CSS/JavaScript supplied in the second attachment provides the visual foundation. The new control-center modules connect it to executable backend actions. The responsive sidebar, dense tables, status badges, empty/error/loading states, focus styles and reduced-motion rules accompany light/dark themes. Command palette, global search, keyboard navigation and saved views support repeated work. Browser local storage contains preferences, never provider keys.

#### Run inspector

The three panes contain an issue/run list; primary detail tabs; and run metadata/evidence. Tabs cover overview, patch, tool timeline, F2P/P2P evidence, conversation, repository context, costs, rewards and failure analysis. Patch controls include file tree, unified/split views and whitespace filtering. Tool rows expose arguments, output, status and duration. Tests expose pass/fail/timeout plus captured final logs. Cost attribution uses per-turn/step usage and recorded pricing. Failure categories cite flags/test evidence; they do not assert an unproven root cause. JSON/artifact export preserves provenance.

#### API and operational scope

See [control-center.md](swe-rl-agent/docs/control-center.md) for lifecycle semantics, storage, all proposal items, credential protections and workload requirements. Read-only inspection works with the lightweight console installation. Workload execution requires full dependencies/Docker/model access; training also needs prepared GPU hardware and explicit opt-in. Host/origin/token checks protect local mutations; this is not a hosted multi-user authentication system.

### Database and observability

- `db.py` defines SQLAlchemy tables for `instances`, `model_checkpoints`, `trajectories`, `rewards`, and `eval_runs`, plus create-all and best-effort save/query helpers.
- `migrations/0001_initial.sql` is the corresponding PostgreSQL DDL. CLI `db migrate` calls SQLAlchemy `create_all`, despite the migration file's production-oriented comment; a migration tool is declared but not wired as a full migration workflow.
- `observability/logging.py` configures structured logs; `metrics.py` defines Prometheus counters/histograms; `sentry_init.py` configures optional Sentry; `observability/__init__.py` initializes them.
- `utils/cost.py` meters estimated token spend and enforces a per-rollout cap. Pool cost coordination is also sketched in Ray, but its guarantee depends on configured rates and successful worker settlement; it is not a provider billing guarantee.

## 5. Command line and common workflows

Run commands from `swe-rl-agent/` after installing dependencies (or through its Makefile). The console entry point is `swe-rl = swe_rl.cli:app`; module invocation is `python -m swe_rl.cli`.

| Command | Purpose / behavior |
|---|---|
| `make setup` | Editable package/dev install and sandbox image build. |
| `make services-up` / `make services-down` | Start/stop local PostgreSQL, Redis, MinIO, and Prometheus compose services. |
| `make migrate` | Create database tables using SQLAlchemy metadata. |
| `make smoke-deps` | Check Docker; probe PostgreSQL (warning-only if unavailable); load one dataset instance. |
| `make smoke` | Gold-patch smoke rollout after smoke dependency checks. |
| `python -m swe_rl.cli rollout one INSTANCE_ID` | Execute one model-driven rollout. |
| `python -m swe_rl.cli rollout pool` | Process a small hard-coded Lite sample via Ray pool. |
| `make warmstart` | HumanEvalPack SFT; script requires explicit heavy-training opt-in. |
| `make train` | GRPO; script requires explicit heavy-training opt-in. |
| `make eval CHECKPOINT=... SUBSET=lite` | Calls the Lite evaluation script; see caveats below. |
| `make leaderboard RUN_ID=...` | Copies prediction/report assets into a tarball submission package. |
| `make dashboard` | Start local dashboard on 127.0.0.1:8088. |
| `make repro RUN_ID=...` | Locate trajectory and run the replay comparison. |
| `make test`, `make test-integration`, `make test-e2e` | Unit, Docker integration, and gold-patch e2e suites. |
| `make lint`, `make fmt`, `make clean` | Ruff/mypy checks, format/fix, and remove build/cache artifacts. |

Useful direct CLI commands include `smoke deps`, `smoke rollout [--gold]`, `db migrate`, `rollout one`, `rollout pool`, `eval lite`, `train sft`, `train grpo`, `repro`, and `dashboard`.

### Setup notes

1. Use Python 3.11 or 3.12 for the full ML package (`pyproject.toml` requires `>=3.11,<3.13`). The lightweight dashboard has a separate requirements file and the README notes it can run on newer Python.
2. Install the project from `swe-rl-agent/`, configure `.env` from `.env.example` as needed, and have Docker available for rollouts.
3. `make setup` builds the sandbox image; this image build downloads Ubuntu/Miniconda and creates several Python environments, so it takes substantial time/storage.
4. For a dashboard-only setup, install `requirements-dashboard.txt`, then run `PYTHONPATH=src .venv-dashboard/bin/python -m uvicorn swe_rl.dashboard.app:app --host 127.0.0.1 --port 8088`.
5. To use local inference, start an OpenAI-compatible vLLM endpoint on the configured host/port. Cloud inference requires the user's own provider key/profile.
6. Model training needs the guarded opt-in `SWE_RL_ALLOW_HEAVY_TRAINING=YES` through the supplied scripts, appropriate model access, and capable compute. Do not enable it casually on a workstation.

## 6. Configuration

`settings.py` uses Pydantic Settings and reads environment variables plus `.env`; settings are case-insensitive. `.env.example` is the inventory. Key groups:

| Area | Important settings |
|---|---|
| Model/Hugging Face | `HF_TOKEN`, `MODEL_NAME`, `MODEL_REVISION`, `TOKENIZER_NAME` |
| Local vLLM | `VLLM_HOST`, `VLLM_PORT`, `VLLM_API_KEY` |
| Shared inference profile | `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_PROFILE_PATH` |
| Sandbox | `SANDBOX_IMAGE`, `SANDBOX_MEM_GB`, `SANDBOX_CPUS`, `SANDBOX_WALLCLOCK_S`, `SANDBOX_MAX_OUTPUT_BYTES`, `SANDBOX_DISABLE_NETWORK_AFTER_INSTALL` |
| PostgreSQL | `DATABASE_URL` or `POSTGRES_HOST/PORT/DB/USER/PASSWORD` |
| Redis/MinIO | `REDIS_URL`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET` |
| Cost | `MAX_DOLLARS_PER_RUN`, `MAX_DOLLARS_PER_POOL`, `DOLLAR_PER_1K_TOKENS_IN`, `DOLLAR_PER_1K_TOKENS_OUT` |
| Observability | `SENTRY_DSN`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `PROMETHEUS_PORT`, `LOG_DIR`, `LOG_LEVEL` |
| Training | `TRAIN_OUTPUT_DIR`, `TRAIN_BATCH_SIZE`, `GRPO_GROUP_SIZE`, `GRPO_KL_COEF`, `GRPO_LR`, `SWE_RL_ALLOW_HEAVY_TRAINING` |

`configs/base.yaml` and subdirectories provide Hydra-style model/algo/reward/eval configurations, but most currently exposed CLI flows construct dataclasses/settings directly; verify the exact consuming entry point before assuming every YAML value affects a command.

## 7. Configuration presets

- `configs/model/qwen25_coder_14b.yaml` — Qwen2.5-Coder-14B-Instruct, bf16, 32k max length, tensor parallel size 1.
- `configs/model/deepseek_coder_v2_lite.yaml` — DeepSeek Coder V2 Lite, bf16, 32k max length, remote-code trust enabled.
- `configs/algo/grpo.yaml` — group size 8, KL 0.04, LR 1e-6, batches/steps/sync intervals.
- `configs/algo/dpo.yaml` — DPO beta/loss/LR and sequence limits.
- `configs/reward/exec_grounded.yaml` — binary/shaped weights, shaped decay, patch penalties, PRM disabled.
- `configs/eval/swebench_lite.yaml` — dataset/split, sample count, prediction/report output paths.

These values are experiment presets; CLI defaults in Python and shell scripts may differ. Example: YAML sets eval max instances to null, whereas the shell script defaults to 300 and CLI defaults to 20.

## 8. Storage and artifacts

| Artifact | Format/location | Notes |
|---|---|---|
| Trajectory | JSONL under a selected output directory | Primary durable local record; contains conversation, calls, patch, reward details and metadata. |
| Replay buffer | Parquet `shard_*.parquet` under a local root | Append-only shard utility; separate from automatic worker JSONL path. |
| Prediction file | JSON list, often `outputs/predictions/<run>.json` | One SWE-bench patch record per instance. |
| Evaluation report | `outputs/reports/<run>/` | Markdown leaderboard and summary JSON. |
| Dashboard run index | Incremental SQLite cache or optional PostgreSQL trajectory index | Canonical evidence is local JSONL; operational jobs merge with indexed records. |
| PostgreSQL | Schema in `db.py` and `migrations/0001_initial.sql` | Worker attempts best-effort metadata/reward save. |
| MinIO/S3 | Configured bucket | Worker attempts a MinIO upload; local record remains if it fails. |
| Model/checkpoint files | Local output/checkpoint directories | Large generated artifacts are gitignored. |

Generated data, environments, model files, and credentials should not be committed. `.gitignore` excludes `.env`, model profile, logs, outputs, caches, weights, and Parquet/JSONL.

## 9. Tests and quality checks

The test suite is arranged as:

- `tests/unit/` — schema, tool paths, safety, patch parsing, reward math, model providers, ReAct loop, trajectory serialization, replay buffer, dashboard API, and data assembly.
- `tests/integration/test_docker_runner.py` — Docker-backed container and runner behavior; requires a working daemon/image.
- `tests/e2e/test_gold_patch_rollout.py` — full gold-patch sandbox rollout; requires dataset/network/Docker setup.
- `tests/*-ui.cjs` — Node/jsdom interaction checks for dashboard and connection UI.
- `tests/conftest.py` — common fixtures and environment setup.

GitHub Actions defines lint/format/unit on pushes and pull requests, a Docker integration job on pushes, and a self-hosted GPU nightly-eval job only if a schedule event is configured. The README/docs report earlier validation, but that historical record is not equivalent to running the current full test matrix in the user's environment.

## 10. Docker, services, and infrastructure

- `docker/sandbox.Dockerfile` — Ubuntu 22.04, common build/test tools, Miniconda environments for Python 3.5–3.11, non-root `agent` user.
- `docker/learner.Dockerfile` — NVIDIA PyTorch base image and project install for training/rollout work.
- `docker/compose.dev.yaml` — local PostgreSQL, Redis, MinIO, and Prometheus services.
- `docker/prometheus.yml` — sample scrape targets for workers/learner.
- `infra/main.tf` — starter AWS Terraform resources: VPC/NAT, S3, RDS PostgreSQL, ElastiCache Redis, ECS cluster, ECR repositories, IAM role, Secrets Manager secrets.
- `infra/rollout_task.tf` — ECS EC2 task definition with privileged Docker access and rollout command.
- `infra/README.md` — deployment notes and intended AWS setup.

The Terraform files are infrastructure scaffolding. Inspect credentials/state/backends, IAM permissions, security groups, Docker-in-Docker privilege, network boundaries, and missing resources before applying to a real AWS account. The `infra` README's described topology can be ahead of what is actually provisioned in Terraform.

## 11. File-by-file inventory

This index covers the meaningful checked-in files under `swe-rl-agent/`; generated caches/logs are omitted.

### Root and packaging

- `README.md` — project overview, operational GUI, setup, current limitations and claims.
- `pyproject.toml` — package metadata, Python range, runtime/dev dependencies, CLI script, Ruff/mypy/pytest settings.
- `Makefile` — setup, service, smoke, training, eval, dashboard, test, lint, format, and clean targets.
- `.env.example` — documented environment variables with example local defaults.
- `.gitignore` — excludes secrets, local environments, outputs, model weights, trajectory records, caches.
- `requirements-dashboard.txt` — small dependency set for dashboard-only use.

### Configurations

- `configs/base.yaml` — experiment defaults and interpolated paths/settings.
- `configs/algo/grpo.yaml`, `configs/algo/dpo.yaml` — algorithm hyperparameters.
- `configs/eval/swebench_lite.yaml` — evaluation dataset and output defaults.
- `configs/model/qwen25_coder_14b.yaml`, `configs/model/deepseek_coder_v2_lite.yaml` — model/server presets.
- `configs/reward/exec_grounded.yaml` — binary/shaped reward and PRM preset.

### Documentation

- `docs/architecture.md` — architecture narrative and known integration gaps.
- `docs/grpo_notes.md` — GRPO rationale, tuning notes, and suggested metrics.
- `docs/gui-validation.md` — dashboard validation record and explicitly unverified areas.
- `docs/model-connections.md` — provider behavior, profile handling, security expectations, and training limits.
- `docs/reward_design.md` — reward objective and design intent.
- `docs/runbook.md` — operational procedures and historical phase log.
- `infra/README.md` — intended infrastructure deployment guide.

### Application modules

- `src/swe_rl/__init__.py` — package marker/version exports.
- `src/swe_rl/cli.py` — Typer CLI command tree and orchestration entry points.
- `src/swe_rl/settings.py` — typed runtime configuration.
- `src/swe_rl/db.py` — SQLAlchemy models, engine/session, instance and trajectory persistence.
- `src/swe_rl/data/__init__.py` — public data exports.
- `src/swe_rl/data/instance_schema.py` — canonical SWE-bench schema/normalization.
- `src/swe_rl/data/swebench_loader.py` — lazy SWE-bench dataset loading and instance lookup.
- `src/swe_rl/data/humanevalpack_loader.py` — lazy HumanEvalPack dataset loader.
- `src/swe_rl/sandbox/docker_runner.py` — Docker container/command/file/diff lifecycle.
- `src/swe_rl/sandbox/repo_setup.py` — target repository clone/install/baseline preparation.
- `src/swe_rl/sandbox/safety.py` — shell command safety rules.
- `src/swe_rl/sandbox/test_executor.py` — targeted tests and outcome parsing.
- `src/swe_rl/agent/llm_client.py` — model HTTP interface and normalized response.
- `src/swe_rl/agent/providers.py` — provider registry and local active connection profiles.
- `src/swe_rl/agent/react_loop.py` — model-driven reasoning/tool loop.
- `src/swe_rl/agent/trajectory.py` — conversation/action/provenance records and JSONL IO.
- `src/swe_rl/agent/prompts/__init__.py` — prompt resource loading.
- `src/swe_rl/agent/prompts/system.md`, `user_intro.md`, `reflection.md` — agent prompt content.
- `src/swe_rl/agent/tools/__init__.py` — default toolset registration.
- `src/swe_rl/agent/tools/base.py` — tool context/result/base class/output truncation.
- `src/swe_rl/agent/tools/bash.py` — sandbox shell tool.
- `src/swe_rl/agent/tools/file_edit.py` — read/write/replace file tools and path containment.
- `src/swe_rl/agent/tools/grep.py` — repository search tool.
- `src/swe_rl/agent/tools/repo_map.py` — repository inventory tool.
- `src/swe_rl/agent/tools/test_run.py` — bounded targeted test tool.
- `src/swe_rl/agent/tools/finish.py` — stop-loop tool.
- `src/swe_rl/reward/exec_reward.py` — primary test-derived success metric.
- `src/swe_rl/reward/shaped_reward.py` — dense auxiliary reward computation.
- `src/swe_rl/reward/prm.py` — optional process reward scorer abstraction.
- `src/swe_rl/rollout/worker.py` — one full rollout and artifact persistence attempts.
- `src/swe_rl/rollout/ray_pool.py` — Ray actors, pool dispatch and cost reservations.
- `src/swe_rl/rollout/replay_buffer.py` — local Parquet shard buffer.
- `src/swe_rl/rollout/repro.py` — recorded-response rerun and invariant comparison.
- `src/swe_rl/train/data_assembly.py` — SFT and DPO dataset conversion.
- `src/swe_rl/train/sft_warmstart.py` — SFT/LoRA training function.
- `src/swe_rl/train/dpo_trainer.py` — DPO training function.
- `src/swe_rl/train/grpo_trainer.py` — TRL GRPO training and environment reward callback.
- `src/swe_rl/train/curriculum.py` — heuristic curriculum ordering.
- `src/swe_rl/serve/vllm_server.py` — constructs/launches local vLLM command.
- `src/swe_rl/serve/learner_sync.py` — vLLM LoRA load/unload requests.
- `src/swe_rl/eval/swebench_eval.py` — rollout predictions and official harness invocation.
- `src/swe_rl/eval/report.py` — report rendering.
- `src/swe_rl/dashboard/app.py` — FastAPI dashboard API, health, run discovery, provider actions.
- `src/swe_rl/dashboard/static/index.html`, `app.js`, `styles.css` — dashboard shell, behavior, styling.
- `src/swe_rl/dashboard/static/connections.js`, `connections.css` — provider settings view and styles.
- `src/swe_rl/observability/__init__.py` — observability initialization.
- `src/swe_rl/observability/logging.py` — structured logging configuration.
- `src/swe_rl/observability/metrics.py` — Prometheus metric definitions/server setup.
- `src/swe_rl/observability/sentry_init.py` — optional error reporting initialization.
- `src/swe_rl/utils/cost.py` — local token-cost meter/cap exception.
- `src/swe_rl/utils/git.py` — git command, diff and current-commit helpers.
- `src/swe_rl/utils/patch.py` — unified diff parsing/statistics/empty detection.
- `src/swe_rl/utils/retry.py` — retry helper.

### Operations, infra, and tests

- `scripts/00_bootstrap.sh` — install, optional HF login, build image and create output directories.
- `scripts/01_build_sandbox.sh` — build/list sandbox image.
- `scripts/02_smoke_rollout.sh` — gold-patch smoke wrapper.
- `scripts/03_warmstart_sft.sh` — guarded SFT wrapper.
- `scripts/04_grpo_train.sh` — guarded GRPO wrapper.
- `scripts/05_eval_lite.sh` — prediction/evaluation/report script.
- `scripts/06_leaderboard_submit.sh` — submission asset copy and tarball packaging.
- `docker/compose.dev.yaml` — developer service stack.
- `docker/sandbox.Dockerfile`, `docker/learner.Dockerfile` — sandbox and learner images.
- `docker/prometheus.yml` — example scrape configuration.
- `infra/main.tf`, `infra/rollout_task.tf` — AWS Terraform starter resources and rollout task.
- `migrations/0001_initial.sql` — PostgreSQL DDL.
- `.github/workflows/ci.yml` — lint/test/integration/nightly workflow definitions.
- `tests/conftest.py` — pytest fixtures.
- `tests/unit/test_cost.py` — cost cap behavior.
- `tests/unit/test_dashboard.py` — dashboard endpoint tests.
- `tests/unit/test_data_assembly.py` — training data conversion tests.
- `tests/unit/test_exec_reward.py`, `test_shaped_reward.py`, `test_grpo_reward.py` — reward behavior.
- `tests/unit/test_gui_regressions.py` — regressions around dashboard discovery/behavior.
- `tests/unit/test_instance_schema.py` — instance validation/normalization.
- `tests/unit/test_patch.py` — patch parsing helpers.
- `tests/unit/test_providers.py` — provider selection/profile/client behavior.
- `tests/unit/test_react_loop.py` — loop/tool interaction.
- `tests/unit/test_replay_buffer.py` — Parquet buffer.
- `tests/unit/test_safety.py` — shell filter behavior.
- `tests/unit/test_test_executor_parsing.py` — test result parsing.
- `tests/unit/test_tool_paths.py` — file path containment.
- `tests/unit/test_trajectory.py` — trajectory serialization.
- `tests/integration/test_docker_runner.py` — Docker runner integration.
- `tests/e2e/test_gold_patch_rollout.py` — end-to-end gold patch path.
- `tests/dashboard-ui.cjs`, `tests/connections-ui.cjs` — JS UI interaction checks.

## 12. Implemented scope and known limitations

1. **Operational local control center:** launch repair/evaluation/guarded training, durable SQLite queue, two subprocess slots, lifecycle actions, persisted SSE events and recovery. Ray is a separate CLI pool; Redis is not the active queue broker.
2. **Evidence and comparison:** enterprise inspector, split/unified patches, complete captured tests/logs, usage/reward components, failure heuristics, matched experiment cohorts and pass@k eligibility. Failed attempts count in repair metrics; training/evaluation jobs do not. Missing costs remain unknown.
3. **Agent enhancements:** versioned strategies/custom prompts, conditional reflection, bounded context/test adaptation, Python AST graph/test recommendations, semantic symbol/docstring relevance search without embeddings, git history and AST edits. Explorer/implementer/reviewer is a staged runtime using the same model, with recorded roles; it does not imply three separate models.
4. **Candidates and minimization:** bounded independent candidates, explicit reranking/critic, hidden-test-safe patch capture and bounded execution-verified hunk minimizer. Selection informed by grading is identified; do not call it official pass@1.
5. **Training:** exposed SFT/DPO/GRPO, callback progress/checkpoints, reviewed export and configured weighted rewards with shaped decay. Linux/GPU/TRL/vLLM compatibility and throughput need target-host validation. Configuring a checkpoint's model identity requires a separate serving reload.
6. **Storage:** JSONL/hashed sidecars/Parquet, local job journal/index, optional PostgreSQL trajectory index and bounded PostgreSQL/MinIO persistence. No distributed atomic transaction across stores; back up local artifacts.
7. **Credentials:** approved OS keyring required for saved keys; environment keys remain supported. Legacy plaintext profiles migrate when read. Model capabilities/pricing are user declarations and API telemetry is measured in the current process.
8. **Benchmark variants:** Lite/Verified/Full automatic prediction generation with actual parallelism and all k candidates. Multilingual/Multimodal harness support accepts imported predictions; the current Python repair runtime does not implement vision or language-specific setup.
9. **Advanced scope:** immutable configuration enables prompt/reward/strategy/model/checkpoint comparisons. No autonomous tournament scheduler, whole-repository embedding store or distributed durable Ray fleet is provided.
10. **Replay:** recorded assistant outputs can be replayed, but exact model/environment determinism is not guaranteed. Docker setup/install may need network before isolation; Docker remains a host trust boundary.
11. **Validation:** Python unit tests and terminal jsdom UI interactions, lint/format/syntax and HTTP smoke checks are used. Real paid provider calls, official Docker grading and GPU training were not run here. No benchmark score, cost efficiency, deployed cloud readiness or GPU throughput is claimed.

## 13. Recommended mental model

`rollout/worker.py` owns repair execution and cleanup. `dashboard/jobs.py` owns local scheduling and durable lifecycle, `dashboard/job_runner.py` connects that lifecycle to workloads, and `dashboard/static/` exposes operations and evidence. `train/` consumes local model weights, prompts and reward callbacks; it needs a validated target environment. `infra/` is starter infrastructure rather than a proven production deployment. Consult [control-center.md](swe-rl-agent/docs/control-center.md) for an explicit mapping of all 33 proposal areas and their scoped limitations.

## 14. Source documents

This overview is a synthesis of the checked-in project documentation and implementation. Useful focused references are [`swe-rl-agent/README.md`](swe-rl-agent/README.md), [`docs/architecture.md`](swe-rl-agent/docs/architecture.md), [`docs/reward_design.md`](swe-rl-agent/docs/reward_design.md), [`docs/grpo_notes.md`](swe-rl-agent/docs/grpo_notes.md), [`docs/runbook.md`](swe-rl-agent/docs/runbook.md), [`docs/model-connections.md`](swe-rl-agent/docs/model-connections.md), and [`docs/gui-validation.md`](swe-rl-agent/docs/gui-validation.md).


## 15. Upgrade source modules and complete callable inventory

The inventory below is generated from source AST declarations, including public and internal functions/methods and route handlers. It is a navigation aid, not proof that every hardware path has run. Source is authoritative for parameter types/defaults and detailed algorithms.

### `swe-rl-agent/src/swe_rl/__init__.py`

swe-rl-agent: Code-Repair Agent with Execution-Grounded RL on SWE-Bench.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/agent/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/agent/credentials.py`

OS credential store. No plaintext or alternate file-backend fallback.

- `_backend()` (line 10) — Implementation function; see source.
- `secret_reference(profile, provider, endpoint)` (line 32) — Implementation function; see source.
- `store_secret(reference, value)` (line 37) — Implementation function; see source.
- `load_secret(reference)` (line 46) — Implementation function; see source.
- `delete_secret(reference)` (line 57) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/intelligence.py`

Bounded Python AST graph and evidence-based test recommendations.

- `analyze_repository(root, max_files)` (line 8) — Implementation function; see source.
- `recommend_tests(graph, changed_files)` (line 64) — Implementation function; see source.
- `semantic_code_search(sources, query, limit)` (line 93) — Rank AST definitions by conceptual term expansion and symbol/docstring relevance.
- `semantic_code_search.terms(text)` (line 104) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/llm_client.py`

Shared inference client for local vLLM and cloud BYOK providers.

- `LLMResponse` (line 17) — Class/data model.
- `LLMClient` (line 25) — Class/data model.
- `LLMClient.__init__(self, base_url, api_key, model, provider, config, timeout, transport, allow_fallback)` (line 26) — Implementation function; see source.
- `LLMClient._headers(self)` (line 64) — Implementation function; see source.
- `LLMClient._request(self, method, path, **kwargs)` (line 69) — Implementation function; see source.
- `LLMClient.models(self)` (line 111) — Implementation function; see source.
- `LLMClient.chat(self, messages, **kwargs)` (line 123) — Use an explicitly configured fallback only for transient failures.
- `LLMClient._chat(self, messages, temperature, top_p, max_tokens, seed, stop)` (line 171) — Implementation function; see source.
- `LLMClient.close(self)` (line 239) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/model_registry.py`

User-declared model capabilities and measured local request telemetry.

- `record_request(provider, model, duration_s, status, tokens_in, tokens_out)` (line 17) — Implementation function; see source.
- `request_telemetry()` (line 52) — Implementation function; see source.
- `_path()` (line 57) — Implementation function; see source.
- `models()` (line 61) — Implementation function; see source.
- `register_model(row)` (line 70) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/profiles.py`

Versioned repair strategies persisted with every trajectory.

- `AgentProfile` (line 7) — Class/data model.
- `get_profile(name)` (line 27) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/prompt_registry.py`

Immutable custom prompt versions with content fingerprints.

- `_connect()` (line 14) — Implementation function; see source.
- `list_prompts()` (line 22) — Implementation function; see source.
- `register_prompt(name, version, content)` (line 38) — Implementation function; see source.
- `get_prompt(identifier)` (line 61) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/prompts/__init__.py`

Prompt templates loaded from .md files in this directory.

- `load(name)` (line 10) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/providers.py`

Provider registry and shared, server-side inference profile.

- `Provider` (line 23) — Class/data model.
- `Connection` (line 113) — Class/data model.
- `Connection.protocol(self)` (line 120) — Implementation function; see source.
- `Connection.public(self)` (line 123) — Implementation function; see source.
- `connection(provider, model, api_key, base_url)` (line 135) — Implementation function; see source.
- `active_connection()` (line 169) — Implementation function; see source.
- `save_connection(config)` (line 197) — Save public configuration atomically; keep keys in the OS credential store.
- `clear_connection()` (line 234) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/react_loop.py`

ReAct loop: planner -> tool dispatch -> observation -> reflection -> next action.

- `ReactConfig` (line 24) — Class/data model.
- `ReactConfig.__post_init__(self)` (line 40) — Implementation function; see source.
- `ReactRunResult` (line 55) — Class/data model.
- `_extract_tool_call(text)` (line 62) — Extract a single JSON tool call from model text. Tolerant of code fences.
- `_format_tools_block(tools)` (line 84) — Implementation function; see source.
- `_format_intro(instance)` (line 100) — Implementation function; see source.
- `_format_system(cfg)` (line 118) — Implementation function; see source.
- `run_react(llm, ctx, instance, trajectory, config, cost, event_callback, control_callback)` (line 130) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/__init__.py`

Agent tool registry. Each tool exposes pydantic input + JSON-schema description + ToolResult.

- `default_toolset(ctx)` (line 18) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/base.py`

Tool base class and shared context.

- `ToolContext` (line 17) — Class/data model.
- `ToolResult` (line 26) — Class/data model.
- `ToolResult.to_dict(self)` (line 33) — Implementation function; see source.
- `Tool` (line 43) — Class/data model.
- `Tool.__init__(self, ctx)` (line 48) — Implementation function; see source.
- `Tool.schema(self)` (line 51) — Implementation function; see source.
- `Tool.__call__(self, **kwargs)` (line 58) — Implementation function; see source.
- `Tool._run(self, args)` (line 66) — Implementation function; see source.
- `truncate(s, limit)` (line 70) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/bash.py`

Bash tool — execute a shell command inside the sandbox.

- `BashArgs` (line 10) — Class/data model.
- `BashTool` (line 16) — Class/data model.
- `BashTool._run(self, args)` (line 26) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/file_edit.py`

File read / write / search-replace edit tools.

- `_resolve(repo_dir, path)` (line 19) — Implementation function; see source.
- `FileReadArgs` (line 26) — Class/data model.
- `FileReadTool` (line 32) — Class/data model.
- `FileReadTool._run(self, args)` (line 37) — Implementation function; see source.
- `FileWriteArgs` (line 55) — Class/data model.
- `FileWriteTool` (line 60) — Class/data model.
- `FileWriteTool._run(self, args)` (line 65) — Implementation function; see source.
- `FileEditArgs` (line 82) — Class/data model.
- `FileEditTool` (line 88) — Class/data model.
- `FileEditTool._run(self, args)` (line 96) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/finish.py`

Finish tool: agent signals completion. Returning ok=True ends the rollout.

- `FinishArgs` (line 10) — Class/data model.
- `FinishTool` (line 15) — Class/data model.
- `FinishTool._run(self, args)` (line 23) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/grep.py`

Ripgrep-backed search across the repo.

- `GrepArgs` (line 13) — Class/data model.
- `GrepTool` (line 21) — Class/data model.
- `GrepTool._run(self, args)` (line 26) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/intelligence.py`

Sandbox repository intelligence with bounded source collection.

- `IntelligenceArgs` (line 14) — Class/data model.
- `RepositoryGraphTool` (line 19) — Class/data model.
- `RepositoryGraphTool._run(self, args)` (line 24) — Implementation function; see source.
- `HistoryArgs` (line 41) — Class/data model.
- `GitHistoryTool` (line 46) — Class/data model.
- `GitHistoryTool._run(self, args)` (line 51) — Implementation function; see source.
- `SymbolEditArgs` (line 63) — Class/data model.
- `SymbolEditTool` (line 69) — Class/data model.
- `SymbolEditTool._run(self, args)` (line 74) — Implementation function; see source.
- `_collect_sources(ctx, max_files)` (line 125) — Implementation function; see source.
- `SemanticSearchArgs` (line 138) — Class/data model.
- `SemanticCodeSearchTool` (line 144) — Class/data model.
- `SemanticCodeSearchTool._run(self, args)` (line 149) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/repo_map.py`

Repo-map tool: produce a compact symbol overview of the repo via ctags.

- `RepoMapArgs` (line 13) — Class/data model.
- `RepoMapTool` (line 19) — Class/data model.
- `RepoMapTool._run(self, args)` (line 27) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/tools/test_run.py`

Test runner tool — capped to N runs per trajectory by ToolContext.

- `TestRunArgs` (line 10) — Class/data model.
- `TestRunTool` (line 19) — Class/data model.
- `TestRunTool._run(self, args)` (line 27) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/agent/trajectory.py`

Serializable rollout trajectory record.

- `Message` (line 14) — Class/data model.
- `ToolCall` (line 21) — Class/data model.
- `Trajectory` (line 33) — Class/data model.
- `Trajectory.new(cls, instance_id, seed, temperature, top_p, model_name, sandbox_image_digest, checkpoint_sha)` (line 55) — Implementation function; see source.
- `Trajectory.add_message(self, role, content, name)` (line 78) — Implementation function; see source.
- `Trajectory.add_tool_call(self, step, tool, arguments, result, duration_s, tokens_in, tokens_out, cost_usd)` (line 81) — Implementation function; see source.
- `Trajectory.to_dict(self)` (line 106) — Implementation function; see source.
- `Trajectory.write_jsonl(self, dest_dir)` (line 109) — Implementation function; see source.
- `Trajectory.from_jsonl(cls, path)` (line 117) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/cli.py`

Top-level CLI for swe-rl-agent.

- `dashboard(host, port)` (line 31) — Start the local operations control center.
- `db_migrate()` (line 45) — Create all tables from the SQLAlchemy schema.
- `smoke_deps()` (line 58) — Verify imports + Docker daemon + DB reachable.
- `smoke_rollout(instance_id, apply_gold, output_dir)` (line 98) — Implementation function; see source.
- `rollout_one(instance_id, output_dir, seed, max_steps, strategy, test_budget, minimize_patch)` (line 131) — Implementation function; see source.
- `rollout_candidates(instance_id, count, workers, selection, output_dir)` (line 160) — Generate independent repair candidates and report measured pass@k.
- `rollout_pool(output_dir, n_workers, seeds_per_instance)` (line 199) — Run a distributed Ray pool to process rollouts.
- `eval_lite(max_instances, k_samples, output_predictions, report_dir, run_id, base_model, invoke_harness, subset, parallelism, model)` (line 232) — Implementation function; see source.
- `train_sft(output_dir, use_humanevalpack, dataset_path)` (line 291) — Implementation function; see source.
- `train_dpo(dataset_path, output_dir)` (line 319) — Train on reviewed preference JSON/JSONL (prompt/chosen/rejected).
- `train_grpo(max_instances, output_dir)` (line 332) — Implementation function; see source.
- `repro(run_id, trajectory_dir)` (line 364) — Replay recorded assistant messages and compare execution invariants.

### `swe-rl-agent/src/swe_rl/dashboard/__init__.py`

Read-only local operations dashboard.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/dashboard/analytics.py`

Evidence-derived summaries. Unknown measurements remain unknown.

- `_duration(row)` (line 12) — Implementation function; see source.
- `failure_analysis(row)` (line 26) — Implementation function; see source.
- `normalize_run(row)` (line 67) — Implementation function; see source.
- `_average(rows, field)` (line 98) — Implementation function; see source.
- `pass_at_k(rows, k)` (line 107) — Unbiased estimator on groups with at least k independent candidates.
- `summarize_runs(rows)` (line 129) — Implementation function; see source.
- `compare_experiments(rows, left, right)` (line 186) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/app.py`

Repair evidence, local model configuration, and provider inference API.

- `safe_validation(request, exc)` (line 43) — Implementation function; see source.
- `protect_settings(request)` (line 48) — Implementation function; see source.
- `ModelSettings` (line 56) — Class/data model.
- `submitted_config(body)` (line 64) — Implementation function; see source.
- `providers()` (line 85) — Implementation function; see source.
- `model_settings(body)` (line 113) — Implementation function; see source.
- `reset_model_settings()` (line 123) — Implementation function; see source.
- `list_models(body)` (line 129) — Implementation function; see source.
- `test_model(body)` (line 140) — Implementation function; see source.
- `AssistRequest` (line 153) — Class/data model.
- `assist(body)` (line 158) — Implementation function; see source.
- `_tcp_status(host, port)` (line 184) — Implementation function; see source.
- `_docker_status()` (line 192) — Implementation function; see source.
- `_trajectory_files()` (line 203) — Implementation function; see source.
- `_load_run(path)` (line 212) — Implementation function; see source.
- `health()` (line 241) — Implementation function; see source.
- `runs()` (line 272) — Implementation function; see source.
- `analytics()` (line 300) — Implementation function; see source.
- `compare(left, right)` (line 305) — Implementation function; see source.
- `index()` (line 315) — Implementation function; see source.
- `asset(asset)` (line 320) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/control.py`

Operational API for local jobs, reproducible experiments and artifact access.

- `Sandbox` (line 23) — Class/data model.
- `RunRequest` (line 30) — Class/data model.
- `RunRequest.validate_reward(cls, value)` (line 66) — Implementation function; see source.
- `EvaluationRequest` (line 76) — Class/data model.
- `TrainingRequest` (line 84) — Class/data model.
- `install(app, protect_settings, load_run, trajectory_files)` (line 100) — Implementation function; see source.
- `install.store()` (line 104) — Implementation function; see source.
- `install.manager()` (line 109) — Implementation function; see source.
- `install.ensure_manager()` (line 113) — Implementation function; see source.
- `install.get_job(job_id)` (line 121) — Implementation function; see source.
- `install.config(data)` (line 127) — Implementation function; see source.
- `install.control_lifespan(application)` (line 216) — Implementation function; see source.
- `install.jobs()` (line 229) — Implementation function; see source.
- `install.create_runs(body)` (line 233) — Implementation function; see source.
- `install.run(job_id)` (line 260) — Implementation function; see source.
- `install.action(job_id, action)` (line 284) — Implementation function; see source.
- `install.events(job_id, request, after)` (line 293) — Implementation function; see source.
- `install.events.stream()` (line 300) — Implementation function; see source.
- `install.experiments()` (line 320) — Implementation function; see source.
- `install.candidate_selection(experiment_id, strategy)` (line 327) — Implementation function; see source.
- `install.selected_candidates(experiment_id, strategy)` (line 410) — Implementation function; see source.
- `install.critic_selection(experiment_id, body)` (line 414) — Implementation function; see source.
- `install.evaluate(body)` (line 488) — Implementation function; see source.
- `install.evaluations()` (line 511) — Implementation function; see source.
- `install.train(body)` (line 515) — Implementation function; see source.
- `install.training()` (line 544) — Implementation function; see source.
- `install.allowed_artifact(path)` (line 547) — Implementation function; see source.
- `install.artifact_rows(max_rows)` (line 562) — Implementation function; see source.
- `install.artifacts(offset, limit)` (line 612) — Implementation function; see source.
- `install.artifact(artifact_id)` (line 618) — Implementation function; see source.
- `install.download_artifact(artifact_id)` (line 631) — Implementation function; see source.
- `install.datasets()` (line 643) — Implementation function; see source.
- `install.dataset_instances(dataset_id, repo, language, difficulty, limit, offset, revision)` (line 664) — Implementation function; see source.
- `install.checkpoints()` (line 717) — Implementation function; see source.
- `install.priority(job_id, body)` (line 736) — Implementation function; see source.
- `install.checkpoint_action(job_id, action)` (line 746) — Implementation function; see source.
- `install.annotation(job_id)` (line 779) — Implementation function; see source.
- `install.annotate(job_id, body)` (line 783) — Implementation function; see source.
- `install.training_metrics(job_id)` (line 796) — Implementation function; see source.
- `install.replay()` (line 823) — Implementation function; see source.
- `install.export_replay(body)` (line 842) — Implementation function; see source.
- `install.infrastructure()` (line 900) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/job_runner.py`

Isolated job entry point: invokes real rollout, harness and trainers.

- `execute(store, job_id)` (line 13) — Implementation function; see source.
- `execute.control()` (line 38) — Implementation function; see source.
- `execute.event(data)` (line 41) — Implementation function; see source.
- `main()` (line 294) — Implementation function; see source.
- `main.terminate(_signum, _frame)` (line 298) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/jobs.py`

Durable local job queue and event journal shared by API and isolated workers.

- `now()` (line 24) — Implementation function; see source.
- `JobStore` (line 28) — Class/data model.
- `JobStore.__init__(self, root)` (line 29) — Implementation function; see source.
- `JobStore.connect(self)` (line 41) — Implementation function; see source.
- `JobStore.get(self, job_id)` (line 46) — Implementation function; see source.
- `JobStore.list(self)` (line 53) — Implementation function; see source.
- `JobStore.experiments(self)` (line 60) — Implementation function; see source.
- `JobStore.experiment(self, config)` (line 64) — Implementation function; see source.
- `JobStore.create(self, kind, config, experiment_id)` (line 80) — Implementation function; see source.
- `JobStore.update(self, job_id, **changes)` (line 107) — Implementation function; see source.
- `JobStore.claim(self, job_id, capacity)` (line 129) — Implementation function; see source.
- `JobStore.event(self, job_id, data)` (line 146) — Implementation function; see source.
- `JobStore.events(self, job_id, after)` (line 153) — Implementation function; see source.
- `JobStore.directory(self, job_id)` (line 163) — Implementation function; see source.
- `JobManager` (line 168) — Bounded subprocess scheduler. Queued jobs survive API restarts.
- `JobManager.__init__(self, store, workers)` (line 171) — Implementation function; see source.
- `JobManager.start(self)` (line 177) — Implementation function; see source.
- `JobManager.stop(self)` (line 182) — Implementation function; see source.
- `JobManager._loop(self)` (line 187) — Implementation function; see source.
- `JobManager.action(self, job_id, action)` (line 265) — Implementation function; see source.
- `JobManager._worker_identity(pid, job_id)` (line 297) — Implementation function; see source.
- `JobManager._terminate_later(self, job_id, pid)` (line 309) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/model_api.py`

Model capability registry and explicit fallback configuration.

- `ModelRecord` (line 19) — Class/data model.
- `FallbackSettings` (line 30) — Class/data model.
- `PromptRecord` (line 39) — Class/data model.
- `fallback_path()` (line 45) — Implementation function; see source.
- `fallback_connection()` (line 49) — Implementation function; see source.
- `build_router(protect)` (line 63) — Implementation function; see source.
- `build_router.prompts()` (line 67) — Implementation function; see source.
- `build_router.save_prompt(body)` (line 73) — Implementation function; see source.
- `build_router.registry()` (line 82) — Implementation function; see source.
- `build_router.register(body)` (line 91) — Implementation function; see source.
- `build_router.telemetry()` (line 97) — Implementation function; see source.
- `build_router.fallback()` (line 101) — Implementation function; see source.
- `build_router.save_fallback(body)` (line 110) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/run_index.py`

Incremental trajectory index with an optional PostgreSQL primary index.

- `indexed_runs(loader)` (line 18) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/dashboard/telemetry.py`

Bounded service probes; report unavailable counters as null.

- `service_metrics()` (line 19) — Implementation function; see source.
- `metrics()` (line 158) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/data/__init__.py`

Dataset schemas and lazy loaders.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/data/humanevalpack_loader.py`

Lazy HumanEvalPack loader used by the optional SFT command.

- `load_humanevalpack(language, split)` (line 9) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/data/instance_schema.py`

Validated, normalized representation of a SWE-bench instance.

- `SWEBenchInstance` (line 11) — Class/data model.
- `SWEBenchInstance._parse_test_list(cls, value)` (line 30) — Implementation function; see source.
- `SWEBenchInstance.from_hf_row(cls, row)` (line 43) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/data/swebench_loader.py`

Lazy loaders for the official SWE-bench datasets.

- `load_swebench(subset, split, max_instances, revision)` (line 18) — Implementation function; see source.
- `load_one(instance_id, subset, revision)` (line 42) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/db.py`

SQLAlchemy schema + engine for persistence.

- `Base` (line 25) — Class/data model.
- `Instance` (line 29) — Class/data model.
- `ModelCheckpoint` (line 43) — Class/data model.
- `Trajectory` (line 54) — Class/data model.
- `Reward` (line 83) — Class/data model.
- `EvalRun` (line 94) — Class/data model.
- `make_engine(url)` (line 110) — Implementation function; see source.
- `init_db(url)` (line 114) — Create tables. For real deployments use alembic migrations under migrations/.
- `get_session(url)` (line 120) — Return a short-lived session for use in a ``with`` block.
- `save_trajectory(session, traj_data, rewards, storage_path)` (line 125) — Save a trajectory and its rewards to PostgreSQL.
- `trajectory_exists(session, instance_id, seed, checkpoint_sha)` (line 178) — Return whether this exact logical rollout has already been persisted.

### `swe-rl-agent/src/swe_rl/eval/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/eval/report.py`

Render a markdown leaderboard report from harness output.

- `render_report(run_id, predictions_path, harness_report, output_dir, base_model, notes)` (line 11) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/eval/swebench_eval.py`

Evaluate a policy on SWE-bench Lite/Verified and emit a `predictions.json`.

- `EvalConfig` (line 33) — Class/data model.
- `InstancePrediction` (line 47) — Class/data model.
- `InstancePrediction.to_dict(self)` (line 53) — Implementation function; see source.
- `generate_predictions(eval_config, react_config, llm, shaped_cfg, instances, instance_ids, event_callback, control_callback)` (line 61) — Implementation function; see source.
- `generate_predictions.emit(data)` (line 93) — Implementation function; see source.
- `generate_predictions.evaluate_one(index, inst)` (line 97) — Implementation function; see source.
- `run_official_harness(predictions_path, subset, split, max_workers, run_id, instance_ids)` (line 195) — Invoke the official swebench evaluation harness.
- `resolved_at_1(report)` (line 267) — Pull resolved@1 out of the official harness report.

### `swe-rl-agent/src/swe_rl/observability/__init__.py`

Observability bootstrap: structlog, Sentry, W&B, Prometheus.

- `init(component, enable_metrics_server)` (line 10) — Initialize logging + Sentry + Prometheus for a long-running entrypoint.

### `swe-rl-agent/src/swe_rl/observability/logging.py`

Structured JSON logging via structlog.

- `init_logging(component)` (line 17) — Implementation function; see source.
- `get_logger(name)` (line 55) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/observability/metrics.py`

Prometheus metrics + W&B run handle.

- `_Metrics` (line 19) — Class/data model.
- `_build_metrics()` (line 32) — Implementation function; see source.
- `init_metrics(component)` (line 59) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/observability/sentry_init.py`

Sentry bootstrap. No-op when DSN is not configured.

- `init_sentry(component)` (line 11) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/reward/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/reward/analysis.py`

Evidence-based failure labels; no model-generated claims are inferred.

- `classify_failure(trajectory)` (line 6) — Implementation function; see source.
- `pass_at_k(n, correct, k)` (line 54) — Unbiased estimator: 1 - C(n-c,k)/C(n,k), only for observed k <= n.

### `swe-rl-agent/src/swe_rl/reward/exec_reward.py`

Binary execution-grounded reward.

- `ExecRewardResult` (line 19) — Class/data model.
- `ExecRewardResult.to_dict(self)` (line 31) — Implementation function; see source.
- `compute_exec_reward(f2p_results, p2p_results, fail_to_pass, pass_to_pass)` (line 46) — Implementation function; see source.
- `ExecReward` (line 84) — Object form for use as a callable reward in trainers.
- `ExecReward.__call__(self, f2p_results, p2p_results, fail_to_pass, pass_to_pass)` (line 87) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/reward/prm.py`

Optional remote Process Reward Model (PRM).

- `PRMConfig` (line 22) — Class/data model.
- `PRMConfig.__post_init__(self)` (line 29) — Implementation function; see source.
- `PRMScorer` (line 38) — Class/data model.
- `PRMScorer.__init__(self, config)` (line 39) — Implementation function; see source.
- `PRMScorer.score(self, messages)` (line 43) — Return a bounded score; judge failures intentionally contribute zero.

### `swe-rl-agent/src/swe_rl/reward/shaped_reward.py`

Dense shaped reward = mix of partial test pass-rates with edit-size penalty.

- `ShapedRewardConfig` (line 22) — Class/data model.
- `ShapedRewardConfig.__post_init__(self)` (line 32) — Implementation function; see source.
- `ShapedRewardResult` (line 40) — Class/data model.
- `ShapedRewardResult.to_dict(self)` (line 44) — Implementation function; see source.
- `compute_shaped_reward(exec_result, patch, timed_out, lint_errors, prm_score, config)` (line 48) — Implementation function; see source.
- `ShapedReward` (line 88) — Class/data model.
- `ShapedReward.__call__(self, exec_result, patch, timed_out, lint_errors)` (line 91) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/rollout/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/rollout/ray_pool.py`

Ray-based distributed rollout pool.

- `PoolConfig` (line 27) — Class/data model.
- `PoolConfig.__post_init__(self)` (line 33) — Implementation function; see source.
- `GlobalCostMeter` (line 41) — Class/data model.
- `GlobalCostMeter.__init__(self, max_dollars)` (line 42) — Implementation function; see source.
- `GlobalCostMeter.reserve(self, cost)` (line 47) — Reserve a worst-case rollout cost before contacting the model.
- `GlobalCostMeter.settle(self, reserved, actual)` (line 59) — Implementation function; see source.
- `GlobalCostMeter.get_total(self)` (line 65) — Implementation function; see source.
- `RolloutActor` (line 70) — Class/data model.
- `RolloutActor.__init__(self, config)` (line 71) — Implementation function; see source.
- `RolloutActor.run(self, instance_dict, seed, react_kwargs, output_dir, shaped_kwargs, apply_gold_patch, checkpoint_sha, cost_meter)` (line 76) — Implementation function; see source.
- `submit_pool(instances, pool, react_config, output_dir, model_name, vllm_base_url, vllm_api_key, seeds_per_instance, apply_gold_patch, checkpoint_sha)` (line 134) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/rollout/replay_buffer.py`

Disk-backed append-only replay buffer (Parquet shards).

- `ReplayBuffer` (line 43) — Class/data model.
- `ReplayBuffer.__post_init__(self)` (line 47) — Implementation function; see source.
- `ReplayBuffer._row_for(self, traj)` (line 51) — Implementation function; see source.
- `ReplayBuffer.add(self, traj)` (line 79) — Implementation function; see source.
- `ReplayBuffer.flush(self)` (line 84) — Implementation function; see source.
- `ReplayBuffer.iter_shards(self)` (line 93) — Implementation function; see source.
- `ReplayBuffer.iter_rows(self, only_resolved)` (line 96) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/rollout/repro.py`

Replay a recorded trajectory deterministically.

- `MockLLMClient` (line 23) — Mocks the LLM by yielding exact responses recorded in a trajectory.
- `MockLLMClient.__init__(self, recorded_messages)` (line 26) — Implementation function; see source.
- `MockLLMClient.chat(self, messages, **kwargs)` (line 32) — Implementation function; see source.
- `replay_trajectory(path, output_dir)` (line 48) — Read a trajectory, parse its instance, and re-execute deterministically.

### `swe-rl-agent/src/swe_rl/rollout/worker.py`

Rollout worker — drives one trajectory end-to-end.

- `RolloutResult` (line 34) — Class/data model.
- `run_rollout(instance, llm, react_config, output_dir, shaped_cfg, apply_gold_patch, candidate_patch, checkpoint_sha, event_callback, control_callback)` (line 43) — Run a single rollout. Container is created and torn down inside this function.
- `run_rollout.emit(kind, **data)` (line 58) — Implementation function; see source.
- `_empty_results()` (line 385) — Implementation function; see source.
- `_instance_metadata(instance)` (line 391) — Persist the evaluation identity required for trustworthy replay/audit.
- `run_candidates(instance, candidates, selection, max_workers, **kwargs)` (line 405) — Independent seeded candidates with measured pass@k and deterministic selection.
- `run_candidates.candidate(index)` (line 431) — Implementation function; see source.
- `run_candidates.behavior(result)` (line 488) — Implementation function; see source.
- `run_candidates.ranking(result)` (line 503) — Implementation function; see source.
- `_minimize_patch(runner, handle, instance, patch, max_trials)` (line 534) — Greedily reverse individual modification hunks and retain only passing removals.
- `_persist_local(traj, output_dir)` (line 622) — Persist uniquely named local repair artifacts and their content hashes.

### `swe-rl-agent/src/swe_rl/sandbox/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/sandbox/docker_runner.py`

Ephemeral Docker container manager for rollouts.

- `ContainerHandle` (line 41) — Class/data model.
- `ExecResult` (line 51) — Class/data model.
- `DockerRunner` (line 61) — Class/data model.
- `DockerRunner.__init__(self, client)` (line 62) — Implementation function; see source.
- `DockerRunner.ensure_image(self)` (line 69) — Implementation function; see source.
- `DockerRunner.start(self, name, env)` (line 78) — Implementation function; see source.
- `DockerRunner.disable_network(self, handle)` (line 121) — Implementation function; see source.
- `DockerRunner.teardown(self, handle)` (line 139) — Implementation function; see source.
- `DockerRunner.session(self, **start_kwargs)` (line 153) — Implementation function; see source.
- `DockerRunner.exec(self, handle, cmd, timeout, workdir, user, check_safety)` (line 162) — Implementation function; see source.
- `DockerRunner.write_file(self, handle, path, content)` (line 247) — Implementation function; see source.
- `DockerRunner.read_file(self, handle, path)` (line 261) — Implementation function; see source.
- `DockerRunner.commit_diff(self, handle, repo_dir)` (line 274) — Implementation function; see source.
- `DockerRunner._get(self, handle)` (line 285) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/sandbox/repo_setup.py`

Clone the target repo at the buggy commit and run instance-specific install.

- `RepoSetupResult` (line 26) — Class/data model.
- `_swebench_specs(repo, version)` (line 32) — Look up SWE-bench install commands from the official package; None on miss.
- `setup_repo(runner, handle, instance, apply_test_patch)` (line 46) — Clone, checkout, install. Records logs to the trajectory layer upstream.
- `setup_repo.run(cmd, timeout)` (line 59) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/sandbox/safety.py`

Allowlist/denylist for shell commands emitted by the LLM.

- `SafetyVerdict` (line 30) — Class/data model.
- `check_command(command)` (line 35) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/sandbox/test_executor.py`

Run the project's test suite inside a sandbox container and parse outcomes.

- `TestResults` (line 33) — Class/data model.
- `TestResults.outcome_for(self, test_id)` (line 42) — Implementation function; see source.
- `TestResults.all_pass(self, test_ids)` (line 45) — Implementation function; see source.
- `TestResults.pass_rate(self, test_ids)` (line 48) — Implementation function; see source.
- `_normalize(test_id)` (line 54) — Normalize pytest node ids — collapse repeated separators, drop leading './'.
- `_parse_json_report(blob)` (line 62) — Implementation function; see source.
- `_parse_verbose(stdout)` (line 84) — Implementation function; see source.
- `TestExecutor` (line 97) — Class/data model.
- `TestExecutor.__init__(self, runner)` (line 98) — Implementation function; see source.
- `TestExecutor.run(self, handle, test_ids, timeout, prefer_json_report)` (line 101) — Implementation function; see source.
- `_count_by_outcome(outcomes)` (line 160) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/serve/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/serve/learner_sync.py`

Sync trained weights from learner to vLLM rollout servers.

- `SyncConfig` (line 23) — Class/data model.
- `sync_lora(adapter_path, config)` (line 30) — Push a LoRA adapter to vLLM. Returns True on success.
- `unload(config)` (line 49) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/serve/vllm_server.py`

Wrapper to launch vLLM's OpenAI-compatible server with our config.

- `VLLMServeConfig` (line 20) — Class/data model.
- `build_command(cfg)` (line 34) — Implementation function; see source.
- `serve(cfg)` (line 63) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/settings.py`

Process-wide settings loaded from environment.

- `Settings` (line 16) — Class/data model.
- `Settings._local_vllm_only_by_default(cls, value)` (line 86) — Implementation function; see source.
- `Settings.vllm_base_url(self)` (line 100) — Implementation function; see source.
- `get_settings()` (line 105) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/train/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/train/curriculum.py`

Easy -> hard curriculum scheduler for SWE-bench instances.

- `difficulty_score(instance)` (line 12) — Lower tuple = easier. Sort key: (files, total_lines, problem_len).
- `sort_by_difficulty(instances)` (line 19) — Implementation function; see source.
- `staged_curriculum(instances, stages)` (line 23) — Default stages: <20 LOC single-file, <80 LOC, anything.
- `curriculum_from_failures(instances, recorded_runs)` (line 44) — Use observed categories to separate environment remediation and repair lessons.

### `swe-rl-agent/src/swe_rl/train/data_assembly.py`

Convert recorded trajectories into TRL-friendly datasets.

- `SFTExample` (line 17) — Class/data model.
- `trajectory_to_sft(messages)` (line 22) — Pair user/system context with each assistant turn.
- `replay_to_sft_dataset(buffer, only_resolved)` (line 34) — Implementation function; see source.
- `replay_to_dpo_pairs(buffer)` (line 45) — Implementation function; see source.
- `_last_assistant(messages)` (line 76) — Implementation function; see source.
- `humanevalpack_to_sft(rows)` (line 83) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/train/dpo_trainer.py`

Offline DPO over preference pairs derived from rollouts.

- `DPOConfig` (line 17) — Class/data model.
- `run_dpo(dataset, config, event_callback, control_callback)` (line 31) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/train/grpo_trainer.py`

GRPO trainer wrapping `trl.GRPOTrainer` with our execution-grounded reward.

- `GRPOConfig` (line 34) — Class/data model.
- `make_reward_fn(instances_by_id, react_config, output_dir, shaped_cfg, vllm_base_url, vllm_api_key, binary_weight, shaped_weight, shaped_decay_steps, prm_weight)` (line 58) — Build a TRL-compatible reward callable.
- `make_reward_fn.reward(prompts, completions, **kwargs)` (line 79) — Implementation function; see source.
- `_completion_text(completion)` (line 143) — Implementation function; see source.
- `_infer_instance_ids(prompts)` (line 154) — Heuristic — extract instance ids from the prompt body. Used only when the
- `run_grpo(train_prompts, instances, config, react_config, shaped_cfg, vllm_base_url, vllm_api_key, event_callback, control_callback)` (line 167) — Run GRPO training. `train_prompts` is a list of dicts with `prompt` + `instance_id`.
- `build_train_prompts(instances)` (line 248) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/train/sft_warmstart.py`

SFT warm-start using TRL's SFTTrainer over HumanEvalPack + curated trajectories.

- `SFTConfig` (line 17) — Class/data model.
- `run_sft(dataset, config, event_callback, control_callback)` (line 34) — Train a model with SFT and save to config.output_dir.
- `run_sft._formatter(example)` (line 86) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/train/telemetry.py`

Optional training callbacks emitting only metrics reported by the trainer.

- `make_callbacks(event_callback, control_callback)` (line 8) — Implementation function; see source.
- `make_callbacks.TelemetryCallback` (line 16) — Class/data model.
- `make_callbacks.TelemetryCallback.on_train_begin(self, args, state, control, **kwargs)` (line 17) — Implementation function; see source.
- `make_callbacks.TelemetryCallback.on_log(self, args, state, control, logs, **kwargs)` (line 28) — Implementation function; see source.
- `make_callbacks.TelemetryCallback.on_save(self, args, state, control, **kwargs)` (line 39) — Implementation function; see source.
- `make_callbacks.TelemetryCallback.on_step_begin(self, args, state, control, **kwargs)` (line 49) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/utils/__init__.py`

Package/module implementation.

No callable declarations; package/constants/schema module.

### `swe-rl-agent/src/swe_rl/utils/cost.py`

Cost guardrail: block rollouts when projected $ exceeds the run cap.

- `CostCapExceeded` (line 14) — Class/data model.
- `CostMeter` (line 19) — Class/data model.
- `CostMeter.dollars(self)` (line 24) — Implementation function; see source.
- `CostMeter.add(self, tokens_in, tokens_out)` (line 30) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/utils/git.py`

Tiny git helpers that shell out, used inside containers.

- `run_git(args, cwd)` (line 10) — Implementation function; see source.
- `diff_against_head(cwd)` (line 20) — Implementation function; see source.
- `current_commit(cwd)` (line 24) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/utils/patch.py`

Unified-diff helpers.

- `parse_patch(text)` (line 8) — Implementation function; see source.
- `patch_stats(text)` (line 12) — Return (files_changed, lines_added, lines_removed).
- `is_empty_patch(text)` (line 25) — Implementation function; see source.

### `swe-rl-agent/src/swe_rl/utils/retry.py`

Retry decorator built on tenacity with structured logging.

- `with_retry(fn, attempts, min_wait, max_wait, name)` (line 21) — Implementation function; see source.
- `with_retry._log_attempt(state)` (line 31) — Implementation function; see source.
- `with_retry.wrapper(*args, **kwargs)` (line 35) — Implementation function; see source.

## 16. Current project file inventory

This lists source/configuration/documentation/test/operations files and excludes generated environments, output artifacts, Git internals and caches.

```text
.gitattributes
.github/workflows/ci.yml
.gitignore
Everything_about_project.md
README.md
swe-rl-agent/.env.example
swe-rl-agent/.github/workflows/ci.yml
swe-rl-agent/.gitignore
swe-rl-agent/Makefile
swe-rl-agent/README.md
swe-rl-agent/configs/algo/dpo.yaml
swe-rl-agent/configs/algo/grpo.yaml
swe-rl-agent/configs/base.yaml
swe-rl-agent/configs/eval/swebench_lite.yaml
swe-rl-agent/configs/model/deepseek_coder_v2_lite.yaml
swe-rl-agent/configs/model/qwen25_coder_14b.yaml
swe-rl-agent/configs/reward/exec_grounded.yaml
swe-rl-agent/docker/compose.dev.yaml
swe-rl-agent/docker/learner.Dockerfile
swe-rl-agent/docker/prometheus.yml
swe-rl-agent/docker/sandbox.Dockerfile
swe-rl-agent/docs/architecture.md
swe-rl-agent/docs/control-center.md
swe-rl-agent/docs/grpo_notes.md
swe-rl-agent/docs/gui-validation.md
swe-rl-agent/docs/model-connections.md
swe-rl-agent/docs/reward_design.md
swe-rl-agent/docs/runbook.md
swe-rl-agent/infra/README.md
swe-rl-agent/infra/main.tf
swe-rl-agent/infra/rollout_task.tf
swe-rl-agent/migrations/0001_initial.sql
swe-rl-agent/pyproject.toml
swe-rl-agent/requirements-dashboard.txt
swe-rl-agent/scripts/00_bootstrap.sh
swe-rl-agent/scripts/01_build_sandbox.sh
swe-rl-agent/scripts/02_smoke_rollout.sh
swe-rl-agent/scripts/03_warmstart_sft.sh
swe-rl-agent/scripts/04_grpo_train.sh
swe-rl-agent/scripts/05_eval_lite.sh
swe-rl-agent/scripts/06_leaderboard_submit.sh
swe-rl-agent/src/swe_rl/__init__.py
swe-rl-agent/src/swe_rl/agent/__init__.py
swe-rl-agent/src/swe_rl/agent/credentials.py
swe-rl-agent/src/swe_rl/agent/intelligence.py
swe-rl-agent/src/swe_rl/agent/llm_client.py
swe-rl-agent/src/swe_rl/agent/model_registry.py
swe-rl-agent/src/swe_rl/agent/profiles.py
swe-rl-agent/src/swe_rl/agent/prompt_registry.py
swe-rl-agent/src/swe_rl/agent/prompts/__init__.py
swe-rl-agent/src/swe_rl/agent/prompts/reflection.md
swe-rl-agent/src/swe_rl/agent/prompts/system.md
swe-rl-agent/src/swe_rl/agent/prompts/user_intro.md
swe-rl-agent/src/swe_rl/agent/providers.py
swe-rl-agent/src/swe_rl/agent/react_loop.py
swe-rl-agent/src/swe_rl/agent/tools/__init__.py
swe-rl-agent/src/swe_rl/agent/tools/base.py
swe-rl-agent/src/swe_rl/agent/tools/bash.py
swe-rl-agent/src/swe_rl/agent/tools/file_edit.py
swe-rl-agent/src/swe_rl/agent/tools/finish.py
swe-rl-agent/src/swe_rl/agent/tools/grep.py
swe-rl-agent/src/swe_rl/agent/tools/intelligence.py
swe-rl-agent/src/swe_rl/agent/tools/repo_map.py
swe-rl-agent/src/swe_rl/agent/tools/test_run.py
swe-rl-agent/src/swe_rl/agent/trajectory.py
swe-rl-agent/src/swe_rl/cli.py
swe-rl-agent/src/swe_rl/dashboard/__init__.py
swe-rl-agent/src/swe_rl/dashboard/analytics.py
swe-rl-agent/src/swe_rl/dashboard/app.py
swe-rl-agent/src/swe_rl/dashboard/control.py
swe-rl-agent/src/swe_rl/dashboard/job_runner.py
swe-rl-agent/src/swe_rl/dashboard/jobs.py
swe-rl-agent/src/swe_rl/dashboard/model_api.py
swe-rl-agent/src/swe_rl/dashboard/run_index.py
swe-rl-agent/src/swe_rl/dashboard/static/api.js
swe-rl-agent/src/swe_rl/dashboard/static/app.js
swe-rl-agent/src/swe_rl/dashboard/static/connections.css
swe-rl-agent/src/swe_rl/dashboard/static/connections.js
swe-rl-agent/src/swe_rl/dashboard/static/control-center.css
swe-rl-agent/src/swe_rl/dashboard/static/control-center.js
swe-rl-agent/src/swe_rl/dashboard/static/index.html
swe-rl-agent/src/swe_rl/dashboard/static/styles.css
swe-rl-agent/src/swe_rl/dashboard/telemetry.py
swe-rl-agent/src/swe_rl/data/__init__.py
swe-rl-agent/src/swe_rl/data/humanevalpack_loader.py
swe-rl-agent/src/swe_rl/data/instance_schema.py
swe-rl-agent/src/swe_rl/data/swebench_loader.py
swe-rl-agent/src/swe_rl/db.py
swe-rl-agent/src/swe_rl/eval/__init__.py
swe-rl-agent/src/swe_rl/eval/report.py
swe-rl-agent/src/swe_rl/eval/swebench_eval.py
swe-rl-agent/src/swe_rl/observability/__init__.py
swe-rl-agent/src/swe_rl/observability/logging.py
swe-rl-agent/src/swe_rl/observability/metrics.py
swe-rl-agent/src/swe_rl/observability/sentry_init.py
swe-rl-agent/src/swe_rl/reward/__init__.py
swe-rl-agent/src/swe_rl/reward/analysis.py
swe-rl-agent/src/swe_rl/reward/exec_reward.py
swe-rl-agent/src/swe_rl/reward/prm.py
swe-rl-agent/src/swe_rl/reward/shaped_reward.py
swe-rl-agent/src/swe_rl/rollout/__init__.py
swe-rl-agent/src/swe_rl/rollout/ray_pool.py
swe-rl-agent/src/swe_rl/rollout/replay_buffer.py
swe-rl-agent/src/swe_rl/rollout/repro.py
swe-rl-agent/src/swe_rl/rollout/worker.py
swe-rl-agent/src/swe_rl/sandbox/__init__.py
swe-rl-agent/src/swe_rl/sandbox/docker_runner.py
swe-rl-agent/src/swe_rl/sandbox/repo_setup.py
swe-rl-agent/src/swe_rl/sandbox/safety.py
swe-rl-agent/src/swe_rl/sandbox/test_executor.py
swe-rl-agent/src/swe_rl/serve/__init__.py
swe-rl-agent/src/swe_rl/serve/learner_sync.py
swe-rl-agent/src/swe_rl/serve/vllm_server.py
swe-rl-agent/src/swe_rl/settings.py
swe-rl-agent/src/swe_rl/train/__init__.py
swe-rl-agent/src/swe_rl/train/curriculum.py
swe-rl-agent/src/swe_rl/train/data_assembly.py
swe-rl-agent/src/swe_rl/train/dpo_trainer.py
swe-rl-agent/src/swe_rl/train/grpo_trainer.py
swe-rl-agent/src/swe_rl/train/sft_warmstart.py
swe-rl-agent/src/swe_rl/train/telemetry.py
swe-rl-agent/src/swe_rl/utils/__init__.py
swe-rl-agent/src/swe_rl/utils/cost.py
swe-rl-agent/src/swe_rl/utils/git.py
swe-rl-agent/src/swe_rl/utils/patch.py
swe-rl-agent/src/swe_rl/utils/retry.py
swe-rl-agent/tests/__init__.py
swe-rl-agent/tests/conftest.py
swe-rl-agent/tests/connections-ui.cjs
swe-rl-agent/tests/control-center-ui.cjs
swe-rl-agent/tests/dashboard-ui.cjs
swe-rl-agent/tests/e2e/__init__.py
swe-rl-agent/tests/e2e/test_gold_patch_rollout.py
swe-rl-agent/tests/integration/__init__.py
swe-rl-agent/tests/integration/test_docker_runner.py
swe-rl-agent/tests/unit/__init__.py
swe-rl-agent/tests/unit/test_control.py
swe-rl-agent/tests/unit/test_control_analytics.py
swe-rl-agent/tests/unit/test_cost.py
swe-rl-agent/tests/unit/test_dashboard.py
swe-rl-agent/tests/unit/test_data_assembly.py
swe-rl-agent/tests/unit/test_evaluation_parallel.py
swe-rl-agent/tests/unit/test_exec_reward.py
swe-rl-agent/tests/unit/test_grpo_reward.py
swe-rl-agent/tests/unit/test_gui_regressions.py
swe-rl-agent/tests/unit/test_instance_schema.py
swe-rl-agent/tests/unit/test_job_runner.py
swe-rl-agent/tests/unit/test_jobs.py
swe-rl-agent/tests/unit/test_patch.py
swe-rl-agent/tests/unit/test_providers.py
swe-rl-agent/tests/unit/test_react_loop.py
swe-rl-agent/tests/unit/test_registry_index.py
swe-rl-agent/tests/unit/test_replay_buffer.py
swe-rl-agent/tests/unit/test_runtime_intelligence.py
swe-rl-agent/tests/unit/test_safety.py
swe-rl-agent/tests/unit/test_shaped_reward.py
swe-rl-agent/tests/unit/test_test_executor_parsing.py
swe-rl-agent/tests/unit/test_tool_paths.py
swe-rl-agent/tests/unit/test_trajectory.py
```

## 17. Complete HTTP endpoint inventory

Generated from FastAPI OpenAPI, including included routers. Mutation routes require the local settings authorization mechanism. `/api/docs` provides schemas.

| Method | Path | Operation |
|---|---|---|
| GET | `/` | Index |
| GET | `/api/analytics` | Analytics |
| GET | `/api/artifacts` | Artifacts |
| GET | `/api/artifacts/{artifact_id}/content` | Artifact |
| GET | `/api/artifacts/{artifact_id}/download` | Download Artifact |
| GET | `/api/checkpoints` | Checkpoints |
| POST | `/api/checkpoints/{job_id}/{action}` | Checkpoint Action |
| GET | `/api/compare` | Compare |
| GET | `/api/datasets` | Datasets |
| GET | `/api/datasets/{dataset_id}/instances` | Dataset Instances |
| GET | `/api/evaluations` | Evaluations |
| POST | `/api/evaluations` | Evaluate |
| GET | `/api/experiments` | Experiments |
| GET | `/api/experiments/{experiment_id}/selection` | Selected Candidates |
| POST | `/api/experiments/{experiment_id}/selection` | Critic Selection |
| GET | `/api/health` | Health |
| GET | `/api/infrastructure` | Infrastructure |
| GET | `/api/infrastructure/metrics` | Metrics |
| GET | `/api/jobs` | Jobs |
| POST | `/api/jobs/{job_id}/priority` | Priority |
| POST | `/api/model/assist` | Assist |
| GET | `/api/model/fallback` | Fallback |
| POST | `/api/model/fallback` | Save Fallback |
| POST | `/api/model/models` | List Models |
| GET | `/api/model/registry` | Registry |
| POST | `/api/model/registry` | Register |
| POST | `/api/model/settings` | Model Settings |
| DELETE | `/api/model/settings` | Reset Model Settings |
| GET | `/api/model/telemetry` | Telemetry |
| POST | `/api/model/test` | Test Model |
| GET | `/api/prompts` | Prompts |
| POST | `/api/prompts` | Save Prompt |
| GET | `/api/providers` | Providers |
| GET | `/api/replay` | Replay |
| POST | `/api/replay/export` | Export Replay |
| GET | `/api/runs` | Runs |
| POST | `/api/runs` | Create Runs |
| GET | `/api/runs/{job_id}` | Run |
| GET | `/api/runs/{job_id}/annotation` | Annotation |
| POST | `/api/runs/{job_id}/annotation` | Annotate |
| GET | `/api/runs/{job_id}/events` | Events |
| POST | `/api/runs/{job_id}/{action}` | Action |
| GET | `/api/training/jobs` | Training |
| POST | `/api/training/jobs` | Train |
| GET | `/api/training/jobs/{job_id}/metrics` | Training Metrics |
| GET | `/{asset}` | Asset |

## 18. In-app dropdown UI

`dashboard/static/selects.js` and `selects.css` provide theme-aware custom dropdowns for every select, including dynamic screens/forms and the run outcome filter. Original form values/events remain intact. Keyboard navigation, type-to-find, selected checkmarks, disabled/grouped options, multiple selections, reset synchronization and viewport positioning are shared throughout the app. The scripts/styles are loaded by `index.html`.
