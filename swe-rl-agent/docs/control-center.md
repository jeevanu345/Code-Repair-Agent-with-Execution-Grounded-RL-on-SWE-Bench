# SWE-RL Console: operation and implementation guide

Updated 2026-10-02. This document maps the supplied upgrade proposal to the implementation. Measurements come from recorded trajectories, job events or bounded service probes. There are no synthetic benchmark scores.

## Start the UI

From the repository root:

```bash
cd swe-rl-agent
python3 -m venv .venv-dashboard
.venv-dashboard/bin/pip install -r requirements-dashboard.txt
PYTHONPATH=src .venv-dashboard/bin/python -m uvicorn swe_rl.dashboard.app:app --host 127.0.0.1 --port 8088
```

Open http://127.0.0.1:8088. This lightweight installation runs the console and configuration interface. Executing workloads needs the full package on Python 3.11–3.12, Docker and dataset access. Training additionally needs compatible GPU drivers, local model weights and `SWE_RL_ALLOW_HEAVY_TRAINING=true`. The training launch form also requires explicit opt-in. Cloud inference credentials cannot train a provider's remote weights.

The application expects its working directory to be `swe-rl-agent`. `TRAIN_OUTPUT_DIR` selects the artifact/control root. Do not run multiple independent schedulers against the same control directory. The scheduler uses two local subprocess worker slots; Ray remains a separate CLI pool.

## Navigation and UI contents

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

### Run inspector

The three panes contain an issue/run list; primary detail tabs; and run metadata/evidence. Tabs cover overview, patch, tool timeline, F2P/P2P evidence, conversation, repository context, costs, rewards and failure analysis. Patch controls include file tree, unified/split views and whitespace filtering. Tool rows expose arguments, output, status and duration. Tests expose pass/fail/timeout plus captured final logs. Cost attribution uses per-turn/step usage and recorded pricing. Failure categories cite flags/test evidence; they do not assert an unproven root cause. JSON/artifact export preserves provenance.

## Lifecycle and durable queue

`dashboard/jobs.py` persists jobs, experiment configurations and event journals in SQLite WAL. Atomic capacity claims bound concurrent workers. Each job executes in a separate Python subprocess, with PID identity checks before signaling. Startup recovery reconciles stale jobs, and failures retain diagnostics.

States: `queued → preparing → running → testing/evaluating → completed`, or `failed/cancelled`. Pause is a cooperative control flag at safe runtime/trainer boundaries. Resume clears it. Cancel requests cooperative termination and can signal a still-running owned worker. Stop-and-evaluate is allowed during preparation/agent execution and asks the worker to grade the current patch. Blocking external operations are not instantaneously pausable. Retry creates a new attempt; requeue returns an eligible job to the queue. Priority changes affect unclaimed jobs.

SSE `/api/runs/{id}/events` replays persisted events after a cursor and streams new ones. It includes state, container, model/tool/usage/test/reward/artifact/failure events. Closing the browser does not remove a persisted job. Worker stderr/stdout are artifacts. A local SQLite control queue is implemented; Redis/Celery is not its broker and PostgreSQL does not store its lifecycle journal.

## Experiment and execution provenance

Launch configuration is snapshotted, fingerprinted and associated with an experiment. Snapshots include dataset/revision, instance, model/provider/checkpoint, strategy/prompt hash, sampling/seed, bounds, sandbox resources and reward mix. Workers store dataset fingerprint/revision, repository commit, image digest, patch, assistant/tool messages, usage and tests. This supports comparison and recorded-message replay; exact model or environment determinism is not guaranteed by a seed.

Agent profiles: standard, plan-execute, test-driven, minimal-patch, high-exploration, low-cost and explorer-reviewer. Conditional reflection follows failed tools/tests or repeated actions. Adaptive compression preserves relevant recent context; test budget exhaustion is enforced. Repository intelligence uses bounded Python AST symbols/imports/call references, related-test recommendations, git history and syntax-validated symbol edits. Static references are not runtime call resolution.

Multiple candidates use independent seeds and bounded parallelism. Selection modes consider execution score, minimal patch size, regression preservation, majority agreement or an explicit critic. A model review is labeled as model evidence. Dashboard pass@k uses `1 - C(n-c,k)/C(n,k)` only where at least k candidates exist; incomplete groups remain unavailable. Generation selection that consults graded outcomes is explicitly labeled and is not an official pass@1 claim. Official benchmark claims require the harness report and declared selection protocol.

Patch minimization greedily tries bounded hunk removals and re-executes grading, preserving successful F2P/P2P outcomes. The agent patch is captured before hidden test-patch application. Hidden grading evidence must not be added to agent-visible discovery context.

## Rewards, review and training

Execution reward is binary resolution; shaped reward exposes partial test rates and edit/lint/empty/timeout penalties; optional PRM is explicitly configured. Rollouts record the weighted mixture plus component values. Defaults for normal rollouts are execution weight 1, shaped/PRM 0. GRPO uses configured execution/shaped/PRM weights, with shaped weight decaying by optimizer step (default horizon 5000). Configured PRM judges require their own model availability.

Trainer callbacks record actual progress, loss/reward metrics where emitted, checkpoint saves and cooperative control. Charts do not fill absent metrics with samples. SFT accepts reviewed JSON/JSONL or HumanEvalPack; DPO needs `prompt/chosen/rejected`; GRPO evaluates each generated patch against its matching issue. The replay export includes only human-reviewed approved trajectories; DPO pairs approved and rejected patches from the same instance. Failure-conditioned curriculum staging uses observed failure categories, not gold patches.

## Storage

| Location | Role |
|---|---|
| `outputs/control/control.sqlite3` | Durable job/configuration/event/annotation journal. |
| `outputs/control/<job-id>/` | Worker configuration, logs, predictions/reports/checkpoints and trajectories. |
| `outputs/registry/prompts.sqlite3` | Immutable custom prompt versions. |
| `outputs/registry/models.json` | User-declared model capabilities/pricing. |
| Output trajectory JSONL and hashed sidecars | Canonical local execution evidence and separately inspectable logs/tests/patches. |
| Output replay Parquet shards | Unique trajectory replay rows, appended by rollout persistence. |
| Output run-index SQLite | Incremental trajectory discovery cache keyed by file identity/stat metadata. |
| PostgreSQL `control_trajectory_index` | Optional primary trajectory index when `CONTROL_DATABASE_URL` is configured; local fallback on service failure. |
| PostgreSQL/MinIO | Bounded best-effort trajectory/artifact persistence hooks; local evidence retained on failure. |
| `.llm-profile.json` / `.llm-fallback.json` | Connection identities and OS secret references, no provider key values. |

Keep local output backups. SQLite, filesystem artifacts, PostgreSQL and MinIO are separate stores; there is no distributed transaction across all of them.

## Credentials and model behavior

Keys are saved using an approved OS keyring backend (macOS Keychain, Windows credential store, Secret Service/libsecret). If none is available, saving credentials fails with an actionable error; there is no plaintext fallback. Legacy plaintext active profiles migrate to keyring references when read. Environment-supplied keys remain supported. Responses redact key values and frontend password fields clear after actions.

Fallback is opt-in, only for transient transport/rate-limit/server errors, and cannot recurse. The actual fallback provider/model is recorded. Registry prices are user declarations, not provider billing guarantees. Rollout caps use configured token rates and account for usage already incurred; they cannot undo a billed request or guarantee a distributed account budget. GUI test/assistant requests are explicit provider calls separate from rollout caps.

The API is for trusted loopback use. Host checks, same-origin checks and an in-memory settings token protect mutations. This is not multi-user authentication, TLS termination, RBAC or a remotely deployable security boundary. Docker setup may access the network before isolation; root Docker access and arbitrary repository setup remain host trust boundaries.

## Upgrade proposal coverage

| Proposal | Implemented entry points and scope |
|---|---|
| 1–3 Control center, New Run, live execution | Operational navigation/forms, durable jobs and event SSE, worker lifecycle. |
| 4–6 Inspector, diff, tests | Three panes, split/unified/file tree, captured test evidence and logs. |
| 7–9 Comparison/config/versioning/benchmarks | Configuration snapshots, matched cohorts, parallel generation, official harness imports and reports. |
| 10–12 Failure/cost/pass@k | Evidence heuristic categories, per-step usage, bounded candidates and qualified estimator. |
| 13–17 Minimizer/test selection/graph/profiles/reflection | Bounded hunk minimizer, AST recommendations, strategy prompts, failure/repetition reflection. |
| 18–20 Training/checkpoints/reward | Guarded SFT/DPO/GRPO workers/callbacks, checkpoint actions, weighted-component analysis. |
| 21–24 Infrastructure/queue/persistence/artifacts | Measured probes, two subprocess slots, durable local journal, optional PG index/MinIO, hashes/paging/downloads. |
| 25–26 Models/key security | Declared capabilities/prices, measured request telemetry, explicit fallback, OS-managed secret references. |
| 27–32 Metrics/polish/palette/search/views/keyboard | Evidence-based overview/charts, supplied enterprise design, accessible navigation and local saved views. |
| 33 Research extensions | Candidate parallelism/reranking/critic, strategy ablations through immutable config, prompt/reward/model/checkpoint experiments, AST/history/context tools, curriculum, annotations/approval/preference export and dataset slices. Explorer-reviewer is a staged strategy in one runtime, not separate independently trained models. |

Multilingual/multimodal variants can grade **imported predictions** through the official harness. The current automatic repair runtime remains Python-oriented; multimodal image perception and language-specific sandbox setup are not implemented. Dataset browsing and declared vision capabilities do not provide that runtime. Checkpoint tournament experiments can be launched and compared, but there is no autonomous tournament scheduler. Repository graph retrieval is bounded static Python analysis, not a whole-repository embedding database. The Ray pool remains a separate basic executor; this upgrade does not turn it into a distributed durable control-plane worker fleet.

## Validation and limits

Unit tests mock external transports, Docker and trainer boundaries to validate local orchestration without consuming provider credits or training hardware. The terminal jsdom suite exercises UI navigation/forms/SSE/controls/escaping/errors. Python lint/format and JavaScript/Bash syntax checks accompany them. Actual Docker grading, official harness reports, paid API calls and full GPU training must be validated on the configured execution host. No benchmark score, GPU throughput or cloud deployment success is claimed here.

References: [official SWE-bench datasets](https://www.swebench.com/SWE-bench/guides/datasets/), [official harness CLI](https://www.swebench.com/SWE-bench/reference/cli/), [keyring documentation](https://keyring.readthedocs.io/en/stable/).

## App dropdown controls

All native select fields are enhanced by shared `static/selects.js` and `selects.css`, including dynamically rendered forms and filters. Menus use the console theme and are rendered above scroll containers. Original selects retain form values, validation and existing input/change handlers. Controls support arrow keys, Home/End, Enter/Space, Escape/Tab, type-to-find, option groups, disabled options, multiple selection and form reset. Programmatic value/index assignments and option changes synchronize the visible control. Menus close on outside interaction and resize; focus stays on an accessible combobox trigger.
