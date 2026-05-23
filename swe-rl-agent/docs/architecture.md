# Architecture

## Data flow

```
                ┌────────────────┐
                │ SWE-bench Lite │
                │  / Verified    │
                └───────┬────────┘
                        │  HF datasets
                        ▼
                ┌────────────────┐
                │ swebench_loader│
                └───────┬────────┘
                        │ SWEBenchInstance
                        ▼
                ┌────────────────┐
                │ DockerRunner   │   ──▶ ephemeral container
                │ + repo_setup   │       (4GB / 2CPU / netns post-install)
                └───────┬────────┘
                        │ ContainerHandle
                        ▼
                ┌────────────────┐
                │ ReAct loop     │ ◀─── vLLM (OpenAI-compat /v1/chat)
                │  + Tools       │
                └───────┬────────┘
                        │ trajectory + final patch
                        ▼
                ┌────────────────┐
                │ test_executor  │   ──▶ FAIL_TO_PASS, PASS_TO_PASS
                └───────┬────────┘
                        │ TestResults
                        ▼
                ┌────────────────┐
                │ exec_reward    │   ──▶ binary 0/1
                │ shaped_reward  │   ──▶ dense
                └───────┬────────┘
                        ▼
                ┌────────────────┐
                │ replay_buffer  │   Parquet shards on disk / S3
                │ + Postgres idx │
                └───────┬────────┘
                        │ groups of K rollouts per instance
                        ▼
                ┌────────────────┐
                │ GRPO trainer   │   group-relative advantages, KL to ref
                │ (TRL)          │
                └───────┬────────┘
                        │ LoRA adapter / full weights
                        ▼
                ┌────────────────┐
                │ learner_sync   │ ──▶ vLLM /v1/load_lora_adapter
                └────────────────┘
```

## Why these choices

- **vLLM for rollouts** — high-throughput batched generation; OpenAI-compatible
  endpoint keeps the agent loop framework-agnostic.
- **Ray actors per Docker socket** — failure isolation per trajectory; one bad
  rollout doesn't corrupt the buffer.
- **Group Relative Policy Optimization (GRPO)** — sample K rollouts per
  prompt; advantage = within-group reward z-score. No critic needed; uses the
  group as its own baseline. KL-anchored to a frozen reference.
- **Binary primary reward + dense shaped auxiliary** — binary is the true
  objective (resolved or not); shaped term gives gradient when nothing fully
  passes. Shaped weight decays over training.
- **Docker, no chroot/firecracker** — broadest compatibility with the
  install-script-driven SWE-bench environments. We layer hardening
  (cap-drop, no-new-privileges, network=none post-install) on top.

## Determinism

All randomness derives from a single `seed`. The rollout records:
- `seed`, `temperature`, `top_p`
- `model_name`, `checkpoint_sha` (sha256 of weights), `model_revision`
- `sandbox_image_digest`
- `base_commit` of the target repo
- container & host tool versions

`make repro RUN_ID=...` re-runs with these recorded values.

## Failure modes & recovery

| Failure | Detection | Recovery |
|---|---|---|
| Container start error | `DockerRunner.start` exception | retry with backoff; if persistent, mark instance as setup_failed |
| Install step nonzero | `repo_setup` log | continue — many SWE-bench installs are noisy but functional |
| Test parse miss | `_parse_json_report` empty + `_parse_verbose` empty | treat all as `missing` (counts as fail in reward) |
| Stuck rollout | wallclock > sandbox cap | container teardown by reaper, trajectory marked timed_out |
| Cost cap exceeded | `CostMeter.add` raises `CostCapExceeded` | rollout pool drains, learner stops requesting new rollouts |
| Reward function bug | Test: gold patch → 1.0, no-op → 0.0 | property-tested in `tests/unit/test_exec_reward.py` |

## Deviations from prompt.md

None yet. Record any future deviation here with rationale and date.
