# Demo GUI and validation

Validated on 2026-09-27 using terminal tools only.

## Delivered

- Responsive repair workbench, searchable runs, outcome filters, light/dark themes.
- Patch highlighting, expandable tool calls, conversation and test inspection.
- Export of selected runs with explicit recorded/synthetic source labeling.
- Illustrative demo fixtures isolated from recorded trajectories and live health.
- Keyboard tab navigation, visible focus, reduced-motion handling, escaped run content.
- Standalone GUI requirements, without ML frameworks or model downloads.

The frontend-design skill informed the repair-focused layout, palette, and hierarchy.
No browser automation or screenshot inspection was used; visual appearance still
needs human review on the intended presentation display.

## Checks

- 46 unit tests passed, including new timeout, PRM configuration, and run-discovery regressions.
- Node/jsdom interaction checks passed: demo, filtering, tabs, theme persistence,
  empty selection, HTML escaping, API failure and retry availability.
- Ruff lint and formatting passed across src and tests.
- Python compilation, JavaScript syntax and Git whitespace checks passed.
- Running localhost server returned the page, health API and runs API successfully.
- Host verification used Python 3.14 with lightweight dependencies, not the full
  project's supported Python 3.11–3.12 training environment.
- One upstream Starlette/httpx test-client deprecation warning remains.

## Bugs corrected

- Dashboard omitted normal rollout directories and failed on non-object JSON records.
- Docker health client had no explicit short timeout or cleanup.
- Timed-out tests could still produce a successful execution reward.
- Nested PRM configuration assigned directly to a frozen dataclass.
- Agent exceptions were recorded but did not force the execution reward to zero.
- Existing lint/import issues and formatting failures.

## Remaining findings and limits

These are not fixed or certified by the GUI work:

- Ray's configured task timeout is not consumed by the coordinator.
- Cost accounting happens after model responses and is not a strict provider-spend guarantee.
- Rollout idempotency does not incorporate every model/configuration/patch parameter.
- The test executor reuses a JSON report path; stale-report protection needs hardening.
- Replay compares patch/reward/tool sequence, but does not establish exact observation replay.
- Docker, Redis, MinIO and the model endpoint were unavailable in the live health check.
  PostgreSQL reported TCP connectivity only; that is not a database migration check.
- Docker/SWE-bench E2E, GPU training, model serving and cloud infrastructure were not tested.

No LLM or model weights were downloaded. The GUI never starts workloads.
