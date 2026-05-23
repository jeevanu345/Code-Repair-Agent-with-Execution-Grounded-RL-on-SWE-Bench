"""Prometheus metrics + W&B run handle.

Metrics defined here are imported and used by rollout/train/eval modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

_log = get_logger(__name__)


@dataclass(frozen=True)
class _Metrics:
    rollout_duration_seconds: Histogram
    tests_run_total: Counter
    reward_value: Histogram
    tool_calls_total: Counter
    container_failures_total: Counter
    tokens_in_total: Counter
    tokens_out_total: Counter
    dollar_cost_estimate: Gauge
    optimizer_steps_total: Counter
    eval_resolved_at_1: Gauge


def _build_metrics() -> _Metrics:
    return _Metrics(
        rollout_duration_seconds=Histogram(
            "rollout_duration_seconds",
            "Wall-clock duration of a single rollout",
            buckets=(5, 15, 30, 60, 120, 300, 600, 1200),
        ),
        tests_run_total=Counter("tests_run_total", "Tests executed in sandbox", ["outcome"]),
        reward_value=Histogram(
            "reward_value",
            "Reward emitted by reward function",
            buckets=(-1, -0.25, 0, 0.25, 0.5, 0.75, 1.0),
        ),
        tool_calls_total=Counter("tool_calls_total", "Tool invocations", ["tool"]),
        container_failures_total=Counter("container_failures_total", "Container errors", ["kind"]),
        tokens_in_total=Counter("tokens_in_total", "Prompt tokens consumed"),
        tokens_out_total=Counter("tokens_out_total", "Completion tokens generated"),
        dollar_cost_estimate=Gauge("dollar_cost_estimate", "Cumulative USD estimate for run"),
        optimizer_steps_total=Counter("optimizer_steps_total", "Optimizer steps taken"),
        eval_resolved_at_1=Gauge("eval_resolved_at_1", "Latest resolved@1 score"),
    )


METRICS = _build_metrics()
_SERVER_STARTED = False


def init_metrics(component: str) -> None:
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return
    port = settings.prometheus_port + (1 if component == "learner" else 0)
    try:
        start_http_server(port)
        _log.info("prometheus.started", port=port, component=component)
        _SERVER_STARTED = True
    except OSError as exc:
        _log.warning("prometheus.bind_failed", port=port, error=str(exc))
