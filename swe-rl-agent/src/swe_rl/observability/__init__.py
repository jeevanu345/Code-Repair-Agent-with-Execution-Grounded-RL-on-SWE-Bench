"""Observability bootstrap: structlog, Sentry, W&B, Prometheus."""

from swe_rl.observability.logging import get_logger, init_logging
from swe_rl.observability.metrics import METRICS, init_metrics
from swe_rl.observability.sentry_init import init_sentry

__all__ = ["init", "init_logging", "get_logger", "init_sentry", "init_metrics", "METRICS"]


def init(component: str, *, enable_metrics_server: bool = True) -> None:
    """Initialize logging + Sentry + Prometheus for a long-running entrypoint."""
    init_logging(component=component)
    init_sentry(component=component)
    if enable_metrics_server:
        init_metrics(component=component)
