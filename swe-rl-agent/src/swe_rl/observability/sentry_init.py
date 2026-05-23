"""Sentry bootstrap. No-op when DSN is not configured."""

from __future__ import annotations

from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

_log = get_logger(__name__)


def init_sentry(component: str) -> None:
    if not settings.sentry_dsn:
        _log.info("sentry.disabled", reason="no DSN configured")
        return
    try:
        import sentry_sdk
    except ImportError:
        _log.warning("sentry.import_failed")
        return

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        traces_sample_rate=0.1,
        profiles_sample_rate=0.0,
        send_default_pii=False,
        environment=component,
    )
    sentry_sdk.set_tag("component", component)
    _log.info("sentry.initialized", component=component)
