"""Retry decorator built on tenacity with structured logging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    RetryCallState,
    Retrying,
    stop_after_attempt,
    wait_exponential,
)

from swe_rl.observability.logging import get_logger

_log = get_logger(__name__)
T = TypeVar("T")


def with_retry(
    fn: Callable[..., T],
    *,
    attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 16.0,
    name: str | None = None,
) -> Callable[..., T]:
    label = name or fn.__name__

    def _log_attempt(state: RetryCallState) -> None:
        if state.attempt_number > 1:
            _log.warning("retry.attempt", op=label, attempt=state.attempt_number)

    def wrapper(*args: Any, **kwargs: Any) -> T:
        for attempt in Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(min=min_wait, max=max_wait),
            reraise=True,
            before=_log_attempt,
        ):
            with attempt:
                return fn(*args, **kwargs)
        raise RuntimeError("unreachable")

    return wrapper
