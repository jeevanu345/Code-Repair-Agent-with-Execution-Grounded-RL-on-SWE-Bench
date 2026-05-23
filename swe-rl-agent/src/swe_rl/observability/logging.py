"""Structured JSON logging via structlog."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from swe_rl.settings import settings

_INITIALIZED = False


def init_logging(component: str) -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{component}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    structlog.contextvars.bind_contextvars(component=component)
    _INITIALIZED = True


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)
