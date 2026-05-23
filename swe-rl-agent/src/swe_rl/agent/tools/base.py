"""Tool base class and shared context."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar, Type

from pydantic import BaseModel

from swe_rl.observability.metrics import METRICS
from swe_rl.sandbox.docker_runner import ContainerHandle, DockerRunner
from swe_rl.sandbox.test_executor import TestExecutor


@dataclass
class ToolContext:
    runner: DockerRunner
    handle: ContainerHandle
    test_executor: TestExecutor
    repo_dir: str = "/workspace/repo"
    test_runs_remaining: int = 6


@dataclass
class ToolResult:
    ok: bool
    output: str
    data: dict[str, Any]
    duration_s: float
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "data": self.data,
            "duration_s": self.duration_s,
            "truncated": self.truncated,
        }


class Tool:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    Args: ClassVar[Type[BaseModel]]

    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.Args.model_json_schema(),
        }

    def __call__(self, **kwargs: Any) -> ToolResult:
        args = self.Args.model_validate(kwargs)
        t0 = time.time()
        result = self._run(args)
        result.duration_s = time.time() - t0
        METRICS.tool_calls_total.labels(tool=self.name).inc()
        return result

    def _run(self, args: BaseModel) -> ToolResult:  # noqa: ARG002
        raise NotImplementedError


def truncate(s: str, limit: int = 16_000) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    head = s[: limit // 2]
    tail = s[-limit // 2 :]
    return head + f"\n…[truncated {len(s) - limit} chars]…\n" + tail, True
