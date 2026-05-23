"""Test runner tool — capped to N runs per trajectory by ToolContext."""

from __future__ import annotations

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult, truncate


class TestRunArgs(BaseModel):
    test_ids: list[str] = Field(
        ...,
        description="Pytest node IDs (e.g. 'tests/test_foo.py::test_bar') to run",
        min_length=1,
    )
    timeout_s: int = Field(300, ge=10, le=600)


class TestRunTool(Tool):
    name = "test_run"
    description = (
        "Run a list of pytest tests inside the sandbox and return pass/fail per test. "
        "Use sparingly — there is a per-trajectory budget. Prefer narrow test selections."
    )
    Args = TestRunArgs

    def _run(self, args: TestRunArgs) -> ToolResult:  # type: ignore[override]
        if self.ctx.test_runs_remaining <= 0:
            return ToolResult(
                ok=False,
                output="test_run budget exhausted for this trajectory",
                data={"budget_remaining": 0},
                duration_s=0.0,
            )
        self.ctx.test_runs_remaining -= 1
        results = self.ctx.test_executor.run(self.ctx.handle, args.test_ids, timeout=args.timeout_s)
        text, truncated = truncate(results.raw_stdout)
        passed = sum(1 for v in results.outcomes.values() if v == "passed")
        failed = sum(1 for v in results.outcomes.values() if v in {"failed", "error", "missing"})
        skipped = sum(1 for v in results.outcomes.values() if v == "skipped")
        summary = (
            f"passed={passed} failed={failed} skipped={skipped} "
            f"timed_out={results.timed_out} duration={results.duration_s:.1f}s\n\n"
        )
        return ToolResult(
            ok=(not results.timed_out),
            output=summary + text,
            data={
                "outcomes": results.outcomes,
                "timed_out": results.timed_out,
                "budget_remaining": self.ctx.test_runs_remaining,
            },
            duration_s=results.duration_s,
            truncated=truncated,
        )
