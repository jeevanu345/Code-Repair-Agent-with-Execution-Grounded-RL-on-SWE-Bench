"""Bash tool — execute a shell command inside the sandbox."""

from __future__ import annotations

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult, truncate


class BashArgs(BaseModel):
    command: str = Field(..., description="Shell command to run inside the repo working directory")
    timeout_s: int = Field(120, ge=1, le=600, description="Per-call timeout in seconds")
    workdir: str | None = Field(None, description="Optional working directory inside the container")


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command inside the sandboxed container. "
        "Use for environment inspection (ls, find, cat) and quick scripts. "
        "Do not use for tests — use the test_run tool for those. "
        "Network is disabled after install; do not attempt curl/wget."
    )
    Args = BashArgs

    def _run(self, args: BashArgs) -> ToolResult:  # type: ignore[override]
        r = self.ctx.runner.exec(
            self.ctx.handle,
            args.command,
            timeout=args.timeout_s,
            workdir=args.workdir or self.ctx.repo_dir,
        )
        text, truncated = truncate(r.stdout)
        return ToolResult(
            ok=(r.exit_code == 0 and not r.timed_out and not r.safety_blocked),
            output=text,
            data={
                "exit_code": r.exit_code,
                "timed_out": r.timed_out,
                "safety_blocked": r.safety_blocked,
                "blocked_reason": r.blocked_reason,
            },
            duration_s=r.duration_s,
            truncated=truncated,
        )
