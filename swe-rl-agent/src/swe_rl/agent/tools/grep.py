"""Ripgrep-backed search across the repo."""

from __future__ import annotations

import shlex

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult, truncate
from swe_rl.agent.tools.file_edit import _resolve


class GrepArgs(BaseModel):
    pattern: str = Field(..., description="Regex pattern (PCRE2)")
    path: str | None = Field(None, description="Optional repo-relative path to limit search")
    glob: str | None = Field(None, description="Optional file glob, e.g. '*.py'")
    case_sensitive: bool = Field(True)
    max_results: int = Field(200, ge=1, le=1000)


class GrepTool(Tool):
    name = "grep"
    description = "Search the repo with ripgrep. Returns file:line:match lines."
    Args = GrepArgs

    def _run(self, args: GrepArgs) -> ToolResult:  # type: ignore[override]
        flags = ["-n", "--no-heading", "--color=never", f"-m{args.max_results}"]
        if not args.case_sensitive:
            flags.append("-i")
        if args.glob:
            flags += ["-g", args.glob]
        target = "." if args.path is None else _resolve(".", args.path).removeprefix("./")
        cmd = f"cd {shlex.quote(self.ctx.repo_dir)} && rg {' '.join(shlex.quote(f) for f in flags)} {shlex.quote(args.pattern)} {shlex.quote(target)} || true"
        r = self.ctx.runner.exec(self.ctx.handle, cmd, timeout=30)
        text, truncated = truncate(r.stdout)
        n_lines = text.count("\n")
        return ToolResult(
            ok=(r.exit_code == 0),
            output=text,
            data={"matches": n_lines},
            duration_s=r.duration_s,
            truncated=truncated,
        )
