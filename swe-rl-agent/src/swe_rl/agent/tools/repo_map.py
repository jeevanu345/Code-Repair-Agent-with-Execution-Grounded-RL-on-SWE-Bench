"""Repo-map tool: produce a compact symbol overview of the repo via ctags."""

from __future__ import annotations

import shlex

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult, truncate


class RepoMapArgs(BaseModel):
    path: str = Field(".", description="Repo-relative subpath to map")
    languages: list[str] = Field(default_factory=lambda: ["python"])
    max_lines: int = Field(400, ge=10, le=4000)


class RepoMapTool(Tool):
    name = "repo_map"
    description = (
        "Produce a compact symbol map (classes/functions per file) using universal-ctags. "
        "Use to orient yourself before diving into specific files."
    )
    Args = RepoMapArgs

    def _run(self, args: RepoMapArgs) -> ToolResult:  # type: ignore[override]
        langs = ",".join(args.languages)
        cmd = (
            f"cd {shlex.quote(self.ctx.repo_dir)} && "
            f"ctags --languages={shlex.quote(langs)} --kinds-Python=cf "
            f"-x --_xformat='%-30N %-8K %-40F %4n' -R {shlex.quote(args.path)} 2>/dev/null "
            f"| head -n {args.max_lines}"
        )
        r = self.ctx.runner.exec(self.ctx.handle, cmd, timeout=60)
        text, truncated = truncate(r.stdout)
        return ToolResult(
            ok=(r.exit_code == 0),
            output=text,
            data={"lines": text.count("\n")},
            duration_s=r.duration_s,
            truncated=truncated,
        )
