"""File read / write / search-replace edit tools.

`FileEditTool` performs a literal `old -> new` replacement and fails when `old`
is not unique in the file (forces the model to provide enough context). For
larger structural rewrites, the model can fall back to bash + heredoc via the
write tool.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult, truncate


def _resolve(repo_dir: str, path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path.strip() in {"", "."}:
        raise ValueError("path must be a non-empty repo-relative path without '..'")
    return f"{repo_dir.rstrip('/')}/{candidate.as_posix()}"


class FileReadArgs(BaseModel):
    path: str = Field(..., description="Repo-relative file path")
    start_line: int = Field(1, ge=1)
    num_lines: int = Field(400, ge=1, le=4000)


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a slice of a file from the repo. Returns numbered lines."
    Args = FileReadArgs

    def _run(self, args: FileReadArgs) -> ToolResult:  # type: ignore[override]
        full = _resolve(self.ctx.repo_dir, args.path)
        end = args.start_line + args.num_lines - 1
        cmd = (
            f"awk 'NR>={args.start_line} && NR<={end} "
            f'{{ printf "%6d  %s\\n", NR, $0 }}\' {shlex.quote(full)}'
        )
        r = self.ctx.runner.exec(self.ctx.handle, cmd, timeout=30)
        text, truncated = truncate(r.stdout)
        return ToolResult(
            ok=(r.exit_code == 0),
            output=text,
            data={"path": full, "start_line": args.start_line, "num_lines": args.num_lines},
            duration_s=r.duration_s,
            truncated=truncated,
        )


class FileWriteArgs(BaseModel):
    path: str = Field(..., description="Repo-relative file path")
    content: str = Field(..., description="Full file content")


class FileWriteTool(Tool):
    name = "file_write"
    description = "Overwrite a file with the provided content. Creates parent dirs as needed."
    Args = FileWriteArgs

    def _run(self, args: FileWriteArgs) -> ToolResult:  # type: ignore[override]
        full = _resolve(self.ctx.repo_dir, args.path)
        # Use docker put_archive via the runner for binary safety.
        self.ctx.runner.exec(
            self.ctx.handle,
            f"mkdir -p {shlex.quote(full.rsplit('/', 1)[0])}",
            timeout=15,
        )
        self.ctx.runner.write_file(self.ctx.handle, full, args.content)
        return ToolResult(
            ok=True,
            output=f"wrote {len(args.content)} bytes to {full}",
            data={"path": full, "bytes": len(args.content)},
            duration_s=0.0,
        )


class FileEditArgs(BaseModel):
    path: str = Field(..., description="Repo-relative file path")
    old: str = Field(..., description="Exact text to replace; must be unique in the file")
    new: str = Field(..., description="Replacement text")


class FileEditTool(Tool):
    name = "file_edit"
    description = (
        "Replace `old` with `new` in a file. Fails if `old` is missing or not unique. "
        "Provide enough surrounding context so the match is unique."
    )
    Args = FileEditArgs

    def _run(self, args: FileEditArgs) -> ToolResult:  # type: ignore[override]
        full = _resolve(self.ctx.repo_dir, args.path)
        try:
            current = self.ctx.runner.read_file(self.ctx.handle, full).decode(
                "utf-8", errors="replace"
            )
        except Exception as e:
            return ToolResult(False, f"file not found: {full} ({e})", {}, 0.0)

        count = current.count(args.old)
        if count == 0:
            return ToolResult(False, f"`old` text not found in {args.path}", {"matches": 0}, 0.0)
        if count > 1:
            return ToolResult(
                False,
                f"`old` text occurs {count} times in {args.path}; provide more context to make it unique",
                {"matches": count},
                0.0,
            )
        new_content = current.replace(args.old, args.new, 1)
        self.ctx.runner.write_file(self.ctx.handle, full, new_content)
        return ToolResult(
            ok=True,
            output=f"replaced 1 occurrence in {args.path}",
            data={"path": full, "old_len": len(args.old), "new_len": len(args.new)},
            duration_s=0.0,
        )
