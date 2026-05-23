"""Agent tool registry. Each tool exposes pydantic input + JSON-schema description + ToolResult."""

from swe_rl.agent.tools.base import Tool, ToolResult, ToolContext
from swe_rl.agent.tools.bash import BashTool
from swe_rl.agent.tools.file_edit import FileEditTool, FileWriteTool, FileReadTool
from swe_rl.agent.tools.grep import GrepTool
from swe_rl.agent.tools.repo_map import RepoMapTool
from swe_rl.agent.tools.test_run import TestRunTool
from swe_rl.agent.tools.finish import FinishTool


def default_toolset(ctx: ToolContext) -> list[Tool]:
    return [
        BashTool(ctx),
        FileReadTool(ctx),
        FileWriteTool(ctx),
        FileEditTool(ctx),
        GrepTool(ctx),
        RepoMapTool(ctx),
        TestRunTool(ctx),
        FinishTool(ctx),
    ]


__all__ = [
    "Tool",
    "ToolResult",
    "ToolContext",
    "BashTool",
    "FileEditTool",
    "FileWriteTool",
    "FileReadTool",
    "GrepTool",
    "RepoMapTool",
    "TestRunTool",
    "FinishTool",
    "default_toolset",
]
