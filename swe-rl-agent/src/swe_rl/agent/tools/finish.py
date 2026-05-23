"""Finish tool: agent signals completion. Returning ok=True ends the rollout."""

from __future__ import annotations

from pydantic import BaseModel, Field

from swe_rl.agent.tools.base import Tool, ToolResult


class FinishArgs(BaseModel):
    summary: str = Field(..., description="One-paragraph summary of the fix")
    confident: bool = Field(True, description="Whether the agent is confident the fix is correct")


class FinishTool(Tool):
    name = "finish"
    description = (
        "Signal that you are done. The current diff in the working tree will be used as the patch. "
        "Only call this after you have verified the fix with test_run."
    )
    Args = FinishArgs

    def _run(self, args: FinishArgs) -> ToolResult:  # type: ignore[override]
        return ToolResult(
            ok=True,
            output=f"FINISH: {args.summary}",
            data={"confident": args.confident, "summary": args.summary, "is_finish": True},
            duration_s=0.0,
        )
