"""ReAct loop: planner -> tool dispatch -> observation -> reflection -> next action."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.prompts import load as load_prompt
from swe_rl.agent.tools import Tool, ToolContext, default_toolset
from swe_rl.agent.trajectory import Trajectory
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.utils.cost import CostMeter

_log = get_logger(__name__)


@dataclass
class ReactConfig:
    max_steps: int = 50
    max_test_runs: int = 6
    max_new_tokens_per_turn: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 17


@dataclass
class ReactRunResult:
    finished: bool
    reason: str
    n_steps: int
    cost: CostMeter = field(default_factory=CostMeter)


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a single JSON tool call from model text. Tolerant of code fences."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if not candidate:
        # last balanced JSON object
        depth = 0
        start: int | None = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : i + 1]
                    # keep going to find the LAST one
                    start = None
        if candidate is None:
            return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj


def _format_tools_block(tools: list[Tool]) -> str:
    items = []
    for t in tools:
        items.append(
            json.dumps(
                {"name": t.name, "description": t.description, "parameters": t.Args.model_json_schema()},
                indent=2,
            )
        )
    return "\n\n".join(items)


def _format_intro(instance: SWEBenchInstance) -> str:
    tpl = load_prompt("user_intro")
    f2p = "\n".join(f"- {t}" for t in instance.fail_to_pass[:50]) or "(none listed)"
    p2p_n = len(instance.pass_to_pass)
    p2p_summary = (
        f"{p2p_n} tests must remain passing (sample: {', '.join(instance.pass_to_pass[:5])})"
        if instance.pass_to_pass
        else "(none listed)"
    )
    return tpl.format(
        repo=instance.repo,
        base_commit=instance.base_commit[:12],
        problem_statement=instance.problem_statement.strip()[:8000],
        fail_to_pass=f2p,
        pass_to_pass_summary=p2p_summary,
    )


def _format_system(cfg: ReactConfig) -> str:
    tpl = load_prompt("system")
    return tpl.format(
        max_steps=cfg.max_steps,
        max_test_runs=cfg.max_test_runs,
        max_tokens=cfg.max_new_tokens_per_turn * cfg.max_steps,
    )


def run_react(
    *,
    llm: LLMClient,
    ctx: ToolContext,
    instance: SWEBenchInstance,
    trajectory: Trajectory,
    config: ReactConfig,
    cost: CostMeter | None = None,
) -> ReactRunResult:
    cost = cost or CostMeter()
    tools_by_name = {t.name: t for t in default_toolset(ctx)}
    ctx.test_runs_remaining = config.max_test_runs

    system = _format_system(config) + "\n\n# Available tools\n\n" + _format_tools_block(list(tools_by_name.values()))
    intro = _format_intro(instance)

    trajectory.add_message("system", system)
    trajectory.add_message("user", intro)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": intro},
    ]

    finished = False
    reason = "max_steps"

    for step in range(1, config.max_steps + 1):
        rsp = llm.chat(
            messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens_per_turn,
            seed=config.seed + step,
        )
        cost.add(rsp.tokens_in, rsp.tokens_out)
        trajectory.tokens_in += rsp.tokens_in
        trajectory.tokens_out += rsp.tokens_out
        trajectory.add_message("assistant", rsp.text)
        messages.append({"role": "assistant", "content": rsp.text})

        call = _extract_tool_call(rsp.text)
        if call is None:
            obs = (
                "Could not parse a tool call from your response. "
                'Reply with exactly one JSON object {"tool": "<name>", "arguments": {...}}.'
            )
            trajectory.add_message("tool", obs)
            messages.append({"role": "user", "content": obs})
            continue

        tool_name = call.get("tool", "")
        arguments = call.get("arguments", {}) or {}
        tool = tools_by_name.get(tool_name)
        if tool is None:
            obs = f"Unknown tool {tool_name!r}. Available: {list(tools_by_name)}"
            trajectory.add_message("tool", obs)
            messages.append({"role": "user", "content": obs})
            continue

        try:
            result = tool(**arguments)
        except Exception as e:  # tool surface errors -> observation
            _log.warning("tool.exception", tool=tool_name, error=str(e))
            obs = f"tool {tool_name} raised: {type(e).__name__}: {e}"
            trajectory.add_message("tool", obs)
            messages.append({"role": "user", "content": obs})
            continue

        trajectory.add_tool_call(
            step=step,
            tool=tool_name,
            arguments=arguments,
            result=result.to_dict(),
            duration_s=result.duration_s,
        )
        observation = result.output if not result.truncated else result.output
        trajectory.add_message("tool", observation, name=tool_name)
        messages.append({"role": "user", "content": f"[{tool_name} result]\n{observation}"})

        if tool_name == "finish" and result.ok:
            finished = True
            reason = "finish"
            break

    trajectory.n_steps = step
    METRICS.optimizer_steps_total  # touch (no-op) to keep import live
    return ReactRunResult(finished=finished, reason=reason, n_steps=step, cost=cost)
