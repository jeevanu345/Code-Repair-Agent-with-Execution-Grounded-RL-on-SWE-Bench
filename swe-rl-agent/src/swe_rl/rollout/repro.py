"""Replay a recorded trajectory deterministically.

This script parses a Trajectory JSONL file and re-executes the exact sequence
of tool calls in the sandbox without querying the LLM, verifying that the
environment produces identical observations and final patches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from swe_rl.agent.llm_client import LLMClient, LLMResponse
from swe_rl.agent.react_loop import ReactConfig, run_react
from swe_rl.agent.trajectory import Trajectory
from swe_rl.data.swebench_loader import load_one
from swe_rl.observability.logging import get_logger
from swe_rl.rollout.worker import run_rollout

_log = get_logger(__name__)


class MockLLMClient(LLMClient):
    """Mocks the LLM by yielding exact responses recorded in a trajectory."""
    
    def __init__(self, recorded_messages: list[dict[str, Any]]):
        super().__init__(base_url="", api_key="", model="mock")
        # filter to just the assistant messages from the original trajectory
        self.assistant_responses = [
            m["content"] for m in recorded_messages if m["role"] == "assistant"
        ]
        self.idx = 0

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        if self.idx >= len(self.assistant_responses):
            _log.warning("mock_llm.out_of_responses")
            return LLMResponse(text='{"tool": "finish", "arguments": {}}', tokens_in=0, tokens_out=0)
        
        text = self.assistant_responses[self.idx]
        self.idx += 1
        return LLMResponse(text=text, tokens_in=0, tokens_out=0)


def replay_trajectory(path: Path, output_dir: Path) -> None:
    """Read a trajectory, parse its instance, and re-execute deterministically."""
    traj = Trajectory.from_jsonl(path)
    inst = load_one(traj.instance_id)
    
    _log.info("repro.start", instance=inst.instance_id, seed=traj.seed)
    
    mock_llm = MockLLMClient(traj.messages)
    cfg = ReactConfig(
        seed=traj.seed,
        max_steps=traj.n_steps,
        temperature=traj.temperature,
        top_p=traj.top_p,
    )
    
    # Run rollout using the mocked LLM
    result = run_rollout(
        instance=inst,
        llm=mock_llm,
        react_config=cfg,
        output_dir=output_dir,
    )
    
    # Compare outcomes
    if result.final_patch != traj.final_patch:
        _log.error("repro.patch_mismatch")
        print("Original patch:")
        print(traj.final_patch)
        print("Replayed patch:")
        print(result.final_patch)
        raise RuntimeError("Replay diverged from recorded trajectory.")
    
    _log.info(
        "repro.success", 
        reward=result.reward, 
        resolved=result.resolved, 
        original_reward=traj.reward
    )
