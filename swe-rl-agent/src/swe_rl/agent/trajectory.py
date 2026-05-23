"""Serializable rollout trajectory record."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    name: str | None = None


@dataclass
class ToolCall:
    step: int
    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    duration_s: float


@dataclass
class Trajectory:
    trajectory_id: str
    instance_id: str
    seed: int
    temperature: float
    top_p: float
    model_name: str
    checkpoint_sha: str | None
    sandbox_image_digest: str
    started_at: str
    finished_at: str | None = None
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_patch: str = ""
    n_steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    reward: float | None = None
    reward_details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        instance_id: str,
        seed: int,
        temperature: float,
        top_p: float,
        model_name: str,
        sandbox_image_digest: str,
        checkpoint_sha: str | None = None,
    ) -> "Trajectory":
        return cls(
            trajectory_id=uuid.uuid4().hex,
            instance_id=instance_id,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            model_name=model_name,
            checkpoint_sha=checkpoint_sha,
            sandbox_image_digest=sandbox_image_digest,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

    def add_message(self, role: str, content: str, *, name: str | None = None) -> None:
        self.messages.append(Message(role=role, content=content, name=name))

    def add_tool_call(
        self, *, step: int, tool: str, arguments: dict[str, Any], result: dict[str, Any], duration_s: float
    ) -> None:
        self.tool_calls.append(
            ToolCall(step=step, tool=tool, arguments=arguments, result=result, duration_s=duration_s)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_jsonl(self, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{self.trajectory_id}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
        return path

    @classmethod
    def from_jsonl(cls, path: Path) -> "Trajectory":
        with path.open() as f:
            data = json.loads(f.readline())
        msgs = [Message(**m) for m in data.pop("messages", [])]
        calls = [ToolCall(**c) for c in data.pop("tool_calls", [])]
        traj = cls(**data)
        traj.messages = msgs
        traj.tool_calls = calls
        return traj
