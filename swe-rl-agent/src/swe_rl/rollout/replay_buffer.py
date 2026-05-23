"""Disk-backed append-only replay buffer (Parquet shards)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from swe_rl.agent.trajectory import Trajectory


_SCHEMA = pa.schema(
    [
        ("trajectory_id", pa.string()),
        ("instance_id", pa.string()),
        ("seed", pa.int64()),
        ("temperature", pa.float32()),
        ("top_p", pa.float32()),
        ("model_name", pa.string()),
        ("checkpoint_sha", pa.string()),
        ("sandbox_image_digest", pa.string()),
        ("n_steps", pa.int32()),
        ("tokens_in", pa.int64()),
        ("tokens_out", pa.int64()),
        ("reward", pa.float32()),
        ("resolved", pa.bool_()),
        ("final_patch", pa.string()),
        ("messages_json", pa.string()),
        ("tool_calls_json", pa.string()),
        ("reward_details_json", pa.string()),
        ("metadata_json", pa.string()),
        ("created_at", pa.float64()),
    ]
)


@dataclass
class ReplayBuffer:
    root: Path
    shard_max_rows: int = 1000

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []

    def _row_for(self, traj: Trajectory) -> dict[str, Any]:
        resolved = bool(traj.reward_details.get("exec", {}).get("resolved", False)) if traj.reward_details else False
        return {
            "trajectory_id": traj.trajectory_id,
            "instance_id": traj.instance_id,
            "seed": int(traj.seed),
            "temperature": float(traj.temperature),
            "top_p": float(traj.top_p),
            "model_name": traj.model_name,
            "checkpoint_sha": traj.checkpoint_sha or "",
            "sandbox_image_digest": traj.sandbox_image_digest,
            "n_steps": int(traj.n_steps),
            "tokens_in": int(traj.tokens_in),
            "tokens_out": int(traj.tokens_out),
            "reward": float(traj.reward or 0.0),
            "resolved": resolved,
            "final_patch": traj.final_patch or "",
            "messages_json": json.dumps([m.__dict__ for m in traj.messages]),
            "tool_calls_json": json.dumps([t.__dict__ for t in traj.tool_calls]),
            "reward_details_json": json.dumps(traj.reward_details),
            "metadata_json": json.dumps(traj.metadata),
            "created_at": time.time(),
        }

    def add(self, traj: Trajectory) -> None:
        self._buffer.append(self._row_for(traj))
        if len(self._buffer) >= self.shard_max_rows:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        table = pa.Table.from_pylist(self._buffer, schema=_SCHEMA)
        path = self.root / f"shard_{int(time.time() * 1000)}.parquet"
        pq.write_table(table, path)
        self._buffer.clear()
        return path

    def iter_shards(self) -> Iterator[Path]:
        yield from sorted(self.root.glob("shard_*.parquet"))

    def iter_rows(self, *, only_resolved: bool | None = None) -> Iterator[dict[str, Any]]:
        for shard in self.iter_shards():
            t = pq.read_table(shard)
            for row in t.to_pylist():
                if only_resolved is not None and bool(row["resolved"]) != only_resolved:
                    continue
                yield row
