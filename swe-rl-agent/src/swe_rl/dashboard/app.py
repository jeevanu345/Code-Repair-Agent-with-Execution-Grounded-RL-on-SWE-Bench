"""Small, read-only control-plane API and bundled web UI."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse

from swe_rl.settings import settings

_STATIC = Path(__file__).with_name("static")
app = FastAPI(title="SWE-RL Control Plane", docs_url="/api/docs")


def _tcp_status(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return "available"
    except OSError:
        return "unavailable"


def _docker_status() -> str:
    try:
        import docker

        getattr(docker, "from_env")().ping()
        return "available"
    except Exception:
        return "unavailable"


def _trajectory_files() -> list[Path]:
    roots = [settings.train_output_dir / "trajectories", settings.train_output_dir / "smoke"]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.jsonl"))
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def _load_run(path: Path) -> dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    except (OSError, IndexError, json.JSONDecodeError):
        return None
    return {
        "run_id": row.get("trajectory_id", path.stem),
        "instance_id": row.get("instance_id", ""),
        "status": "completed" if row.get("finished_at") else "incomplete",
        "model": row.get("model_name", ""),
        "steps": row.get("n_steps", 0),
        "reward": row.get("reward"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "tokens_in": row.get("tokens_in", 0),
        "tokens_out": row.get("tokens_out", 0),
        "tool_calls": row.get("tool_calls", []),
        "messages": row.get("messages", []),
        "reward_details": row.get("reward_details", {}),
        "final_patch": row.get("final_patch", ""),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    runs = [run for path in _trajectory_files() if (run := _load_run(path)) is not None]
    return {
        "api": "available",
        "docker": _docker_status(),
        "postgresql": _tcp_status(settings.postgres_host, settings.postgres_port),
        "redis": _tcp_status("127.0.0.1", 6379),
        "minio": _tcp_status("127.0.0.1", 9000),
        "model_endpoint": _tcp_status(settings.vllm_host, settings.vllm_port),
        "runs": {
            "active": sum(run["status"] == "incomplete" for run in runs),
            "completed": sum(run["status"] == "completed" for run in runs),
            "failed": sum(run["reward"] == 0 for run in runs if run["reward"] is not None),
        },
    }


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    return [run for path in _trajectory_files() if (run := _load_run(path)) is not None]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/{asset:path}")
def asset(asset: str) -> FileResponse:
    target = (_STATIC / asset).resolve()
    if _STATIC.resolve() not in target.parents:
        return FileResponse(_STATIC / "index.html")
    return FileResponse(target if target.is_file() else _STATIC / "index.html")
