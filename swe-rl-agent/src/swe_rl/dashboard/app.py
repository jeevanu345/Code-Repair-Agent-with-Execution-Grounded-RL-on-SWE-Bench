"""Repair evidence, local model configuration, and provider inference API."""

from __future__ import annotations

import json
import secrets
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.providers import (
    PROVIDERS,
    Connection,
    active_connection,
    clear_connection,
    connection,
    save_connection,
)
from swe_rl.settings import settings

_STATIC = Path(__file__).with_name("static")
app = FastAPI(title="SWE-RL Control Plane", docs_url="/api/docs")
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
)
_token = secrets.token_urlsafe(32)


@app.exception_handler(RequestValidationError)
async def safe_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default validation response can reflect raw secret input.
    return JSONResponse(status_code=422, content={"detail": "Invalid request fields"})


def protect_settings(request: Request) -> None:
    if not secrets.compare_digest(request.headers.get("x-settings-token", ""), _token):
        raise HTTPException(403, "Refresh the page to authorize model settings")
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Cross-origin model requests are not allowed")


class ModelSettings(BaseModel):
    provider: str
    model: str = Field("", max_length=256)
    base_url: str = Field("", max_length=2048)
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    clear_key: bool = False


def submitted_config(body: ModelSettings) -> Connection:
    try:
        current = active_connection()
    except ValueError:
        current = None
    key = body.api_key.get_secret_value()
    if (
        not key
        and not body.clear_key
        and current
        and body.provider == current.provider
        and (not body.base_url or body.base_url.rstrip("/") == current.base_url)
    ):
        key = current.api_key
    try:
        return connection(body.provider, body.model, key, body.base_url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@app.get("/api/providers")
def providers() -> dict[str, Any]:
    try:
        active = active_connection().public()
    except ValueError:
        active = {
            "provider": "vllm",
            "model": "",
            "base_url": settings.vllm_base_url,
            "has_key": False,
            "ready": False,
        }
    return {
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "base_url": p.base_url,
                "docs": p.docs,
                "requires_key": p.requires_key,
            }
            for p in PROVIDERS.values()
        ],
        "active": active,
        "settings_token": _token,
    }


@app.post("/api/model/settings", dependencies=[Depends(protect_settings)])
def model_settings(body: ModelSettings) -> dict[str, object]:
    config = submitted_config(body)
    try:
        save_connection(config)
    except (OSError, ValueError):
        raise HTTPException(500, "Could not save the local model profile") from None
    return config.public()


@app.delete("/api/model/settings", dependencies=[Depends(protect_settings)])
def reset_model_settings() -> dict[str, object]:
    clear_connection()
    return active_connection().public()


@app.post("/api/model/models", dependencies=[Depends(protect_settings)])
def list_models(body: ModelSettings) -> dict[str, object]:
    client = LLMClient(config=submitted_config(body), timeout=15)
    try:
        return {"models": client.models()}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        client.close()


@app.post("/api/model/test", dependencies=[Depends(protect_settings)])
def test_model(body: ModelSettings) -> dict[str, object]:
    client = LLMClient(config=submitted_config(body), timeout=30)
    try:
        result = client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=256)
        if not result.text.strip():
            raise HTTPException(400, "Provider returned no text; try a different model")
        return {"ok": True, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        client.close()


class AssistRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)


@app.post("/api/model/assist", dependencies=[Depends(protect_settings)])
def assist(body: AssistRequest) -> dict[str, object]:
    client = LLMClient(timeout=60)
    try:
        result = client.chat(
            [
                {
                    "role": "system",
                    "content": "You are a code repair assistant. Explain the issue and suggest a focused fix. Do not claim to have executed tests.",
                },
                {"role": "user", "content": body.prompt},
            ],
            max_tokens=1024,
        )
        return {
            "text": result.text,
            "provider": client.provider,
            "model": client.model,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
        }
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        client.close()


def _tcp_status(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return "available"
    except OSError:
        return "unavailable"


def _docker_status() -> str:
    try:
        import docker

        with getattr(docker, "from_env")(timeout=2) as client:
            client.ping()
        return "available"
    except Exception:
        return "unavailable"


def _trajectory_files() -> list[Path]:
    roots = [settings.train_output_dir]
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
    if not isinstance(row, dict) or not row.get("trajectory_id"):
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
        "metadata": row.get("metadata", {}),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    runs = [run for path in _trajectory_files() if (run := _load_run(path)) is not None]
    try:
        selected = active_connection()
        endpoint = urlsplit(selected.base_url)
        model_status = (
            _tcp_status(endpoint.hostname or "127.0.0.1", endpoint.port or 8000)
            if selected.provider in {"vllm", "custom"}
            and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
            else "configured"
            if selected.public()["ready"]
            else "not configured"
        )
    except ValueError:
        model_status = "invalid configuration"
    return {
        "api": "available",
        "docker": _docker_status(),
        "postgresql": _tcp_status(settings.postgres_host, settings.postgres_port),
        "redis": _tcp_status("127.0.0.1", 6379),
        "minio": _tcp_status("127.0.0.1", 9000),
        "model_endpoint": model_status,
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
