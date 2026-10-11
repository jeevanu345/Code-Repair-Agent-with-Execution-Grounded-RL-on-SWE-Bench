"""Model capability registry and explicit fallback configuration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from swe_rl.agent.model_registry import models, register_model, request_telemetry
from swe_rl.agent.providers import PROVIDERS, active_connection, connection
from swe_rl.settings import settings


class ModelRecord(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=256)
    context_limit: int | None = Field(None, ge=1)
    tool_calling: bool | None = None
    json_mode: bool | None = None
    streaming: bool | None = None
    input_price_per_1k: float | None = Field(None, ge=0)
    output_price_per_1k: float | None = Field(None, ge=0)


class FallbackSettings(BaseModel):
    enabled: bool = False
    provider: str = "vllm"
    model: str = Field("", max_length=256)
    base_url: str = ""
    # Fallback only uses an environment/OS-managed credential reference.
    secret_ref: str = ""


class PromptRecord(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    content: str = Field(min_length=1, max_length=32000)


def fallback_path() -> Path:
    return settings.llm_profile_path.with_name(".llm-fallback.json")


def fallback_connection() -> Any:
    from swe_rl.agent.credentials import load_secret

    try:
        data = json.loads(fallback_path().read_text())
    except FileNotFoundError:
        return None
    if not data.get("enabled"):
        return None
    reference = data.get("secret_ref", "")
    key = load_secret(reference) if reference else ""
    return connection(data["provider"], data["model"], key, data.get("base_url", ""))


def build_router(protect: Any) -> APIRouter:
    router = APIRouter()

    @router.get("/api/prompts")
    def prompts() -> list[dict[str, Any]]:
        from swe_rl.agent.prompt_registry import list_prompts

        return list_prompts()

    @router.post("/api/prompts", dependencies=[Depends(protect)])
    def save_prompt(body: PromptRecord) -> dict[str, Any]:
        from swe_rl.agent.prompt_registry import register_prompt

        try:
            return register_prompt(body.name, body.version, body.content)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @router.get("/api/model/registry")
    def registry() -> dict[str, Any]:
        return {
            "models": models(),
            "telemetry": request_telemetry(),
            "capability_source": "user_declared",
            "telemetry_scope": "dashboard process; not distributed provider telemetry",
        }

    @router.post("/api/model/registry", dependencies=[Depends(protect)])
    def register(body: ModelRecord) -> dict[str, Any]:
        if body.provider not in PROVIDERS:
            raise HTTPException(400, "Unknown provider")
        return register_model(body.model_dump())

    @router.get("/api/model/telemetry")
    def telemetry() -> list[dict[str, Any]]:
        return request_telemetry()

    @router.get("/api/model/fallback")
    def fallback() -> dict[str, Any]:
        try:
            row = json.loads(fallback_path().read_text())
            row.pop("secret_ref", None)
            return row
        except (OSError, ValueError):
            return {"enabled": False}

    @router.post("/api/model/fallback", dependencies=[Depends(protect)])
    def save_fallback(body: FallbackSettings) -> dict[str, Any]:
        from swe_rl.agent.credentials import secret_reference, store_secret

        data = body.model_dump()
        if body.enabled:
            current = active_connection()
            config = connection(body.provider, body.model, "", body.base_url)
            if (
                current.provider == config.provider
                and current.base_url == config.base_url
                and current.api_key
            ):
                reference = secret_reference(fallback_path(), config.provider, config.base_url)
                try:
                    store_secret(reference, current.api_key)
                except RuntimeError as exc:
                    raise HTTPException(400, str(exc)) from None
                data["secret_ref"] = reference
            elif body.secret_ref:
                try:
                    from swe_rl.agent.credentials import load_secret

                    load_secret(body.secret_ref)
                except (RuntimeError, ValueError) as exc:
                    raise HTTPException(400, str(exc)) from None
            elif PROVIDERS[body.provider].requires_key:
                raise HTTPException(
                    400,
                    "Save this provider as the active connection first to copy its OS-managed credential",
                )
        path = fallback_path()
        if path.is_symlink():
            raise HTTPException(400, "Fallback profile must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".fallback-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream)
            Path(temporary).chmod(0o600)
            Path(temporary).replace(path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {key: value for key, value in data.items() if key != "secret_ref"}

    return router
