"""LLM client — talks to vLLM's OpenAI-compatible /v1/chat/completions.

Stateless. The ReAct loop owns conversation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

_log = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    finish_reason: str
    raw: dict[str, Any]


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.api_key = api_key or settings.vllm_api_key
        self.model = model or settings.model_name
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        seed: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop

        url = f"{self.base_url}/v1/chat/completions"
        r = self._client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice["message"]["content"] or "",
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    def close(self) -> None:
        self._client.close()
