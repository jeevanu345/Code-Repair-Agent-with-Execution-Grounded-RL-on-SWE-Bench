"""Shared inference client for local vLLM and cloud BYOK providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from swe_rl.agent.providers import PROVIDERS, Connection, active_connection, connection
from swe_rl.settings import settings


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
        provider: str | None = None,
        config: Connection | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config is not None:
            selected = config
        elif provider is not None or base_url:
            # Explicit endpoints preserve existing vLLM/custom caller behavior.
            selected = connection(
                provider or "custom", model or settings.model_name, api_key or "", base_url or ""
            )
        else:
            selected = active_connection()
            if model is not None or api_key is not None:
                selected = connection(
                    selected.provider,
                    model or selected.model,
                    api_key if api_key is not None else selected.api_key,
                    selected.base_url,
                )
        self.config = selected
        self.provider = selected.provider
        self.base_url = selected.base_url
        self.api_key = selected.api_key
        self.model = selected.model
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

    def _headers(self) -> dict[str, str]:
        if self.config.protocol == "anthropic":
            return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            result = self._client.request(
                method, f"{self.base_url}/{path}", headers=self._headers(), **kwargs
            )
            if result.status_code >= 300:
                # Never surface provider response bodies, request objects, or keys.
                status = result.status_code
                reason = {
                    401: "Invalid API key",
                    403: "Access denied",
                    404: "Model or endpoint unavailable",
                    429: "Rate limit or quota exceeded",
                }.get(status, "Provider request failed")
                raise RuntimeError(f"{reason} (HTTP {status})")
            body = result.json()
            if not isinstance(body, dict):
                raise ValueError("Expected JSON object")
            return body
        except httpx.TimeoutException:
            raise RuntimeError("Provider request timed out") from None
        except httpx.HTTPError:
            raise RuntimeError("Could not connect to the provider") from None
        except ValueError:
            raise RuntimeError("Provider returned invalid JSON") from None

    def models(self) -> list[str]:
        if PROVIDERS[self.provider].requires_key and not self.api_key:
            raise RuntimeError("Configure your provider API key before fetching models")
        data = self._request("GET", "models")
        return sorted(
            {
                item["id"]
                for item in data.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
        )[:1000]

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
        if not self.config.public()["ready"]:
            raise RuntimeError("Select a model and configure its API key in model settings")
        if self.config.protocol == "anthropic":
            system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
            chat_messages = [
                {
                    "role": "assistant" if m["role"] == "assistant" else "user",
                    "content": str(m["content"]),
                }
                for m in messages
                if m["role"] != "system"
            ]
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": chat_messages,
                "max_tokens": max_tokens,
            }
            if system:
                payload["system"] = system
            if stop:
                payload["stop_sequences"] = stop
            data = self._request("POST", "messages", json=payload)
            usage = data.get("usage", {})
            return LLMResponse(
                text="".join(
                    part.get("text", "")
                    for part in data.get("content", [])
                    if part.get("type") == "text"
                ),
                tokens_in=int(usage.get("input_tokens", 0)),
                tokens_out=int(usage.get("output_tokens", 0)),
                finish_reason=data.get("stop_reason") or "stop",
                raw=data,
            )
        payload = {"model": self.model, "messages": messages}
        payload["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = max_tokens
        # Cloud models differ in supported sampling parameters (especially reasoning
        # models); use provider defaults instead of sending vLLM-only options.
        if self.provider in {"vllm", "custom"}:
            payload.update(temperature=temperature, top_p=top_p)
            if seed is not None:
                payload["seed"] = seed
        if stop and self.provider in {"vllm", "custom"}:
            payload["stop"] = stop
        data = self._request("POST", "chat/completions", json=payload)
        try:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return LLMResponse(
                text=choice["message"].get("content") or "",
                tokens_in=int(usage.get("prompt_tokens", 0)),
                tokens_out=int(usage.get("completion_tokens", 0)),
                finish_reason=choice.get("finish_reason") or "stop",
                raw=data,
            )
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Provider returned an invalid completion") from None

    def close(self) -> None:
        self._client.close()
