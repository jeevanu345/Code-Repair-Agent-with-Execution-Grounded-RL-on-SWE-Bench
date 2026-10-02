"""Provider registry and shared, server-side inference profile.

Cloud endpoints use TLS. vLLM keeps the existing local configuration.
No cloud model is selected or contacted until the user supplies a model and key.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from swe_rl.settings import settings


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str
    docs: str
    protocol: str = "openai"
    requires_key: bool = True


PROVIDERS = {
    item.id: item
    for item in [
        Provider(
            "vllm",
            "Local vLLM",
            "",
            "https://docs.vllm.ai/en/latest/serving/openai_compatible_server/",
            requires_key=False,
        ),
        Provider(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            "https://platform.openai.com/docs/api-reference/chat",
        ),
        Provider(
            "anthropic",
            "Anthropic / Claude",
            "https://api.anthropic.com/v1",
            "https://platform.claude.com/docs/en/api/overview",
            protocol="anthropic",
        ),
        Provider(
            "nvidia",
            "NVIDIA NIM",
            "https://integrate.api.nvidia.com/v1",
            "https://docs.api.nvidia.com/nim/reference/llm-apis",
        ),
        Provider(
            "google",
            "Google / Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://ai.google.dev/gemini-api/docs/openai",
        ),
        Provider(
            "qwen",
            "Alibaba Cloud / Qwen",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "https://www.alibabacloud.com/help/en/model-studio/base-url",
        ),
        Provider(
            "deepseek", "DeepSeek", "https://api.deepseek.com/v1", "https://api-docs.deepseek.com/"
        ),
        Provider("mistral", "Mistral", "https://api.mistral.ai/v1", "https://docs.mistral.ai/api"),
        Provider(
            "xai",
            "xAI / Grok",
            "https://api.x.ai/v1",
            "https://docs.x.ai/developers/rest-api-reference/inference",
        ),
        Provider(
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1",
            "https://console.groq.com/docs/api-reference",
        ),
        Provider(
            "together",
            "Together AI",
            "https://api.together.xyz/v1",
            "https://www.together.ai/serverless-inference",
        ),
        Provider(
            "fireworks",
            "Fireworks AI",
            "https://api.fireworks.ai/inference/v1",
            "https://docs.fireworks.ai/",
        ),
        Provider(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "https://openrouter.ai/docs/quickstart",
        ),
        Provider("custom", "Custom OpenAI-compatible", "", "", requires_key=False),
    ]
}


@dataclass(frozen=True)
class Connection:
    provider: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)

    @property
    def protocol(self) -> str:
        return PROVIDERS[self.provider].protocol

    def public(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "has_key": bool(self.api_key),
            "ready": bool(
                self.model and (self.api_key or not PROVIDERS[self.provider].requires_key)
            ),
        }


def connection(provider: str, model: str, api_key: str = "", base_url: str = "") -> Connection:
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    definition = PROVIDERS[provider]
    url = (
        base_url or definition.base_url or (settings.vllm_base_url if provider == "vllm" else "")
    ).rstrip("/")
    if provider in {"vllm", "custom"} and not urlsplit(url).path.rstrip("/"):
        url += "/v1"
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Enter a base URL without credentials, query, or fragment")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Endpoint must use HTTP or HTTPS")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote endpoints must use HTTPS")
    if provider not in {"vllm", "custom", "qwen"} and url != definition.base_url:
        raise ValueError("Use the provider's official endpoint or select Custom")
    if provider == "qwen" and url not in {
        definition.base_url,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    }:
        raise ValueError("Select a supported Qwen regional endpoint")
    if any(char in api_key for char in "\r\n") or len(api_key) > 4096:
        raise ValueError("Invalid API key format")
    if len(model) > 256:
        raise ValueError("Model ID is too long")
    return Connection(provider, url, model.strip(), api_key.strip())


_lock = RLock()


def active_connection() -> Connection:
    with _lock:
        path = settings.llm_profile_path
        if path.is_symlink():
            raise ValueError("Profile file must not be a symlink")
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return connection(**data)
            except (OSError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid inference profile; reset it in model settings") from exc
        provider = settings.llm_provider
        return connection(
            provider,
            settings.llm_model or (settings.model_name if provider == "vllm" else ""),
            settings.llm_api_key or (settings.vllm_api_key if provider == "vllm" else ""),
            settings.llm_base_url or "",
        )


def save_connection(config: Connection) -> None:
    """Atomically save an explicitly supplied key, with owner-only permissions."""
    with _lock:
        path = settings.llm_profile_path
        if path.is_symlink():
            raise ValueError("Profile file must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".llm-profile-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(config), stream)
                stream.flush()
                os.fsync(stream.fileno())
            Path(name).chmod(0o600)
            Path(name).replace(path)
        finally:
            Path(name).unlink(missing_ok=True)


def clear_connection() -> None:
    with _lock:
        if settings.llm_profile_path.is_symlink():
            raise ValueError("Profile file must not be a symlink")
        settings.llm_profile_path.unlink(missing_ok=True)
