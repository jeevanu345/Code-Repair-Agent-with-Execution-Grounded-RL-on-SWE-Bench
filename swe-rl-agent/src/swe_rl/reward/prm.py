"""Optional remote Process Reward Model (PRM).

The PRM never replaces execution reward. It is disabled by default and does
not load weights locally; an enabled PRM uses an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

_SCORE_RE = re.compile(r"(?:^|\s)(0(?:\.\d+)?|1(?:\.0+)?)(?:\s|$)")
_log = get_logger(__name__)


@dataclass(frozen=True)
class PRMConfig:
    enabled: bool = False
    model_name: str | None = None
    endpoint_url: str | None = None
    weight: float = 0.0
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("PRM weight must be in [0, 1]")
        if self.timeout_s <= 0:
            raise ValueError("PRM timeout must be positive")
        if self.enabled and not self.model_name:
            raise ValueError("an enabled PRM requires model_name")


class PRMScorer:
    def __init__(self, config: PRMConfig) -> None:
        self.config = config
        self.base_url = (config.endpoint_url or settings.vllm_base_url).rstrip("/")

    def score(self, messages: list[dict[str, str]]) -> float:
        """Return a bounded score; judge failures intentionally contribute zero."""
        if not self.config.enabled:
            return 0.0
        prompt = [
            {
                "role": "system",
                "content": "Score the latest repair step. Return only one number from 0 to 1.",
            },
            *messages,
        ]
        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
                json={
                    "model": self.config.model_name,
                    "messages": prompt,
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
                timeout=self.config.timeout_s,
            )
            response.raise_for_status()
            content = str(response.json()["choices"][0]["message"]["content"]).strip()
            match = _SCORE_RE.search(content)
            if match is None:
                raise ValueError("judge did not return a scalar score")
            return min(1.0, max(0.0, float(match.group(1))))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            _log.warning("prm.score_failed", error=str(exc))
            return 0.0
