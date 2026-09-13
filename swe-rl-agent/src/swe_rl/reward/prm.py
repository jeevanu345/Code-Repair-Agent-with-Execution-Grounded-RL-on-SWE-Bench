"""Optional Process Reward Model (PRM) — frozen judge over intermediate steps.

Off by default. Enable via configs/reward/exec_grounded.yaml `prm.enabled: true`
and supply a checkpoint path. The PRM scores each step's (state, action) and the
trainer blends the average step-score with the terminal exec reward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRMConfig:
    enabled: bool = False
    model_path: str | None = None
    weight: float = 0.0


class PRMScorer:
    def __init__(self, config: PRMConfig) -> None:
        self.config = config
        # Rather than loading heavy weights locally, the PRM is hosted on a vLLM instance.
        # We will use the model_path as the base URL or rely on global settings.
        from swe_rl.settings import settings
        self.base_url = self.config.model_path or settings.vllm_base_url
        self.api_key = settings.vllm_api_key

    def score(self, messages: list[dict[str, str]]) -> float:
        if not self.config.enabled or not self.base_url:
            return 0.0
            
        import httpx
        try:
            # We assume a standard chat completions endpoint that acts as a judge,
            # or a custom /score endpoint if the PRM is a classification model.
            # Here we structure it as a prompt to the judge model.
            judge_prompt = [
                {"role": "system", "content": "You are a Process Reward Model. Evaluate the last step of the trajectory. Output a score between 0.0 and 1.0 representing the likelihood of success."},
            ] + messages
            
            # This is a structural implementation that sends the request to the vLLM server.
            # In a real environment, we would parse specific tokens or logprobs.
            response = httpx.post(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "prm-model", # The specific PRM model name
                    "messages": judge_prompt,
                    "max_tokens": 10,
                    "temperature": 0.0
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Simple parsing of the score
            import re
            match = re.search(r"(\d+\.\d+)", content)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            # PRM scoring shouldn't crash the rollout
            from swe_rl.observability.logging import get_logger
            _log = get_logger(__name__)
            _log.warning("prm.score_failed", error=str(e))
            return 0.0
