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
        self._model = None
        if config.enabled and config.model_path:
            self._load()

    def _load(self) -> None:
        # Intentionally lazy. Real PRM would be a small reward model fine-tuned
        # to predict resolved outcome from partial trajectories.
        # Left as an extension point per Phase I.
        return None

    def score(self, messages: list[dict[str, str]]) -> float:
        if not self.config.enabled or self._model is None:
            return 0.0
        # Placeholder for the actual PRM forward pass.
        return 0.0
