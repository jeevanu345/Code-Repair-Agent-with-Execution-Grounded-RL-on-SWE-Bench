"""Cost guardrail: block rollouts when projected $ exceeds the run cap."""

from __future__ import annotations

from dataclasses import dataclass

from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.settings import settings

_log = get_logger(__name__)


class CostCapExceeded(RuntimeError):
    pass


@dataclass
class CostMeter:
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def dollars(self) -> float:
        return (
            self.tokens_in * settings.dollar_per_1k_tokens_in / 1000.0
            + self.tokens_out * settings.dollar_per_1k_tokens_out / 1000.0
        )

    def add(self, tokens_in: int, tokens_out: int) -> None:
        if tokens_in < 0 or tokens_out < 0:
            raise ValueError("token counts must be non-negative")
        projected_in = self.tokens_in + tokens_in
        projected_out = self.tokens_out + tokens_out
        projected = (
            projected_in * settings.dollar_per_1k_tokens_in / 1000.0
            + projected_out * settings.dollar_per_1k_tokens_out / 1000.0
        )
        if projected > settings.max_dollars_per_run:
            raise CostCapExceeded(
                f"Cost cap would be exceeded: ${projected:.2f} > "
                f"${settings.max_dollars_per_run:.2f}"
            )
        self.tokens_in = projected_in
        self.tokens_out = projected_out
        METRICS.tokens_in_total.inc(tokens_in)
        METRICS.tokens_out_total.inc(tokens_out)
        METRICS.dollar_cost_estimate.set(self.dollars)
