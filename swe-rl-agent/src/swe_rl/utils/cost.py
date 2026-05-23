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
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        METRICS.tokens_in_total.inc(tokens_in)
        METRICS.tokens_out_total.inc(tokens_out)
        METRICS.dollar_cost_estimate.set(self.dollars)
        if self.dollars > settings.max_dollars_per_run:
            _log.error(
                "cost.cap_exceeded",
                dollars=self.dollars,
                cap=settings.max_dollars_per_run,
            )
            raise CostCapExceeded(
                f"Cost cap exceeded: ${self.dollars:.2f} > ${settings.max_dollars_per_run:.2f}"
            )
