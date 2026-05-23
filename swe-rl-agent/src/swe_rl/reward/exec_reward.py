"""Binary execution-grounded reward.

`resolved` ⇔ every FAIL_TO_PASS test now passes AND every PASS_TO_PASS test still passes.
Reward = 1.0 if resolved else 0.0.

Pure function of test outcomes — no network, no LLM. Reward must be reproducible
from saved trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

from swe_rl.observability.metrics import METRICS
from swe_rl.sandbox.test_executor import TestResults


@dataclass
class ExecRewardResult:
    value: float
    resolved: bool
    fail_to_pass_pass_rate: float
    pass_to_pass_pass_rate: float
    n_fail_to_pass: int
    n_pass_to_pass: int
    f2p_passed: list[str]
    f2p_failed: list[str]
    p2p_passed: list[str]
    p2p_failed: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "resolved": self.resolved,
            "fail_to_pass_pass_rate": self.fail_to_pass_pass_rate,
            "pass_to_pass_pass_rate": self.pass_to_pass_pass_rate,
            "n_fail_to_pass": self.n_fail_to_pass,
            "n_pass_to_pass": self.n_pass_to_pass,
            "f2p_passed": self.f2p_passed,
            "f2p_failed": self.f2p_failed,
            "p2p_passed": self.p2p_passed,
            "p2p_failed": self.p2p_failed,
        }


def compute_exec_reward(
    f2p_results: TestResults,
    p2p_results: TestResults,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> ExecRewardResult:
    f2p_pass = [t for t in fail_to_pass if f2p_results.outcome_for(t) == "passed"]
    f2p_fail = [t for t in fail_to_pass if f2p_results.outcome_for(t) != "passed"]
    p2p_pass = [t for t in pass_to_pass if p2p_results.outcome_for(t) == "passed"]
    p2p_fail = [t for t in pass_to_pass if p2p_results.outcome_for(t) != "passed"]

    f2p_rate = len(f2p_pass) / len(fail_to_pass) if fail_to_pass else 1.0
    p2p_rate = len(p2p_pass) / len(pass_to_pass) if pass_to_pass else 1.0

    resolved = (not f2p_fail) and (not p2p_fail) and bool(fail_to_pass)
    value = 1.0 if resolved else 0.0
    METRICS.reward_value.observe(value)
    return ExecRewardResult(
        value=value,
        resolved=resolved,
        fail_to_pass_pass_rate=f2p_rate,
        pass_to_pass_pass_rate=p2p_rate,
        n_fail_to_pass=len(fail_to_pass),
        n_pass_to_pass=len(pass_to_pass),
        f2p_passed=f2p_pass,
        f2p_failed=f2p_fail,
        p2p_passed=p2p_pass,
        p2p_failed=p2p_fail,
    )


@dataclass
class ExecReward:
    """Object form for use as a callable reward in trainers."""

    def __call__(
        self,
        f2p_results: TestResults,
        p2p_results: TestResults,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
    ) -> ExecRewardResult:
        return compute_exec_reward(f2p_results, p2p_results, fail_to_pass, pass_to_pass)
