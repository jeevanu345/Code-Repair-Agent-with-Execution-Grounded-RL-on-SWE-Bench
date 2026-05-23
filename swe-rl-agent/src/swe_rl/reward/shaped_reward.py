"""Dense shaped reward = mix of partial test pass-rates with edit-size penalty.

reward = 0.5 * f2p_rate + 0.5 * p2p_rate
       - lambda * size_penalty
       - mu * lint_penalty
       (+ no_patch_penalty if patch is empty)

Combined with binary at trainer level: total = binary + alpha * shaped, alpha
decayed over training to push the model from partial credit to full resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

from swe_rl.reward.exec_reward import ExecRewardResult
from swe_rl.utils.patch import is_empty_patch, patch_stats


@dataclass(frozen=True)
class ShapedRewardConfig:
    edit_size_penalty_lambda: float = 0.001
    lint_penalty_mu: float = 0.05
    no_patch_penalty: float = -0.1
    test_timeout_penalty: float = -0.05


@dataclass
class ShapedRewardResult:
    value: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "components": self.components}


def compute_shaped_reward(
    exec_result: ExecRewardResult,
    *,
    patch: str,
    timed_out: bool,
    lint_errors: int = 0,
    config: ShapedRewardConfig | None = None,
) -> ShapedRewardResult:
    cfg = config or ShapedRewardConfig()

    base = 0.5 * exec_result.fail_to_pass_pass_rate + 0.5 * exec_result.pass_to_pass_pass_rate

    files, added, removed = patch_stats(patch)
    size_penalty = cfg.edit_size_penalty_lambda * (added + removed)
    lint_penalty = cfg.lint_penalty_mu * float(lint_errors)
    no_patch = cfg.no_patch_penalty if is_empty_patch(patch) else 0.0
    timeout_pen = cfg.test_timeout_penalty if timed_out else 0.0

    value = base - size_penalty - lint_penalty + no_patch + timeout_pen

    return ShapedRewardResult(
        value=value,
        components={
            "base": base,
            "size_penalty": size_penalty,
            "lint_penalty": lint_penalty,
            "no_patch": no_patch,
            "timeout_penalty": timeout_pen,
            "files": float(files),
            "lines_added": float(added),
            "lines_removed": float(removed),
        },
    )


@dataclass
class ShapedReward:
    config: ShapedRewardConfig = ShapedRewardConfig()

    def __call__(
        self,
        exec_result: ExecRewardResult,
        *,
        patch: str,
        timed_out: bool,
        lint_errors: int = 0,
    ) -> ShapedRewardResult:
        return compute_shaped_reward(
            exec_result,
            patch=patch,
            timed_out=timed_out,
            lint_errors=lint_errors,
            config=self.config,
        )
