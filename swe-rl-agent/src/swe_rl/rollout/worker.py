"""Rollout worker — drives one trajectory end-to-end.

A rollout = (instance, seed) -> Trajectory + reward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.react_loop import ReactConfig, run_react
from swe_rl.agent.tools import ToolContext
from swe_rl.agent.trajectory import Trajectory
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.reward.exec_reward import compute_exec_reward
from swe_rl.reward.shaped_reward import ShapedRewardConfig, compute_shaped_reward
from swe_rl.sandbox.docker_runner import DockerRunner
from swe_rl.sandbox.repo_setup import REPO_DIR, setup_repo
from swe_rl.sandbox.test_executor import TestExecutor
from swe_rl.utils.cost import CostMeter

_log = get_logger(__name__)


@dataclass
class RolloutResult:
    trajectory: Trajectory
    reward: float
    resolved: bool
    final_patch: str
    duration_s: float
    details: dict[str, Any]


def run_rollout(
    instance: SWEBenchInstance,
    *,
    llm: LLMClient,
    react_config: ReactConfig,
    output_dir: Path,
    shaped_cfg: ShapedRewardConfig | None = None,
    apply_gold_patch: bool = False,
    checkpoint_sha: str | None = None,
) -> RolloutResult:
    """Run a single rollout. Container is created and torn down inside this function."""
    runner = DockerRunner()
    t0 = time.time()
    handle = runner.start()
    try:
        setup = setup_repo(runner, handle, instance)
        if not setup.ok:
            METRICS.container_failures_total.labels(kind="setup").inc()
            traj = Trajectory.new(
                instance_id=instance.instance_id,
                seed=react_config.seed,
                temperature=react_config.temperature,
                top_p=react_config.top_p,
                model_name=llm.model,
                sandbox_image_digest=handle.image_digest,
                checkpoint_sha=checkpoint_sha,
            )
            traj.metadata["setup_failed"] = True
            traj.metadata["install_log"] = setup.install_log[-4000:]
            traj.write_jsonl(output_dir)
            return RolloutResult(
                trajectory=traj,
                reward=0.0,
                resolved=False,
                final_patch="",
                duration_s=time.time() - t0,
                details={"setup_failed": True},
            )

        traj = Trajectory.new(
            instance_id=instance.instance_id,
            seed=react_config.seed,
            temperature=react_config.temperature,
            top_p=react_config.top_p,
            model_name=llm.model,
            sandbox_image_digest=handle.image_digest,
            checkpoint_sha=checkpoint_sha,
        )

        ctx = ToolContext(
            runner=runner,
            handle=handle,
            test_executor=TestExecutor(runner),
            repo_dir=REPO_DIR,
            test_runs_remaining=react_config.max_test_runs,
        )

        cost = CostMeter()

        if apply_gold_patch and instance.patch:
            # Sanity path used by smoke tests: skip the LLM, apply gold, evaluate.
            runner.write_file(handle, f"{REPO_DIR}/.gold.patch", instance.patch)
            runner.exec(
                handle,
                f"cd {REPO_DIR} && git apply --allow-empty -v .gold.patch || "
                f"git apply --allow-empty --reject -v .gold.patch",
                check_safety=False,
            )
            traj.metadata["mode"] = "gold_patch"
        else:
            run_react(
                llm=llm,
                ctx=ctx,
                instance=instance,
                trajectory=traj,
                config=react_config,
                cost=cost,
            )

        # Capture diff against the baseline commit we made during setup.
        diff = runner.commit_diff(handle, repo_dir=REPO_DIR)
        traj.final_patch = diff

        # Run F2P + P2P. Use one combined call when feasible to save time.
        executor = TestExecutor(runner)
        f2p_results = executor.run(handle, instance.fail_to_pass, timeout=300) if instance.fail_to_pass else _empty_results()
        p2p_results = executor.run(handle, instance.pass_to_pass[:50], timeout=300) if instance.pass_to_pass else _empty_results()

        exec_r = compute_exec_reward(
            f2p_results=f2p_results,
            p2p_results=p2p_results,
            fail_to_pass=instance.fail_to_pass,
            pass_to_pass=instance.pass_to_pass[:50],
        )
        shaped = compute_shaped_reward(
            exec_r,
            patch=diff,
            timed_out=f2p_results.timed_out or p2p_results.timed_out,
            config=shaped_cfg,
        )
        reward = exec_r.value  # primary signal stored on trajectory; trainer blends shaped separately
        traj.reward = reward
        traj.reward_details = {
            "exec": exec_r.to_dict(),
            "shaped": shaped.to_dict(),
        }
        traj.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = traj.write_jsonl(output_dir)
        _log.info(
            "rollout.done",
            instance=instance.instance_id,
            reward=reward,
            resolved=exec_r.resolved,
            n_steps=traj.n_steps,
            patch_bytes=len(diff),
            path=str(path),
        )
        METRICS.rollout_duration_seconds.observe(time.time() - t0)
        return RolloutResult(
            trajectory=traj,
            reward=reward,
            resolved=exec_r.resolved,
            final_patch=diff,
            duration_s=time.time() - t0,
            details={"exec": exec_r.to_dict(), "shaped": shaped.to_dict()},
        )
    finally:
        runner.teardown(handle)


def _empty_results() -> Any:
    from swe_rl.sandbox.test_executor import TestResults

    return TestResults()
