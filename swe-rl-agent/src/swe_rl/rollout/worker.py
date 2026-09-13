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
from swe_rl.reward.prm import PRMScorer
from swe_rl.reward.shaped_reward import ShapedRewardConfig, compute_shaped_reward
from swe_rl.sandbox.docker_runner import DockerRunner
from swe_rl.sandbox.repo_setup import REPO_DIR, setup_repo
from swe_rl.sandbox.test_executor import TestExecutor
from swe_rl.settings import settings
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
    candidate_patch: str | None = None,
    checkpoint_sha: str | None = None,
) -> RolloutResult:
    """Run a single rollout. Container is created and torn down inside this function."""
    runner = DockerRunner()
    t0 = time.time()
    handle = runner.start()
    try:
        # Hidden evaluation tests must not be visible to the agent.
        setup = setup_repo(runner, handle, instance, apply_test_patch=False)
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
            traj.metadata.update(_instance_metadata(instance))
            traj.metadata["setup_failed"] = True
            traj.metadata["install_log"] = setup.install_log[-4000:]
            traj.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
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
        traj.metadata.update(
            {
                "model_revision": settings.model_revision,
                "network_disabled": handle.network_disabled,
            } | _instance_metadata(instance)
        )

        ctx = ToolContext(
            runner=runner,
            handle=handle,
            test_executor=TestExecutor(runner),
            repo_dir=REPO_DIR,
            test_runs_remaining=react_config.max_test_runs,
        )

        cost = CostMeter()
        patch_apply_ok = True

        if apply_gold_patch and candidate_patch is not None:
            raise ValueError("apply_gold_patch and candidate_patch are mutually exclusive")

        patch_to_apply = instance.patch if apply_gold_patch else candidate_patch
        if patch_to_apply is not None:
            # Sanity path used by smoke tests: skip the LLM, apply gold, evaluate.
            patch_name = ".gold.patch" if apply_gold_patch else ".candidate.patch"
            runner.write_file(handle, f"/tmp/{patch_name}", patch_to_apply)
            applied = runner.exec(
                handle,
                f"cd {REPO_DIR} && git apply --check --allow-empty /tmp/{patch_name} && "
                f"git apply --allow-empty -v /tmp/{patch_name}",
                check_safety=False,
            )
            if applied.exit_code != 0:
                traj.metadata["patch_apply_failed"] = True
                patch_apply_ok = False
            traj.metadata["mode"] = "gold_patch" if apply_gold_patch else "candidate_patch"
        else:
            try:
                run_react(
                    llm=llm,
                    ctx=ctx,
                    instance=instance,
                    trajectory=traj,
                    config=react_config,
                    cost=cost,
                )
            except Exception as exc:
                # A cost limit or unavailable endpoint must produce a saved,
                # zero-reward trajectory rather than silently dropping work.
                traj.metadata["agent_failed"] = type(exc).__name__
                traj.metadata["agent_error"] = str(exc)[:1000]

        # Capture diff against the baseline commit we made during setup.
        diff = runner.commit_diff(handle, repo_dir=REPO_DIR)
        traj.final_patch = diff

        test_patch_ok = True
        if instance.test_patch:
            runner.write_file(handle, "/tmp/.swe_test.patch", instance.test_patch)
            applied_tests = runner.exec(
                handle,
                f"cd {REPO_DIR} && git apply --check --allow-empty /tmp/.swe_test.patch && "
                "git apply --allow-empty -v /tmp/.swe_test.patch",
                check_safety=False,
            )
            test_patch_ok = applied_tests.exit_code == 0
            if not test_patch_ok:
                traj.metadata["test_patch_apply_failed"] = True

        # Run F2P + P2P. Use one combined call when feasible to save time.
        executor = TestExecutor(runner)
        f2p_results = (
            executor.run(handle, instance.fail_to_pass, timeout=300)
            if instance.fail_to_pass
            else _empty_results()
        )
        p2p_results = (
            executor.run(handle, instance.pass_to_pass, timeout=300)
            if instance.pass_to_pass
            else _empty_results()
        )

        exec_r = compute_exec_reward(
            f2p_results=f2p_results,
            p2p_results=p2p_results,
            fail_to_pass=instance.fail_to_pass,
            pass_to_pass=instance.pass_to_pass,
        )
        if not test_patch_ok or not patch_apply_ok:
            exec_r.resolved = False
            exec_r.value = 0.0
        prm_score = PRMScorer((shaped_cfg or ShapedRewardConfig()).prm).score(
            [{"role": message.role, "content": message.content} for message in traj.messages]
        )
        shaped = compute_shaped_reward(
            exec_r,
            patch=diff,
            timed_out=f2p_results.timed_out or p2p_results.timed_out,
            config=shaped_cfg,
            prm_score=prm_score,
        )
        reward = (
            exec_r.value
        )  # primary signal stored on trajectory; trainer blends shaped separately
        traj.reward = reward
        traj.reward_details = {
            "exec": exec_r.to_dict(),
            "shaped": shaped.to_dict(),
        }
        traj.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path = traj.write_jsonl(output_dir)
        
        # Upload to the configured MinIO bucket. Failure is explicit in the
        # trajectory metadata but does not discard the local durable artifact.
        s3_path = str(path)
        try:
            from minio import Minio
            client = Minio(
                settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=False,
            )
            if not client.bucket_exists(settings.minio_bucket):
                client.make_bucket(settings.minio_bucket)
            client.fput_object(settings.minio_bucket, f"{traj.trajectory_id}.jsonl", str(path))
            s3_path = f"s3://{settings.minio_bucket}/{traj.trajectory_id}.jsonl"
        except Exception as e:
            _log.warning("rollout.minio_upload_failed", error=str(e))
            traj.metadata["object_storage_error"] = str(e)[:500]

        # Save to PostgreSQL
        try:
            from swe_rl.db import get_session, save_trajectory
            with get_session() as session:
                save_trajectory(session, traj.__dict__, {"exec": exec_r.value, "shaped": shaped.value}, s3_path)
        except Exception as e:
            _log.error("rollout.db_save_failed", error=str(e))

        _log.info(
            "rollout.done",
            instance=instance.instance_id,
            reward=reward,
            resolved=exec_r.resolved,
            n_steps=traj.n_steps,
            patch_bytes=len(diff),
            path=s3_path,
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


def _instance_metadata(instance: SWEBenchInstance) -> dict[str, Any]:
    """Persist the evaluation identity required for trustworthy replay/audit."""
    return {
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "problem_statement": instance.problem_statement,
        "fail_to_pass": instance.fail_to_pass,
        "pass_to_pass": instance.pass_to_pass,
        "test_patch": instance.test_patch,
        "environment_setup_commit": instance.environment_setup_commit,
    }
