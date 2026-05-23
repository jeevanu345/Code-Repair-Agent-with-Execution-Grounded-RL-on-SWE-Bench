"""Ray-based distributed rollout pool.

Each Ray actor runs a `run_rollout` against its own DockerRunner. Use Ray
because (a) it gives us cheap actor-per-Docker isolation, (b) we get failure
isolation per task, and (c) it scales to multi-node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.react_loop import ReactConfig
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.observability.logging import get_logger
from swe_rl.reward.shaped_reward import ShapedRewardConfig
from swe_rl.rollout.worker import RolloutResult, run_rollout

_log = get_logger(__name__)


@dataclass
class PoolConfig:
    n_workers: int = 8
    max_concurrent_per_worker: int = 1


@ray.remote
class RolloutActor:
    def __init__(self, model_name: str, vllm_base_url: str, vllm_api_key: str) -> None:
        self.llm = LLMClient(base_url=vllm_base_url, api_key=vllm_api_key, model=model_name)

    def run(
        self,
        instance_dict: dict[str, Any],
        seed: int,
        react_kwargs: dict[str, Any],
        output_dir: str,
        shaped_kwargs: dict[str, Any] | None,
        apply_gold_patch: bool,
        checkpoint_sha: str | None,
    ) -> dict[str, Any]:
        instance = SWEBenchInstance.model_validate(instance_dict)
        cfg = ReactConfig(seed=seed, **react_kwargs)
        shaped_cfg = ShapedRewardConfig(**(shaped_kwargs or {}))
        result = run_rollout(
            instance,
            llm=self.llm,
            react_config=cfg,
            output_dir=Path(output_dir),
            shaped_cfg=shaped_cfg,
            apply_gold_patch=apply_gold_patch,
            checkpoint_sha=checkpoint_sha,
        )
        return {
            "trajectory_id": result.trajectory.trajectory_id,
            "instance_id": instance.instance_id,
            "reward": result.reward,
            "resolved": result.resolved,
            "final_patch": result.final_patch,
            "duration_s": result.duration_s,
        }


def submit_pool(
    instances: list[SWEBenchInstance],
    *,
    pool: PoolConfig,
    react_config: ReactConfig,
    output_dir: Path,
    model_name: str,
    vllm_base_url: str,
    vllm_api_key: str,
    seeds_per_instance: int = 1,
    apply_gold_patch: bool = False,
    checkpoint_sha: str | None = None,
) -> list[dict[str, Any]]:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    actors = [
        RolloutActor.remote(model_name, vllm_base_url, vllm_api_key)
        for _ in range(pool.n_workers)
    ]
    react_kwargs = {
        k: v for k, v in react_config.__dict__.items() if k not in {"seed"}
    }

    futures = []
    for i, inst in enumerate(instances):
        for k in range(seeds_per_instance):
            actor = actors[i % len(actors)]
            futures.append(
                actor.run.remote(
                    inst.model_dump(),
                    react_config.seed + k * 1000 + i,
                    react_kwargs,
                    str(output_dir),
                    None,
                    apply_gold_patch,
                    checkpoint_sha,
                )
            )

    results: list[dict[str, Any]] = ray.get(futures)
    _log.info("rollout_pool.done", n=len(results), workers=pool.n_workers)
    return results
