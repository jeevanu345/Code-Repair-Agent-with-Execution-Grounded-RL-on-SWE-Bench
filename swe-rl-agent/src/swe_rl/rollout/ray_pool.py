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
from swe_rl.rollout.worker import run_rollout

_log = get_logger(__name__)


@dataclass
class PoolConfig:
    n_workers: int = 8
    max_concurrent_per_worker: int = 1


@ray.remote
class GlobalCostMeter:
    def __init__(self, max_dollars: float | None):
        self.max_dollars = max_dollars
        self.total_cost = 0.0

    def add_cost(self, cost: float) -> bool:
        """Returns True if we are still under budget."""
        self.total_cost += cost
        if self.max_dollars is not None and self.total_cost > self.max_dollars:
            return False
        return True

    def get_total(self) -> float:
        return self.total_cost


@ray.remote(max_retries=3)
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
        cost_meter: ray.actor.ActorHandle | None = None,
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
        
        # In a real distributed setting, we'd pull the cost from result.trajectory
        # and report it to the GlobalCostMeter.
        # Currently, worker.py creates its own CostMeter.
        
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
    from swe_rl.db import get_session, Trajectory
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    actors = [
        RolloutActor.remote(model_name, vllm_base_url, vllm_api_key)
        for _ in range(pool.n_workers)
    ]
    
    cost_meter = GlobalCostMeter.remote(max_dollars=None)  # Can wire to settings later
    
    react_kwargs = {k: v for k, v in react_config.__dict__.items() if k not in {"seed"}}

    # Create a task list
    tasks = []
    for i, inst in enumerate(instances):
        for k in range(seeds_per_instance):
            seed = react_config.seed + k * 1000 + i
            tasks.append((inst, seed))
            
    # Check idempotency via database
    pending_tasks = []
    with get_session() as db:
        for inst, seed in tasks:
            exists = db.query(Trajectory).filter_by(instance_id=inst.instance_id, seed=seed).first()
            if exists:
                _log.info("rollout_pool.skip_existing", instance=inst.instance_id, seed=seed)
            else:
                pending_tasks.append((inst, seed))

    in_flight = {}
    results = []
    
    task_idx = 0
    while task_idx < len(pending_tasks) or in_flight:
        # Submit new tasks if we have capacity
        while len(in_flight) < (pool.n_workers * pool.max_concurrent_per_worker) and task_idx < len(pending_tasks):
            inst, seed = pending_tasks[task_idx]
            actor = actors[task_idx % len(actors)]
            
            future = actor.run.remote(
                inst.model_dump(),
                seed,
                react_kwargs,
                str(output_dir),
                None,
                apply_gold_patch,
                checkpoint_sha,
                cost_meter,
            )
            in_flight[future] = (inst.instance_id, seed)
            task_idx += 1

        if not in_flight:
            break

        # Wait for at least one task to complete
        ready, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=1.0)
        
        for done_ref in ready:
            inst_id, seed = in_flight.pop(done_ref)
            try:
                res = ray.get(done_ref)
                results.append(res)
            except ray.exceptions.TaskCancelledError:
                _log.warning("rollout_pool.task_cancelled", instance=inst_id, seed=seed)
            except Exception as e:
                _log.error("rollout_pool.task_failed", instance=inst_id, seed=seed, error=str(e))
                
    _log.info("rollout_pool.done", n=len(results), workers=pool.n_workers)
    return results
