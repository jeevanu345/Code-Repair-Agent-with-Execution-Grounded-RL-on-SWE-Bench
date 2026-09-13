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
    task_timeout_s: float = 3_600.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.n_workers < 1 or self.max_concurrent_per_worker < 1:
            raise ValueError("worker counts must be positive")
        if self.task_timeout_s <= 0 or self.max_retries < 0:
            raise ValueError("task_timeout_s must be positive and max_retries non-negative")


@ray.remote
class GlobalCostMeter:
    def __init__(self, max_dollars: float | None):
        self.max_dollars = max_dollars
        self.total_cost = 0.0
        self.reserved_cost = 0.0

    def reserve(self, cost: float) -> bool:
        """Reserve a worst-case rollout cost before contacting the model."""
        if cost < 0:
            raise ValueError("cost must be non-negative")
        if self.max_dollars is not None and self.total_cost + self.reserved_cost + cost > self.max_dollars:
            return False
        self.reserved_cost += cost
        return True

    def settle(self, reserved: float, actual: float) -> None:
        if reserved < 0 or actual < 0 or actual > reserved:
            raise ValueError("invalid cost settlement")
        self.reserved_cost -= reserved
        self.total_cost += actual

    def get_total(self) -> float:
        return self.total_cost


@ray.remote(max_retries=2)
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
        from swe_rl.settings import settings

        reservation = settings.max_dollars_per_run
        if not ray.get(cost_meter.reserve.remote(reservation)):
            return {
                "instance_id": instance_dict["instance_id"],
                "reward": 0.0,
                "resolved": False,
                "skipped": True,
                "error": "pool cost cap would be exceeded",
                "tokens_in": 0,
                "tokens_out": 0,
            }
        instance = SWEBenchInstance.model_validate(instance_dict)
        cfg = ReactConfig(seed=seed, **react_kwargs)
        shaped_cfg = ShapedRewardConfig(**(shaped_kwargs or {}))
        actual_cost = 0.0
        try:
            result = run_rollout(
                instance,
                llm=self.llm,
                react_config=cfg,
                output_dir=Path(output_dir),
                shaped_cfg=shaped_cfg,
                apply_gold_patch=apply_gold_patch,
                checkpoint_sha=checkpoint_sha,
            )
            actual_cost = (
                result.trajectory.tokens_in * settings.dollar_per_1k_tokens_in / 1000.0
                + result.trajectory.tokens_out * settings.dollar_per_1k_tokens_out / 1000.0
            )
        finally:
            ray.get(cost_meter.settle.remote(reservation, actual_cost))
        
        return {
            "trajectory_id": result.trajectory.trajectory_id,
            "instance_id": instance.instance_id,
            "reward": result.reward,
            "resolved": result.resolved,
            "final_patch": result.final_patch,
            "duration_s": result.duration_s,
            "tokens_in": result.trajectory.tokens_in,
            "tokens_out": result.trajectory.tokens_out,
            "cost_accepted": True,
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
    from swe_rl.db import get_session, trajectory_exists
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    actors = [
        RolloutActor.remote(model_name, vllm_base_url, vllm_api_key)
        for _ in range(pool.n_workers)
    ]
    
    from swe_rl.settings import settings

    cost_meter = GlobalCostMeter.remote(max_dollars=settings.max_dollars_per_pool)
    
    react_kwargs = {k: v for k, v in react_config.__dict__.items() if k not in {"seed"}}

    # Create a task list
    tasks = []
    for i, inst in enumerate(instances):
        for k in range(seeds_per_instance):
            seed = react_config.seed + k * 1000 + i
            tasks.append((inst, seed))
            
    # Do not resubmit already-persisted logical rollouts. This check is best
    # effort when Postgres is intentionally unavailable for local-only runs.
    pending_tasks = list(tasks)
    try:
        with get_session() as db:
            pending_tasks = []
            for inst, seed in tasks:
                if trajectory_exists(
                    db, instance_id=inst.instance_id, seed=seed, checkpoint_sha=checkpoint_sha
                ):
                    _log.info("rollout_pool.skip_existing", instance=inst.instance_id, seed=seed)
                else:
                    pending_tasks.append((inst, seed))
    except Exception as exc:
        # The pool remains usable in explicitly local/file-backed mode, but it
        # is honest that database idempotency was unavailable.
        _log.warning("rollout_pool.idempotency_unavailable", error=str(exc))

    in_flight: dict[Any, tuple[SWEBenchInstance, int, int]] = {}
    results: list[dict[str, Any]] = []
    
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
            in_flight[future] = (inst, seed, 0)
            task_idx += 1

        if not in_flight:
            break

        # Wait for at least one task to complete
        ready, _ = ray.wait(list(in_flight.keys()), num_returns=1, timeout=1.0)
        
        for done_ref in ready:
            inst, seed, attempts = in_flight.pop(done_ref)
            try:
                res = ray.get(done_ref)
                results.append(res)
            except ray.exceptions.TaskCancelledError:
                _log.warning("rollout_pool.task_cancelled", instance=inst.instance_id, seed=seed)
            except Exception as e:
                if attempts < pool.max_retries:
                    actor = actors[task_idx % len(actors)]
                    retry = actor.run.remote(
                        inst.model_dump(), seed, react_kwargs, str(output_dir), None,
                        apply_gold_patch, checkpoint_sha, cost_meter,
                    )
                    in_flight[retry] = (inst, seed, attempts + 1)
                    _log.warning(
                        "rollout_pool.retrying", instance=inst.instance_id, seed=seed,
                        attempt=attempts + 1, error=str(e),
                    )
                else:
                    _log.error(
                        "rollout_pool.task_failed", instance=inst.instance_id, seed=seed, error=str(e)
                    )
                    results.append({
                        "instance_id": inst.instance_id, "reward": 0.0, "resolved": False,
                        "error": str(e), "attempts": attempts + 1,
                    })
                
    _log.info("rollout_pool.done", n=len(results), workers=pool.n_workers)
    return results
