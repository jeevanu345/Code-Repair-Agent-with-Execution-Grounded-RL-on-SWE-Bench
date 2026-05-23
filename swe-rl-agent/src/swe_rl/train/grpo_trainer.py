"""GRPO trainer wrapping `trl.GRPOTrainer` with our execution-grounded reward.

The TRL trainer expects:
  - a base policy + a reference policy
  - a list of prompts
  - a callable `reward_funcs` that takes (prompts, completions, **kw) -> list[float]

Our reward is execution-grounded, so the function spins up a Docker sandbox per
completion and runs the project's tests. To stay within rollout-server budgets,
we keep group_size small (default 8) and run reward computation in parallel via
a Ray pool.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from swe_rl.agent.react_loop import ReactConfig
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.reward.shaped_reward import ShapedRewardConfig

_log = get_logger(__name__)


@dataclass
class GRPOConfig:
    model_name: str
    output_dir: Path
    group_size: int = 8
    kl_coef: float = 0.04
    clip_ratio: float = 0.2
    learning_rate: float = 1e-6
    lr_min: float = 1e-7
    global_batch_trajectories: int = 64
    gradient_accumulation_steps: int = 8
    per_device_batch_size: int = 1
    max_optimizer_steps: int = 5000
    warmup_steps: int = 50
    save_steps: int = 200
    seed: int = 17
    bf16: bool = True
    weight_sync_every_steps: int = 50
    extra: dict[str, Any] = field(default_factory=dict)


def make_reward_fn(
    instances_by_id: dict[str, SWEBenchInstance],
    *,
    react_config: ReactConfig,
    output_dir: Path,
    shaped_cfg: ShapedRewardConfig,
    vllm_base_url: str,
    vllm_api_key: str,
) -> Callable[..., list[float]]:
    """Build a TRL-compatible reward callable.

    Each prompt is keyed (in metadata) to a SWEBenchInstance; the reward function
    submits the K completions to the rollout pool and returns binary rewards.
    """
    from swe_rl.rollout.ray_pool import PoolConfig, submit_pool

    def reward(prompts: list[str], completions: list[str], **kwargs: Any) -> list[float]:
        instance_ids: list[str] = kwargs.get("instance_ids") or _infer_instance_ids(prompts)
        instances = [instances_by_id[i] for i in instance_ids]
        # Run rollouts in parallel; the *completions* serve as initial assistant
        # plans which the rollout worker may continue or honor as final patch.
        # In practice with GRPO we re-run the agent per completion seed.
        results = submit_pool(
            instances,
            pool=PoolConfig(n_workers=min(len(instances), 8)),
            react_config=react_config,
            output_dir=output_dir,
            model_name=react_config.__dict__.get("model_name", ""),
            vllm_base_url=vllm_base_url,
            vllm_api_key=vllm_api_key,
            seeds_per_instance=1,
        )
        return [float(r["reward"]) for r in results]

    return reward


def _infer_instance_ids(prompts: list[str]) -> list[str]:
    """Heuristic — extract instance ids from the prompt body. Used only when the
    caller didn't pass them explicitly via kwargs."""
    out: list[str] = []
    for p in prompts:
        if "instance_id=" in p:
            out.append(p.split("instance_id=", 1)[1].split()[0])
        else:
            out.append("__unknown__")
    return out


def run_grpo(
    *,
    train_prompts: list[dict[str, Any]],
    instances: list[SWEBenchInstance],
    config: GRPOConfig,
    react_config: ReactConfig,
    shaped_cfg: ShapedRewardConfig,
    vllm_base_url: str,
    vllm_api_key: str,
) -> Path:
    """Run GRPO training. `train_prompts` is a list of dicts with `prompt` + `instance_id`."""
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    tok = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype="bfloat16" if config.bf16 else "auto"
    )

    args = TRLGRPOConfig(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_steps=config.max_optimizer_steps,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        logging_steps=10,
        bf16=config.bf16,
        seed=config.seed,
        report_to=["wandb"],
        num_generations=config.group_size,
        beta=config.kl_coef,
    )

    instances_by_id = {i.instance_id: i for i in instances}
    reward_fn = make_reward_fn(
        instances_by_id,
        react_config=react_config,
        output_dir=config.output_dir / "rollouts",
        shaped_cfg=shaped_cfg,
        vllm_base_url=vllm_base_url,
        vllm_api_key=vllm_api_key,
    )

    ds = Dataset.from_list(train_prompts)

    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=[reward_fn],
        train_dataset=ds,
        tokenizer=tok,
    )
    t0 = time.time()
    trainer.train()
    METRICS.optimizer_steps_total.inc(config.max_optimizer_steps)
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    _log.info("grpo.saved", path=str(out), wall_s=time.time() - t0)
    return out


def build_train_prompts(instances: list[SWEBenchInstance]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for inst in instances:
        rows.append(
            {
                "prompt": json.dumps(
                    {
                        "instance_id": inst.instance_id,
                        "repo": inst.repo,
                        "problem_statement": inst.problem_statement[:4000],
                    }
                ),
                "instance_id": inst.instance_id,
            }
        )
    return rows
