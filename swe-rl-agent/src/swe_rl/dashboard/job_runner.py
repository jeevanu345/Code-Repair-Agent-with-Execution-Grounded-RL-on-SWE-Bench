"""Isolated job entry point: invokes real rollout, harness and trainers."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

from swe_rl.dashboard.jobs import JobStore


def execute(store: JobStore, job_id: str):
    job = store.get(job_id)
    cfg = job["config"]
    output = store.directory(job_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    from swe_rl.observability import init
    from swe_rl.settings import settings

    init("control-worker", enable_metrics_server=False)
    sandbox = cfg.get("sandbox", {})
    settings.sandbox_cpus = sandbox.get("cpus", settings.sandbox_cpus)
    settings.sandbox_mem_gb = sandbox.get("memory", settings.sandbox_mem_gb)
    settings.sandbox_wallclock_s = sandbox.get("timeout", settings.sandbox_wallclock_s)
    settings.sandbox_disable_network_after_install = sandbox.get("network_disabled", True)
    settings.dollar_per_1k_tokens_in = cfg.get(
        "input_price_per_1k", settings.dollar_per_1k_tokens_in
    )
    settings.dollar_per_1k_tokens_out = cfg.get(
        "output_price_per_1k", settings.dollar_per_1k_tokens_out
    )
    settings.max_dollars_per_run = cfg.get("max_cost", settings.max_dollars_per_run)
    if cfg.get("model"):
        settings.model_name = cfg["model"]

    def control():
        return store.get(job_id).get("command")

    def event(data):
        store.event(job_id, data)
        if data.get("type") == "step":
            store.update(
                job_id,
                steps=data["step"],
                tokens_in=data.get("tokens_in", 0),
                tokens_out=data.get("tokens_out", 0),
                cost_usd=data.get("cost_usd"),
                test_budget_remaining=data.get("test_budget_remaining"),
            )
        state = data.get("status") or data.get("state") or data.get("type")
        if state in {"repository_prepared", "reasoning", "tool_completed"}:
            state = "running"
        if state in {"preparing", "running", "testing", "evaluating"}:
            store.update(job_id, status=state)

    if control() == "cancel":
        store.update(job_id, status="cancelled")
        return
    store.update(job_id, status="running")
    if job["kind"] == "evaluation" and cfg.get("predictions_path"):
        from swe_rl.eval.swebench_eval import run_official_harness

        store.update(job_id, status="evaluating")
        data = run_official_harness(
            Path(cfg["predictions_path"]),
            subset=cfg["dataset"],
            max_workers=cfg["parallelism"],
            run_id=job_id,
            instance_ids=cfg.get("instance_ids") or None,
        )
        (output / "evaluation.json").write_text(json.dumps(data, indent=2))
        store.update(
            job_id, result=data, status="cancelled" if control() == "cancel" else "completed"
        )
        return
    if job["kind"] in {"run", "evaluation"}:
        from swe_rl.agent.llm_client import LLMClient
        from swe_rl.agent.providers import active_connection, connection
        from swe_rl.agent.react_loop import ReactConfig
        from swe_rl.data.swebench_loader import load_one
        from swe_rl.rollout.worker import run_rollout

        selected = active_connection()
        if (cfg.get("provider") and cfg["provider"] != selected.provider) or (
            cfg.get("base_url") and cfg["base_url"] != selected.base_url
        ):
            raise ValueError("Save the selected provider connection before launching")
        client = LLMClient(
            config=connection(
                selected.provider,
                cfg.get("model") or selected.model,
                selected.api_key,
                cfg.get("base_url") or selected.base_url,
            )
        )
        react = ReactConfig(
            max_steps=cfg.get("max_steps", 30),
            max_test_runs=cfg.get("test_budget", 5),
            seed=cfg.get("seed", 17),
            temperature=cfg.get("temperature", 0.2),
            top_p=cfg.get("top_p", 0.95),
            profile=cfg.get("strategy", "standard").replace("_", "-"),
            prompt_content=cfg.get("prompt_content"),
            prompt_version=cfg.get("prompt_version"),
            max_cost_usd=cfg.get("max_cost"),
            minimize_patch=cfg.get("minimize_patch", False),
        )
        from swe_rl.reward.shaped_reward import ShapedRewardConfig

        reward = cfg.get("reward", {})
        from swe_rl.reward.prm import PRMConfig

        shaped = ShapedRewardConfig(
            prm=PRMConfig(enabled=bool(cfg.get("prm_model")), model_name=cfg.get("prm_model")),
            **{
                key: reward[key]
                for key in ["binary_weight", "shaped_weight", "prm_weight"]
                if key in reward
            },
        )
        try:
            if job["kind"] == "run":
                instance = load_one(
                    cfg["instance_id"],
                    subset=cfg["dataset"],
                    revision=cfg.get("dataset_revision", "main"),
                )
                result = run_rollout(
                    instance,
                    llm=client,
                    react_config=react,
                    output_dir=output,
                    shaped_cfg=shaped,
                    event_callback=event,
                    control_callback=control,
                )
                result.trajectory.metadata.update(
                    {"experiment_id": job["experiment_id"], "job_id": job_id, "config": cfg}
                )
                result.trajectory.write_jsonl(output)
                data = {
                    "trajectory_id": result.trajectory.trajectory_id,
                    "resolved": result.resolved,
                    "reward": result.reward,
                    "duration_s": result.duration_s,
                    "details": result.details,
                }
                store.update(
                    job_id, result=data, reward=result.reward, steps=result.trajectory.n_steps
                )
                if control() == "cancel":
                    store.update(job_id, status="cancelled")
                    return
                if result.trajectory.metadata.get("setup_failed") or result.trajectory.metadata.get(
                    "agent_failed"
                ):
                    store.update(
                        job_id, status="failed", error="Repair failed; inspect trajectory evidence"
                    )
                    return
            else:
                from swe_rl.eval.swebench_eval import (
                    EvalConfig,
                    generate_predictions,
                    run_official_harness,
                )

                predictions = output / "predictions.json"
                eval_cfg = EvalConfig(
                    subset=cfg["dataset"],
                    dataset_revision=cfg.get("dataset_revision", "main"),
                    max_instances=cfg["max_instances"],
                    k_samples=cfg["candidates"],
                    parallelism=cfg["parallelism"],
                    output_predictions=predictions,
                    report_dir=output,
                    model_name_for_predictions=client.model,
                )
                generate_predictions(
                    eval_config=eval_cfg,
                    react_config=react,
                    llm=client,
                    shaped_cfg=shaped,
                    event_callback=event,
                    control_callback=control,
                    instance_ids=cfg.get("instance_ids") or None,
                )
                if control() == "cancel":
                    store.update(job_id, status="cancelled")
                    return
                store.update(job_id, status="evaluating")
                data = (
                    run_official_harness(
                        predictions,
                        subset=cfg["dataset"],
                        max_workers=cfg["parallelism"],
                        run_id=job_id,
                    )
                    if cfg["invoke_harness"]
                    else {"status": "predictions_generated", "official_grading": False}
                )
                (output / "evaluation.json").write_text(json.dumps(data, indent=2))
                store.update(job_id, result=data)
        finally:
            client.close()
    else:
        if not cfg.get("allow_heavy_training") or not settings.swe_rl_allow_heavy_training:
            raise ValueError(
                "Heavy training requires both explicit submission opt-in and SWE_RL_ALLOW_HEAVY_TRAINING=true on the worker"
            )
        from datasets import load_dataset

        algorithm = cfg["algorithm"]
        if algorithm in {"sft", "dpo"}:
            if not cfg.get("dataset_path"):
                if algorithm == "dpo":
                    raise ValueError(
                        "DPO requires a JSONL dataset with prompt/chosen/rejected columns"
                    )
                from swe_rl.data.humanevalpack_loader import load_humanevalpack
                from swe_rl.train.data_assembly import humanevalpack_to_sft

                dataset = humanevalpack_to_sft(load_humanevalpack(language="python", split="test"))
            else:
                dataset = load_dataset("json", data_files=cfg["dataset_path"], split="train")
            if algorithm == "sft":
                from swe_rl.train.sft_warmstart import SFTConfig, run_sft

                path = run_sft(
                    dataset,
                    SFTConfig(
                        model_name=cfg["model"],
                        output_dir=output / "checkpoint",
                        learning_rate=cfg["learning_rate"],
                    ),
                    event_callback=event,
                    control_callback=control,
                )
            else:
                from swe_rl.train.dpo_trainer import DPOConfig, run_dpo

                path = run_dpo(
                    dataset,
                    DPOConfig(
                        model_name=cfg["model"],
                        output_dir=output / "checkpoint",
                        learning_rate=cfg["learning_rate"],
                    ),
                    event_callback=event,
                    control_callback=control,
                )
        else:
            from swe_rl.agent.react_loop import ReactConfig
            from swe_rl.data.swebench_loader import load_swebench
            from swe_rl.reward.prm import PRMConfig
            from swe_rl.reward.shaped_reward import ShapedRewardConfig
            from swe_rl.train.grpo_trainer import GRPOConfig, build_train_prompts, run_grpo

            instances = list(
                load_swebench(subset=cfg["dataset"], max_instances=cfg["max_instances"])
            )
            path = run_grpo(
                train_prompts=build_train_prompts(instances),
                instances=instances,
                config=GRPOConfig(
                    model_name=cfg["model"],
                    output_dir=output / "checkpoint",
                    learning_rate=cfg["learning_rate"],
                    group_size=cfg.get("group_size", 8),
                    binary_reward_weight=cfg.get("binary_weight", 1),
                    shaped_reward_weight=cfg.get("shaped_weight", 0),
                    prm_reward_weight=cfg.get("prm_weight", 0),
                ),
                react_config=ReactConfig(),
                shaped_cfg=ShapedRewardConfig(
                    binary_weight=cfg.get("binary_weight", 1),
                    shaped_weight=cfg.get("shaped_weight", 0),
                    prm_weight=cfg.get("prm_weight", 0),
                    prm=PRMConfig(
                        enabled=bool(cfg.get("prm_model")), model_name=cfg.get("prm_model")
                    ),
                ),
                vllm_base_url=settings.vllm_base_url,
                vllm_api_key=settings.vllm_api_key,
                event_callback=event,
                control_callback=control,
            )
        store.update(job_id, result={"checkpoint": str(path)})
    store.update(job_id, status="cancelled" if control() == "cancel" else "completed")


def main():
    store = JobStore(Path(sys.argv[1]))
    job_id = sys.argv[2]

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("Worker cancellation")

    signal.signal(signal.SIGTERM, terminate)
    try:
        execute(store, job_id)
    except BaseException as exc:
        store.event(job_id, {"type": "error", "error_type": type(exc).__name__})
        store.update(
            job_id,
            status="cancelled" if store.get(job_id).get("command") == "cancel" else "failed",
            error=f"{type(exc).__name__}: {str(exc)[:1200]}",
        )
        raise


if __name__ == "__main__":
    main()
