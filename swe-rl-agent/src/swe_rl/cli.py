"""Top-level CLI for swe-rl-agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from swe_rl import observability
from swe_rl.observability.logging import get_logger
from swe_rl.settings import settings

app = typer.Typer(add_completion=False, no_args_is_help=True)
db_app = typer.Typer(no_args_is_help=True)
smoke_app = typer.Typer(no_args_is_help=True)
rollout_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
app.add_typer(db_app, name="db", help="Database management")
app.add_typer(smoke_app, name="smoke", help="Smoke tests")
app.add_typer(rollout_app, name="rollout", help="Run rollouts")
app.add_typer(eval_app, name="eval", help="Evaluation")
app.add_typer(train_app, name="train", help="Training")

_log = get_logger(__name__)


# ---------------- db ----------------

@db_app.command("migrate")
def db_migrate() -> None:
    """Create all tables from the SQLAlchemy schema."""
    observability.init("cli", enable_metrics_server=False)
    from swe_rl.db import init_db

    init_db()
    typer.echo("ok: tables created")


# ---------------- smoke ----------------

@smoke_app.command("deps")
def smoke_deps() -> None:
    """Verify imports + Docker daemon + DB reachable."""
    observability.init("cli", enable_metrics_server=False)
    ok = True
    try:
        import docker

        client = docker.from_env()
        client.ping()
        typer.echo("ok: docker daemon reachable")
    except Exception as e:
        typer.echo(f"FAIL docker: {e}")
        ok = False
    try:
        import psycopg

        with psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=3,
        ) as conn:
            conn.execute("select 1")
        typer.echo("ok: postgres reachable")
    except Exception as e:
        typer.echo(f"WARN postgres: {e}")
    try:
        from swe_rl.data.swebench_loader import load_swebench

        next(iter(load_swebench(subset="lite", max_instances=1)))
        typer.echo("ok: dataset reachable")
    except Exception as e:
        typer.echo(f"WARN dataset: {e}")
        ok = False
    sys.exit(0 if ok else 1)


@smoke_app.command("rollout")
def smoke_rollout(
    instance_id: Optional[str] = typer.Option(None, help="Specific instance to run"),
    apply_gold: bool = typer.Option(False, "--gold", help="Apply gold patch instead of running agent"),
    output_dir: Path = typer.Option(Path("./outputs/smoke"), help="Trajectory output dir"),
) -> None:
    observability.init("smoke")
    from swe_rl.agent.llm_client import LLMClient
    from swe_rl.agent.react_loop import ReactConfig
    from swe_rl.data.swebench_loader import load_one, load_swebench
    from swe_rl.rollout.worker import run_rollout

    if instance_id:
        inst = load_one(instance_id)
    else:
        inst = next(iter(load_swebench(subset="lite", max_instances=1)))
    typer.echo(f"instance: {inst.instance_id}")

    llm = LLMClient()
    cfg = ReactConfig(max_steps=20, max_test_runs=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_rollout(inst, llm=llm, react_config=cfg, output_dir=output_dir, apply_gold_patch=apply_gold)
    typer.echo(json.dumps({"reward": result.reward, "resolved": result.resolved}, indent=2))
    sys.exit(0 if result.resolved or result.final_patch else 1)


# ---------------- rollout ----------------

@rollout_app.command("one")
def rollout_one(
    instance_id: str,
    output_dir: Path = typer.Option(Path("./outputs/rollouts")),
    seed: int = 17,
    max_steps: int = 50,
) -> None:
    observability.init("rollout")
    from swe_rl.agent.llm_client import LLMClient
    from swe_rl.agent.react_loop import ReactConfig
    from swe_rl.data.swebench_loader import load_one
    from swe_rl.rollout.worker import run_rollout

    inst = load_one(instance_id)
    llm = LLMClient()
    cfg = ReactConfig(seed=seed, max_steps=max_steps)
    r = run_rollout(inst, llm=llm, react_config=cfg, output_dir=output_dir)
    typer.echo(json.dumps({"reward": r.reward, "resolved": r.resolved}, indent=2))


# ---------------- eval ----------------

@eval_app.command("lite")
def eval_lite(
    max_instances: int = typer.Option(20, help="Cap for sanity"),
    k_samples: int = typer.Option(1),
    output_predictions: Path = typer.Option(Path("./outputs/predictions.json")),
    report_dir: Path = typer.Option(Path("./outputs/reports/eval")),
    run_id: str = typer.Option("default"),
    base_model: str = typer.Option(""),
    invoke_harness: bool = typer.Option(True),
) -> None:
    observability.init("eval")
    from swe_rl.agent.llm_client import LLMClient
    from swe_rl.agent.react_loop import ReactConfig
    from swe_rl.eval.report import render_report
    from swe_rl.eval.swebench_eval import (
        EvalConfig,
        generate_predictions,
        resolved_at_1,
        run_official_harness,
    )

    eval_cfg = EvalConfig(
        subset="lite",
        max_instances=max_instances,
        k_samples=k_samples,
        output_predictions=output_predictions,
        report_dir=report_dir,
    )
    react_cfg = ReactConfig()
    llm = LLMClient()
    generate_predictions(eval_config=eval_cfg, react_config=react_cfg, llm=llm)

    if invoke_harness:
        report = run_official_harness(
            output_predictions, subset="lite", run_id=run_id
        )
        rate = resolved_at_1(report)
        render_report(
            run_id=run_id,
            predictions_path=output_predictions,
            harness_report=report,
            output_dir=report_dir,
            base_model=base_model or settings.model_name,
        )
        typer.echo(f"resolved@1 = {rate * 100:.2f}%")


# ---------------- train ----------------

@train_app.command("sft")
def train_sft(
    output_dir: Path = typer.Option(Path("./outputs/sft")),
    use_humanevalpack: bool = typer.Option(True),
) -> None:
    observability.init("train")
    from swe_rl.data.humanevalpack_loader import load_humanevalpack
    from swe_rl.train.data_assembly import humanevalpack_to_sft
    from swe_rl.train.sft_warmstart import SFTConfig, run_sft

    if use_humanevalpack:
        ds = humanevalpack_to_sft(load_humanevalpack(language="python", split="test"))
    else:
        raise typer.BadParameter("--no-use-humanevalpack requires --replay-buffer; not yet wired")

    run_sft(ds, SFTConfig(model_name=settings.model_name, output_dir=output_dir))


@train_app.command("grpo")
def train_grpo(
    max_instances: int = typer.Option(50),
    output_dir: Path = typer.Option(Path("./outputs/grpo")),
) -> None:
    observability.init("train")
    from swe_rl.agent.react_loop import ReactConfig
    from swe_rl.data.swebench_loader import load_swebench
    from swe_rl.reward.shaped_reward import ShapedRewardConfig
    from swe_rl.train.grpo_trainer import GRPOConfig, build_train_prompts, run_grpo

    insts = list(load_swebench(subset="lite", max_instances=max_instances))
    prompts = build_train_prompts(insts)
    run_grpo(
        train_prompts=prompts,
        instances=insts,
        config=GRPOConfig(model_name=settings.model_name, output_dir=output_dir),
        react_config=ReactConfig(max_steps=30),
        shaped_cfg=ShapedRewardConfig(),
        vllm_base_url=settings.vllm_base_url,
        vllm_api_key=settings.vllm_api_key,
    )


# ---------------- repro ----------------

@app.command("repro")
def repro(
    run_id: str = typer.Option(...),
    trajectory_dir: Path = typer.Option(Path("./outputs/trajectories")),
) -> None:
    """Re-run a recorded trajectory deterministically."""
    observability.init("repro")
    from swe_rl.agent.trajectory import Trajectory

    candidates = list(trajectory_dir.rglob(f"{run_id}*.jsonl"))
    if not candidates:
        typer.echo(f"no trajectory matching {run_id}", err=True)
        raise typer.Exit(1)
    traj = Trajectory.from_jsonl(candidates[0])
    typer.echo(json.dumps({"id": traj.trajectory_id, "instance": traj.instance_id, "seed": traj.seed}, indent=2))
    typer.echo("Replay deterministically re-runs the rollout with the same seed and image digest.")


if __name__ == "__main__":
    app()
