"""Evaluate a policy on SWE-bench Lite/Verified and emit a `predictions.json`.

The official `swebench` package ships an evaluation harness that runs each
prediction's diff in its own Docker container against the gold tests. We:

1. Iterate the dataset, run our agent per instance (or load saved patches).
2. Build the leaderboard-format predictions file.
3. Optionally invoke the `swebench.harness.run_evaluation` entrypoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.react_loop import ReactConfig
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.data.swebench_loader import load_swebench
from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.reward.shaped_reward import ShapedRewardConfig
from swe_rl.rollout.worker import run_rollout

_log = get_logger(__name__)


@dataclass
class EvalConfig:
    subset: str = "lite"  # "lite" | "verified"
    split: str = "test"
    max_instances: int | None = None
    k_samples: int = 1
    parallelism: int = 4
    output_predictions: Path = Path("./outputs/predictions.json")
    report_dir: Path = Path("./outputs/reports/eval")
    apply_gold_patch: bool = False
    model_name_for_predictions: str = "swe-rl-agent"
    dataset_revision: str = "main"


@dataclass
class InstancePrediction:
    instance_id: str
    model_name_or_path: str
    model_patch: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "model_name_or_path": self.model_name_or_path,
            "model_patch": self.model_patch,
        }


def generate_predictions(
    *,
    eval_config: EvalConfig,
    react_config: ReactConfig,
    llm: LLMClient,
    shaped_cfg: ShapedRewardConfig | None = None,
    instances: Iterable[SWEBenchInstance] | None = None,
    instance_ids: list[str] | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    control_callback: Callable[[], str | None] | None = None,
) -> list[InstancePrediction]:
    insts = list(
        instances
        if instances is not None
        else load_swebench(
            subset=eval_config.subset,  # type: ignore[arg-type]
            split=eval_config.split,
            max_instances=eval_config.max_instances,
            revision=eval_config.dataset_revision,
        )
    )
    out_dir = eval_config.report_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    if eval_config.k_samples < 1 or eval_config.parallelism < 1:
        raise ValueError("Candidates and parallelism must be positive")
    if instance_ids:
        selected_ids = set(instance_ids)
        insts = [item for item in insts if item.instance_id in selected_ids]
        if selected_ids - {item.instance_id for item in insts}:
            raise ValueError("Some requested instances are absent from the selected dataset slice")

    def emit(data: dict[str, Any]) -> None:
        if event_callback:
            event_callback(data)

    def evaluate_one(
        index: int, inst: SWEBenchInstance
    ) -> tuple[int, InstancePrediction, list[dict[str, Any]]]:
        candidates: list[Any] = []
        for k in range(eval_config.k_samples):
            if control_callback and control_callback() == "cancel":
                raise RuntimeError("Evaluation cancelled")
            cfg = ReactConfig(
                **{**react_config.__dict__, "seed": react_config.seed + k * 1000 + index}
            )

            def progress(data: dict[str, Any], candidate: int = k) -> None:
                emit({**data, "instance_id": inst.instance_id, "candidate": candidate})

            result = run_rollout(
                inst,
                llm=llm,
                react_config=cfg,
                output_dir=out_dir,
                shaped_cfg=shaped_cfg,
                apply_gold_patch=eval_config.apply_gold_patch,
                event_callback=progress,
                control_callback=control_callback,
            )
            candidates.append(result)
        # k=1 is one independent candidate; k>1 is explicitly execution-selected,
        # not an official resolved@1 claim about the first candidate.
        best = max(
            candidates,
            key=lambda item: (item.resolved, bool(item.final_patch), -len(item.final_patch)),
        )
        summary = [
            {
                "instance_id": inst.instance_id,
                "trajectory_id": item.trajectory.trajectory_id,
                "resolved": item.resolved,
                "reward": item.reward,
                "candidate": k,
            }
            for k, item in enumerate(candidates)
        ]
        return (
            index,
            InstancePrediction(
                inst.instance_id, eval_config.model_name_for_predictions, best.final_patch
            ),
            summary,
        )

    ordered: dict[int, InstancePrediction] = {}
    candidate_records: list[dict[str, Any]] = []
    emit({"type": "evaluation_progress", "completed": 0, "total": len(insts), "status": "running"})
    with ThreadPoolExecutor(max_workers=eval_config.parallelism) as executor:
        futures = {
            executor.submit(evaluate_one, index, inst): inst for index, inst in enumerate(insts)
        }
        try:
            for future in as_completed(futures):
                index, prediction, records = future.result()
                ordered[index] = prediction
                candidate_records.extend(records)
                emit(
                    {
                        "type": "evaluation_progress",
                        "completed": len(ordered),
                        "total": len(insts),
                        "resolved_candidates": sum(item["resolved"] for item in candidate_records),
                    }
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    preds = [ordered[index] for index in sorted(ordered)]
    (eval_config.report_dir / "candidates.json").write_text(
        json.dumps(
            {
                "selection": "execution_score_then_minimal_patch",
                "samples_per_instance": eval_config.k_samples,
                "official_pass_at_1": eval_config.k_samples == 1,
                "candidates": candidate_records,
            },
            indent=2,
        )
    )
    eval_config.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    eval_config.output_predictions.write_text(
        json.dumps([p.to_dict() for p in preds], indent=2),
        encoding="utf-8",
    )
    _log.info(
        "eval.predictions_saved",
        path=str(eval_config.output_predictions),
        n=len(preds),
    )
    return preds


def run_official_harness(
    predictions_path: Path,
    *,
    subset: str = "lite",
    split: str = "test",
    max_workers: int = 4,
    run_id: str = "default",
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke the official swebench evaluation harness.

    Returns the parsed report dict on success.
    """
    try:
        from swebench.harness.run_evaluation import main as run_eval_main
    except ImportError as e:
        raise RuntimeError(
            "`swebench` is not installed. `pip install swebench` and try again."
        ) from e

    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
        "multilingual": "SWE-bench/SWE-bench_Multilingual",
        "multimodal": "SWE-bench/SWE-bench_Multimodal",
    }.get(subset)
    if not dataset_name:
        raise ValueError("Unsupported benchmark dataset")

    arguments = dict(
        dataset_name=dataset_name,
        split=split,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
        namespace=None,
    )
    # Installed harness releases expose different signatures. Keep supported
    # arguments only and let required parameters raise actionable errors.
    import inspect

    parameters = inspect.signature(run_eval_main).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        run_eval_main(**arguments)
    else:
        run_eval_main(**{key: value for key, value in arguments.items() if key in parameters})
    model_names = {
        row.get("model_name_or_path", "") for row in json.loads(predictions_path.read_text())
    }
    candidates = [Path(f"{model.replace('/', '__')}.{run_id}.json") for model in model_names]
    candidates.extend(
        [
            Path(f"{run_id}.{predictions_path.stem}.json"),
            Path("logs/evaluation") / run_id / "report.json",
        ]
    )
    for report_path in candidates:
        if report_path.is_file():
            parsed: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
            return {**parsed, "report_path": str(report_path), "official_grading": True}
    raise RuntimeError(
        "Harness exited without a discoverable report; inspect worker logs before claiming a benchmark score"
    )


def resolved_at_1(report: dict[str, Any]) -> float:
    """Pull resolved@1 out of the official harness report."""
    if "resolved_instances" in report and "submitted_instances" in report:
        resolved = report.get("resolved_instances", 0)
        submitted = report.get("submitted_instances", 0)
        n_resolved = float(len(resolved) if isinstance(resolved, list) else resolved)
        n_submitted = float(len(submitted) if isinstance(submitted, list) else submitted)
        rate = n_resolved / n_submitted if n_submitted else 0.0
        METRICS.eval_resolved_at_1.set(rate)
        return rate
    return 0.0
