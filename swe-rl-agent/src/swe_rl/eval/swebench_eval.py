"""Evaluate a policy on SWE-bench Lite/Verified and emit a `predictions.json`.

The official `swebench` package ships an evaluation harness that runs each
prediction's diff in its own Docker container against the gold tests. We:

1. Iterate the dataset, run our agent per instance (or load saved patches).
2. Build the leaderboard-format predictions file.
3. Optionally invoke the `swebench.harness.run_evaluation` entrypoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
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
) -> list[InstancePrediction]:
    insts = list(
        instances
        if instances is not None
        else load_swebench(
            subset=eval_config.subset,  # type: ignore[arg-type]
            split=eval_config.split,
            max_instances=eval_config.max_instances,
        )
    )
    out_dir = eval_config.report_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds: list[InstancePrediction] = []
    for i, inst in enumerate(insts):
        _log.info("eval.run", i=i, instance=inst.instance_id)
        # Per-instance: take the best of k_samples by the agent's own self-test
        # (we run reward downstream so the model picks the patch via majority later).
        best_patch = ""
        for k in range(eval_config.k_samples):
            cfg = ReactConfig(**{**react_config.__dict__, "seed": react_config.seed + k * 1000 + i})
            r = run_rollout(
                inst,
                llm=llm,
                react_config=cfg,
                output_dir=out_dir,
                shaped_cfg=shaped_cfg,
                apply_gold_patch=eval_config.apply_gold_patch,
            )
            if r.resolved or (best_patch == "" and r.final_patch):
                best_patch = r.final_patch
                if r.resolved:
                    break
        preds.append(
            InstancePrediction(
                instance_id=inst.instance_id,
                model_name_or_path=eval_config.model_name_for_predictions,
                model_patch=best_patch,
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
) -> dict[str, Any]:
    """Invoke the official swebench evaluation harness.

    Returns the parsed report dict on success.
    """
    try:
        from swebench.harness.run_evaluation import main as run_eval_main  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "`swebench` is not installed. `pip install swebench` and try again."
        ) from e

    dataset_name = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }[subset]

    run_eval_main(
        dataset_name=dataset_name,
        split=split,
        instance_ids=None,
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
    report_path = Path(f"{run_id}.{predictions_path.stem}.json")
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"status": "completed", "warning": "report file not found at default path"}


def resolved_at_1(report: dict[str, Any]) -> float:
    """Pull resolved@1 out of the official harness report."""
    if "resolved_instances" in report and "submitted_instances" in report:
        n_resolved = float(report.get("resolved_instances", 0))
        n_submitted = float(report.get("submitted_instances", 0))
        rate = n_resolved / n_submitted if n_submitted else 0.0
        METRICS.eval_resolved_at_1.set(rate)
        return rate
    return 0.0
