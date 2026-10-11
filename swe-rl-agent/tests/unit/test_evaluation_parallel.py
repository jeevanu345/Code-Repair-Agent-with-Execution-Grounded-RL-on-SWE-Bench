"""Validate scheduling/order/report semantics without Docker or model calls."""

import json
from types import SimpleNamespace

import pytest

from swe_rl.agent.react_loop import ReactConfig
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.eval.swebench_eval import EvalConfig, generate_predictions, resolved_at_1


def test_parallel_evaluation_preserves_order_and_independent_candidates(tmp_path, monkeypatch):
    import swe_rl.eval.swebench_eval as evaluation

    seen = []

    def rollout(instance, **kwargs):
        seed = kwargs["react_config"].seed
        seen.append((instance.instance_id, seed))
        return SimpleNamespace(
            resolved=seed % 2 == 0,
            final_patch=f"patch:{seed}",
            reward=float(seed % 2 == 0),
            trajectory=SimpleNamespace(trajectory_id=f"t:{seed}"),
        )

    monkeypatch.setattr(evaluation, "run_rollout", rollout)
    instances = [
        SWEBenchInstance(
            instance_id=f"repo-{i}", repo="a/b", base_commit="abc", problem_statement="fix"
        )
        for i in range(3)
    ]
    cfg = EvalConfig(
        k_samples=2,
        parallelism=2,
        report_dir=tmp_path,
        output_predictions=tmp_path / "predictions.json",
    )
    events = []
    results = generate_predictions(
        eval_config=cfg,
        react_config=ReactConfig(),
        llm=SimpleNamespace(),
        instances=instances,
        event_callback=events.append,
    )
    assert [result.instance_id for result in results] == [item.instance_id for item in instances]
    assert len(seen) == 6
    assert len(set(seen)) == 6
    assert events[-1]["completed"] == 3
    assert not json.loads((tmp_path / "candidates.json").read_text())["official_pass_at_1"]


def test_cancel_does_not_generate_predictions(tmp_path):
    instances = [
        SWEBenchInstance(
            instance_id="repo-1", repo="a/b", base_commit="abc", problem_statement="fix"
        )
    ]
    cfg = EvalConfig(report_dir=tmp_path, output_predictions=tmp_path / "predictions.json")
    with pytest.raises(RuntimeError, match="cancelled"):
        generate_predictions(
            eval_config=cfg,
            react_config=ReactConfig(),
            llm=SimpleNamespace(),
            instances=instances,
            control_callback=lambda: "cancel",
        )
    assert not cfg.output_predictions.exists()


def test_harness_list_or_count_resolution():
    assert resolved_at_1({"resolved_instances": ["a"], "submitted_instances": ["a", "b"]}) == 0.5
    assert resolved_at_1({"resolved_instances": 1, "submitted_instances": 2}) == 0.5
