from pathlib import Path
from types import SimpleNamespace

import pytest

from swe_rl.agent.react_loop import ReactConfig
from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.reward.shaped_reward import ShapedRewardConfig
from swe_rl.train.grpo_trainer import make_reward_fn


def test_reward_evaluates_the_matching_completion(monkeypatch, tmp_path: Path):
    instance = SWEBenchInstance(
        instance_id="repo__one-1",
        repo="repo/one",
        base_commit="abc",
        problem_statement="fix",
        fail_to_pass=["test_one"],
    )
    seen = []

    def fake_rollout(inst, **kwargs):
        seen.append((inst.instance_id, kwargs["candidate_patch"]))
        return SimpleNamespace(reward=1.0)

    monkeypatch.setattr("swe_rl.rollout.worker.run_rollout", fake_rollout)
    reward = make_reward_fn(
        {instance.instance_id: instance},
        react_config=ReactConfig(),
        output_dir=tmp_path,
        shaped_cfg=ShapedRewardConfig(),
        vllm_base_url="http://127.0.0.1:1",
        vllm_api_key="unused",
    )
    assert reward(
        ['{"instance_id":"repo__one-1"}'],
        ["diff --git a/a b/a"],
    ) == [1.0]
    assert seen == [("repo__one-1", "diff --git a/a b/a")]


def test_reward_blends_shaped_and_decays_from_trainer_state(monkeypatch, tmp_path):
    instance = SWEBenchInstance(
        instance_id="repo__one-1", repo="repo/one", base_commit="abc", problem_statement="fix"
    )
    monkeypatch.setattr(
        "swe_rl.rollout.worker.run_rollout",
        lambda *a, **k: SimpleNamespace(reward=1.0, details={"shaped": {"value": 0.5}}),
    )
    reward = make_reward_fn(
        {instance.instance_id: instance},
        react_config=ReactConfig(),
        output_dir=tmp_path,
        shaped_cfg=ShapedRewardConfig(),
        vllm_base_url="http://127.0.0.1:1",
        vllm_api_key="unused",
        shaped_weight=0.2,
        shaped_decay_steps=100,
    )
    reward.trainer_state = SimpleNamespace(global_step=50)
    assert reward(["{}"], ["patch"], instance_id=[instance.instance_id]) == [1.05]
    reward.trainer_state.global_step = 100
    assert reward(["{}"], ["patch"], instance_id=[instance.instance_id]) == [1.0]


def test_reward_uses_configured_execution_and_process_weights(monkeypatch, tmp_path):
    instance = SWEBenchInstance(
        instance_id="repo__one-1", repo="repo/one", base_commit="abc", problem_statement="fix"
    )
    monkeypatch.setattr(
        "swe_rl.rollout.worker.run_rollout",
        lambda *a, **k: SimpleNamespace(
            reward=99,
            details={"exec": {"value": 1}, "shaped": {"value": 0.5}},
            trajectory=SimpleNamespace(metadata={"prm": {"score": 0.4}}),
        ),
    )
    reward = make_reward_fn(
        {instance.instance_id: instance},
        react_config=ReactConfig(),
        output_dir=tmp_path,
        shaped_cfg=ShapedRewardConfig(),
        vllm_base_url="http://127.0.0.1:1",
        vllm_api_key="unused",
        binary_weight=2,
        shaped_weight=0.2,
        prm_weight=0.5,
    )
    assert reward(["{}"], ["patch"], instance_id=[instance.instance_id]) == pytest.approx([2.3])
