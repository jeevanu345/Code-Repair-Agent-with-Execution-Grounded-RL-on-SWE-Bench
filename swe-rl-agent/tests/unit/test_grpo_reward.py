from pathlib import Path
from types import SimpleNamespace

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
