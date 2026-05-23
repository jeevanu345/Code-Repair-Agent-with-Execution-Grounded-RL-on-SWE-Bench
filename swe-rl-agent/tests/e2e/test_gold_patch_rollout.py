"""E2E: pick the first SWE-bench Lite instance, apply the gold patch,
run F2P + P2P, expect reward = 1.0.

Requires Docker, network access for the install phase, and the sandbox image.
"""

import pytest

pytestmark = pytest.mark.e2e


def _prereqs_or_skip():
    docker = pytest.importorskip("docker")
    pytest.importorskip("datasets")
    try:
        c = docker.from_env()
        c.ping()
        c.images.get("swe-rl/sandbox:latest")
    except Exception as e:
        pytest.skip(f"prereqs missing: {e}")


def test_gold_patch_resolves(tmp_path):
    _prereqs_or_skip()
    from swe_rl.agent.llm_client import LLMClient
    from swe_rl.agent.react_loop import ReactConfig
    from swe_rl.data.swebench_loader import load_swebench
    from swe_rl.rollout.worker import run_rollout

    inst = next(iter(load_swebench(subset="lite", max_instances=1)))
    llm = LLMClient()
    cfg = ReactConfig(max_steps=1)
    result = run_rollout(
        inst, llm=llm, react_config=cfg, output_dir=tmp_path, apply_gold_patch=True
    )
    assert result.resolved is True
    assert result.reward == 1.0
