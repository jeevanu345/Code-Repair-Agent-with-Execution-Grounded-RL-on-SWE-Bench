import json

from fastapi.testclient import TestClient

from swe_rl.dashboard.app import app
from swe_rl.reward.exec_reward import compute_exec_reward
from swe_rl.reward.shaped_reward import ShapedRewardConfig
from swe_rl.sandbox.test_executor import TestResults
from swe_rl.settings import settings


def test_timeout_cannot_resolve_passing_results():
    result = compute_exec_reward(
        TestResults(outcomes={"a": "passed"}, timed_out=True), TestResults(), ["a"], []
    )
    assert not result.resolved
    assert result.value == 0


def test_nested_prm_configuration():
    config = ShapedRewardConfig(prm={"enabled": False})
    assert config.prm.enabled is False


def test_dashboard_discovers_rollouts_and_ignores_invalid_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "train_output_dir", tmp_path)
    folder = tmp_path / "rollouts"
    folder.mkdir()
    (folder / "run.jsonl").write_text(json.dumps({"trajectory_id": "example"}))
    (folder / "invalid.jsonl").write_text("[]")
    (folder / "broken.jsonl").write_text("{")
    result = TestClient(app).get("/api/runs")
    assert result.status_code == 200
    assert [row["run_id"] for row in result.json()] == ["example"]
