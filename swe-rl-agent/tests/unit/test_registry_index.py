import json

import pytest

from swe_rl.agent.prompt_registry import get_prompt, register_prompt
from swe_rl.dashboard.app import _load_run
from swe_rl.dashboard.run_index import indexed_runs
from swe_rl.settings import settings


def test_prompt_versions_are_immutable_and_fingerprinted(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "train_output_dir", tmp_path)
    row = register_prompt("repair", "v1", "Find the root cause.")
    assert get_prompt("repair@v1") == row
    assert len(row["sha256"]) == 64
    with pytest.raises(ValueError, match="already exists"):
        register_prompt("repair", "v1", "Changed content")


def test_incremental_index_updates_and_rejects_external_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(settings, "train_output_dir", root)
    path = root / "record.jsonl"
    path.write_text(json.dumps({"trajectory_id": "t1", "instance_id": "i1", "reward": 0}) + "\n")
    outside = tmp_path / "other.jsonl"
    outside.write_text(json.dumps({"trajectory_id": "private"}) + "\n")
    (root / "external.jsonl").symlink_to(outside)
    assert [row["run_id"] for row in indexed_runs(_load_run)] == ["t1"]
    path.write_text(json.dumps({"trajectory_id": "t2", "instance_id": "i1", "reward": 1}) + "\n")
    assert [row["run_id"] for row in indexed_runs(_load_run)] == ["t2"]
    path.unlink()
    assert indexed_runs(_load_run) == []
