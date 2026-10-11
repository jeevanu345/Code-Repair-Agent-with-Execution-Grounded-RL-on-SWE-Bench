import hashlib

from fastapi.testclient import TestClient

from swe_rl.dashboard.app import app
from swe_rl.dashboard.jobs import JobManager
from swe_rl.settings import settings


def test_control_requires_token_and_persists_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "train_output_dir", tmp_path)
    monkeypatch.setattr(JobManager, "start", lambda self: None)
    if hasattr(app.state, "job_manager"):
        monkeypatch.delattr(app.state, "job_manager")
    with TestClient(app) as client:
        assert client.post("/api/runs", json={"instance_id": "test__repo-1"}).status_code == 403
        token = client.get("/api/providers").json()["settings_token"]
        headers = {"x-settings-token": token}
        response = client.post(
            "/api/runs", headers=headers, json={"instance_id": "test__repo-1", "candidates": 2}
        )
        assert response.status_code == 200
        body = response.json()
        assert len(body["runs"]) == 2
        assert len(body["fingerprint"]) == 64
        job_id = body["runs"][0]["id"]
        assert client.get("/api/runs/" + job_id).json()["status"] == "queued"
        assert (
            client.post("/api/runs/" + job_id + "/cancel", headers=headers).json()["status"]
            == "cancelled"
        )
        assert (
            client.post("/api/runs/" + job_id + "/retry", headers=headers).json()["status"]
            == "queued"
        )
        assert (
            client.post(
                "/api/runs/" + job_id + "/annotation",
                headers=headers,
                json={"decision": "approved", "notes": "reviewed"},
            ).status_code
            == 200
        )
        assert client.get("/api/runs/" + job_id + "/annotation").json()["decision"] == "approved"
        assert client.get("/api/experiments").json()[0]["experiment_id"] == body["experiment_id"]
        assert (
            client.post(
                "/api/training/jobs", headers=headers, json={"algorithm": "grpo", "model": "test"}
            ).status_code
            == 400
        )
        assert client.get("/api/runs/job-missing").status_code == 404


def test_artifacts_reject_external_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    root.mkdir()
    monkeypatch.setattr(settings, "train_output_dir", root)
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    (root / "leak.txt").symlink_to(outside)
    (root / "report.txt").write_text("report")
    client = TestClient(app)
    rows = client.get("/api/artifacts").json()
    assert [row["name"] for row in rows] == ["report.txt"]
    assert client.get("/api/artifacts/" + rows[0]["id"] + "/content").text == "report"
    assert client.get("/api/artifacts/../secret/content").status_code != 200


def test_artifact_preview_blocks_guessed_secret_and_database_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "train_output_dir", tmp_path)
    for name in [".env", "prompts.sqlite3", "secret.key", "credentials.json"]:
        (tmp_path / name).write_text("private")
    client = TestClient(app)
    assert client.get("/api/artifacts").json() == []
    for name in [".env", "prompts.sqlite3", "secret.key", "credentials.json"]:
        artifact_id = hashlib.sha256(name.encode()).hexdigest()[:24]
        assert client.get("/api/artifacts/" + artifact_id + "/content").status_code == 404


def test_candidate_selection_uses_recorded_execution_and_patch_size(tmp_path, monkeypatch):
    import json

    from swe_rl.dashboard.jobs import JobStore

    monkeypatch.setattr(settings, "train_output_dir", tmp_path)
    store = JobStore(tmp_path / "control")
    experiment = store.experiment({"selection": "minimal-patch"})
    identities = []
    for count in [3, 1]:
        job = store.create("run", {"instance_id": "test__repo-1"}, experiment["id"])
        store.update(job["id"], status="completed", result={"resolved": True})
        folder = store.directory(job["id"])
        folder.mkdir()
        (folder / "trajectory.jsonl").write_text(
            json.dumps(
                {
                    "trajectory_id": "trajectory-" + job["id"],
                    "instance_id": "test__repo-1",
                    "finished_at": "2026-10-02",
                    "final_patch": "+line\n" * count,
                    "reward_details": {
                        "exec": {"resolved": True, "fail_to_pass_pass_rate": 1.0, "p2p_failed": []}
                    },
                }
            )
            + "\n"
        )
        identities.append(job["id"])
    client = TestClient(app)
    response = client.get("/api/experiments/" + experiment["id"] + "/selection")
    assert response.status_code == 200
    assert response.json()["selections"][0]["selected_run_id"] == identities[1]
    token = client.get("/api/providers").json()["settings_token"]
    assert (
        client.post(
            "/api/experiments/" + experiment["id"] + "/selection",
            headers={"x-settings-token": token},
            json={"strategy": "critic-ranking"},
        ).status_code
        == 400
    )
