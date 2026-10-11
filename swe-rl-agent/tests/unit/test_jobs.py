"""Queue persistence and lifecycle tests; never launch inference or training."""

from pathlib import Path

import pytest

from swe_rl.dashboard.jobs import JobManager, JobStore


def test_jobs_and_events_survive_store_restart(tmp_path: Path):
    store = JobStore(tmp_path)
    exp = store.experiment({"seed": 17, "model": "test"})
    job = store.create("run", {"instance_id": "test__repo-1"}, exp["id"])
    store.update(job["id"], status="preparing")
    reopened = JobStore(tmp_path)
    assert reopened.get(job["id"])["status"] == "preparing"
    events = reopened.events(job["id"])
    assert [event["status"] for event in events] == ["queued", "preparing"]
    assert reopened.events(job["id"], events[-1]["seq"]) == []
    assert reopened.experiments()[0]["fingerprint"] == exp["fingerprint"]


def test_terminal_state_cannot_be_overwritten(tmp_path: Path):
    store = JobStore(tmp_path)
    job = store.create("run", {}, "exp")
    store.update(job["id"], status="cancelled")
    store.update(job["id"], status="completed", result={"late": True})
    assert store.get(job["id"])["status"] == "cancelled"


def test_cancel_queued_retry_and_pause(tmp_path: Path):
    store = JobStore(tmp_path)
    manager = JobManager(store)
    job = store.create("run", {"seed": 3}, "exp")
    assert manager.action(job["id"], "pause")["paused"]
    assert not manager.action(job["id"], "resume")["paused"]
    assert manager.action(job["id"], "cancel")["status"] == "cancelled"
    retry = manager.action(job["id"], "retry")
    assert retry["id"] != job["id"]
    assert retry["config"] == job["config"]
    assert retry["status"] == "queued"
    with pytest.raises(ValueError):
        manager.action(retry["id"], "retry")


def test_job_directory_requires_existing_identity(tmp_path: Path):
    with pytest.raises(KeyError):
        JobStore(tmp_path).directory("../outside")


def test_atomic_claim_enforces_capacity_and_paused_queue(tmp_path):
    store = JobStore(tmp_path)
    first = store.create("run", {}, "exp")
    second = store.create("run", {}, "exp")
    assert store.claim(first["id"], 1)
    assert not JobStore(tmp_path).claim(first["id"], 2)
    assert not store.claim(second["id"], 1)
    store.update(first["id"], status="completed")
    store.update(second["id"], paused=True)
    assert not store.claim(second["id"], 1)
    store.update(second["id"], paused=False)
    assert store.claim(second["id"], 1)


def test_scheduler_priority_capacity_and_reaps_completed_process(tmp_path, monkeypatch):
    from swe_rl.dashboard import jobs

    store = JobStore(tmp_path)
    low = store.create("run", {"seed": 1}, "exp")
    high = store.create("run", {"seed": 2}, "exp")
    third = store.create("run", {"seed": 3}, "exp")
    store.update(high["id"], priority=10)
    done = store.create("run", {}, "exp")
    store.update(done["id"], status="completed")
    spawned = []

    class Process:
        def __init__(self, pid, finished=False):
            self.pid, self.finished, self.waited = pid, finished, False

        def poll(self):
            return 0 if self.finished else None

        def wait(self):
            self.waited = True

    def popen(command, **kwargs):
        spawned.append(command[-1])
        return Process(100 + len(spawned))

    monkeypatch.setattr(jobs.subprocess, "Popen", popen)
    manager = JobManager(store)
    completed_process = Process(99, finished=True)
    manager.processes[done["id"]] = completed_process

    class OneIteration:
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            return self.calls > 1

    manager.halt = OneIteration()
    manager._loop()
    assert completed_process.waited
    assert spawned == [high["id"], low["id"]]
    assert store.get(third["id"])["status"] == "queued"
