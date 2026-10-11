"""Durable local job queue and event journal shared by API and isolated workers."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {"completed", "failed", "cancelled"}
STATES = {"queued", "preparing", "running", "testing", "evaluating"} | TERMINAL


def now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "control.sqlite3"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS events_job ON events(job_id, seq);
            """)

    def connect(self):
        db = sqlite3.connect(self.db, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def get(self, job_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def list(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT data FROM jobs").fetchall()
        return sorted(
            (json.loads(row[0]) for row in rows), key=lambda job: job["created_at"], reverse=True
        )

    def experiments(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT data FROM experiments")]

    def experiment(self, config: dict[str, Any]) -> dict[str, Any]:
        fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        exp = {
            "id": "exp-" + uuid.uuid4().hex[:12],
            "experiment_id": "",
            "fingerprint": fingerprint,
            "config": config,
            "created_at": now(),
        }
        exp["experiment_id"] = exp["id"]
        with self.connect() as db:
            db.execute("INSERT INTO experiments VALUES (?,?)", (exp["id"], json.dumps(exp)))
        return exp

    def create(self, kind: str, config: dict[str, Any], experiment_id: str) -> dict[str, Any]:
        job_id = "job-" + uuid.uuid4().hex
        job = {
            "id": job_id,
            "run_id": job_id,
            "kind": kind,
            "status": "queued",
            "config": config,
            "experiment_id": experiment_id,
            "created_at": now(),
            "updated_at": now(),
            "pid": None,
            "result": None,
            "error": None,
            "paused": False,
            "command": None,
            "instance_id": config.get("instance_id", ""),
            "model": config.get("model", ""),
            "reward": None,
            "steps": 0,
            "priority": 0,
        }
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?)", (job_id, json.dumps(job)))
        self.event(job_id, {"type": "state", "status": "queued"})
        return job

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        # A write transaction prevents API cancellation and worker updates losing each other.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            job = json.loads(row[0])
            if (
                job["status"] in TERMINAL
                and "status" in changes
                and changes["status"] != job["status"]
            ):
                changes.pop("status")
            if "status" in changes and changes["status"] not in STATES:
                raise ValueError("Invalid job state")
            job.update(changes, updated_at=now())
            db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(job), job_id))
        if "status" in changes:
            self.event(job_id, {"type": "state", "status": job["status"]})
        return job

    def claim(self, job_id: str, capacity: int) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return False
            job = json.loads(row[0])
            active = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE json_extract(data, '$.status') IN ('preparing','running','testing','evaluating')"
            ).fetchone()[0]
            if job["status"] != "queued" or job.get("paused") or active >= capacity:
                return False
            job.update(status="preparing", updated_at=now())
            db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(job), job_id))
        self.event(job_id, {"type": "state", "status": "preparing"})
        return True

    def event(self, job_id: str, data: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(job_id,data) VALUES (?,?)",
                (job_id, json.dumps({"created_at": now(), **data}, default=str)),
            )

    def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                {"seq": seq, **json.loads(data)}
                for seq, data in db.execute(
                    "SELECT seq,data FROM events WHERE job_id=? AND seq>? ORDER BY seq LIMIT 200",
                    (job_id, after),
                )
            ]

    def directory(self, job_id: str) -> Path:
        self.get(job_id)
        return self.root / job_id


class JobManager:
    """Bounded subprocess scheduler. Queued jobs survive API restarts."""

    def __init__(self, store: JobStore, workers: int = 2):
        self.store, self.workers = store, workers
        self.processes: dict[str, subprocess.Popen] = {}
        self.halt = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        # Workers can survive an API restart and continue writing their own events.
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.halt.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _loop(self):
        while not self.halt.wait(0.3):
            try:
                jobs = self.store.list()
                for job in jobs:
                    if (
                        job["status"] == "preparing"
                        and not job.get("pid")
                        and job["id"] not in self.processes
                        and (
                            datetime.now(UTC) - datetime.fromisoformat(job["updated_at"])
                        ).total_seconds()
                        > 10
                    ):
                        self.store.update(
                            job["id"],
                            status="failed",
                            error="Worker startup interrupted; retry the job",
                        )
                        continue
                    if job["status"] in TERMINAL:
                        process = self.processes.get(job["id"])
                        if process and process.poll() is not None:
                            process.wait()
                            self.processes.pop(job["id"], None)
                        continue
                    process = self.processes.get(job["id"])
                    if process and process.poll() is not None:
                        self.store.update(
                            job["id"],
                            status="failed",
                            error=f"Worker exited with code {process.returncode}",
                        )
                    elif job["status"] != "queued" and job.get("pid"):
                        try:
                            os.kill(job["pid"], 0)
                            if not self._worker_identity(job["pid"], job["id"]):
                                raise ProcessLookupError("Worker identity no longer matches")
                        except ProcessLookupError:
                            self.store.update(
                                job["id"],
                                status="failed",
                                error="Worker interrupted; retry the job",
                            )
                active = sum(
                    job["status"] not in TERMINAL | {"queued"} for job in self.store.list()
                )
                for job in sorted(
                    jobs, key=lambda item: (-item.get("priority", 0), item["created_at"])
                ):
                    if active >= self.workers:
                        break
                    if job["status"] != "queued" or job.get("paused"):
                        continue
                    folder = self.store.directory(job["id"])
                    folder.mkdir(parents=True, exist_ok=True)
                    if not self.store.claim(job["id"], self.workers):
                        continue
                    with (folder / "worker.log").open("ab") as log:
                        proc = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "swe_rl.dashboard.job_runner",
                                str(self.store.root),
                                job["id"],
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    self.processes[job["id"]] = proc
                    self.store.update(job["id"], pid=proc.pid)
                    active += 1
            except Exception:
                # An unavailable filesystem should not kill the queue thread.
                continue

    def action(self, job_id: str, action: str) -> dict[str, Any]:
        job = self.store.get(job_id)
        if action in {"retry", "requeue"}:
            if job["status"] not in TERMINAL:
                raise ValueError("Only finished jobs can be retried")
            return self.store.create(job["kind"], job["config"], job["experiment_id"])
        if job["status"] in TERMINAL:
            raise ValueError("Job is already finished")
        if action == "cancel":
            if job["status"] == "queued":
                return self.store.update(job_id, status="cancelled", command="cancel")
            self.store.update(job_id, command="cancel", paused=False)
            # Cooperatively clean up Docker first; hard termination bounds unresponsive calls.
            threading.Thread(
                target=self._terminate_later, args=(job_id, job.get("pid")), daemon=True
            ).start()
        elif action in {"pause", "resume"}:
            if job["status"] not in {"queued", "preparing", "running"}:
                raise ValueError("Pause is supported during agent execution only")
            self.store.update(
                job_id, paused=action == "pause", command="pause" if action == "pause" else None
            )
        elif action == "stop-evaluate":
            if job["kind"] != "run" or job["status"] not in {"preparing", "running"}:
                raise ValueError("Stop and evaluate requires an active repair run")
            self.store.update(job_id, command="stop_evaluate", paused=False)
        else:
            raise ValueError("Unknown action")
        self.store.event(job_id, {"type": "control", "action": action})
        return self.store.get(job_id)

    @staticmethod
    def _worker_identity(pid: int, job_id: str) -> bool:
        try:
            args = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "args="],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return "swe_rl.dashboard.job_runner" in args and job_id in args
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _terminate_later(self, job_id: str, pid: int | None):
        time.sleep(10)
        job = self.store.get(job_id)
        if job["status"] not in TERMINAL and job["command"] == "cancel":
            process = self.processes.get(job_id)
            owned = process is not None and process.pid == pid and process.poll() is None
            if pid and (owned or self._worker_identity(pid, job_id)):
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)
            self.store.update(
                job_id,
                status="cancelled",
                error="Cancellation requested; worker terminated after cleanup grace period",
            )
