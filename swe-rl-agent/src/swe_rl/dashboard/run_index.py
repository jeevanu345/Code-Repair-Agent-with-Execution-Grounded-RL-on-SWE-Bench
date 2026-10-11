"""Incremental trajectory index with an optional PostgreSQL primary index.

JSONL remains a portable artifact. The dashboard queries indexed records after
incremental ingestion rather than reparsing every record on every request.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from swe_rl.settings import settings


def indexed_runs(loader: Callable[[Path], dict[str, Any] | None]) -> list[dict[str, Any]]:
    root = settings.train_output_dir.resolve()
    if not root.exists():
        return []
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(control / "trajectory-index.sqlite3", timeout=10) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS trajectories(path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER, data TEXT)"
        )
        previous = {
            path: (mtime, size)
            for path, mtime, size in db.execute("SELECT path,mtime_ns,size FROM trajectories")
        }
        seen = set()
        for path in root.rglob("*.jsonl"):
            if path.is_symlink() or root not in path.resolve().parents:
                continue
            name = str(path.relative_to(root))
            seen.add(name)
            try:
                info = path.stat()
                if previous.get(name) == (info.st_mtime_ns, info.st_size):
                    continue
                row = loader(path)
                if row is not None:
                    row["artifact_path"] = name
                db.execute(
                    "INSERT OR REPLACE INTO trajectories VALUES(?,?,?,?)",
                    (name, info.st_mtime_ns, info.st_size, json.dumps(row)),
                )
            except OSError:
                continue
        for name in set(previous) - seen:
            db.execute("DELETE FROM trajectories WHERE path=?", (name,))
        local = [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM trajectories ORDER BY mtime_ns DESC")
            if row[0] != "null"
        ]
    if settings.control_database_url:
        # Explicit opt-in: PostgreSQL holds indexed normalized evidence; artifacts remain local/object storage.
        from sqlalchemy import JSON, Column, MetaData, String, Table, create_engine, select

        engine = create_engine(
            settings.control_database_url,
            connect_args={"connect_timeout": 2}
            if settings.control_database_url.startswith("postgresql")
            else {},
        )
        try:
            metadata = MetaData()
            table = Table(
                "control_trajectory_index",
                metadata,
                Column("id", String, primary_key=True),
                Column("data", JSON, nullable=False),
            )
            metadata.create_all(engine)
            with engine.begin() as connection:
                existing = set(connection.execute(select(table.c.id)).scalars())
                for row in local:
                    identifier = row["run_id"]
                    if identifier in existing:
                        connection.execute(
                            table.update().where(table.c.id == identifier).values(data=row)
                        )
                    else:
                        connection.execute(table.insert().values(id=identifier, data=row))
                        existing.add(identifier)
                return list(connection.execute(select(table.c.data)).scalars())
        finally:
            engine.dispose()
    return local
