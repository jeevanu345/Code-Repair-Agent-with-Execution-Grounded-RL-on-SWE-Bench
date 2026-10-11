"""Immutable custom prompt versions with content fingerprints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from swe_rl.agent.profiles import PROFILES
from swe_rl.settings import settings


def _connect() -> sqlite3.Connection:
    root = settings.train_output_dir / "registry"
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "prompts.sqlite3", timeout=10)
    db.execute("CREATE TABLE IF NOT EXISTS prompts(id TEXT PRIMARY KEY,data TEXT NOT NULL)")
    return db


def list_prompts() -> list[dict[str, Any]]:
    builtin = [
        {
            "id": f"{profile.name}@{profile.version}",
            "name": profile.name,
            "version": profile.version,
            "content": profile.instructions,
            "sha256": hashlib.sha256(profile.instructions.encode()).hexdigest(),
            "source": "builtin",
        }
        for profile in PROFILES.values()
    ]
    with _connect() as db:
        return builtin + [json.loads(row[0]) for row in db.execute("SELECT data FROM prompts")]


def register_prompt(name: str, version: str, content: str) -> dict[str, Any]:
    identifier = f"{name}@{version}"
    if identifier in {item["id"] for item in list_prompts()}:
        raise ValueError("Prompt version already exists; use a new version")
    import time

    row = {
        "id": identifier,
        "name": name,
        "version": version,
        "content": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source": "custom",
        "created_at": time.time(),
    }
    with _connect() as db:
        try:
            db.execute("INSERT INTO prompts VALUES(?,?)", (identifier, json.dumps(row)))
        except sqlite3.IntegrityError:
            raise ValueError("Prompt version already exists") from None
    return row


def get_prompt(identifier: str) -> dict[str, Any]:
    for row in list_prompts():
        if row["id"] == identifier:
            return row
    raise ValueError("Unknown prompt version")
