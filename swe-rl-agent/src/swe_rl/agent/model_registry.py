"""User-declared model capabilities and measured local request telemetry."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from swe_rl.settings import settings

_lock = threading.RLock()
_telemetry: dict[str, dict[str, Any]] = {}


def record_request(
    provider: str,
    model: str,
    duration_s: float,
    *,
    status: int | None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    with _lock:
        key = f"{provider}/{model}"
        row = _telemetry.setdefault(
            key,
            {
                "provider": provider,
                "model": model,
                "requests": 0,
                "failures": 0,
                "total_duration_s": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )
        row["requests"] += 1
        row["failures"] += int(status is None or status >= 300)
        row["total_duration_s"] += duration_s
        row["latency_ms"] = duration_s * 1000
        row["average_latency_ms"] = row["total_duration_s"] / row["requests"] * 1000
        row["last_request_at"] = time.time()
        row["last_http_status"] = status
        row["rate_limited"] = status == 429
        row["tokens_in"] += tokens_in
        row["tokens_out"] += tokens_out


def request_telemetry() -> list[dict[str, Any]]:
    with _lock:
        return [dict(row) for row in _telemetry.values()]


def _path() -> Path:
    return settings.train_output_dir / "registry" / "models.json"


def models() -> list[dict[str, Any]]:
    with _lock:
        try:
            rows = json.loads(_path().read_text())
            return rows if isinstance(rows, list) else []
        except (OSError, ValueError):
            return []


def register_model(row: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        rows = [
            item
            for item in models()
            if (item["provider"], item["model"]) != (row["provider"], row["model"])
        ]
        row = dict(row, source="user_declared", updated_at=time.time())
        rows.append(row)
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(path)
        return row
