"""Sync trained weights from learner to vLLM rollout servers.

Strategy: every N optimizer steps, dump LoRA adapter (or full weights) to a
shared directory, then signal the vLLM server to reload via its admin endpoint.
For LoRA, we use vLLM's `--enable-lora` and the `/v1/load_lora_adapter` endpoint
where available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from swe_rl.observability.logging import get_logger

_log = get_logger(__name__)


@dataclass
class SyncConfig:
    vllm_base_url: str
    vllm_api_key: str
    adapter_name: str = "policy"
    timeout_s: float = 60.0


def sync_lora(adapter_path: Path, config: SyncConfig) -> bool:
    """Push a LoRA adapter to vLLM. Returns True on success."""
    url = f"{config.vllm_base_url.rstrip('/')}/v1/load_lora_adapter"
    payload = {
        "lora_name": config.adapter_name,
        "lora_path": str(adapter_path),
    }
    headers = {"Authorization": f"Bearer {config.vllm_api_key}"}
    t0 = time.time()
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=config.timeout_s)
        r.raise_for_status()
    except Exception as e:
        _log.error("learner_sync.failed", error=str(e), url=url)
        return False
    _log.info("learner_sync.done", path=str(adapter_path), wall_s=time.time() - t0)
    return True


def unload(config: SyncConfig) -> bool:
    url = f"{config.vllm_base_url.rstrip('/')}/v1/unload_lora_adapter"
    headers = {"Authorization": f"Bearer {config.vllm_api_key}"}
    try:
        httpx.post(url, json={"lora_name": config.adapter_name}, headers=headers, timeout=15.0)
    except Exception as e:
        _log.warning("learner_sync.unload_failed", error=str(e))
        return False
    return True
