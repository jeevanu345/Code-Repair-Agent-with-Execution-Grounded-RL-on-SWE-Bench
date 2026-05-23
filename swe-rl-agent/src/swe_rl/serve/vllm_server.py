"""Wrapper to launch vLLM's OpenAI-compatible server with our config.

We don't reimplement vLLM serving — we shell out so we can pin the exact CLI
args and re-launch on weight sync.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swe_rl.observability.logging import get_logger

_log = get_logger(__name__)


@dataclass
class VLLMServeConfig:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = "local-dev"
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    download_dir: Path | None = None
    trust_remote_code: bool = False


def build_command(cfg: VLLMServeConfig) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--api-key",
        cfg.api_key,
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(cfg.tensor_parallel_size),
    ]
    if cfg.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if cfg.download_dir:
        cmd += ["--download-dir", str(cfg.download_dir)]
    if cfg.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def serve(cfg: VLLMServeConfig) -> subprocess.Popen[bytes]:
    cmd = build_command(cfg)
    _log.info("vllm.serve", cmd=shlex.join(cmd))
    return subprocess.Popen(cmd)
