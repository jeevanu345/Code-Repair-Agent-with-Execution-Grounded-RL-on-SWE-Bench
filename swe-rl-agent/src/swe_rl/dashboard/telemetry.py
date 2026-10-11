"""Bounded service probes; report unavailable counters as null."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter

from swe_rl.agent.model_registry import request_telemetry
from swe_rl.agent.providers import active_connection
from swe_rl.settings import settings


def service_metrics() -> dict[str, Any]:
    data: dict[str, Any] = {
        "source": "measured_service_probes",
        "checked_at": time.time(),
        "postgresql": {"ready": False},
        "minio": {"ready": False},
        "docker": {"ready": False},
        "gpu": None,
        "model_requests": request_telemetry(),
    }
    started = time.monotonic()
    try:
        import psycopg

        with psycopg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            connect_timeout=2,
        ) as connection:
            connection.execute("SET statement_timeout = '2000ms'")
            tables = connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            ).fetchall()
            sessions = connection.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database()"
            ).fetchone()
            data["postgresql"] = {
                "ready": True,
                "latency_ms": (time.monotonic() - started) * 1000,
                "connections": sessions[0] if sessions else None,
                "schema_ready": {"instances", "trajectories", "rewards"}.issubset(
                    {row[0] for row in tables}
                ),
            }
    except Exception as exc:
        data["postgresql"]["error_type"] = type(exc).__name__
    try:
        import docker

        with docker.from_env(timeout=2) as client:
            containers = client.containers.list(filters={"label": "project=swe-rl-agent"})
            measured = []
            for container in containers[:20]:
                stats = container.stats(stream=False)
                cpu = stats.get("cpu_stats", {})
                previous = stats.get("precpu_stats", {})
                delta = cpu.get("cpu_usage", {}).get("total_usage", 0) - previous.get(
                    "cpu_usage", {}
                ).get("total_usage", 0)
                system_delta = cpu.get("system_cpu_usage", 0) - previous.get("system_cpu_usage", 0)
                cores = cpu.get("online_cpus", 1)
                measured.append(
                    {
                        "id": container.short_id,
                        "cpu_percent": delta / system_delta * cores * 100
                        if system_delta > 0
                        else None,
                        "memory_bytes": stats.get("memory_stats", {}).get("usage"),
                        "memory_limit_bytes": stats.get("memory_stats", {}).get("limit"),
                    }
                )
            data["docker"] = {
                "ready": True,
                "running_sandboxes": len(containers),
                "containers": measured,
            }
    except Exception as exc:
        data["docker"]["error_type"] = type(exc).__name__
    try:
        import urllib3
        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=False,
            http_client=urllib3.PoolManager(
                timeout=urllib3.Timeout(connect=1, read=2), retries=False
            ),
        )
        if client.bucket_exists(settings.minio_bucket):
            objects = []
            for item in client.list_objects(settings.minio_bucket, recursive=True):
                objects.append(item)
                if len(objects) >= 10000:
                    break
            data["minio"] = {
                "ready": True,
                "objects_sampled": len(objects),
                "storage_bytes_sampled": sum(item.size for item in objects),
                "truncated": len(objects) == 10000,
            }
    except Exception as exc:
        data["minio"]["error_type"] = type(exc).__name__
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            output = subprocess.check_output(
                [
                    executable,
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            data["gpu"] = [
                dict(
                    zip(
                        ("index", "utilization_percent", "memory_used_mib", "memory_total_mib"),
                        [float(value.strip()) for value in line.split(",")],
                        strict=True,
                    )
                )
                for line in output.splitlines()
            ]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        selected = active_connection()
        endpoint = urlsplit(selected.base_url)
        if selected.provider == "vllm" and endpoint.hostname in {"localhost", "127.0.0.1", "::1"}:
            with httpx.Client(timeout=2) as client:
                response = client.get(selected.base_url.removesuffix("/v1") + "/metrics")
                response.raise_for_status()
                data["vllm"] = {"metrics": response.text[:200000], "source": "vllm_prometheus"}
    except (RuntimeError, ValueError, httpx.HTTPError):
        data["vllm"] = {"metrics": None}
    return data


router = APIRouter()


@router.get("/api/infrastructure/metrics")
def metrics() -> dict[str, Any]:
    return service_metrics()
