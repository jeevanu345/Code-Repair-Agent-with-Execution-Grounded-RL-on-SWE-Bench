"""Ephemeral Docker container manager for rollouts.

Public API:
    DockerRunner().start(workdir) -> ContainerHandle
    runner.exec(handle, cmd, ...) -> ExecResult
    runner.write_file(handle, path, content)
    runner.read_file(handle, path) -> bytes
    runner.commit_diff(handle) -> str  (unified diff against initial commit)
    runner.disable_network(handle)
    runner.teardown(handle)

Always wrap usage in try/finally; teardown is idempotent.
"""

from __future__ import annotations

import io
import shlex
import tarfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container

from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.sandbox.safety import check_command
from swe_rl.settings import settings

_log = get_logger(__name__)


@dataclass
class ContainerHandle:
    container_id: str
    image_digest: str
    started_at: float
    network_disabled: bool = False
    workdir: str = "/workspace"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    safety_blocked: bool = False
    blocked_reason: str | None = None


class DockerRunner:
    def __init__(self, client: docker.DockerClient | None = None) -> None:
        self.client = client or docker.from_env()
        self.image = settings.sandbox_image
        self._containers: dict[str, Container] = {}

    # --- lifecycle ---------------------------------------------------------

    def ensure_image(self) -> str:
        try:
            img = self.client.images.get(self.image)
        except ImageNotFound as exc:
            raise RuntimeError(
                f"Sandbox image {self.image!r} missing. Build with `make build-sandbox`."
            ) from exc
        digest = img.id or ""
        return digest

    def start(self, *, name: str | None = None, env: dict[str, str] | None = None) -> ContainerHandle:
        digest = self.ensure_image()
        cname = name or f"swe-rl-{uuid.uuid4().hex[:10]}"
        mem_bytes = settings.sandbox_mem_gb * 1024 * 1024 * 1024
        cpus = settings.sandbox_cpus
        nano_cpus = int(cpus * 1e9)
        try:
            container = self.client.containers.run(
                image=self.image,
                name=cname,
                command=["sleep", str(settings.sandbox_wallclock_s + 60)],
                detach=True,
                tty=False,
                stdin_open=False,
                user="agent",
                working_dir="/workspace",
                mem_limit=mem_bytes,
                memswap_limit=mem_bytes,
                nano_cpus=nano_cpus,
                pids_limit=512,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                read_only=False,
                tmpfs={"/tmp": "size=512m"},
                network_mode="bridge",  # install needs network; will detach later
                environment=env or {},
                labels={"project": "swe-rl-agent", "kind": "sandbox"},
            )
        except APIError as e:
            METRICS.container_failures_total.labels(kind="start").inc()
            raise RuntimeError(f"Failed to start container: {e}") from e
        self._containers[container.id] = container
        _log.info("sandbox.started", container_id=container.id, image=self.image, name=cname)
        return ContainerHandle(
            container_id=container.id,
            image_digest=digest,
            started_at=time.time(),
        )

    def disable_network(self, handle: ContainerHandle) -> None:
        if handle.network_disabled:
            return
        c = self._get(handle)
        try:
            for net in self.client.networks.list():
                try:
                    net.disconnect(c, force=True)
                except (APIError, NotFound):
                    pass
            handle.network_disabled = True
            _log.info("sandbox.network_disabled", container_id=handle.container_id)
        except APIError as e:
            _log.warning("sandbox.network_disable_failed", error=str(e))

    def teardown(self, handle: ContainerHandle) -> None:
        c = self._containers.pop(handle.container_id, None)
        if c is None:
            try:
                c = self.client.containers.get(handle.container_id)
            except NotFound:
                return
        try:
            c.kill()
        except (APIError, NotFound):
            pass
        try:
            c.remove(force=True)
        except (APIError, NotFound):
            pass
        _log.info("sandbox.teardown", container_id=handle.container_id)

    @contextmanager
    def session(self, **start_kwargs: Any) -> Iterator[ContainerHandle]:
        handle = self.start(**start_kwargs)
        try:
            yield handle
        finally:
            self.teardown(handle)

    # --- exec --------------------------------------------------------------

    def exec(
        self,
        handle: ContainerHandle,
        cmd: str | list[str],
        *,
        timeout: int | None = None,
        workdir: str | None = None,
        user: str = "agent",
        check_safety: bool = True,
    ) -> ExecResult:
        cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)
        if check_safety:
            verdict = check_command(cmd_str)
            if not verdict.allowed:
                _log.warning("sandbox.safety_blocked", reason=verdict.reason, cmd=cmd_str)
                return ExecResult(
                    exit_code=126,
                    stdout="",
                    stderr=f"BLOCKED: {verdict.reason}",
                    duration_s=0.0,
                    safety_blocked=True,
                    blocked_reason=verdict.reason,
                )

        c = self._get(handle)
        wrapped = ["bash", "-lc", cmd_str]
        timeout_s = timeout or settings.sandbox_wallclock_s

        start = time.time()
        try:
            api = self.client.api
            exec_id = api.exec_create(
                c.id,
                cmd=wrapped,
                user=user,
                workdir=workdir or handle.workdir,
                stdout=True,
                stderr=True,
                tty=False,
            )["Id"]

            output_chunks: list[bytes] = []
            stream = api.exec_start(exec_id, stream=True, demux=False)
            timed_out = False
            for chunk in stream:
                output_chunks.append(chunk)
                if time.time() - start > timeout_s:
                    timed_out = True
                    break

            inspect = api.exec_inspect(exec_id)
            exit_code = -1 if timed_out else int(inspect.get("ExitCode") or 0)
            output = b"".join(output_chunks).decode("utf-8", errors="replace")
            duration = time.time() - start

            return ExecResult(
                exit_code=exit_code,
                stdout=output,
                stderr="",
                duration_s=duration,
                timed_out=timed_out,
            )
        except APIError as e:
            METRICS.container_failures_total.labels(kind="exec").inc()
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"docker error: {e}",
                duration_s=time.time() - start,
            )

    # --- file IO -----------------------------------------------------------

    def write_file(self, handle: ContainerHandle, path: str, content: str | bytes) -> None:
        c = self._get(handle)
        if isinstance(content, str):
            content = content.encode("utf-8")
        target = Path(path)
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=target.name)
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))
        tar_buf.seek(0)
        c.put_archive(str(target.parent), tar_buf.getvalue())

    def read_file(self, handle: ContainerHandle, path: str) -> bytes:
        c = self._get(handle)
        bits, _ = c.get_archive(path)
        buf = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=buf, mode="r") as tar:
            members = tar.getmembers()
            if not members:
                raise FileNotFoundError(path)
            f = tar.extractfile(members[0])
            return f.read() if f else b""

    # --- diff --------------------------------------------------------------

    def commit_diff(self, handle: ContainerHandle, repo_dir: str = "/workspace/repo") -> str:
        result = self.exec(
            handle,
            f"git -C {shlex.quote(repo_dir)} add -A && "
            f"git -C {shlex.quote(repo_dir)} diff --no-color --staged --unified=3",
            check_safety=False,
        )
        return result.stdout

    # --- internal ----------------------------------------------------------

    def _get(self, handle: ContainerHandle) -> Container:
        c = self._containers.get(handle.container_id)
        if c is None:
            c = self.client.containers.get(handle.container_id)
            self._containers[handle.container_id] = c
        return c
