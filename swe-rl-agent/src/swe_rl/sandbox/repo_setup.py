"""Clone the target repo at the buggy commit and run instance-specific install.

We rely on the official `swebench` package's per-repo specs when available, and
fall back to a generic `pip install -e .` install. Network is enabled during
install and disabled afterward (controlled by sandbox config).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any

from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.sandbox.docker_runner import ContainerHandle, DockerRunner, ExecResult
from swe_rl.settings import settings

_log = get_logger(__name__)

REPO_DIR = "/workspace/repo"


@dataclass
class RepoSetupResult:
    ok: bool
    install_log: str
    duration_s: float


def _swebench_specs(repo: str, version: str | None) -> dict[str, Any] | None:
    """Look up SWE-bench install commands from the official package; None on miss."""
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except Exception:
        return None
    repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo)
    if not repo_specs:
        return None
    if version and version in repo_specs:
        return repo_specs[version]  # type: ignore[no-any-return]
    return None


def setup_repo(
    runner: DockerRunner,
    handle: ContainerHandle,
    instance: SWEBenchInstance,
    *,
    apply_test_patch: bool = True,
) -> RepoSetupResult:
    """Clone, checkout, install. Records logs to the trajectory layer upstream."""
    import time

    t0 = time.time()
    logs: list[str] = []

    def run(cmd: str, *, timeout: int = 600) -> ExecResult:
        r = runner.exec(handle, cmd, timeout=timeout, check_safety=False)
        logs.append(f"$ {cmd}\n[exit={r.exit_code} t={r.duration_s:.1f}s]\n{r.stdout}\n")
        return r

    repo_url = f"https://github.com/{instance.repo}.git"

    r = run(f"mkdir -p {shlex.quote(REPO_DIR)}")
    if r.exit_code != 0:
        return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    r = run(f"git clone --quiet {shlex.quote(repo_url)} {shlex.quote(REPO_DIR)}", timeout=900)
    if r.exit_code != 0:
        METRICS.container_failures_total.labels(kind="git_clone").inc()
        return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    r = run(
        f"git -C {shlex.quote(REPO_DIR)} fetch --quiet origin {shlex.quote(instance.base_commit)} && "
        f"git -C {shlex.quote(REPO_DIR)} checkout --quiet {shlex.quote(instance.base_commit)}"
    )
    if r.exit_code != 0:
        return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    # Install via SWE-bench specs if available.
    specs = _swebench_specs(instance.repo, instance.version)
    install_cmds: list[str] = []
    python_version = "3.11"
    
    if specs:
        python_version = str(specs.get("python", "3.11"))
        pip_packages = specs.get("pip_packages", [])
        if pip_packages:
            quoted = " ".join(shlex.quote(str(package)) for package in pip_packages)
            install_cmds.append(f"python -m pip install {quoted}")
        if "install" in specs:
            install = specs["install"] if isinstance(specs["install"], list) else [specs["install"]]
            install_cmds.extend(install)
        if "pre_install" in specs:
            pre = specs["pre_install"]
            install_cmds = (pre if isinstance(pre, list) else [pre]) + install_cmds

    handle.extra["python_version"] = python_version

    if not install_cmds:
        install_cmds = [
            "python -m pip install --upgrade pip",
            "pip install -e . || pip install .",
        ]

    install_ok = True
    for cmd in install_cmds:
        r = run(f"cd {shlex.quote(REPO_DIR)} && {cmd}", timeout=900)
        if r.exit_code != 0:
            _log.warning("repo_setup.install_step_failed", cmd=cmd, exit=r.exit_code)
            install_ok = False

    # Apply the SWE-bench test patch (adds the failing tests we need to flip).
    if apply_test_patch and instance.test_patch:
        runner.write_file(handle, f"{REPO_DIR}/.swe_test.patch", instance.test_patch)
        r = run(
            f"cd {shlex.quote(REPO_DIR)} && git apply --allow-empty -v .swe_test.patch || "
            f"git apply --allow-empty --reject -v .swe_test.patch"
        )
        if r.exit_code != 0:
            _log.warning("repo_setup.test_patch_failed")
            return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    # Snapshot a clean baseline commit so commit_diff later shows ONLY the agent's edits.
    r = run(
        f"cd {shlex.quote(REPO_DIR)} && "
        "git -c user.email=swe-rl@local -c user.name=swe-rl "
        "add -A && git -c user.email=swe-rl@local -c user.name=swe-rl "
        'commit --allow-empty -m "swe-rl baseline" -q'
    )
    if r.exit_code != 0 or not install_ok:
        return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    if settings.sandbox_disable_network_after_install:
        try:
            runner.disable_network(handle)
        except RuntimeError as exc:
            logs.append(f"network isolation failed: {exc}")
            return RepoSetupResult(False, "\n".join(logs), time.time() - t0)

    return RepoSetupResult(True, "\n".join(logs), time.time() - t0)
