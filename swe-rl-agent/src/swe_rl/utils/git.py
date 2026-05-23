"""Tiny git helpers that shell out, used inside containers."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_git(args: list[str], cwd: Path) -> str:
    cmd = ["git", *args]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"git {shlex.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def diff_against_head(cwd: Path) -> str:
    return run_git(["diff", "--no-color", "--unified=3"], cwd=cwd)


def current_commit(cwd: Path) -> str:
    return run_git(["rev-parse", "HEAD"], cwd=cwd).strip()
