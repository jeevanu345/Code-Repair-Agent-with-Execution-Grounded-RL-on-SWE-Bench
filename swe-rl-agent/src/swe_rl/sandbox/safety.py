"""Allowlist/denylist for shell commands emitted by the LLM.

Containers are already sandboxed (network=none, cap-drop=ALL, non-root, tmpfs).
This is a *defense in depth* check that runs *before* exec, to refuse obviously
malicious commands so we can fail loudly with a clean reason in trajectories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-rf\s+/(?!\w)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;?\s*:"),  # fork bomb
    re.compile(r"\bcurl\b[^|]*\|\s*(?:bash|sh|zsh)\b"),
    re.compile(r"\bwget\b[^|]*\|\s*(?:bash|sh|zsh)\b"),
    re.compile(r"/var/run/docker\.sock"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if=.+\s+of=/dev/"),
    re.compile(r"\bchroot\b"),
    re.compile(r"\bmount\b\s+--bind"),
    re.compile(r"\binsmod\b|\brmmod\b|\bmodprobe\b"),
    re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"),
    re.compile(r"\bnc\b\s+-l"),
]


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    reason: str | None = None


def check_command(command: str) -> SafetyVerdict:
    cmd = command.strip()
    if not cmd:
        return SafetyVerdict(False, "empty command")
    for pat in DENY_PATTERNS:
        if pat.search(cmd):
            return SafetyVerdict(False, f"denied by pattern: {pat.pattern}")
    return SafetyVerdict(True)
