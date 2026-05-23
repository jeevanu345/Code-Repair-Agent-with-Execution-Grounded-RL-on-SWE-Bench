"""Run the project's test suite inside a sandbox container and parse outcomes.

We honor SWE-bench's `FAIL_TO_PASS` and `PASS_TO_PASS` semantics:
- The instance is `resolved` iff every FAIL_TO_PASS test is now PASSING and
  every PASS_TO_PASS test is still PASSING.

Output parsing prefers pytest's machine-readable JSON report. We try, in order:
  1) `pytest --json-report` if available (reliable)
  2) `pytest -v` with regex parsing (fallback)
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from typing import Literal

from swe_rl.observability.logging import get_logger
from swe_rl.observability.metrics import METRICS
from swe_rl.sandbox.docker_runner import ContainerHandle, DockerRunner

_log = get_logger(__name__)

REPO_DIR = "/workspace/repo"
JSON_REPORT_PATH = "/tmp/pytest-report.json"

TestOutcome = Literal["passed", "failed", "error", "skipped", "missing"]


@dataclass
class TestResults:
    outcomes: dict[str, TestOutcome] = field(default_factory=dict)
    raw_stdout: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    runner: str = "pytest"

    def outcome_for(self, test_id: str) -> TestOutcome:
        return self.outcomes.get(test_id, self.outcomes.get(_normalize(test_id), "missing"))

    def all_pass(self, test_ids: list[str]) -> bool:
        return all(self.outcome_for(t) == "passed" for t in test_ids)

    def pass_rate(self, test_ids: list[str]) -> float:
        if not test_ids:
            return 1.0
        return sum(1 for t in test_ids if self.outcome_for(t) == "passed") / len(test_ids)


def _normalize(test_id: str) -> str:
    """Normalize pytest node ids — collapse repeated separators, drop leading './'."""
    s = test_id.strip()
    if s.startswith("./"):
        s = s[2:]
    return s


def _parse_json_report(blob: str) -> dict[str, TestOutcome]:
    out: dict[str, TestOutcome] = {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return out
    for t in data.get("tests", []):
        nid = t.get("nodeid")
        outcome = t.get("outcome")
        if not nid or not outcome:
            continue
        if outcome in {"passed", "failed", "error", "skipped"}:
            out[_normalize(nid)] = outcome  # type: ignore[assignment]
    return out


_VERBOSE_RE = re.compile(
    r"^(?P<nid>\S+::\S+)\s+(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)",
    re.MULTILINE,
)


def _parse_verbose(stdout: str) -> dict[str, TestOutcome]:
    out: dict[str, TestOutcome] = {}
    for m in _VERBOSE_RE.finditer(stdout):
        status = m.group("status").lower()
        if status == "xfail":
            status = "skipped"
        if status == "xpass":
            status = "passed"
        if status in {"passed", "failed", "error", "skipped"}:
            out[_normalize(m.group("nid"))] = status  # type: ignore[assignment]
    return out


class TestExecutor:
    def __init__(self, runner: DockerRunner) -> None:
        self.runner = runner

    def run(
        self,
        handle: ContainerHandle,
        test_ids: list[str],
        *,
        timeout: int = 600,
        prefer_json_report: bool = True,
    ) -> TestResults:
        if not test_ids:
            return TestResults()

        # Try json-report; install plugin lazily (silent fail -> verbose fallback).
        cmd_parts = ["cd", shlex.quote(REPO_DIR), "&&"]
        if prefer_json_report:
            cmd_parts += [
                "python -m pip install --quiet pytest-json-report >/dev/null 2>&1 || true",
                "&&",
            ]
            pytest_cmd = (
                f"python -m pytest --json-report --json-report-file={JSON_REPORT_PATH} "
                f"-q --tb=short --no-header "
                + " ".join(shlex.quote(t) for t in test_ids)
            )
        else:
            pytest_cmd = (
                "python -m pytest -v --tb=short --no-header "
                + " ".join(shlex.quote(t) for t in test_ids)
            )
        cmd_parts.append(pytest_cmd)
        full_cmd = " ".join(cmd_parts)

        result = self.runner.exec(handle, full_cmd, timeout=timeout, check_safety=False)

        outcomes: dict[str, TestOutcome] = {}
        if prefer_json_report:
            blob = self.runner.exec(
                handle, f"cat {JSON_REPORT_PATH} 2>/dev/null || echo ''", check_safety=False
            ).stdout
            if blob.strip():
                outcomes = _parse_json_report(blob)

        if not outcomes:
            outcomes = _parse_verbose(result.stdout)

        # Tests we asked for but didn't see -> missing (treated as fail by reward).
        for tid in test_ids:
            outcomes.setdefault(_normalize(tid), "missing")

        for outcome, count in _count_by_outcome(outcomes).items():
            METRICS.tests_run_total.labels(outcome=outcome).inc(count)

        return TestResults(
            outcomes=outcomes,
            raw_stdout=result.stdout,
            duration_s=result.duration_s,
            timed_out=result.timed_out,
        )


def _count_by_outcome(outcomes: dict[str, TestOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in outcomes.values():
        counts[v] = counts.get(v, 0) + 1
    return counts
