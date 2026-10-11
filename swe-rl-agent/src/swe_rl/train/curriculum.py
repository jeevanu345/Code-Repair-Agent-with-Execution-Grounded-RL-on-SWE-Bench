"""Easy -> hard curriculum scheduler for SWE-bench instances."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from swe_rl.data.instance_schema import SWEBenchInstance
from swe_rl.utils.patch import patch_stats


def difficulty_score(instance: SWEBenchInstance) -> tuple[int, int, int]:
    """Lower tuple = easier. Sort key: (files, total_lines, problem_len)."""
    files, added, removed = patch_stats(instance.patch or "")
    total_lines = added + removed
    return (files or 999, total_lines or 999, len(instance.problem_statement))


def sort_by_difficulty(instances: Iterable[SWEBenchInstance]) -> list[SWEBenchInstance]:
    return sorted(instances, key=difficulty_score)


def staged_curriculum(
    instances: Iterable[SWEBenchInstance],
    stages: list[dict[str, int]] | None = None,
) -> list[list[SWEBenchInstance]]:
    """Default stages: <20 LOC single-file, <80 LOC, anything."""
    stages = stages or [
        {"max_files": 1, "max_lines": 20},
        {"max_files": 2, "max_lines": 80},
        {"max_files": 99, "max_lines": 9999},
    ]
    sorted_inst = sort_by_difficulty(instances)
    out: list[list[SWEBenchInstance]] = [[] for _ in stages]
    for inst in sorted_inst:
        files, added, removed = patch_stats(inst.patch or "")
        for i, s in enumerate(stages):
            if files <= s["max_files"] and (added + removed) <= s["max_lines"]:
                out[i].append(inst)
                break
    return out


def curriculum_from_failures(
    instances: Iterable[SWEBenchInstance],
    recorded_runs: list[dict[str, Any]],
) -> dict[str, list[SWEBenchInstance]]:
    """Use observed categories to separate environment remediation and repair lessons.

    No gold patch is inspected; this is a measured-category curriculum, not a
    learned difficulty estimate. Previously solved tasks are a retention stage.
    """
    by_instance: dict[str, set[str]] = {}
    for row in recorded_runs:
        evidence = (
            row.get("failure_analysis") or (row.get("metadata") or {}).get("failure_analysis") or {}
        )
        category = evidence.get("category", "unobserved")
        by_instance.setdefault(row.get("instance_id", ""), set()).add(category)
    stages: dict[str, list[SWEBenchInstance]] = {
        "environment_remediation": [],
        "repair_practice": [],
        "retention": [],
        "unobserved": [],
    }
    for instance in instances:
        categories = by_instance.get(instance.instance_id, set())
        if categories & {"repository_setup", "evaluation_setup", "tests_unavailable"}:
            stage = "environment_remediation"
        elif categories & {"resolved"}:
            stage = "retention"
        elif categories:
            stage = "repair_practice"
        else:
            stage = "unobserved"
        stages[stage].append(instance)
    return stages
