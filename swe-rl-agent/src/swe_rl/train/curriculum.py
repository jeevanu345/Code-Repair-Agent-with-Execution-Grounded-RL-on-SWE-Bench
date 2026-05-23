"""Easy -> hard curriculum scheduler for SWE-bench instances."""

from __future__ import annotations

from collections.abc import Iterable

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
