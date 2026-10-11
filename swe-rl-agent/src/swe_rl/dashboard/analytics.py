"""Evidence-derived summaries. Unknown measurements remain unknown."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any


def _duration(row: dict[str, Any]) -> float | None:
    metadata = row.get("metadata") or {}
    if metadata.get("duration_s") is not None:
        return float(metadata["duration_s"])
    if row.get("duration_s") is not None:
        return float(row["duration_s"])
    try:
        start = datetime.fromisoformat(row["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(row["finished_at"].replace("Z", "+00:00"))
        return max(0, (end - start).total_seconds())
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def failure_analysis(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    evidence = (row.get("reward_details") or {}).get("exec") or {}
    if evidence.get("resolved") or row.get("resolved"):
        return {"category": "resolved", "source": "recorded_evidence", "evidence": []}
    findings: list[tuple[str, str]] = []
    for flag, category in [
        ("setup_failed", "repository_setup"),
        ("patch_apply_failed", "invalid_patch"),
        ("test_patch_apply_failed", "evaluation_setup"),
        ("agent_failed", "agent_error"),
    ]:
        if metadata.get(flag):
            findings.append((category, f"Recorded {flag}: {metadata[flag]}"))
    if metadata.get("timed_out") or any(
        "timeout" in str(value).lower()
        for value in [metadata.get("agent_failed", ""), metadata.get("stop_reason", "")]
    ):
        findings.append(("timeout", "Timeout recorded by runtime"))
    if evidence.get("p2p_failed"):
        findings.append(("regression", f"{len(evidence['p2p_failed'])} preservation tests failed"))
    if not row.get("final_patch"):
        findings.append(("empty_patch", "No final patch recorded"))
    if metadata.get("stop_reason") in {"max_steps", "step_limit", "context_limit"}:
        findings.append(("agent_loop_exhausted", f"Stopped at {metadata['stop_reason']}"))
    if evidence.get("f2p_failed"):
        findings.append(
            (
                "incorrect_implementation",
                f"{len(evidence['f2p_failed'])} required tests remain failing",
            )
        )
    category = findings[0][0] if findings else "unknown"
    return {
        "category": category,
        "source": "recorded_evidence",
        "evidence": [item[1] for item in findings],
        "inference": "Heuristic classification from recorded flags and tests; localization/root cause is not established.",
    }


def normalize_run(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    metadata = result.get("metadata") or {}
    evidence = (result.get("reward_details") or {}).get("exec") or {}
    result["run_id"] = result.get("run_id") or result.get("trajectory_id") or ""
    result["repository"] = result.get("repository") or metadata.get("repo", "")
    result["problem_statement"] = result.get("problem_statement") or metadata.get(
        "problem_statement", ""
    )
    result["model"] = result.get("model") or result.get("model_name", "")
    result["steps"] = result.get("steps", result.get("n_steps", 0))
    result["tokens_in"] = result.get("tokens_in", result.get("input_tokens", 0))
    result["tokens_out"] = result.get("tokens_out", result.get("output_tokens", 0))
    result["input_tokens"] = result["tokens_in"]
    result["output_tokens"] = result["tokens_out"]
    result["resolved"] = bool(evidence.get("resolved", result.get("resolved", False)))
    result["duration_s"] = _duration(result)
    result["cost"] = result.get("cost", metadata.get("cost_usd", metadata.get("cost_dollars")))
    result["experiment_id"] = result.get("experiment_id") or metadata.get("experiment_id", "")
    result["fail_to_pass"] = {
        "passed": evidence.get("f2p_passed", []),
        "failed": evidence.get("f2p_failed", []),
    }
    result["pass_to_pass"] = {
        "passed": evidence.get("p2p_passed", []),
        "failed": evidence.get("p2p_failed", []),
    }
    result["failure_analysis"] = metadata.get("failure_analysis") or failure_analysis(result)
    return result


def _average(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
    ]
    return mean(values) if values else None


def pass_at_k(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Unbiased estimator on groups with at least k independent candidates."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind", "run") == "run" and row.get("status") in {"completed", "failed"}:
            groups[
                str(row.get("experiment_id", "")) + ":" + str(row.get("instance_id", ""))
            ].append(row)
    values = []
    for candidates in groups.values():
        n = len(candidates)
        c = sum(bool(row.get("resolved")) for row in candidates)
        if n >= k:
            values.append(1.0 if n - c < k else 1 - math.comb(n - c, k) / math.comb(n, k))
    return {
        "k": k,
        "value": mean(values) if values else None,
        "eligible_instances": len(values),
        "method": "1 - C(n-c,k)/C(n,k); only groups with n>=k",
    }


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [normalize_run(row) for row in rows if row.get("kind", "run") == "run"]
    completed = [row for row in runs if row.get("status") in {"completed", "failed"}]
    resolved = [row for row in completed if row["resolved"]]
    categories: dict[str, int] = defaultdict(int)
    for row in completed:
        categories[row["failure_analysis"]["category"]] += 1
    breakdowns = {}
    for field in ["model", "repository", "experiment_id"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in completed:
            grouped[str(row.get(field) or "Unrecorded")].append(row)
        breakdowns[field] = [
            {
                "name": name,
                "runs": len(items),
                "resolved": sum(item["resolved"] for item in items),
                "resolution_rate": sum(item["resolved"] for item in items) / len(items),
            }
            for name, items in grouped.items()
        ]
    total_cost = sum(row["cost"] for row in completed if isinstance(row.get("cost"), (int, float)))
    known_costs = sum(isinstance(row.get("cost"), (int, float)) for row in completed)
    return {
        "runs": len(runs),
        "completed": len(completed),
        "resolved": len(resolved),
        "resolution_rate": len(resolved) / len(completed) if completed else None,
        "pass_at_k": [pass_at_k(runs, k) for k in (1, 2, 4, 8, 16)],
        "avg_cost": _average(completed, "cost"),
        "avg_runtime_s": _average(completed, "duration_s"),
        "avg_steps": _average(completed, "steps"),
        "avg_tokens": mean(row["tokens_in"] + row["tokens_out"] for row in completed)
        if completed
        else None,
        "cost_per_resolved": total_cost / len(resolved)
        if resolved and known_costs == len(completed)
        else None,
        "cost_coverage": known_costs,
        "failure_categories": dict(categories),
        "breakdowns": breakdowns,
        "timeline": [
            {
                "run_id": row["run_id"],
                "started_at": row.get("started_at"),
                "resolved": row["resolved"],
                "cost": row.get("cost"),
                "duration_s": row.get("duration_s"),
                "steps": row["steps"],
                "tokens": row["tokens_in"] + row["tokens_out"],
            }
            for row in completed
        ],
        "benchmark_score": False,
    }


def compare_experiments(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    a = [
        normalize_run(row)
        for row in rows
        if row.get("kind", "run") == "run"
        and (row.get("experiment_id") or (row.get("metadata") or {}).get("experiment_id")) == left
    ]
    b = [
        normalize_run(row)
        for row in rows
        if row.get("kind", "run") == "run"
        and (row.get("experiment_id") or (row.get("metadata") or {}).get("experiment_id")) == right
    ]
    ia = {row["instance_id"] for row in a}
    ib = {row["instance_id"] for row in b}
    shared = ia & ib
    sa = {row["instance_id"] for row in a if row["resolved"]}
    sb = {row["instance_id"] for row in b if row["resolved"]}
    cohorts = {
        "only_a": sorted((sa - sb) & shared),
        "only_b": sorted((sb - sa) & shared),
        "both": sorted(sa & sb & shared),
        "neither": sorted(shared - sa - sb),
    }
    return {
        "left": summarize_runs(a),
        "right": summarize_runs(b),
        "matched_left": summarize_runs([row for row in a if row["instance_id"] in shared]),
        "matched_right": summarize_runs([row for row in b if row["instance_id"] in shared]),
        "shared_instances": len(shared),
        "cohorts": cohorts,
        "comparable": ia == ib and bool(shared),
    }
