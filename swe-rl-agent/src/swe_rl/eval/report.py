"""Render a markdown leaderboard report from harness output."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def render_report(
    *,
    run_id: str,
    predictions_path: Path,
    harness_report: dict[str, Any],
    output_dir: Path,
    base_model: str,
    notes: str = "",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preds = json.loads(predictions_path.read_text(encoding="utf-8"))

    n = len(preds)
    n_resolved = int(harness_report.get("resolved_instances", 0))
    rate = n_resolved / n if n else 0.0

    by_repo = defaultdict(lambda: {"n": 0, "resolved": 0})
    resolved_ids = set(harness_report.get("resolved_ids", []))
    for p in preds:
        repo = p["instance_id"].split("__")[0] if "__" in p["instance_id"] else "?"
        by_repo[repo]["n"] += 1
        if p["instance_id"] in resolved_ids:
            by_repo[repo]["resolved"] += 1

    lines: list[str] = []
    lines.append(f"# SWE-bench Leaderboard — `{run_id}`\n")
    lines.append(f"- Base model: `{base_model}`")
    lines.append(f"- Predictions: `{predictions_path}`")
    lines.append(f"- Instances evaluated: **{n}**")
    lines.append(f"- Resolved: **{n_resolved}**")
    lines.append(f"- **resolved@1: {rate * 100:.2f}%**\n")

    if notes:
        lines.append("## Notes\n")
        lines.append(notes + "\n")

    lines.append("## Per-repo breakdown\n")
    lines.append("| Repo | N | Resolved | Rate |")
    lines.append("|---|---:|---:|---:|")
    for repo, c in sorted(by_repo.items()):
        r = c["resolved"] / c["n"] if c["n"] else 0.0
        lines.append(f"| {repo} | {c['n']} | {c['resolved']} | {r * 100:.1f}% |")

    out_path = output_dir / "leaderboard.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "base_model": base_model,
                "n_instances": n,
                "n_resolved": n_resolved,
                "resolved_at_1": rate,
                "by_repo": dict(by_repo),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path
