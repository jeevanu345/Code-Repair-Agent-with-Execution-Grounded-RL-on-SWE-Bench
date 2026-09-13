"""Convert recorded trajectories into TRL-friendly datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from swe_rl.rollout.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from datasets import Dataset


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str


def trajectory_to_sft(messages: list[dict[str, Any]]) -> list[SFTExample]:
    """Pair user/system context with each assistant turn."""
    out: list[SFTExample] = []
    accum: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "assistant":
            prompt = "\n\n".join(f"<{x['role']}>\n{x['content']}" for x in accum)
            out.append(SFTExample(prompt=prompt, completion=m["content"]))
        accum.append(m)
    return out


def replay_to_sft_dataset(buffer: ReplayBuffer, *, only_resolved: bool = True) -> Dataset:
    from datasets import Dataset

    rows: list[dict[str, str]] = []
    for r in buffer.iter_rows(only_resolved=only_resolved if only_resolved else None):
        msgs = json.loads(r["messages_json"])
        for ex in trajectory_to_sft(msgs):
            rows.append({"prompt": ex.prompt, "completion": ex.completion})
    return Dataset.from_list(rows)


def replay_to_dpo_pairs(buffer: ReplayBuffer) -> Dataset:
    from datasets import Dataset

    """Group rollouts by instance; pair best vs worst (by reward) within each group."""
    by_inst: dict[str, list[dict[str, Any]]] = {}
    for r in buffer.iter_rows():
        by_inst.setdefault(r["instance_id"], []).append(r)
    pairs: list[dict[str, str]] = []
    for inst_rows in by_inst.values():
        if len(inst_rows) < 2:
            continue
        sorted_rows = sorted(inst_rows, key=lambda r: r["reward"])
        worst, best = sorted_rows[0], sorted_rows[-1]
        if best["reward"] <= worst["reward"]:
            continue
        bw = json.loads(best["messages_json"])
        ww = json.loads(worst["messages_json"])
        # Use the system+user prefix as the prompt; final patch as response proxy.
        prompt = "\n\n".join(
            f"<{m['role']}>\n{m['content']}" for m in bw if m["role"] in {"system", "user"}
        )
        pairs.append(
            {
                "prompt": prompt,
                "chosen": best["final_patch"] or _last_assistant(bw),
                "rejected": worst["final_patch"] or _last_assistant(ww),
            }
        )
    return Dataset.from_list(pairs)


def _last_assistant(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m["role"] == "assistant":
            return str(m["content"])
    return ""


def humanevalpack_to_sft(rows: Iterable[dict[str, Any]]) -> Dataset:
    from datasets import Dataset

    items: list[dict[str, str]] = []
    for r in rows:
        prompt = (
            "Fix the following Python code so all tests pass.\n\n"
            f"# Buggy code\n```python\n{r.get('declaration', '')}{r.get('buggy_solution', '')}\n```\n\n"
            f"# Tests\n```python\n{r.get('test', '')}\n```\n\n"
            "Return only the corrected code."
        )
        completion = f"```python\n{r.get('declaration', '')}{r.get('canonical_solution', '')}\n```"
        items.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(items)
