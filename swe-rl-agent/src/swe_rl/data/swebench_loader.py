"""Lazy loaders for the official SWE-bench datasets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

from swe_rl.data.instance_schema import SWEBenchInstance

Subset = Literal["lite", "verified", "full"]
_DATASETS: dict[Subset, str] = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def load_swebench(
    *, subset: Subset = "lite", split: str = "test", max_instances: int | None = None
) -> Iterator[SWEBenchInstance]:
    if subset not in _DATASETS:
        raise ValueError(f"unsupported SWE-bench subset: {subset}")
    if max_instances is not None and max_instances < 0:
        raise ValueError("max_instances must be non-negative or None")
    from datasets import load_dataset

    rows = load_dataset(_DATASETS[subset], split=split)
    for index, row in enumerate(rows):
        if max_instances is not None and index >= max_instances:
            break
        yield SWEBenchInstance.from_hf_row(dict(row))


def load_one(instance_id: str, *, subset: Subset | None = None) -> SWEBenchInstance:
    subsets: tuple[Subset, ...] = (subset,) if subset else ("lite", "verified", "full")
    for name in subsets:
        for instance in load_swebench(subset=name):
            if instance.instance_id == instance_id:
                return instance
    raise LookupError(f"SWE-bench instance not found: {instance_id}")
