"""Lazy HumanEvalPack loader used by the optional SFT command."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def load_humanevalpack(
    *, language: str = "python", split: str = "test"
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("bigcode/humanevalpack", language, split=split)
    return (dict(row) for row in dataset)
