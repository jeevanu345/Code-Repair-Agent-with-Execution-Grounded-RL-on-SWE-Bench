"""Prompt templates loaded from .md files in this directory."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def load(name: str) -> str:
    p = _HERE / f"{name}.md"
    return p.read_text(encoding="utf-8")
