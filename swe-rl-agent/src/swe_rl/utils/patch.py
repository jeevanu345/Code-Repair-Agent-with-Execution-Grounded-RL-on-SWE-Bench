"""Unified-diff helpers."""

from __future__ import annotations

from unidiff import PatchSet


def parse_patch(text: str) -> PatchSet:
    return PatchSet(text.splitlines(keepends=True))


def patch_stats(text: str) -> tuple[int, int, int]:
    """Return (files_changed, lines_added, lines_removed)."""
    if not text.strip():
        return 0, 0, 0
    try:
        ps = parse_patch(text)
    except Exception:
        return 0, 0, 0
    added = sum(f.added for f in ps)
    removed = sum(f.removed for f in ps)
    return len(ps), added, removed


def is_empty_patch(text: str) -> bool:
    files, added, removed = patch_stats(text)
    return files == 0 and added == 0 and removed == 0
