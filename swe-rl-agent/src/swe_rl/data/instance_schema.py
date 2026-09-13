"""Validated, normalized representation of a SWE-bench instance."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SWEBenchInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    base_commit: str = Field(min_length=1)
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    version: str | None = None
    environment_setup_commit: str | None = None
    hints_text: str | None = None
    created_at: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fail_to_pass", "pass_to_pass", mode="before")
    @classmethod
    def _parse_test_list(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("test list must be a JSON array") from exc
        if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) for x in value):
            raise ValueError("test list must contain only strings")
        return list(dict.fromkeys(x.strip() for x in value if x.strip()))

    @classmethod
    def from_hf_row(cls, row: dict[str, Any]) -> SWEBenchInstance:
        aliases = {
            "FAIL_TO_PASS": "fail_to_pass",
            "PASS_TO_PASS": "pass_to_pass",
        }
        known = set(cls.model_fields) - {"extra"}
        normalized: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in row.items():
            target = aliases.get(key, key)
            if target in known:
                normalized[target] = value
            else:
                extra[key] = value
        normalized["extra"] = extra
        return cls.model_validate(normalized)
