"""Versioned repair strategies persisted with every trajectory."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    name: str
    version: str
    instructions: str


PROFILES = {
    name: AgentProfile(name, "1", instructions)
    for name, instructions in {
        "explorer-reviewer": "Work in three roles: explorer localizes evidence, implementer applies repair, reviewer independently checks the patch before finish.",
        "standard": "Inspect relevant code, make a repair, and validate focused tests.",
        "plan-execute": "Write an explicit plan before making changes; revise it using tool evidence.",
        "test-driven": "Reproduce the failure with a focused test before editing and retest after edits.",
        "minimal-patch": "Prefer the smallest change that repairs the cause and preserves existing behavior.",
        "high-exploration": "Inspect symbols, imports, call references and neighboring implementations before editing.",
        "low-cost": "Use narrow reads and targeted searches. Avoid repeated broad observations and unnecessary tests.",
    }.items()
}


def get_profile(name: str) -> AgentProfile:
    aliases = {
        "react": "standard",
        "plan_execute": "plan-execute",
        "test_driven": "test-driven",
        "minimal_patch": "minimal-patch",
        "high_exploration": "high-exploration",
        "low_cost": "low-cost",
    }
    key = aliases.get(name, name)
    if key not in PROFILES:
        raise ValueError(f"unknown agent profile: {name}")
    return PROFILES[key]
