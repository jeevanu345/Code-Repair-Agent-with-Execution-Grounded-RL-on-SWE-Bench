"""Evidence-based failure labels; no model-generated claims are inferred."""

from typing import Any


def classify_failure(trajectory: Any) -> dict[str, Any]:
    m = trajectory.metadata
    evidence = []
    category = "other"
    if trajectory.reward_details.get("exec", {}).get("resolved"):
        return {"category": "resolved", "evidence": [], "source": "execution"}
    for key, label in [
        ("setup_failed", "repository_setup"),
        ("patch_apply_failed", "invalid_patch"),
        ("test_patch_apply_failed", "tests_unavailable"),
        ("agent_failed", "agent_failure"),
    ]:
        if m.get(key):
            category = label
            evidence.append({"metadata": key, "value": m[key]})
            break
    else:
        tests = m.get("final_tests", {})
        if any(t.get("timed_out") for t in tests.values()):
            category = "timeout"
        elif any(
            v in {"missing", "error"}
            for t in tests.values()
            for v in t.get("outcomes", {}).values()
        ):
            category = "tests_unavailable"
        elif any(v != "passed" for v in tests.get("pass_to_pass", {}).get("outcomes", {}).values()):
            category = "regression"
        elif not trajectory.final_patch.strip():
            category = "empty_patch"
        elif m.get("terminal_reason") == "max_steps":
            category = "agent_loop_exhausted"
        else:
            category = "incorrect_implementation"
        evidence.append(
            {
                "terminal_reason": m.get("terminal_reason"),
                "test_outcomes": {k: v.get("outcomes", {}) for k, v in tests.items()},
            }
        )
    return {
        "category": category,
        "evidence": evidence,
        "source": "execution",
        "model_generated": False,
    }


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c,k)/C(n,k), only for observed k <= n."""
    import math

    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("require 0 <= correct <= n and 1 <= k <= n")
    return 1.0 if n - correct < k else 1.0 - math.comb(n - correct, k) / math.comb(n, k)
