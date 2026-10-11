import httpx
import pytest

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.providers import connection, save_connection
from swe_rl.dashboard.analytics import compare_experiments, normalize_run, pass_at_k, summarize_runs


def record(instance, experiment, resolved, cost=None):
    return {
        "instance_id": instance,
        "experiment_id": experiment,
        "status": "completed",
        "resolved": resolved,
        "cost": cost,
        "steps": 3,
        "final_patch": "diff",
        "tokens_in": 10,
        "tokens_out": 5,
    }


def test_unknown_costs_and_insufficient_pass_k_are_not_zero():
    result = summarize_runs([record("a", "exp", True)])
    assert result["avg_cost"] is None
    assert result["cost_per_resolved"] is None
    assert pass_at_k([record("a", "exp", True)], 4)["value"] is None
    assert result["resolution_rate"] == 1


def test_pass_at_k_estimator_and_matched_comparison():
    rows = [record("a", "left", index == 0) for index in range(4)]
    assert pass_at_k(rows, 2)["value"] == pytest.approx(0.5)
    rows += [record("a", "right", False), record("b", "right", True)]
    comparison = compare_experiments(rows, "left", "right")
    assert comparison["shared_instances"] == 1
    assert comparison["cohorts"]["only_a"] == ["a"]
    assert comparison["comparable"] is False


def test_normalized_failure_evidence():
    row = normalize_run(
        {
            "trajectory_id": "t",
            "metadata": {"repo": "a/b", "setup_failed": True},
            "reward_details": {},
        }
    )
    assert row["repository"] == "a/b"
    assert row["failure_analysis"]["category"] == "repository_setup"


def test_fallback_is_explicit_and_only_transient(monkeypatch):
    from swe_rl.dashboard import model_api

    monkeypatch.setattr(
        model_api,
        "fallback_connection",
        lambda: connection("custom", "fallback", "", "http://127.0.0.1:9000/v1"),
    )
    seen = []

    def transport(request):
        seen.append(str(request.url))
        if request.url.port == 8000:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "repair"}}]})

    client = LLMClient(
        config=connection("vllm", "primary", "", "http://127.0.0.1:8000/v1"),
        transport=httpx.MockTransport(transport),
    )
    response = client.chat([{"role": "user", "content": "fix"}])
    assert response.text == "repair"
    assert response.raw["swe_rl_inference"]["fallback"]
    assert len(seen) == 2
    client.close()


def test_credentials_unavailable_never_writes_plaintext(tmp_path, monkeypatch):
    from swe_rl.agent import credentials
    from swe_rl.settings import settings

    monkeypatch.setattr(settings, "llm_profile_path", tmp_path / "profile.json")

    def unavailable():
        raise RuntimeError("No OS credential backend")

    monkeypatch.setattr(credentials, "_backend", unavailable)
    with pytest.raises(RuntimeError):
        save_connection(connection("openai", "model", "secret-test"))
    assert not settings.llm_profile_path.exists()


def test_repair_metrics_include_failed_attempts_and_exclude_training():
    from swe_rl.dashboard.analytics import summarize_runs

    report = summarize_runs(
        [
            {"run_id": "ok", "instance_id": "a", "status": "completed", "resolved": True},
            {
                "run_id": "bad",
                "instance_id": "b",
                "status": "failed",
                "metadata": {"setup_failed": True},
            },
            {"run_id": "train", "kind": "training", "status": "completed", "resolved": True},
        ]
    )
    assert report["runs"] == 2
    assert report["resolution_rate"] == 0.5
    assert report["failure_categories"]["repository_setup"] == 1
