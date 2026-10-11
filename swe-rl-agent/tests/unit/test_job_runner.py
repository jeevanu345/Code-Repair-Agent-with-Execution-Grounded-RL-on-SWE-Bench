"""Exercise the isolated worker pipeline without downloads, inference or containers."""

from types import SimpleNamespace

from swe_rl.agent.trajectory import Trajectory
from swe_rl.dashboard.job_runner import execute
from swe_rl.dashboard.jobs import JobStore
from swe_rl.settings import settings


def test_worker_invokes_runtime_and_persists_live_evidence(tmp_path, monkeypatch):
    import swe_rl.observability
    from swe_rl.agent import llm_client, providers
    from swe_rl.data import swebench_loader
    from swe_rl.rollout import worker

    monkeypatch.setattr(swe_rl.observability, "init", lambda *args, **kwargs: None)
    # execute changes process-local settings in production; restore them in this in-process test.
    for name in [
        "model_name",
        "sandbox_cpus",
        "sandbox_mem_gb",
        "sandbox_wallclock_s",
        "sandbox_disable_network_after_install",
        "max_dollars_per_run",
    ]:
        monkeypatch.setattr(settings, name, getattr(settings, name))
    selected = providers.connection("vllm", "test-model", "", "http://127.0.0.1:8000")
    monkeypatch.setattr(providers, "active_connection", lambda: selected)
    closed = []
    monkeypatch.setattr(
        llm_client,
        "LLMClient",
        lambda **kwargs: SimpleNamespace(model="test-model", close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(
        swebench_loader,
        "load_one",
        lambda instance_id, **kwargs: SimpleNamespace(instance_id=instance_id),
    )
    calls = []

    def fake_rollout(instance, **kwargs):
        calls.append(kwargs)
        kwargs["event_callback"]({"type": "preparing"})
        kwargs["event_callback"](
            {
                "type": "step",
                "step": 2,
                "tokens_in": 20,
                "tokens_out": 3,
                "cost_usd": 0.01,
                "test_budget_remaining": 2,
            }
        )
        trajectory = Trajectory.new(
            instance_id=instance.instance_id,
            seed=17,
            temperature=0.2,
            top_p=0.95,
            model_name="test-model",
            sandbox_image_digest="sha256:test",
        )
        trajectory.n_steps = 2
        trajectory.finished_at = "2026-10-02T00:00:00Z"
        return SimpleNamespace(
            trajectory=trajectory,
            resolved=True,
            reward=1.0,
            duration_s=3.0,
            details={"exec": {"resolved": True}},
        )

    monkeypatch.setattr(worker, "run_rollout", fake_rollout)
    store = JobStore(tmp_path)
    job = store.create(
        "run",
        {
            "instance_id": "test__repo-1",
            "dataset": "lite",
            "dataset_revision": "pinned",
            "model": "test-model",
            "provider": "vllm",
            "base_url": selected.base_url,
            "prompt_content": "custom instructions",
            "prompt_version": "custom@1",
        },
        "exp",
    )
    execute(store, job["id"])
    result = store.get(job["id"])
    assert result["status"] == "completed"
    assert result["result"]["resolved"] is True
    assert result["tokens_in"] == 20
    assert calls[0]["react_config"].prompt_content == "custom instructions"
    assert calls[0]["control_callback"]() is None
    assert closed == [True]
    assert list(store.directory(job["id"]).glob("*.jsonl"))
    assert any(event["type"] == "step" for event in store.events(job["id"]))


def test_imported_evaluation_invokes_official_harness(tmp_path, monkeypatch):
    import swe_rl.observability
    from swe_rl.eval import swebench_eval

    monkeypatch.setattr(swe_rl.observability, "init", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(
        swebench_eval,
        "run_official_harness",
        lambda path, **kwargs: (
            calls.append((path, kwargs)) or {"resolved_instances": 1, "submitted_instances": 2}
        ),
    )
    store = JobStore(tmp_path)
    predictions = tmp_path / "predictions.json"
    predictions.write_text("[]")
    job = store.create(
        "evaluation",
        {
            "predictions_path": str(predictions),
            "dataset": "multilingual",
            "parallelism": 3,
            "instance_ids": ["test__repo-1"],
        },
        "exp",
    )
    execute(store, job["id"])
    assert store.get(job["id"])["status"] == "completed"
    assert calls[0][1]["max_workers"] == 3
    assert calls[0][1]["instance_ids"] == ["test__repo-1"]
    assert (store.directory(job["id"]) / "evaluation.json").is_file()
