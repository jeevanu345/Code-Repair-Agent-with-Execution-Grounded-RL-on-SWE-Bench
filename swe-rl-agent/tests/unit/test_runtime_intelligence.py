import pytest

from swe_rl.agent.intelligence import analyze_repository, recommend_tests
from swe_rl.agent.profiles import get_profile
from swe_rl.reward.analysis import pass_at_k


def test_graph_links_tests_to_changed_symbols(tmp_path):
    (tmp_path / "parser.py").write_text("def parse(value):\n return value\n")
    (tmp_path / "test_parser.py").write_text(
        "from parser import parse\ndef test_parse():\n assert parse(1)==1\n"
    )
    graph = analyze_repository(tmp_path)
    result = recommend_tests(graph, ["parser.py"])
    assert result[0]["test_id"] == "test_parser.py"
    assert result[0]["score"] == 3


def test_pass_at_k_is_observed_estimator():
    assert pass_at_k(4, 1, 1) == 0.25
    assert pass_at_k(4, 1, 4) == 1
    with pytest.raises(ValueError):
        pass_at_k(1, 1, 4)


def test_profiles_are_versioned():
    assert get_profile("low_cost").name == "low-cost"
    assert get_profile("standard").version == "1"
    with pytest.raises(ValueError):
        get_profile("invented")


def test_cancel_before_first_turn_is_serializable():
    from types import SimpleNamespace

    from swe_rl.agent.react_loop import ReactConfig, run_react
    from swe_rl.agent.trajectory import Trajectory
    from swe_rl.data.instance_schema import SWEBenchInstance

    trajectory = Trajectory.new(
        instance_id="repo__one-1",
        seed=1,
        temperature=0.2,
        top_p=0.95,
        model_name="fake",
        sandbox_image_digest="test",
    )
    instance = SWEBenchInstance(
        instance_id="repo__one-1", repo="repo/one", base_commit="abc", problem_statement="fix"
    )
    context = SimpleNamespace(test_runs_remaining=1)
    result = run_react(
        llm=SimpleNamespace(),
        ctx=context,
        instance=instance,
        trajectory=trajectory,
        config=ReactConfig(),
        control_callback=lambda: "cancel",
    )
    assert result.reason == "cancelled"
    assert result.n_steps == 0
    assert trajectory.metadata["terminal_reason"] == "cancelled"


def test_step_usage_and_live_events(monkeypatch):
    from types import SimpleNamespace

    from pydantic import BaseModel

    from swe_rl.agent.react_loop import ReactConfig, run_react
    from swe_rl.agent.tools.base import ToolResult
    from swe_rl.agent.trajectory import Trajectory
    from swe_rl.data.instance_schema import SWEBenchInstance

    class Arguments(BaseModel):
        pass

    class Finish:
        name = "finish"
        description = "finish"
        Args = Arguments

        def __call__(self, **kwargs):
            return ToolResult(True, "done", {}, 0.01)

    monkeypatch.setattr("swe_rl.agent.react_loop.default_toolset", lambda ctx: [Finish()])
    trajectory = Trajectory.new(
        instance_id="repo__one-1",
        seed=1,
        temperature=0.2,
        top_p=0.95,
        model_name="fake",
        sandbox_image_digest="test",
    )
    instance = SWEBenchInstance(
        instance_id="repo__one-1", repo="repo/one", base_commit="abc", problem_statement="fix"
    )
    events = []
    llm = SimpleNamespace(
        chat=lambda *args, **kwargs: SimpleNamespace(
            text='{"tool":"finish"}', tokens_in=20, tokens_out=10
        )
    )
    run_react(
        llm=llm,
        ctx=SimpleNamespace(test_runs_remaining=1),
        instance=instance,
        trajectory=trajectory,
        config=ReactConfig(),
        event_callback=events.append,
    )
    assert trajectory.tool_calls[0].tokens_in == 20
    assert trajectory.tokens_out == 10
    assert events[-1]["type"] == "step"
    assert events[-1]["tokens_in"] == 20


def test_tool_call_parser_preserves_braces_inside_code():
    from swe_rl.agent.react_loop import _extract_tool_call

    assert (
        _extract_tool_call('{"tool":"file_write","arguments":{"content":"return {a: 1} # }"}}')[
            "arguments"
        ]["content"]
        == "return {a: 1} # }"
    )


def test_minimizer_removes_hidden_tests_only_during_clean_diff_capture():
    from types import SimpleNamespace
    from unittest.mock import patch

    from swe_rl.rollout.worker import _minimize_patch
    from swe_rl.sandbox.test_executor import TestResults

    repair = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    commands = []

    class Runner:
        def write_file(self, *args):
            pass

        def exec(self, handle, command, **kwargs):
            commands.append(command)
            return SimpleNamespace(exit_code=0)

        def commit_diff(self, *args, **kwargs):
            return ""

    instance = SimpleNamespace(test_patch="hidden", fail_to_pass=["test"], pass_to_pass=[])
    with patch("swe_rl.rollout.worker.TestExecutor") as executor:
        executor.return_value.run.return_value = TestResults(outcomes={"test": "passed"})
        result, details = _minimize_patch(Runner(), object(), instance, repair)
    assert result == ""
    assert details["hunks_removed"] == 1
    assert any("git apply --reverse /tmp/.swe_test.patch" in command for command in commands)
    assert commands[-1].endswith("git apply /tmp/.swe_test.patch")


def test_training_telemetry_reports_trainer_logs(monkeypatch):
    import sys
    from types import SimpleNamespace

    from swe_rl.train.telemetry import make_callbacks

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(TrainerCallback=object))
    events = []
    callback = make_callbacks(events.append)[0]
    callback.on_log(
        SimpleNamespace(), SimpleNamespace(global_step=20, epoch=0.5), None, logs={"loss": 0.7}
    )
    assert events == [
        {"type": "training_metrics", "step": 20, "epoch": 0.5, "metrics": {"loss": 0.7}}
    ]


def test_custom_prompt_size_is_bounded():
    from swe_rl.agent.react_loop import ReactConfig

    assert (
        ReactConfig(prompt_content="repair minimally", prompt_version="v2").prompt_version == "v2"
    )
    with pytest.raises(ValueError):
        ReactConfig(prompt_content="x" * 32001)


def test_local_artifacts_are_hashed_and_isolated(tmp_path):
    import hashlib

    from swe_rl.agent.trajectory import Trajectory
    from swe_rl.rollout.worker import _persist_local

    trajectory = Trajectory.new(
        instance_id="repo__one-1",
        seed=1,
        temperature=0.2,
        top_p=0.95,
        model_name="fake",
        sandbox_image_digest="test",
    )
    trajectory.final_patch = "repair"
    _persist_local(trajectory, tmp_path)
    artifact = trajectory.metadata["artifacts"]["final_patch.diff"]
    assert artifact["sha256"] == hashlib.sha256(b"repair").hexdigest()
    assert trajectory.trajectory_id in artifact["path"]
    assert (tmp_path / "artifacts" / trajectory.trajectory_id / "final_tests.json").exists()


def test_semantic_search_ranks_docstring_symbol_evidence():
    from swe_rl.agent.intelligence import semantic_code_search

    sources = {
        "parser.py": 'def decode_payload(value):\n """Parse encoded input into a structured object."""\n return value\n',
        "unrelated.py": "def calculate_total(items):\n return sum(items)\n",
    }
    result = semantic_code_search(sources, "deserialize input", limit=1)
    assert result["matches"][0]["symbol"] == "decode_payload"
    assert "decode" in result["matches"][0]["evidence"]["related_terms"]["symbol"]
    assert "no embeddings" in result["method"]
    assert result["files_processed"] == 2


def test_semantic_search_skips_invalid_and_large_sources():
    from swe_rl.agent.intelligence import semantic_code_search

    result = semantic_code_search({"broken.py": "def x(", "large.py": "x" * 100001}, "parser")
    assert result["matches"] == []
    assert result["skipped"] == ["broken.py", "large.py"]
    with pytest.raises(ValueError):
        semantic_code_search({}, " ", limit=1)


def test_role_strategy_records_real_phase_transitions(monkeypatch):
    from types import SimpleNamespace

    from pydantic import BaseModel

    from swe_rl.agent.react_loop import ReactConfig, run_react
    from swe_rl.agent.tools.base import ToolResult
    from swe_rl.agent.trajectory import Trajectory
    from swe_rl.data.instance_schema import SWEBenchInstance

    class Arguments(BaseModel):
        pass

    class FakeTool:
        Args = Arguments
        description = "observe"

        def __init__(self, name):
            self.name = name

        def __call__(self, **kwargs):
            return ToolResult(True, "observed", {}, 0.01)

    monkeypatch.setattr(
        "swe_rl.agent.react_loop.default_toolset",
        lambda ctx: [FakeTool("repo_map"), FakeTool("finish")],
    )
    replies = iter(
        [
            '{"tool":"repo_map"}',
            '{"tool":"repo_map"}',
            '{"tool":"finish"}',
            '{"approved":true,"feedback":"plausible"}',
        ]
    )
    llm = SimpleNamespace(
        model="selected-model",
        chat=lambda *a, **k: SimpleNamespace(
            text=next(replies), tokens_in=10, tokens_out=5, raw={}
        ),
    )
    trajectory = Trajectory.new(
        instance_id="repo__one-1",
        seed=1,
        temperature=0.2,
        top_p=0.95,
        model_name=llm.model,
        sandbox_image_digest="test",
    )
    instance = SWEBenchInstance(
        instance_id="repo__one-1", repo="repo/one", base_commit="abc", problem_statement="fix"
    )
    context = SimpleNamespace(
        test_runs_remaining=1,
        repo_dir="/repo",
        handle=object(),
        runner=SimpleNamespace(commit_diff=lambda *a, **k: "patch"),
    )
    result = run_react(
        llm=llm,
        ctx=context,
        instance=instance,
        trajectory=trajectory,
        config=ReactConfig(profile="explorer-reviewer"),
    )
    assert result.finished
    assert [phase["role"] for phase in trajectory.metadata["role_transitions"]] == [
        "explorer",
        "implementer",
        "reviewer",
    ]
    assert all(
        phase["model"] == "selected-model" for phase in trajectory.metadata["role_transitions"]
    )
    assert trajectory.metadata["turn_usage"][-1]["agent_role"] == "implementer"
    assert trajectory.tokens_in == 40
