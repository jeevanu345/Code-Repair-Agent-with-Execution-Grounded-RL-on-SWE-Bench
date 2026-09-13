from pathlib import Path

from swe_rl.agent.trajectory import Trajectory


def test_round_trip_jsonl(tmp_path: Path):
    t = Trajectory.new(
        instance_id="x",
        seed=17,
        temperature=0.8,
        top_p=0.95,
        model_name="qwen",
        sandbox_image_digest="sha256:xxx",
    )
    t.add_message("system", "sys")
    t.add_message("user", "task")
    t.add_message("assistant", '{"tool": "bash", "arguments": {"command": "ls"}}')
    t.add_tool_call(
        step=1, tool="bash", arguments={"command": "ls"}, result={"ok": True}, duration_s=0.1
    )
    t.final_patch = "diff..."
    t.reward = 1.0
    path = t.write_jsonl(tmp_path)

    loaded = Trajectory.from_jsonl(path)
    assert loaded.trajectory_id == t.trajectory_id
    assert loaded.instance_id == "x"
    assert len(loaded.messages) == 3
    assert len(loaded.tool_calls) == 1
    assert loaded.tool_calls[0].tool == "bash"
    assert loaded.reward == 1.0


def test_curriculum_difficulty_ordering():
    from swe_rl.data.instance_schema import SWEBenchInstance
    from swe_rl.train.curriculum import sort_by_difficulty

    easy = SWEBenchInstance(
        instance_id="easy",
        repo="r",
        base_commit="c",
        problem_statement="p",
        patch="diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-a\n+b\n",
    )
    hard = SWEBenchInstance(
        instance_id="hard",
        repo="r",
        base_commit="c",
        problem_statement="p" * 5000,
        patch=(
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1,5 +1,5 @@\n"
            "-a\n-b\n-c\n-d\n-e\n+1\n+2\n+3\n+4\n+5\n"
            "diff --git a/y b/y\n--- a/y\n+++ b/y\n@@ -1,1 +1,1 @@\n-z\n+w\n"
        ),
    )
    out = sort_by_difficulty([hard, easy])
    assert out[0].instance_id == "easy"
    assert out[1].instance_id == "hard"
