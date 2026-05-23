from pathlib import Path

from swe_rl.agent.trajectory import Trajectory
from swe_rl.rollout.replay_buffer import ReplayBuffer


def _mk_traj(instance_id: str, reward: float, resolved: bool) -> Trajectory:
    t = Trajectory.new(
        instance_id=instance_id,
        seed=1,
        temperature=0.8,
        top_p=0.95,
        model_name="m",
        sandbox_image_digest="d",
    )
    t.add_message("user", "p")
    t.add_message("assistant", "a")
    t.reward = reward
    t.reward_details = {"exec": {"resolved": resolved}}
    t.final_patch = "diff..."
    return t


def test_buffer_roundtrip(tmp_path: Path):
    buf = ReplayBuffer(root=tmp_path, shard_max_rows=2)
    buf.add(_mk_traj("a", 1.0, True))
    buf.add(_mk_traj("b", 0.0, False))  # triggers flush
    buf.add(_mk_traj("c", 1.0, True))
    buf.flush()

    rows = list(buf.iter_rows())
    assert len(rows) == 3

    resolved = list(buf.iter_rows(only_resolved=True))
    assert {r["instance_id"] for r in resolved} == {"a", "c"}
