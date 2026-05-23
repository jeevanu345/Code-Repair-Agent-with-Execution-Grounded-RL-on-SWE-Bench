from swe_rl.train.data_assembly import trajectory_to_sft


def test_trajectory_to_sft_pairs_each_assistant_turn():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "step 1"},
        {"role": "tool", "content": "obs 1"},
        {"role": "assistant", "content": "step 2"},
    ]
    pairs = trajectory_to_sft(msgs)
    assert len(pairs) == 2
    assert "task" in pairs[0].prompt
    assert pairs[0].completion == "step 1"
    assert "obs 1" in pairs[1].prompt
    assert pairs[1].completion == "step 2"
