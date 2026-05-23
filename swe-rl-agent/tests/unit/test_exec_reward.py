from swe_rl.reward.exec_reward import compute_exec_reward
from swe_rl.sandbox.test_executor import TestResults


def _results(outcomes: dict[str, str]) -> TestResults:
    return TestResults(outcomes=outcomes)  # type: ignore[arg-type]


def test_resolved_when_all_pass():
    f2p = ["a", "b"]
    p2p = ["c"]
    f2p_r = _results({"a": "passed", "b": "passed"})
    p2p_r = _results({"c": "passed"})
    r = compute_exec_reward(f2p_r, p2p_r, f2p, p2p)
    assert r.value == 1.0
    assert r.resolved is True
    assert r.fail_to_pass_pass_rate == 1.0
    assert r.pass_to_pass_pass_rate == 1.0


def test_unresolved_when_f2p_fails():
    f2p = ["a", "b"]
    p2p = ["c"]
    f2p_r = _results({"a": "passed", "b": "failed"})
    p2p_r = _results({"c": "passed"})
    r = compute_exec_reward(f2p_r, p2p_r, f2p, p2p)
    assert r.value == 0.0
    assert r.resolved is False
    assert r.fail_to_pass_pass_rate == 0.5


def test_unresolved_when_p2p_breaks():
    f2p = ["a"]
    p2p = ["c", "d"]
    f2p_r = _results({"a": "passed"})
    p2p_r = _results({"c": "passed", "d": "failed"})
    r = compute_exec_reward(f2p_r, p2p_r, f2p, p2p)
    assert r.value == 0.0
    assert r.resolved is False
    assert r.pass_to_pass_pass_rate == 0.5


def test_missing_test_treated_as_fail():
    f2p_r = _results({})
    p2p_r = _results({})
    r = compute_exec_reward(f2p_r, p2p_r, ["a"], ["b"])
    assert r.value == 0.0
    assert r.resolved is False


def test_empty_f2p_is_unresolved():
    # Without any failing-test target, we shouldn't claim resolved.
    f2p_r = _results({})
    p2p_r = _results({"c": "passed"})
    r = compute_exec_reward(f2p_r, p2p_r, [], ["c"])
    assert r.resolved is False
