from swe_rl.reward.exec_reward import compute_exec_reward
from swe_rl.reward.shaped_reward import ShapedRewardConfig, compute_shaped_reward
from swe_rl.sandbox.test_executor import TestResults


def _r(d):
    return TestResults(outcomes=d)


PATCH = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,1 +1,1 @@
-old
+new
"""


def test_shaped_full_credit():
    exec_r = compute_exec_reward(_r({"a": "passed"}), _r({"b": "passed"}), ["a"], ["b"])
    s = compute_shaped_reward(exec_r, patch=PATCH, timed_out=False)
    assert s.value > 0.9
    assert "size_penalty" in s.components


def test_shaped_no_patch_penalty():
    exec_r = compute_exec_reward(_r({}), _r({"b": "passed"}), ["a"], ["b"])
    s = compute_shaped_reward(exec_r, patch="", timed_out=False)
    assert s.components["no_patch"] == ShapedRewardConfig().no_patch_penalty
    assert s.value < 0.6


def test_shaped_timeout_penalty():
    exec_r = compute_exec_reward(_r({"a": "passed"}), _r({"b": "passed"}), ["a"], ["b"])
    s = compute_shaped_reward(exec_r, patch=PATCH, timed_out=True)
    assert s.components["timeout_penalty"] < 0
