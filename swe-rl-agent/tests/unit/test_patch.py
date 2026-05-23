from swe_rl.utils.patch import is_empty_patch, patch_stats


SAMPLE = """\
diff --git a/foo.py b/foo.py
index abcd123..def4567 100644
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,3 @@
 def f():
-    return 1
+    return 2
+    # added
"""


def test_patch_stats_basic():
    files, added, removed = patch_stats(SAMPLE)
    assert files == 1
    assert added == 2
    assert removed == 1


def test_patch_stats_empty():
    assert patch_stats("") == (0, 0, 0)
    assert is_empty_patch("")
    assert is_empty_patch("   \n  ")


def test_patch_stats_garbage():
    # Malformed input should not raise.
    files, added, removed = patch_stats("not a real patch")
    assert (files, added, removed) == (0, 0, 0)
