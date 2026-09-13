import pytest

from swe_rl.agent.tools.file_edit import _resolve


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "a/../../secret", ""])
def test_resolve_rejects_paths_outside_repo(path):
    with pytest.raises(ValueError):
        _resolve("/workspace/repo", path)


def test_resolve_accepts_repo_relative_path():
    assert _resolve("/workspace/repo", "src/app.py") == "/workspace/repo/src/app.py"
