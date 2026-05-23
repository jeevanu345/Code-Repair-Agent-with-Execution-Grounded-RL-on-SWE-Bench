from swe_rl.agent.react_loop import _extract_tool_call


def test_extract_tool_call_fenced():
    text = """I will now read a file.
```json
{"tool": "file_read", "arguments": {"path": "src/foo.py", "start_line": 1, "num_lines": 50}}
```
"""
    out = _extract_tool_call(text)
    assert out is not None
    assert out["tool"] == "file_read"
    assert out["arguments"]["path"] == "src/foo.py"


def test_extract_tool_call_bare_json():
    text = '{"tool": "bash", "arguments": {"command": "ls"}}'
    out = _extract_tool_call(text)
    assert out is not None
    assert out["tool"] == "bash"


def test_extract_tool_call_picks_last_json():
    text = (
        'first thinking {"plan": "look around"} '
        'then {"tool": "grep", "arguments": {"pattern": "foo"}}'
    )
    out = _extract_tool_call(text)
    assert out is not None
    assert out["tool"] == "grep"


def test_extract_tool_call_no_json():
    assert _extract_tool_call("just prose") is None
    assert _extract_tool_call("") is None


def test_extract_tool_call_invalid_shape():
    # Valid JSON but no `tool` field -> reject.
    out = _extract_tool_call('{"plan": "do stuff"}')
    assert out is None
