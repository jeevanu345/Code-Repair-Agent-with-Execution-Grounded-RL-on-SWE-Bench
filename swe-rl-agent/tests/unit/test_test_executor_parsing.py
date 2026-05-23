import json

from swe_rl.sandbox.test_executor import _parse_json_report, _parse_verbose


def test_parse_json_report():
    blob = json.dumps(
        {
            "tests": [
                {"nodeid": "tests/test_a.py::test_x", "outcome": "passed"},
                {"nodeid": "tests/test_a.py::test_y", "outcome": "failed"},
                {"nodeid": "tests/test_b.py::test_z", "outcome": "skipped"},
            ]
        }
    )
    out = _parse_json_report(blob)
    assert out["tests/test_a.py::test_x"] == "passed"
    assert out["tests/test_a.py::test_y"] == "failed"
    assert out["tests/test_b.py::test_z"] == "skipped"


def test_parse_verbose():
    stdout = """
tests/test_a.py::test_x PASSED                      [ 33%]
tests/test_a.py::test_y FAILED                      [ 66%]
tests/test_b.py::test_z SKIPPED (some reason)       [100%]
"""
    out = _parse_verbose(stdout)
    assert out["tests/test_a.py::test_x"] == "passed"
    assert out["tests/test_a.py::test_y"] == "failed"
    assert out["tests/test_b.py::test_z"] == "skipped"


def test_parse_json_report_invalid():
    assert _parse_json_report("not json") == {}
    assert _parse_json_report("") == {}
