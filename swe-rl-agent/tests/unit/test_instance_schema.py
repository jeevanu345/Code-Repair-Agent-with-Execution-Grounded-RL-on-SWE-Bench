from swe_rl.data.instance_schema import SWEBenchInstance


def test_from_hf_row_minimal():
    row = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "deadbeef",
        "problem_statement": "fix the bug",
        "FAIL_TO_PASS": '["t1", "t2"]',
        "PASS_TO_PASS": ["a", "b"],
    }
    inst = SWEBenchInstance.from_hf_row(row)
    assert inst.instance_id == "django__django-12345"
    assert inst.fail_to_pass == ["t1", "t2"]
    assert inst.pass_to_pass == ["a", "b"]


def test_from_hf_row_extra_fields():
    row = {
        "instance_id": "x",
        "repo": "r",
        "base_commit": "c",
        "problem_statement": "p",
        "version": "1.0",
        "weird_field": 42,
    }
    inst = SWEBenchInstance.from_hf_row(row)
    assert inst.version == "1.0"
    assert inst.extra.get("weird_field") == 42
