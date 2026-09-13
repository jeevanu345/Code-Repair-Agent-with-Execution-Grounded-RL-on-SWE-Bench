"""Integration: requires a running Docker daemon and a built sandbox image."""

import pytest

pytestmark = pytest.mark.integration


def _docker_or_skip():
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        pytest.skip(f"docker not reachable: {e}")
    try:
        client.images.get("swe-rl/sandbox:latest")
    except Exception:
        pytest.skip("sandbox image missing — run `make build-sandbox`")


def test_start_exec_teardown():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        r = runner.exec(h, "echo hello && python -c 'print(2+2)'")
        assert r.exit_code == 0
        assert "hello" in r.stdout
        assert "4" in r.stdout


def test_safety_blocks_rm_rf():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        r = runner.exec(h, "rm -rf /")
        assert r.safety_blocked
        assert r.exit_code != 0


def test_write_and_read_file():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        runner.write_file(h, "/tmp/hello.txt", "world\n")
        out = runner.read_file(h, "/tmp/hello.txt")
        assert out == b"world\n"


def test_timeout_terminates_exec():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        result = runner.exec(h, "sleep 20", timeout=1)
        assert result.timed_out
        assert result.duration_s < 5


def test_resource_and_host_isolation_configuration():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        container = runner.client.containers.get(h.container_id)
        config = container.attrs["HostConfig"]
        assert config["Memory"] > 0
        assert config["NanoCpus"] > 0
        assert config["PidsLimit"] == 512
        assert config["CapDrop"] == ["ALL"]
        assert config["Privileged"] is False
        assert not config["Binds"]


def test_network_can_be_verified_disconnected():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner

    runner = DockerRunner()
    with runner.session() as h:
        runner.disable_network(h)
        container = runner.client.containers.get(h.container_id)
        container.reload()
        assert container.attrs["NetworkSettings"]["Networks"] == {}


def test_execution_grounded_test_parser_after_network_is_disabled():
    _docker_or_skip()
    from swe_rl.sandbox.docker_runner import DockerRunner
    from swe_rl.sandbox.test_executor import TestExecutor

    runner = DockerRunner()
    with runner.session() as h:
        runner.exec(h, "mkdir -p /workspace/repo", check_safety=False)
        runner.write_file(
            h,
            "/workspace/repo/test_sample.py",
            "def test_pass():\n    assert True\n\ndef test_fail():\n    assert False\n",
        )
        runner.disable_network(h)
        results = TestExecutor(runner).run(
            h, ["test_sample.py::test_pass", "test_sample.py::test_fail"]
        )
        assert results.outcomes == {
            "test_sample.py::test_pass": "passed",
            "test_sample.py::test_fail": "failed",
        }
