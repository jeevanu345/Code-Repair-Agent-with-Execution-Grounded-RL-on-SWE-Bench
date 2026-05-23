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
