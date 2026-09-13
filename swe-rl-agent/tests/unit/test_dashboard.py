from fastapi.testclient import TestClient

from swe_rl.dashboard.app import app


def test_dashboard_and_health_are_available():
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "SWE-RL Control Plane" in page.text
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["api"] == "available"


def test_dashboard_rejects_static_path_traversal():
    response = TestClient(app).get("/../pyproject.toml")
    assert "swe-rl-agent" not in response.text
