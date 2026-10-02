import json
import stat

import httpx
import pytest
from fastapi.testclient import TestClient

from swe_rl.agent.llm_client import LLMClient
from swe_rl.agent.providers import PROVIDERS, active_connection, connection, save_connection
from swe_rl.dashboard.app import app
from swe_rl.settings import settings


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "llm_profile_path", tmp_path / ".llm-profile.json")


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "anthropic"])
def test_compatible_provider_routing(provider):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "repair"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        )

    url = "http://127.0.0.1:8000" if provider in {"vllm", "custom"} else ""
    config = connection(provider, "account-model", "synthetic-test-key", url)
    client = LLMClient(config=config, transport=httpx.MockTransport(respond))
    result = client.chat([{"role": "user", "content": "fix"}], seed=17)
    client.close()
    assert str(seen[0].url) == config.base_url + "/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer synthetic-test-key"
    payload = json.loads(seen[0].content)
    assert payload["model"] == "account-model"
    assert payload.get("seed") == (17 if provider in {"vllm", "custom"} else None)
    assert result.text == "repair"
    assert (result.tokens_in, result.tokens_out) == (11, 3)


def test_anthropic_translation():
    def respond(request):
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-test-key"
        body = json.loads(request.content)
        assert body["system"] == "system rules"
        assert body["messages"] == [{"role": "user", "content": "issue"}]
        assert "top_p" not in body and "seed" not in body
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "text", "text": "fix"},
                ],
                "usage": {"input_tokens": 7, "output_tokens": 4},
                "stop_reason": "end_turn",
            },
        )

    client = LLMClient(
        config=connection("anthropic", "model", "synthetic-test-key"),
        transport=httpx.MockTransport(respond),
    )
    result = client.chat(
        [{"role": "system", "content": "system rules"}, {"role": "user", "content": "issue"}]
    )
    client.close()
    assert result.text == "fix"
    assert result.tokens_in == 7


def test_saved_profile_drives_default_client():
    config = connection("openai", "model", "synthetic-test-key")
    save_connection(config)
    assert active_connection() == config
    assert stat.S_IMODE(settings.llm_profile_path.stat().st_mode) == 0o600
    assert "synthetic-test-key" not in repr(config)
    client = LLMClient()
    assert client.provider == "openai" and client.model == "model"
    client.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.example/v1",
        "https://user:pass@example.com/v1",
        "https://example.com/v1?key=abc",
        "file:///tmp/x",
    ],
)
def test_reject_unsafe_endpoint(url):
    with pytest.raises(ValueError):
        connection("custom", "model", "", url)


def test_failure_does_not_echo_secret_response():
    client = LLMClient(
        config=connection("openai", "model", "synthetic-test-key"),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="synthetic-test-key private diagnostic")
        ),
    )
    with pytest.raises(RuntimeError, match="Invalid API key") as raised:
        client.chat([{"role": "user", "content": "test"}])
    assert "synthetic-test-key" not in str(raised.value)
    client.close()


def test_cloud_without_key_fails_before_http():
    def forbidden(request):
        raise AssertionError("Must not make request without key")

    client = LLMClient(
        config=connection("openai", "model"), transport=httpx.MockTransport(forbidden)
    )
    with pytest.raises(RuntimeError, match="configure"):
        client.chat([{"role": "user", "content": "test"}])
    client.close()


def test_profile_api_security_and_retention():
    client = TestClient(app)
    body = {"provider": "openai", "model": "model", "api_key": "synthetic-test-key"}
    assert client.post("/api/model/settings", json=body).status_code == 403
    token = client.get("/api/providers").json()["settings_token"]
    headers = {"x-settings-token": token}
    saved = client.post("/api/model/settings", json=body, headers=headers)
    assert saved.status_code == 200
    assert "synthetic-test-key" not in saved.text
    assert saved.json()["has_key"]
    assert "synthetic-test-key" not in client.get("/api/providers").text
    body["api_key"] = ""
    assert client.post("/api/model/settings", json=body, headers=headers).status_code == 200
    assert active_connection().api_key == "synthetic-test-key"
    headers["origin"] = "https://untrusted.example"
    assert client.post("/api/model/settings", json=body, headers=headers).status_code == 403
    del headers["origin"]
    assert client.delete("/api/model/settings", headers=headers).status_code == 200
    assert not settings.llm_profile_path.exists()


def test_invalid_input_does_not_reflect_key():
    client = TestClient(app)
    token = client.get("/api/providers").json()["settings_token"]
    response = client.post(
        "/api/model/settings",
        json={"provider": "openai", "model": {"secret": "synthetic-test-key"}},
        headers={"x-settings-token": token},
    )
    assert response.status_code == 422
    assert "synthetic-test-key" not in response.text


def test_assistant_uses_saved_provider(monkeypatch):
    save_connection(connection("openai", "model", "synthetic-test-key"))
    from swe_rl.agent.llm_client import LLMResponse

    seen = []

    def fake_chat(self, messages, **kwargs):
        seen.append((self.provider, self.model, messages[-1]["content"]))
        return LLMResponse("Fix the boundary", 10, 4, "stop", {})

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    client = TestClient(app)
    token = client.get("/api/providers").json()["settings_token"]
    response = client.post(
        "/api/model/assist", json={"prompt": "Fix this error"}, headers={"x-settings-token": token}
    )
    assert response.status_code == 200
    assert seen == [("openai", "model", "Fix this error")]
    assert response.json()["text"] == "Fix the boundary"
