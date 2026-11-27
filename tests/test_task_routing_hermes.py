"""Exercise the native Hermes router with a local, deterministic HTTP provider."""

import json

import pytest

pytest.importorskip("hermes_constants")

import httpx
import yaml

import tag_rewriter


@pytest.mark.parametrize("route,host,key_name", [
    ({"provider": "openai", "model": "gpt-6-luna"}, "api.openai.com", "OPENAI_API_KEY"),
    ({"provider": "custom", "model": "gpt-6-luna", "base_url": "https://voice-model.test/v1",
      "key_env": "VOICE_MODEL_KEY"}, "voice-model.test", "VOICE_MODEL_KEY"),
])
@pytest.mark.parametrize("status", [200, 503, "timeout"])
def test_native_task_route_uses_its_credentials_and_one_attempt(tmp_path, monkeypatch, route, host, key_name, status):
    from agent import auxiliary_client
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    source = "Bom, vamos conferir isso juntos?"
    tagged = "[warmly] Bom, vamos conferir isso juntos?"
    config = {
        "model": {"provider": "openai", "default": "gpt-6-luna"},
        "auxiliary": {
            tag_rewriter.AUXILIARY_TASK: route,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    requests = []

    def respond(request):
        requests.append(request)
        if status == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "at capacity"}})
        return httpx.Response(200, json={
            "id": "local-completion", "object": "chat.completion", "created": 1,
            "model": "gpt-6-luna",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": tagged},
                         "finish_reason": "stop"}],
        })

    original_factory = auxiliary_client._create_openai_client

    def client_factory(**kwargs):
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(respond))
        return original_factory(**kwargs)

    monkeypatch.setattr(auxiliary_client, "_create_openai_client", client_factory)
    home_scope = set_hermes_home_override(tmp_path)
    secret_scope = set_secret_scope({key_name: "synthetic-profile-key"}, profile_home=str(tmp_path))
    try:
        assert tag_rewriter.rewrite_audio_script(source, max_tags=2) == (tagged if status == 200 else source)
        assert len(requests) == 1
        request = requests[0]
        assert request.url.host == host
        assert request.headers["Authorization"] == "Bearer synthetic-profile-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-luna"
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        assert "temperature" not in body
        assert body["reasoning_effort"] == "none"
        assert "reasoning" not in body
        assert source in body["messages"][-1]["content"]
    finally:
        reset_secret_scope(secret_scope)
        reset_hermes_home_override(home_scope)
