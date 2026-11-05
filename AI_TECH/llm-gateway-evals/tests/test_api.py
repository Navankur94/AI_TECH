from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.providers import ProviderError

HEADERS = {"x-api-key": "secret"}
BODY = {"messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def client(gateway, settings):
    return TestClient(create_app(gateway, settings))


def test_health_is_open_other_routes_need_a_key(client):
    assert client.get("/health").status_code == 200
    assert client.post("/v1/chat", json=BODY).status_code == 401
    assert client.get("/v1/usage", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/metrics").status_code == 401


def test_server_without_keys_refuses(gateway, settings):
    app = create_app(gateway, replace(settings, api_keys={}))
    assert TestClient(app).post("/v1/chat", headers=HEADERS, json=BODY).status_code == 503


def test_chat_returns_text_tokens_and_cost(client):
    r = client.post("/v1/chat", headers=HEADERS, json=BODY)
    data = r.json()
    assert r.status_code == 200 and data["text"] == "ok"
    assert data["input_tokens"] == 1000 and data["cost_usd"] == pytest.approx(0.0012)


def test_usage_and_metrics_reflect_traffic(client):
    client.post("/v1/chat", headers=HEADERS, json=BODY)
    client.post("/v1/chat", headers=HEADERS, json=BODY)  # cache hit
    usage = client.get("/v1/usage", headers=HEADERS).json()
    assert usage["client"] == "team-a" and usage["requests"] == 2 and usage["cache_hits"] == 1
    text = client.get("/metrics", headers=HEADERS).text
    assert 'gateway_requests_total{status="ok"} 2' in text and "gateway_cache_hits_total 1" in text


def test_validation_errors(client):
    assert client.post("/v1/chat", headers=HEADERS, json={"messages": []}).status_code == 422
    bad_role = {"messages": [{"role": "root", "content": "x"}]}
    assert client.post("/v1/chat", headers=HEADERS, json=bad_role).status_code == 422
    assert client.post("/v1/chat", headers=HEADERS, json={**BODY, "temperature": 5}).status_code == 422


def test_disallowed_model_is_400(client):
    r = client.post("/v1/chat", headers=HEADERS, json={**BODY, "model": "other"})
    assert r.status_code == 400


def test_rate_limit_is_429_with_retry_after(client):
    for i in range(5):
        client.post("/v1/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": f"q{i}"}]})
    r = client.post("/v1/chat", headers=HEADERS, json={"messages": [{"role": "user", "content": "q-extra"}]})
    assert r.status_code == 429 and r.headers["retry-after"] == "60"


def test_provider_outage_is_502(gateway, settings, provider):
    provider.errors = [ProviderError("down", True)] * 3
    r = TestClient(create_app(gateway, settings)).post("/v1/chat", headers=HEADERS, json=BODY)
    assert r.status_code == 502 and "down" not in r.text  # provider details are not leaked
