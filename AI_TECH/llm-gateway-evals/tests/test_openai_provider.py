"""Real OpenAI SDK against a mocked HTTP transport: request shape, parsing and error mapping.

This does NOT call OpenAI, so it says nothing about live behaviour, quality or latency.
"""
import json

import httpx
import pytest
from openai import OpenAI

from app.config import Settings
from app.gateway import Gateway
from app.providers import OpenAIProvider, ProviderError
from app.usage import UsageStore

from .conftest import make_request

OK_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4.1-mini",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi there"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}


def provider_for(handler) -> OpenAIProvider:
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    return OpenAIProvider(client)


def test_success_maps_text_and_token_usage():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK_BODY)

    result = provider_for(handler).chat(make_request(json_mode=True))
    assert (result.text, result.input_tokens, result.output_tokens) == ("hi there", 12, 3)
    assert seen["response_format"] == {"type": "json_object"} and seen["max_tokens"] == 100


@pytest.mark.parametrize("status,retryable", [(429, True), (500, True), (503, True), (400, False), (401, False)])
def test_http_errors_are_classified_as_retryable_or_not(status, retryable):
    provider = provider_for(lambda request: httpx.Response(status, json={"error": {"message": "x", "type": "t"}}))
    with pytest.raises(ProviderError) as info:
        provider.chat(make_request())
    assert info.value.retryable is retryable


def test_connection_failure_is_retryable():
    def handler(request):
        raise httpx.ConnectError("network down")

    with pytest.raises(ProviderError) as info:
        provider_for(handler).chat(make_request())
    assert info.value.retryable is True


def test_gateway_recovers_from_a_429_through_the_real_sdk():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down", "type": "rate_limit"}})
        return httpx.Response(200, json=OK_BODY)

    sleeps: list[float] = []
    gw = Gateway(provider_for(handler), Settings(max_retries=2, backoff_base_seconds=0.5), UsageStore(":memory:"),
                 sleep=sleeps.append)
    response = gw.chat("team-a", make_request())
    assert response.text == "hi there" and response.attempts == 2 and sleeps == [0.5]
    assert response.cost_usd == pytest.approx((12 * 0.40 + 3 * 1.60) / 1_000_000)
