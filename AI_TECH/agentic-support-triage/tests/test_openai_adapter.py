"""Runs the real OpenAI SDK against a mocked HTTP transport (no network, no live-model claims)."""
import json

import httpx
import pytest
from openai import OpenAI

from app.llm import OpenAIJSON


def client_returning(content: str, captured: dict | None = None) -> OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}})

    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_json_mode_request_and_parse():
    captured: dict = {}
    llm = OpenAIJSON(client_returning('{"category": "billing"}', captured), "gpt-4.1-mini")
    assert llm.complete_json("sys", "user") == {"category": "billing"}
    assert captured["response_format"] == {"type": "json_object"} and captured["temperature"] == 0.0


def test_non_object_json_is_rejected():
    with pytest.raises(ValueError):
        OpenAIJSON(client_returning("[1, 2]")).complete_json("sys", "user")


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        OpenAIJSON(client_returning("not json")).complete_json("sys", "user")
