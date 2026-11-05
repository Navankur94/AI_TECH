"""Run the real OpenAI SDK against a mocked HTTP transport.

This proves request building and response parsing for the adapters. It does NOT prove
anything about the live OpenAI service, quality, or latency.
"""
import json

import httpx
from openai import OpenAI

from app.embeddings import OpenAIEmbedder
from app.llm import OpenAIChat


def make_client(handler):
    return OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_embedder_batches_and_restores_order():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body["input"])
        data = [{"object": "embedding", "index": i, "embedding": [float(len(t)), 0.0]} for i, t in enumerate(body["input"])]
        data.reverse()  # the API may return items out of order; the adapter must sort by index
        return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    embedder = OpenAIEmbedder(make_client(handler), "text-embedding-3-small", batch_size=2)
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert seen == [["a", "bb"], ["ccc"]]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_chat_adapter_sends_expected_request_and_parses_reply():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        captured["auth"] = request.headers["authorization"]
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4.1-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": " Hello [1] "}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})

    chat = OpenAIChat(make_client(handler), "gpt-4.1-mini", temperature=0.1, max_tokens=300)
    assert chat.complete("be brief", "hi") == "Hello [1]"
    assert captured["model"] == "gpt-4.1-mini" and captured["max_tokens"] == 300
    assert captured["messages"][0] == {"role": "system", "content": "be brief"}
    assert captured["auth"] == "Bearer test"
