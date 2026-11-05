"""JSON-returning chat model interface used by the graph."""
from __future__ import annotations

import json
from typing import Protocol


class JSONLLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class OpenAIJSON:
    def __init__(self, client, model: str = "gpt-4.1-mini", temperature: float = 0.0, max_tokens: int = 500) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete_json(self, system: str, user: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(data, dict):
            raise ValueError("model did not return a JSON object")
        return data
