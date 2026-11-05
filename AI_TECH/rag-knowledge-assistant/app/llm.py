"""Chat model wrapper. The service depends on this small interface, not on a vendor SDK."""
from __future__ import annotations

from typing import Protocol


class ChatLLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    def __init__(self, client, model: str = "gpt-4.1-mini", temperature: float = 0.1, max_tokens: int = 700) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()
