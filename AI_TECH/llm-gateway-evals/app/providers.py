"""Provider interface, the OpenAI implementation, and a scriptable fake for tests and demos."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]
    model: str
    temperature: float = 0.0
    max_tokens: int = 500
    json_mode: bool = False


@dataclass(frozen=True)
class ProviderResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Provider(Protocol):
    def chat(self, request: ChatRequest) -> ProviderResult: ...


class OpenAIProvider:
    def __init__(self, client) -> None:
        self.client = client

    def chat(self, request: ChatRequest) -> ProviderResult:
        import openai

        kwargs: dict = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
            # APITimeoutError is a subclass of APIConnectionError
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=True) from exc
        except openai.OpenAIError as exc:  # bad request, auth, not found: retrying cannot help
            raise ProviderError(f"{type(exc).__name__}: {exc}", retryable=False) from exc
        usage = response.usage
        return ProviderResult(
            text=(response.choices[0].message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=response.model or request.model,
        )


@dataclass
class FakeProvider:
    """Scripted provider: `errors` are raised first (one per call), then `responder` answers."""

    responder: Callable[[ChatRequest], str] = lambda req: "ok"
    errors: list[ProviderError] = field(default_factory=list)
    input_tokens: int = 100
    output_tokens: int = 50
    calls: list[ChatRequest] = field(default_factory=list)

    def chat(self, request: ChatRequest) -> ProviderResult:
        self.calls.append(request)
        if self.errors:
            raise self.errors.pop(0)
        return ProviderResult(self.responder(request), self.input_tokens, self.output_tokens, request.model)
