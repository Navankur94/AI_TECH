"""FastAPI entrypoint. Run: uvicorn app.main:app --host 0.0.0.0 --port 8000"""
from __future__ import annotations

import hmac
import logging
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .cache import TTLCache
from .config import Settings
from .gateway import Gateway, ModelNotAllowed, RateLimited, UpstreamError
from .providers import ChatRequest
from .ratelimit import SlidingWindowLimiter
from .usage import UsageStore

logger = logging.getLogger("gateway")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=20000)


class ChatBody(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    json_mode: bool = False


def build_gateway(settings: Settings) -> Gateway:
    from openai import OpenAI

    from .providers import OpenAIProvider

    client = OpenAI(api_key=settings.openai_api_key or "unset", timeout=settings.request_timeout, max_retries=0)
    return Gateway(
        OpenAIProvider(client),
        settings,
        UsageStore(settings.db_path),
        cache=TTLCache(settings.cache_ttl_seconds),
        limiter=SlidingWindowLimiter(settings.rate_limit_per_minute),
    )


def create_app(gateway: Gateway | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="LLM Gateway", version="1.0.0")
    holder: dict[str, Gateway | None] = {"gateway": gateway}

    def get_gateway() -> Gateway:
        if holder["gateway"] is None:
            holder["gateway"] = build_gateway(settings)
        return holder["gateway"]  # type: ignore[return-value]

    def client_name(x_api_key: str | None = Header(default=None)) -> str:
        if not settings.api_keys:
            raise HTTPException(status_code=503, detail="Server has no API_KEYS configured")
        if x_api_key:
            for secret, name in settings.api_keys.items():
                if hmac.compare_digest(x_api_key, secret):
                    return name
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat")
    def chat(body: ChatBody, client: str = Depends(client_name)):
        request = ChatRequest(
            messages=[m.model_dump() for m in body.messages],
            model=body.model or settings.default_model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            json_mode=body.json_mode,
        )
        try:
            return get_gateway().chat(client, request).to_dict()
        except ModelNotAllowed:
            raise HTTPException(status_code=400, detail=f"Model not allowed. Allowed: {list(settings.allowed_models)}")
        except RateLimited:
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})
        except UpstreamError:
            logger.exception("upstream failure")
            raise HTTPException(status_code=502, detail="Upstream provider error")

    @app.get("/v1/usage")
    def usage(client: str = Depends(client_name)):
        return {"client": client, **get_gateway().usage.summary(client)}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(client: str = Depends(client_name)):
        s = get_gateway().usage.summary()
        lines = [
            "# TYPE gateway_requests_total counter",
            f'gateway_requests_total{{status="ok"}} {s["ok"]}',
            f'gateway_requests_total{{status="error"}} {s["errors"]}',
            f'gateway_requests_total{{status="rate_limited"}} {s["rate_limited"]}',
            "# TYPE gateway_cache_hits_total counter",
            f"gateway_cache_hits_total {s['cache_hits']}",
            "# TYPE gateway_tokens_total counter",
            f'gateway_tokens_total{{direction="input"}} {s["input_tokens"]}',
            f'gateway_tokens_total{{direction="output"}} {s["output_tokens"]}',
            "# TYPE gateway_cost_usd_total counter",
            f"gateway_cost_usd_total {s['cost_usd']}",
        ]
        return "\n".join(lines) + "\n"

    return app


app = create_app()
