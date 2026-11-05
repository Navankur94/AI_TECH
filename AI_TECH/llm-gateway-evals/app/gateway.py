"""Gateway: rate limit -> cache -> provider with retries -> optional fallback -> usage log."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

from .cache import TTLCache, request_key
from .config import Settings
from .pricing import PriceTable
from .providers import ChatRequest, Provider, ProviderError
from .ratelimit import SlidingWindowLimiter
from .usage import UsageRecord, UsageStore


class RateLimited(Exception):
    pass


class ModelNotAllowed(Exception):
    pass


class UpstreamError(Exception):
    pass


@dataclass(frozen=True)
class GatewayResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    attempts: int
    used_fallback: bool

    def to_dict(self) -> dict:
        return asdict(self)


class Gateway:
    def __init__(
        self,
        provider: Provider,
        settings: Settings,
        usage: UsageStore,
        cache: TTLCache | None = None,
        limiter: SlidingWindowLimiter | None = None,
        fallback: tuple[Provider, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.usage = usage
        self.cache = cache
        self.limiter = limiter
        self.fallback = fallback
        self.sleep = sleep
        self.clock = clock
        self.prices = PriceTable(settings.pricing)
        self.prices.require(settings.allowed_models)

    def chat(self, client: str, request: ChatRequest) -> GatewayResponse:
        if request.model not in self.settings.allowed_models:
            raise ModelNotAllowed(request.model)
        if self.limiter and not self.limiter.allow(client):
            self.usage.log(UsageRecord(client, request.model, "rate_limited"))
            raise RateLimited(client)

        started = self.clock()
        cacheable = self.cache is not None and request.temperature == 0
        key = request_key(request) if cacheable else ""
        if cacheable:
            cached = self.cache.get(key)  # type: ignore[union-attr]
            if cached is not None:
                latency = (self.clock() - started) * 1000
                self.usage.log(UsageRecord(client, request.model, "ok", cache_hit=True, latency_ms=latency))
                return GatewayResponse(cached.text, cached.model, 0, 0, 0.0, latency, True, 0, False)

        try:
            result, attempts, used_fallback = self._call_with_retries(request)
        except ProviderError as exc:
            latency = (self.clock() - started) * 1000
            self.usage.log(UsageRecord(client, request.model, "error", latency_ms=latency,
                                       attempts=getattr(exc, "attempts", 1)))
            raise UpstreamError(str(exc)) from exc

        latency = (self.clock() - started) * 1000
        model_for_price = self.fallback[1] if used_fallback and self.fallback else request.model
        cost = self.prices.cost(model_for_price, result.input_tokens, result.output_tokens) \
            if model_for_price in self.settings.pricing else 0.0
        response = GatewayResponse(result.text, result.model, result.input_tokens, result.output_tokens,
                                   cost, latency, False, attempts, used_fallback)
        self.usage.log(UsageRecord(client, model_for_price, "ok", result.input_tokens, result.output_tokens,
                                   cost, latency, False, attempts))
        if cacheable:
            self.cache.set(key, result)  # type: ignore[union-attr]
        return response

    def _call_with_retries(self, request: ChatRequest):
        last_error: ProviderError | None = None
        attempts = 0
        for attempt in range(self.settings.max_retries + 1):
            attempts += 1
            try:
                return self.provider.chat(request), attempts, False
            except ProviderError as exc:
                exc.attempts = attempts  # type: ignore[attr-defined]
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt < self.settings.max_retries:
                    self.sleep(self.settings.backoff_base_seconds * (2 ** attempt))
        if self.fallback is not None:
            provider, model = self.fallback
            fallback_request = ChatRequest(request.messages, model, request.temperature,
                                           request.max_tokens, request.json_mode)
            try:
                return provider.chat(fallback_request), attempts + 1, True
            except ProviderError as exc:
                exc.attempts = attempts + 1  # type: ignore[attr-defined]
                raise
        assert last_error is not None
        raise last_error
