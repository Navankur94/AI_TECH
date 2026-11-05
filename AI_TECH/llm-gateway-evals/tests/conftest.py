import pytest

from app.cache import TTLCache
from app.config import Settings
from app.gateway import Gateway
from app.providers import ChatRequest, FakeProvider
from app.ratelimit import SlidingWindowLimiter
from app.usage import UsageStore


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_request(text: str = "hello", **overrides) -> ChatRequest:
    base = dict(messages=[{"role": "user", "content": text}], model="gpt-4.1-mini", temperature=0.0, max_tokens=100)
    base.update(overrides)
    return ChatRequest(**base)


@pytest.fixture
def settings():
    return Settings(api_keys={"secret": "team-a"}, rate_limit_per_minute=5, max_retries=2, backoff_base_seconds=0.5)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def provider():
    return FakeProvider(input_tokens=1000, output_tokens=500)


@pytest.fixture
def gateway(provider, settings, clock, sleeps):
    return Gateway(
        provider, settings, UsageStore(":memory:"),
        cache=TTLCache(60, clock=clock), limiter=SlidingWindowLimiter(5, 60, clock=clock),
        sleep=sleeps.append,
    )
