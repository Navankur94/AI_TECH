import pytest

from app.gateway import Gateway, ModelNotAllowed, RateLimited, UpstreamError
from app.providers import FakeProvider, ProviderError
from app.usage import UsageStore

from .conftest import make_request


def test_success_computes_cost_from_token_counts(gateway):
    response = gateway.chat("team-a", make_request())
    assert response.text == "ok" and response.cache_hit is False
    assert response.cost_usd == pytest.approx(0.0012)  # (1000 * 0.40 + 500 * 1.60) / 1,000,000
    assert gateway.usage.summary("team-a")["cost_usd"] == pytest.approx(0.0012)


def test_identical_deterministic_request_is_served_from_cache(gateway, provider):
    first = gateway.chat("team-a", make_request())
    second = gateway.chat("team-a", make_request())
    assert len(provider.calls) == 1
    assert second.cache_hit is True and second.cost_usd == 0.0 and second.text == first.text
    assert gateway.usage.summary()["cache_hits"] == 1


def test_cache_entry_expires(gateway, provider, clock):
    gateway.chat("team-a", make_request())
    clock.now += 61
    gateway.chat("team-a", make_request())
    assert len(provider.calls) == 2


def test_non_deterministic_requests_are_not_cached(gateway, provider):
    gateway.chat("team-a", make_request(temperature=0.7))
    gateway.chat("team-a", make_request(temperature=0.7))
    assert len(provider.calls) == 2


def test_retryable_errors_back_off_exponentially_then_succeed(gateway, provider, sleeps):
    provider.errors = [ProviderError("429", True), ProviderError("503", True)]
    response = gateway.chat("team-a", make_request())
    assert response.attempts == 3 and sleeps == [0.5, 1.0]


def test_exhausted_retries_raise_upstream_error_and_are_logged(gateway, provider, sleeps):
    provider.errors = [ProviderError("down", True)] * 3
    with pytest.raises(UpstreamError):
        gateway.chat("team-a", make_request())
    summary = gateway.usage.summary("team-a")
    assert summary["errors"] == 1 and len(provider.calls) == 3 and sleeps == [0.5, 1.0]


def test_non_retryable_error_is_not_retried(gateway, provider, sleeps):
    provider.errors = [ProviderError("400 bad request", False)]
    with pytest.raises(UpstreamError):
        gateway.chat("team-a", make_request())
    assert len(provider.calls) == 1 and sleeps == []


def test_fallback_provider_used_after_retries_fail(provider, settings, sleeps):
    provider.errors = [ProviderError("down", True)] * 3
    backup = FakeProvider(responder=lambda r: "from backup")
    gw = Gateway(provider, settings, UsageStore(":memory:"), fallback=(backup, "gpt-4.1-mini"), sleep=sleeps.append)
    response = gw.chat("team-a", make_request())
    assert response.text == "from backup" and response.used_fallback is True and response.attempts == 4


def test_non_retryable_error_skips_the_fallback(provider, settings):
    provider.errors = [ProviderError("401", False)]
    backup = FakeProvider()
    gw = Gateway(provider, settings, UsageStore(":memory:"), fallback=(backup, "gpt-4.1-mini"), sleep=lambda s: None)
    with pytest.raises(UpstreamError):
        gw.chat("team-a", make_request())
    assert backup.calls == []


def test_rate_limit_blocks_then_recovers(gateway, clock):
    for i in range(5):
        gateway.chat("team-a", make_request(f"q{i}"))
    with pytest.raises(RateLimited):
        gateway.chat("team-a", make_request("q-extra"))
    assert gateway.usage.summary("team-a")["rate_limited"] == 1
    gateway.chat("team-b", make_request("other client"))  # limits are per client
    clock.now += 61
    gateway.chat("team-a", make_request("after window"))


def test_disallowed_model_is_rejected_before_any_call(gateway, provider):
    with pytest.raises(ModelNotAllowed):
        gateway.chat("team-a", make_request(model="gpt-unknown"))
    assert provider.calls == []


def test_allowed_model_without_a_price_fails_at_startup(provider, settings):
    from dataclasses import replace

    bad = replace(settings, allowed_models=("gpt-4.1-mini", "mystery-model"))
    with pytest.raises(ValueError, match="mystery-model"):
        Gateway(provider, bad, UsageStore(":memory:"))
