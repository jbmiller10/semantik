from webui.config.rate_limits import RateLimitConfig


def test_get_rate_limit_string_defaults():
    assert RateLimitConfig.get_rate_limit_string("unknown") == RateLimitConfig.DEFAULT_LIMIT


def test_bypass_rate_limit_returns_false_without_token(monkeypatch):
    monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", None)

    assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer token"}) is False


def test_bypass_rate_limit_accepts_matching_token(monkeypatch):
    monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")

    assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer secret"}) is True


def test_bypass_rate_limit_rejects_mismatched_token(monkeypatch):
    monkeypatch.setattr(RateLimitConfig, "BYPASS_TOKEN", "secret")

    assert RateLimitConfig.bypass_rate_limit({"authorization": "Bearer nope"}) is False
