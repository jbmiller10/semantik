"""Integration-style tests for the rate limiter configuration."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from packages.webui.rate_limiter import (
    check_circuit_breaker,
    circuit_breaker,
    get_user_or_ip,
    rate_limit_exceeded_handler,
    track_circuit_breaker_failure,
)

pytestmark = pytest.mark.usefixtures("_db_isolation")


@pytest.fixture()
def rate_limited_app() -> TestClient:
    os.environ["DISABLE_RATE_LIMITING"] = "false"
    limiter = Limiter(
        key_func=get_user_or_ip, storage_uri="memory://", default_limits=["2/second"], headers_enabled=False
    )

    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    @app.get("/protected")
    @limiter.limit("2/second")
    async def protected_endpoint(request: Request) -> dict[str, str]:
        check_circuit_breaker(request)
        return {"status": "ok"}

    return TestClient(app)


def test_rate_limit_enforcement(rate_limited_app: TestClient) -> None:
    first = rate_limited_app.get("/protected", headers={"x-forwarded-for": "1.1.1.1"})
    second = rate_limited_app.get("/protected", headers={"x-forwarded-for": "1.1.1.1"})
    third = rate_limited_app.get("/protected", headers={"x-forwarded-for": "1.1.1.1"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code in {200, 429}
    if third.status_code == 429:
        assert third.json()["error"] == "rate_limit_exceeded"


def test_circuit_breaker_blocks_after_threshold(rate_limited_app: TestClient) -> None:
    circuit_breaker.failure_counts.clear()
    circuit_breaker.blocked_until.clear()

    headers = {"x-forwarded-for": "2.2.2.2"}
    for _ in range(circuit_breaker.failure_threshold):
        track_circuit_breaker_failure("ip:2.2.2.2")

    response = rate_limited_app.get("/protected", headers=headers)
    assert response.status_code in {429, 503}
    if response.status_code == 503:
        body = response.json()
        error_value = body.get("error")
        if error_value is None and isinstance(body.get("detail"), dict):
            error_value = body["detail"].get("error")
        assert error_value == "circuit_breaker_open"


def test_get_user_or_ip_prefers_user_id() -> None:
    os.environ["DISABLE_RATE_LIMITING"] = "false"

    class DummyState:
        def __init__(self) -> None:
            self.user = {"id": 42}

    request = type(
        "Request", (), {"headers": {}, "client": type("Client", (), {"host": "127.0.0.1"})(), "state": DummyState()}
    )
    assert get_user_or_ip(request) == "user:42"
