"""Integration tests for the rate limiter using FastAPI endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI, Request
from httpx import AsyncClient
from slowapi.errors import RateLimitExceeded

from packages.webui.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture()
def rate_limited_app(monkeypatch):
    monkeypatch.setenv("DISABLE_RATE_LIMITING", "false")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/11")
    monkeypatch.setenv("RATE_LIMIT_BYPASS_TOKEN", "bypass-token")

    from packages.webui import rate_limiter

    importlib.reload(rate_limiter)

    app = FastAPI()
    app.state.limiter = rate_limiter.limiter
    app.add_exception_handler(RateLimitExceeded, rate_limiter.rate_limit_exceeded_handler)
    app.add_middleware(RateLimitMiddleware)

    @app.get("/limited")
    @app.state.limiter.limit("2/second")
    async def limited_endpoint(request: Request):  # noqa: ARG001
        return {"status": "ok"}

    yield app

    monkeypatch.setenv("DISABLE_RATE_LIMITING", "true")
    importlib.reload(rate_limiter)


@pytest.mark.asyncio()
async def test_rate_limiter_enforces_limits(rate_limited_app):
    async with AsyncClient(app=rate_limited_app, base_url="http://test") as client:
        first = await client.get("/limited")
        second = await client.get("/limited")
        third = await client.get("/limited")

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["error"] == "rate_limit_exceeded"


@pytest.mark.asyncio()
async def test_rate_limiter_bypass_token(rate_limited_app):
    async with AsyncClient(app=rate_limited_app, base_url="http://test") as client:
        for _ in range(3):
            response = await client.get("/limited", headers={"Authorization": "Bearer bypass-token"})
            assert response.status_code == 200
            assert response.json().get("status") == "ok"
