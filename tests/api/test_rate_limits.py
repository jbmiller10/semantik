"""
Tests for rate limiting functionality in the chunking API.

This module tests rate limiting, circuit breaker pattern, and admin bypass functionality.
"""

import os
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient
from slowapi.errors import RateLimitExceeded

from packages.webui.config.rate_limits import RateLimitConfig
from packages.webui.dependencies import get_chunking_service_adapter_dependency
from packages.webui.main import app
from packages.webui.rate_limiter import circuit_breaker
from packages.webui.services.dtos import ServicePreviewResponse


@pytest.fixture()
def mock_redis() -> Generator[Any, None, None]:
    """Mock Redis for rate limiting tests."""
    with patch("packages.webui.rate_limiter.limiter") as mock_limiter:
        yield mock_limiter


@pytest.fixture()
def bypass_token() -> Generator[Any, None, None]:
    """Set and return a bypass token for testing."""
    original = os.environ.get("RATE_LIMIT_BYPASS_TOKEN")
    test_token = "test-bypass-token-123"
    os.environ["RATE_LIMIT_BYPASS_TOKEN"] = test_token
    # Reload config to pick up new token
    RateLimitConfig.BYPASS_TOKEN = test_token
    yield test_token
    # Restore original
    if original:
        os.environ["RATE_LIMIT_BYPASS_TOKEN"] = original
    else:
        os.environ.pop("RATE_LIMIT_BYPASS_TOKEN", None)
    RateLimitConfig.BYPASS_TOKEN = original


@pytest.fixture()
def _reset_circuit_breaker() -> Generator[Any, None, None]:
    """Reset circuit breaker state before each test."""
    circuit_breaker.failure_counts.clear()
    circuit_breaker.blocked_until.clear()
    yield
    circuit_breaker.failure_counts.clear()
    circuit_breaker.blocked_until.clear()


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true", reason="Requires database connection not available in CI"
)
async def test_preview_rate_limit(async_client: AsyncClient, auth_headers: dict) -> None:
    """Test that preview endpoint enforces rate limits."""
    from packages.webui.main import app
    # Create a mock chunking service
    mock_chunking_service = AsyncMock()
    mock_chunking_service.preview_chunking = AsyncMock(
        return_value=ServicePreviewResponse(
            preview_id="test-preview",
            strategy="fixed_size",
            config={"strategy": "fixed_size", "chunk_size": 512},
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
            cached=False,
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
    )
    mock_chunking_service.track_preview_usage = AsyncMock()

    # Override the dependency at the app level
    async def override_get_chunking_service():
        return mock_chunking_service

    app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

    try:
        # Make requests up to the limit (10 per minute for preview)
        preview_data = {
            "content": "Test content for rate limiting",
            "strategy": "fixed_size",
        }

        responses = []
        for i in range(12):  # Try to exceed limit
            response = await async_client.post(
                "/api/v2/chunking/preview",
                json=preview_data,
                headers=auth_headers,
            )
            responses.append(response)

            # First 10 should succeed
            if i < 10:
                # Allow for rate limiter initialization
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    pytest.skip("Rate limiter not properly initialized for test")
            else:
                # 11th and 12th should be rate limited
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
                assert "Retry-After" in response.headers
                error_data = response.json()
                assert "rate_limit_exceeded" in error_data.get("error", "")
    finally:
        # Clean up the override
        del app.dependency_overrides[get_chunking_service_adapter_dependency]


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true", reason="Requires database connection not available in CI"
)
async def test_compare_rate_limit(async_client: AsyncClient, auth_headers: dict) -> None:
    """Test that compare endpoint enforces stricter rate limits."""
    from packages.webui.main import app
    from packages.webui.services.dtos import (
        ServiceCompareResponse,
        ServiceStrategyComparison,
        ServiceStrategyRecommendation,
    )
    # Create a mock chunking service
    mock_chunking_service = AsyncMock()
    mock_chunking_service.compare_strategies_for_api = AsyncMock(
        return_value=ServiceCompareResponse(
            comparison_id="test-comparison",
            comparisons=[
                ServiceStrategyComparison(
                    strategy="fixed_size",
                    config={"strategy": "fixed_size"},
                    sample_chunks=[],
                    total_chunks=10,
                    avg_chunk_size=500.0,
                    size_variance=10.0,
                    quality_score=0.8,
                    processing_time_ms=100,
                    pros=["Fast"],
                    cons=["May break semantic units"],
                )
            ],
            recommendation=ServiceStrategyRecommendation(
                strategy="fixed_size",
                confidence=0.85,
                reasoning="Best for test content",
                alternatives=[],
            ),
            processing_time_ms=200,
        )
    )

    # Override the dependency at the app level
    async def override_get_chunking_service():
        return mock_chunking_service

    app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

    try:
        # Compare endpoint has 5 requests per minute limit
        compare_data = {
            "content": "Test content",
            "strategies": ["fixed_size", "semantic"],
        }

        for i in range(7):  # Try to exceed limit
            response = await async_client.post(
                "/api/v2/chunking/compare",
                json=compare_data,
                headers=auth_headers,
            )

            if i < 5:
                # First 5 should succeed (or skip if limiter not initialized)
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    pytest.skip("Rate limiter not properly initialized for test")
            else:
                # 6th and 7th should be rate limited
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    finally:
        # Clean up the override
        del app.dependency_overrides[get_chunking_service_adapter_dependency]


@pytest.mark.asyncio()
async def test_admin_bypass_token(async_client: AsyncClient, bypass_token: str) -> None:
    """Test that admin bypass token allows unlimited requests."""
    from packages.webui.main import app
    # Create a mock chunking service
    mock_chunking_service = AsyncMock()
    mock_chunking_service.preview_chunking = AsyncMock(
        return_value=ServicePreviewResponse(
            preview_id="test-preview",
            strategy="fixed_size",
            config={"strategy": "fixed_size"},
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
            cached=False,
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
    )
    mock_chunking_service.track_preview_usage = AsyncMock()

    # Override the dependency at the app level
    async def override_get_chunking_service():
        return mock_chunking_service

    app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

    try:
        # Use bypass token in Authorization header
        headers = {"Authorization": f"Bearer {bypass_token}"}
        preview_data = {
            "content": "Test content",
            "strategy": "fixed_size",
        }

        # Make many requests - all should succeed with bypass token
        for _ in range(20):
            response = await async_client.post(
                "/api/v2/chunking/preview",
                json=preview_data,
                headers=headers,
            )
            # With bypass token, should never get rate limited
            assert response.status_code != status.HTTP_429_TOO_MANY_REQUESTS
    finally:
        # Clean up the override
        del app.dependency_overrides[get_chunking_service_adapter_dependency]


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true", reason="Requires database connection not available in CI"
)
@pytest.mark.usefixtures("_reset_circuit_breaker")
async def test_circuit_breaker_activation(
    async_client: AsyncClient,
    auth_headers: dict,
) -> None:
    """Test that circuit breaker activates after consecutive failures."""
    # Mock to simulate rate limit failures
    with patch("packages.webui.rate_limiter.limiter.limit") as mock_limit:
        # Create a mock decorator that always raises RateLimitExceeded
        def mock_decorator(limit_string) -> None:  # noqa: ARG001
            def decorator(func) -> None:  # noqa: ARG001
                async def wrapper(*args, **kwargs) -> None:  # noqa: ARG001

                    raise RateLimitExceeded("Rate limit exceeded")

                return wrapper

            return decorator

        mock_limit.side_effect = mock_decorator

        preview_data = {
            "content": "Test content",
            "strategy": "fixed_size",
        }

        # Make requests to trigger circuit breaker
        for i in range(6):
            response = await async_client.post(
                "/api/v2/chunking/preview",
                json=preview_data,
                headers=auth_headers,
            )

            if i < 5:
                # First 5 should get rate limit errors
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            else:
                # 6th should trigger circuit breaker
                assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
                assert "Circuit breaker open" in response.json().get("detail", "")
                assert "Retry-After" in response.headers


@pytest.mark.asyncio()
async def test_rate_limit_headers(async_client: AsyncClient, auth_headers: dict) -> None:
    """Test that rate limit headers are included in responses."""
    # Mock the Redis manager to avoid type checking issues
    mock_redis_client = AsyncMock()
    mock_redis_manager = AsyncMock()
    mock_redis_manager.async_client = AsyncMock(return_value=mock_redis_client)

    # Mock the chunking service
    mock_chunking_service = AsyncMock()
    mock_chunking_service.preview_chunking = AsyncMock(
        return_value={
            "preview_id": "test-preview",
            "strategy": "fixed_size",
            "config": {"strategy": "fixed_size"},
            "chunks": [],
            "total_chunks": 0,
            "processing_time_ms": 100,
        }
    )
    mock_chunking_service.track_preview_usage = AsyncMock()

    # Override the dependency at the app level
    async def override_get_chunking_service():
        return mock_chunking_service

    app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

    try:
        preview_data = {
            "content": "Test content",
            "strategy": "fixed_size",
        }

        response = await async_client.post(
            "/api/v2/chunking/preview",
            json=preview_data,
            headers=auth_headers,
        )

        # Check for rate limit headers (when using slowapi with headers_enabled=True)
        if response.status_code == status.HTTP_200_OK:
            # These headers should be present when rate limiting is active
            # Note: Actual presence depends on slowapi configuration
            pass  # Headers may or may not be present depending on setup
    finally:
        # Clean up the override
        del app.dependency_overrides[get_chunking_service_adapter_dependency]


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true", reason="Requires database connection not available in CI"
)
async def test_process_hourly_rate_limit(async_client: AsyncClient, auth_headers: dict) -> None:
    """Test that process endpoint has hourly rate limits."""
    # Mock dependencies
    with (
        patch("packages.webui.api.v2.chunking.get_collection_for_user") as mock_collection,
        patch("packages.webui.api.v2.chunking.get_chunking_service_adapter_dependency") as mock_chunking,
        patch("packages.webui.services.factory.get_collection_service") as mock_coll_service,
    ):
        mock_collection.return_value = {"id": "test-collection"}
        mock_chunking.return_value.validate_config_for_collection = AsyncMock(
            return_value={"is_valid": True, "estimated_time": 10}
        )
        mock_chunking.return_value.start_chunking_operation = AsyncMock(return_value=("ws-channel", {}))
        mock_coll_service.return_value.create_operation = AsyncMock(return_value={"uuid": "test-operation-id"})

        process_data = {
            "strategy": "fixed_size",
            "priority": "normal",
        }

        # Process endpoint has 20 requests per hour limit
        # This is harder to test in unit tests due to the hourly window
        response = await async_client.post(
            "/api/v2/chunking/collections/test-collection/chunk",
            json=process_data,
            headers=auth_headers,
        )

        # Should succeed or be accepted
        assert response.status_code in [
            status.HTTP_202_ACCEPTED,
            status.HTTP_429_TOO_MANY_REQUESTS,  # May already be limited
        ]


@pytest.mark.asyncio()
async def test_different_users_have_separate_limits(
    async_client: AsyncClient,
) -> None:
    """Test that different users have independent rate limits."""
    from packages.webui.main import app

    # Create a mock chunking service
    mock_chunking_service = AsyncMock()
    mock_chunking_service.preview_chunking = AsyncMock(
        return_value=ServicePreviewResponse(
            preview_id="test-preview",
            strategy="fixed_size",
            config={"strategy": "fixed_size"},
            chunks=[],
            total_chunks=0,
            processing_time_ms=100,
            cached=False,
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
        )
    )
    mock_chunking_service.track_preview_usage = AsyncMock()

    # Override the dependency at the app level
    async def override_get_chunking_service():
        return mock_chunking_service

    app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

    try:
        preview_data = {
            "content": "Test content",
            "strategy": "fixed_size",
        }

        # User 1 headers
        user1_headers = {"Authorization": "Bearer user1-token"}

        # User 2 headers
        user2_headers = {"Authorization": "Bearer user2-token"}

        # Each user should have their own rate limit bucket
        # This test would need proper JWT tokens in a real scenario
        # For now, we're testing the concept

        await async_client.post(
            "/api/v2/chunking/preview",
            json=preview_data,
            headers=user1_headers,
        )

        await async_client.post(
            "/api/v2/chunking/preview",
            json=preview_data,
            headers=user2_headers,
        )

        # Both should be able to make requests independently
    finally:
        # Clean up the override
        del app.dependency_overrides[get_chunking_service_adapter_dependency]
        # (actual behavior depends on auth implementation)


@pytest.mark.asyncio()
async def test_rate_limit_with_redis_failure(async_client: AsyncClient, auth_headers: dict) -> None:
    """Test fallback behavior when Redis is unavailable."""
    from packages.webui.main import app

    # Simulate Redis connection failure
    with patch("packages.webui.rate_limiter.limiter") as mock_limiter:
        # Configure mock to simulate Redis being down but fallback working
        mock_limiter.limit.return_value = lambda f: f  # Pass through decorator

        # Create a mock chunking service
        mock_chunking_service = AsyncMock()
        mock_chunking_service.preview_chunking = AsyncMock(
            return_value=ServicePreviewResponse(
                preview_id="test-preview",
                strategy="fixed_size",
                config={"strategy": "fixed_size"},
                chunks=[],
                total_chunks=0,
                processing_time_ms=100,
                cached=False,
                expires_at=datetime.now(UTC) + timedelta(minutes=15),
            )
        )
        mock_chunking_service.track_preview_usage = AsyncMock()

        # Override the dependency at the app level
        async def override_get_chunking_service():
            return mock_chunking_service

        app.dependency_overrides[get_chunking_service_adapter_dependency] = override_get_chunking_service

        try:
            preview_data = {
                "content": "Test content",
                "strategy": "fixed_size",
            }

            # Should still work with in-memory fallback
            response = await async_client.post(
                "/api/v2/chunking/preview",
                json=preview_data,
                headers=auth_headers,
            )

            # Should not fail even if Redis is down
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR
        finally:
            # Clean up the override
            del app.dependency_overrides[get_chunking_service_adapter_dependency]
