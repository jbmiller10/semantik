"""
Rate limiting implementation for the webui API.

This module provides distributed rate limiting using Redis backend,
circuit breaker pattern, and admin bypass functionality.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from packages.webui.config.rate_limits import CircuitBreakerConfig, RateLimitConfig

logger = logging.getLogger(__name__)


# Circuit breaker instance
circuit_breaker = CircuitBreakerConfig()


def get_user_or_ip(request: Request) -> str:
    """
    Get rate limit key based on user ID or IP address.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit key (user_id or IP address)
    """
    # Check for admin bypass token first
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer ") and RateLimitConfig.BYPASS_TOKEN:
        token = auth_header.split(" ", 1)[1]
        if token == RateLimitConfig.BYPASS_TOKEN:
            # Return a special key that has unlimited rate limit
            return "admin_bypass"

    # Try to get user from request state (set by auth middleware)
    if hasattr(request.state, "user") and request.state.user:
        user = request.state.user
        if isinstance(user, dict) and "id" in user:
            return f"user:{user['id']}"

    # Fallback to IP address
    return get_remote_address(request)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.

    Args:
        request: FastAPI request object
        exc: RateLimitExceeded exception

    Returns:
        JSONResponse with 429 status and proper headers
    """
    # Extract retry_after from the exception message
    retry_after = 60  # Default to 60 seconds
    try:
        # The exception message usually contains the retry time
        if hasattr(exc, "retry_after"):
            retry_after = int(exc.retry_after)
    except (AttributeError, ValueError):
        pass

    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": "Rate limit exceeded",
            "error": "rate_limit_exceeded",
            "retry_after": retry_after,
            "limit": str(getattr(exc, "limit", "unknown")),
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(getattr(exc, "limit", "unknown")),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + retry_after),
        },
    )

    # Track circuit breaker failures
    key = get_user_or_ip(request)
    if key != "admin_bypass":
        track_circuit_breaker_failure(key)

    return response


def track_circuit_breaker_failure(key: str) -> None:
    """
    Track rate limit failures for circuit breaker pattern.

    Args:
        key: User or IP key that hit rate limit
    """
    current_time = time.time()

    # Check if already blocked
    if key in circuit_breaker.blocked_until:
        if current_time < circuit_breaker.blocked_until[key]:
            return  # Still blocked
        # Unblock and reset counter
        del circuit_breaker.blocked_until[key]
        circuit_breaker.failure_counts[key] = 0

    # Increment failure count
    circuit_breaker.failure_counts[key] = circuit_breaker.failure_counts.get(key, 0) + 1

    # Check if threshold reached
    if circuit_breaker.failure_counts[key] >= circuit_breaker.failure_threshold:
        # Block for timeout period
        circuit_breaker.blocked_until[key] = current_time + circuit_breaker.timeout_seconds
        logger.warning(
            f"Circuit breaker activated for {key}: {circuit_breaker.failure_counts[key]} consecutive failures"
        )
        # Reset failure count
        circuit_breaker.failure_counts[key] = 0


def check_circuit_breaker(request: Request) -> None:
    """
    Check if request should be blocked by circuit breaker.

    Args:
        request: FastAPI request object

    Raises:
        HTTPException: If circuit breaker is open
    """
    key = get_user_or_ip(request)

    # Admin bypass always allowed
    if key == "admin_bypass":
        return

    current_time = time.time()

    if key in circuit_breaker.blocked_until:
        if current_time < circuit_breaker.blocked_until[key]:
            remaining = int(circuit_breaker.blocked_until[key] - current_time)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Circuit breaker open. Service temporarily unavailable. Retry after {remaining} seconds.",
                headers={
                    "Retry-After": str(remaining),
                    "X-Circuit-Breaker": "open",
                },
            )
        # Unblock
        del circuit_breaker.blocked_until[key]
        if key in circuit_breaker.failure_counts:
            del circuit_breaker.failure_counts[key]


def create_rate_limit_decorator(limit: str) -> Callable:
    """
    Create a rate limit decorator with circuit breaker check.

    Args:
        limit: Rate limit string (e.g., "10/minute")

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Apply the rate limit
        limited_func = limiter.limit(limit)(func)

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check circuit breaker first
            request = kwargs.get("request")
            if request:
                check_circuit_breaker(request)

            # Then apply rate limit
            return await limited_func(*args, **kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return wrapper

    return decorator


# Initialize the limiter with Redis backend
try:
    limiter = Limiter(
        key_func=get_user_or_ip,
        storage_uri=RateLimitConfig.REDIS_URL,
        default_limits=[RateLimitConfig.DEFAULT_LIMIT],
        headers_enabled=True,  # Add rate limit headers to responses
        swallow_errors=False,  # Don't silently fail on Redis errors
    )
    logger.info(f"Rate limiter initialized with Redis backend: {RateLimitConfig.REDIS_URL}")
except Exception as e:
    logger.error(f"Failed to initialize rate limiter with Redis: {e}")
    # Fallback to in-memory limiter if Redis is not available
    limiter = Limiter(
        key_func=get_user_or_ip,
        default_limits=[RateLimitConfig.DEFAULT_LIMIT],
        headers_enabled=True,
    )
    logger.warning("Rate limiter falling back to in-memory storage")
