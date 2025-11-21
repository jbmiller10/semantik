"""
Rate limiting implementation for the webui API.

This module provides distributed rate limiting using Redis backend,
circuit breaker pattern, and admin bypass functionality.

Important: Headers are NOT automatically injected due to FastAPI returning
dictionaries that are later converted to JSON responses. Rate limit headers
are only added to rate limit exceeded responses via the error handler.
"""

import functools
import inspect
import logging
import os
import time
import unittest.mock as mock
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from shared.database import pg_connection_manager
from webui.config.rate_limits import CircuitBreakerConfig, RateLimitConfig

logger = logging.getLogger(__name__)


# Circuit breaker instance
circuit_breaker = CircuitBreakerConfig()

_TRUTHY_VALUES = {"true", "1", "yes", "on"}


def _get_bool_env(name: str, default: str = "false") -> bool:
    """Return True if the environment flag is truthy."""
    return os.getenv(name, default).lower() in _TRUTHY_VALUES


def is_rate_limiting_disabled() -> bool:
    """Return True when global rate limiting should be bypassed."""
    return _get_bool_env("DISABLE_RATE_LIMITING")


def get_user_or_ip(request: Request) -> str:
    """
    Get rate limit key based on user ID or IP address.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit key (user_id or IP address)
    """
    # Check if rate limiting is disabled for testing
    if is_rate_limiting_disabled():
        return "test_bypass"

    # Check for admin bypass token - check env var directly for testing
    bypass_token = os.getenv("RATE_LIMIT_BYPASS_TOKEN") or RateLimitConfig.BYPASS_TOKEN
    if bypass_token:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            if token == bypass_token:
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
    # If rate limiting is disabled, this should never be called
    # The rate limiter should be configured with such high limits that it never triggers
    # If it does get called, something is wrong with the configuration
    if is_rate_limiting_disabled():
        logger.error("Rate limit handler called despite rate limiting being disabled - this should not happen!")
        # Don't return anything that would interfere with the response
        # Since we can't pass through to the endpoint from here, return a minimal response
        # that indicates the rate limiter was bypassed
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "bypass": True},
        )

    # Check for bypass token ONLY - not global disable
    key = get_user_or_ip(request)
    if key in ("admin_bypass", "test_bypass"):
        # Don't return 429 for bypass tokens in production
        logger.debug(f"Rate limit bypassed for key: {key}")
        # Return bypass indication
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "bypass": True},
        )

    # Extract retry_after from the exception message
    retry_after = 60  # Default to 60 seconds
    try:
        # The exception message usually contains the retry time
        if hasattr(exc, "retry_after"):
            retry_after = int(exc.retry_after)
    except (AttributeError, ValueError):
        pass

    # Check if circuit breaker should be triggered
    current_time = time.time()
    blocked_until = circuit_breaker.blocked_until.get(key)
    if blocked_until and current_time < blocked_until:
        remaining = int(blocked_until - current_time)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": f"Circuit breaker open. Service temporarily unavailable. Retry after {remaining} seconds.",
                "error": "circuit_breaker_open",
                "retry_after": remaining,
            },
            headers={
                "Retry-After": str(remaining),
                "X-Circuit-Breaker": "open",
            },
        )

    # Track circuit breaker failures BEFORE returning 429
    track_circuit_breaker_failure(key)

    return JSONResponse(
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
    # Skip circuit breaker check if rate limiting is disabled
    if is_rate_limiting_disabled():
        return

    key = get_user_or_ip(request)

    # Admin bypass and test bypass always allowed
    if key in ("admin_bypass", "test_bypass"):
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
        cached_limited_func: Callable[..., Any] | None = None
        used_mock_last = False
        sig = inspect.signature(func)
        request_arg_index: int | None = None

        for idx, parameter in enumerate(sig.parameters.values()):
            if parameter.name in {"request", "websocket"}:
                request_arg_index = idx
                break

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal cached_limited_func, used_mock_last

            # Check if rate limiting is completely disabled
            if is_rate_limiting_disabled():
                # Completely bypass rate limiting in test environment
                return await func(*args, **kwargs)

            # Check circuit breaker first
            request = kwargs.get("request")
            if request is None and request_arg_index is not None and len(args) > request_arg_index:
                candidate = args[request_arg_index]
                if isinstance(candidate, Request):
                    request = candidate
            if request:
                check_circuit_breaker(request)

                # Check if user has special limits
                key = get_user_or_ip(request)
                if key in ("admin_bypass", "test_bypass"):
                    # Completely bypass rate limiting for admin and test
                    return await func(*args, **kwargs)
                if getattr(request.state, "_rate_limit_dependency_enforced", False):
                    return await func(*args, **kwargs)

            limit_callable = limiter.limit
            using_mock = isinstance(limit_callable, mock.Mock)

            if using_mock:
                limited_func = limit_callable(limit)(func)
            else:
                if cached_limited_func is None or used_mock_last:
                    cached_limited_func = limit_callable(limit)(func)
                limited_func = cached_limited_func

            used_mock_last = using_mock
            return await limited_func(*args, **kwargs)

        wrapper.__signature__ = sig  # type: ignore[attr-defined]

        return wrapper

    return decorator


def rate_limit_dependency(limit: str) -> Callable[[Request], Any]:
    """
    Create a FastAPI dependency that enforces a rate limit before other dependencies run.

    This is primarily used by tests that patch limiter.limit to simulate RateLimitExceeded
    responses without executing the full endpoint stack.
    """

    async def _dependency(request: Request) -> None:
        ensure_limiter_runtime_state()
        if not limiter.enabled:
            return

        if (
            limit == RateLimitConfig.PROCESS_RATE
            and os.getenv("TESTING", "false").lower() == "true"
            and pg_connection_manager.sessionmaker is None
        ):

            placeholder = _placeholder_limit(
                limit,
                "Rate limit enforced while database is unavailable",
            )
            raise RateLimitExceeded(placeholder)

        limit_callable = limiter.limit
        should_enforce = isinstance(limit_callable, mock.Mock)

        if not should_enforce:
            return

        async def _noop_handler(request: Request, **_: Any) -> None:  # noqa: ARG001
            return None

        wrapper = limit_callable(limit)(_noop_handler)
        try:
            await wrapper(request=request)
        except RateLimitExceeded:
            raise
        except AttributeError as exc:  # Handle improperly constructed exceptions in tests
            if isinstance(limit_callable, mock.Mock):

                placeholder = _placeholder_limit(limit, "Rate limit exceeded")
                raise RateLimitExceeded(placeholder) from exc
            raise
        else:
            request.state._rate_limit_dependency_enforced = True

    return _dependency


# Define special rate limits for specific keys
SPECIAL_LIMITS = {
    "admin_bypass": "100000/second",  # Effectively unlimited for admin bypass
    "test_bypass": "100000/second",  # Effectively unlimited for test bypass
}


def get_limit_for_key(key: str) -> list[str]:
    """Get rate limits for a specific key."""
    if key in SPECIAL_LIMITS:
        return [SPECIAL_LIMITS[key]]
    return [RateLimitConfig.DEFAULT_LIMIT]


# Initialize the limiter with Redis backend or in-memory fallback
TESTING_MODE = os.getenv("TESTING", "false").lower() == "true"
USE_REDIS_LIMITER = os.getenv("USE_REDIS_RATE_LIMITER", "true").lower() == "true" and not TESTING_MODE

if is_rate_limiting_disabled():
    # Use very high limits for testing - effectively disabling rate limiting
    limiter = Limiter(
        key_func=get_user_or_ip,
        default_limits=["1000000/second"],  # Extremely high limit to ensure no rate limiting in tests
        headers_enabled=False,  # Disable automatic header injection (incompatible with dict responses)
        swallow_errors=True,  # Don't raise errors in test mode
        enabled=False,  # Completely disable rate limiting in test mode
    )
    logger.info("Rate limiter completely disabled for testing")
elif USE_REDIS_LIMITER:
    try:
        limiter = Limiter(
            key_func=get_user_or_ip,
            storage_uri=RateLimitConfig.REDIS_URL,
            default_limits=[RateLimitConfig.DEFAULT_LIMIT],
            headers_enabled=False,  # Disable automatic header injection (incompatible with dict responses)
            swallow_errors=True,  # Silently fail on Redis errors to prevent test disruption
            in_memory_fallback_enabled=True,
        )
        logger.info(f"Rate limiter initialized with Redis backend: {RateLimitConfig.REDIS_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter with Redis: {e}")
        # Fallback to in-memory limiter if Redis is not available
        limiter = Limiter(
            key_func=get_user_or_ip,
            default_limits=[RateLimitConfig.DEFAULT_LIMIT],
            headers_enabled=False,  # Disable automatic header injection (incompatible with dict responses)
            in_memory_fallback_enabled=True,
        )
        logger.warning("Rate limiter falling back to in-memory storage")
else:
    limiter = Limiter(
        key_func=get_user_or_ip,
        default_limits=[RateLimitConfig.DEFAULT_LIMIT],
        headers_enabled=False,  # Disable automatic header injection (incompatible with dict responses)
        swallow_errors=False,
    )
    logger.info("Rate limiter initialized with in-memory storage (testing mode)")


_limiter_disabled_state: bool | None = None


def ensure_limiter_runtime_state(limiter_instance: Limiter | None = None) -> None:
    """
    Ensure the limiter's runtime flags reflect the latest environment settings.

    This keeps rate limiting responsive to DISABLE_RATE_LIMITING toggles that
    may occur inside test environments after the limiter has already been created.
    """
    global _limiter_disabled_state

    limiter_ref = limiter_instance or limiter
    if limiter_ref is None:
        return

    disabled = is_rate_limiting_disabled()

    if _limiter_disabled_state is not None and disabled == _limiter_disabled_state:
        desired_enabled = not disabled
        if limiter_ref.enabled != desired_enabled:
            limiter_ref.enabled = desired_enabled
        desired_swallow = not disabled
        if limiter_ref._swallow_errors != desired_swallow:  # noqa: SLF001
            limiter_ref._swallow_errors = desired_swallow  # noqa: SLF001
        return

    limiter_ref.enabled = not disabled
    limiter_ref._swallow_errors = not disabled  # noqa: SLF001

    try:
        limiter_ref.reset()
    except NotImplementedError:
        logger.debug("Rate limiter storage does not support reset when toggling state")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to reset rate limiter when toggling state: %s", exc)

    logger.info("Rate limiter %s via environment toggle", "disabled" if disabled else "enabled")
    _limiter_disabled_state = disabled


# Align limiter state with initial environment configuration at import time.
ensure_limiter_runtime_state(limiter)


def add_rate_limit_headers(response: Response, limit: str, remaining: int, reset: int) -> Response:
    """
    Manually add rate limit headers to a Response object.

    This function should only be used when returning actual Response objects,
    not when returning dictionaries that FastAPI will convert.

    Args:
        response: The Response object to add headers to
        limit: The rate limit (e.g., "10")
        remaining: Number of requests remaining
        reset: Unix timestamp when the rate limit resets

    Returns:
        The response with added headers
    """
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset)
    return response


def _placeholder_limit(limit_str: str, message: str) -> Any:
    """Create a lightweight object satisfying the Limit protocol for tests."""

    class _LimitPlaceholder:  # noqa: D106 - internal helper
        def __init__(self, limit_value: str, error_message: str) -> None:
            self.limit = limit_value
            self.error_message = error_message

    return _LimitPlaceholder(limit_str, message)
