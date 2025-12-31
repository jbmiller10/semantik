"""Database retry utilities for handling database locking issues"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")

RETRYABLE_ERROR_PATTERNS = [
    # SQLite
    "database is locked",
    # PostgreSQL connection errors
    "connection refused",
    "could not connect",
    "connection timed out",
    "server closed the connection unexpectedly",
    "ssl connection has been closed unexpectedly",
    "connection reset by peer",
    "no connection to the server",
    "terminating connection due to administrator command",
    # PostgreSQL transaction errors
    "deadlock detected",
    "serialization failure",
    "could not serialize access",
    # General transient errors
    "temporary failure",
    "connection aborted",
]


def _is_retryable_error(error: Exception) -> bool:
    """Check if an OperationalError is transient and safe to retry."""
    error_str = str(error).lower()
    return any(pattern in error_str for pattern in RETRYABLE_ERROR_PATTERNS)


def with_db_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
) -> Callable[[Callable[P, Any]], Callable[P, Any]]:
    """
    Decorator to retry database operations on lock errors.

    Args:
        retries: Number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for exponential backoff
        max_delay: Maximum delay between retries
    """

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    if not _is_retryable_error(e) or attempt == retries:
                        raise

                    last_exception = e
                    logger.warning(
                        "Retryable database error (attempt %d/%d): %s",
                        attempt + 1,
                        retries + 1,
                        e,
                        exc_info=True,
                    )

                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if not _is_retryable_error(e) or attempt == retries:
                        raise

                    last_exception = e
                    logger.warning(
                        "Retryable database error (attempt %d/%d): %s",
                        attempt + 1,
                        retries + 1,
                        e,
                        exc_info=True,
                    )

                    import time

                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
