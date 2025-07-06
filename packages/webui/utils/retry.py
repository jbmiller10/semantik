"""
Retry utilities for handling transient failures
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")


def exponential_backoff_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function that retries on failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries + 1} attempts: {e}")
                        raise

                    delay = min(initial_delay * (exponential_base**attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("No exception was caught")

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)  # type: ignore[misc]
                    return cast(T, result)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries + 1} attempts: {e}")
                        raise

                    delay = min(initial_delay * (exponential_base**attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("No exception was caught")

        # Return appropriate wrapper based on function type
        if is_async:
            return cast(Callable[..., T], async_wrapper)
        return sync_wrapper

    return decorator
