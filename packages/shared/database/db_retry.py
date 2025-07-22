"""Database retry utilities for handling database locking issues"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_db_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 30.0,
):
    """
    Decorator to retry database operations on lock errors.

    Args:
        retries: Number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for exponential backoff
        max_delay: Maximum delay between retries
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    if "database is locked" not in str(e) or attempt == retries:
                        raise

                    last_exception = e
                    logger.warning(
                        f"Database locked on attempt {attempt + 1}/{retries + 1}, "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )

                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    if "database is locked" not in str(e) or attempt == retries:
                        raise

                    last_exception = e
                    logger.warning(
                        f"Database locked on attempt {attempt + 1}/{retries + 1}, "
                        f"retrying in {current_delay:.1f}s: {e}"
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
        else:
            return sync_wrapper

    return decorator
