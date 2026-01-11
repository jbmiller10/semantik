import asyncio
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError

from shared.database.db_retry import (
    _calculate_next_delay,
    _is_retryable_error,
    _log_retry,
    _should_retry,
    with_db_retry,
)


def _make_operational_error(message: str) -> OperationalError:
    return OperationalError("SELECT 1", {}, Exception(message))


@pytest.mark.parametrize(
    ("error_message", "should_retry"),
    [
        ("database is locked", True),
        ("connection refused", True),
        ("could not connect to server", True),
        ("server closed the connection unexpectedly", True),
        ("SSL connection has been closed unexpectedly", True),
        ("deadlock detected", True),
        ("serialization failure", True),
        ("could not serialize access due to read/write dependencies", True),
        ("syntax error at or near", False),
        ("unique constraint violated", False),
        ("foreign key constraint", False),
    ],
)
def test_retryable_errors(error_message: str, should_retry: bool) -> None:
    error = _make_operational_error(error_message)
    assert _is_retryable_error(error) is should_retry


class TestCalculateNextDelay:
    """Tests for _calculate_next_delay helper function."""

    def test_basic_backoff(self) -> None:
        """Delay should multiply by backoff factor."""
        assert _calculate_next_delay(1.0, 2.0, 30.0) == 2.0
        assert _calculate_next_delay(2.0, 2.0, 30.0) == 4.0
        assert _calculate_next_delay(4.0, 2.0, 30.0) == 8.0

    def test_respects_max_delay(self) -> None:
        """Delay should not exceed max_delay."""
        assert _calculate_next_delay(20.0, 2.0, 30.0) == 30.0
        assert _calculate_next_delay(100.0, 2.0, 30.0) == 30.0

    def test_fractional_backoff(self) -> None:
        """Should work with fractional backoff values."""
        assert _calculate_next_delay(1.0, 1.5, 30.0) == 1.5
        assert _calculate_next_delay(2.0, 1.5, 30.0) == 3.0

    def test_no_backoff(self) -> None:
        """Backoff of 1.0 should not change delay."""
        assert _calculate_next_delay(5.0, 1.0, 30.0) == 5.0


class TestShouldRetry:
    """Tests for _should_retry helper function."""

    def test_retryable_error_not_exhausted(self) -> None:
        """Should return True for retryable error with attempts remaining."""
        error = _make_operational_error("database is locked")
        assert _should_retry(error, attempt=0, max_retries=3) is True
        assert _should_retry(error, attempt=1, max_retries=3) is True
        assert _should_retry(error, attempt=2, max_retries=3) is True

    def test_retryable_error_exhausted(self) -> None:
        """Should return False when max retries reached."""
        error = _make_operational_error("database is locked")
        assert _should_retry(error, attempt=3, max_retries=3) is False

    def test_non_retryable_error(self) -> None:
        """Should return False for non-retryable errors regardless of attempts."""
        error = _make_operational_error("syntax error")
        assert _should_retry(error, attempt=0, max_retries=3) is False
        assert _should_retry(error, attempt=1, max_retries=3) is False


class TestLogRetry:
    """Tests for _log_retry helper function."""

    def test_logs_with_correct_format(self) -> None:
        """Should log with attempt+1 and max_retries+1 format."""
        error = _make_operational_error("connection refused")
        with patch("shared.database.db_retry.logger") as mock_logger:
            _log_retry(attempt=0, max_retries=3, error=error)
            mock_logger.warning.assert_called_once()
            args = mock_logger.warning.call_args[0]
            # Should log "attempt 1/4" for attempt=0, max_retries=3
            assert "1/4" in args[0] % args[1:]

    def test_includes_exc_info(self) -> None:
        """Should include exc_info=True for traceback logging."""
        error = _make_operational_error("deadlock detected")
        with patch("shared.database.db_retry.logger") as mock_logger:
            _log_retry(attempt=1, max_retries=3, error=error)
            kwargs = mock_logger.warning.call_args[1]
            assert kwargs.get("exc_info") is True


class TestWithDbRetryDecorator:
    """Tests for with_db_retry decorator behavior."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self) -> None:
        """Async function succeeding on first try should not retry."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await async_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retries_on_retryable_error(self) -> None:
        """Async function should retry on retryable errors."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _make_operational_error("database is locked")
            return "success"

        result = await async_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_raises_after_max_retries(self) -> None:
        """Async function should raise after exhausting retries."""

        @with_db_retry(retries=2, delay=0.01)
        async def async_func() -> str:
            raise _make_operational_error("database is locked")

        with pytest.raises(OperationalError):
            await async_func()

    @pytest.mark.asyncio
    async def test_async_raises_immediately_for_non_retryable(self) -> None:
        """Async function should not retry non-retryable errors."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            raise _make_operational_error("syntax error")

        with pytest.raises(OperationalError):
            await async_func()
        assert call_count == 1

    def test_sync_success_no_retry(self) -> None:
        """Sync function succeeding on first try should not retry."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        def sync_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = sync_func()
        assert result == "success"
        assert call_count == 1

    def test_sync_retries_on_retryable_error(self) -> None:
        """Sync function should retry on retryable errors."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        def sync_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _make_operational_error("connection refused")
            return "success"

        result = sync_func()
        assert result == "success"
        assert call_count == 3

    def test_sync_raises_after_max_retries(self) -> None:
        """Sync function should raise after exhausting retries."""

        @with_db_retry(retries=2, delay=0.01)
        def sync_func() -> str:
            raise _make_operational_error("database is locked")

        with pytest.raises(OperationalError):
            sync_func()

    def test_sync_raises_immediately_for_non_retryable(self) -> None:
        """Sync function should not retry non-retryable errors."""
        call_count = 0

        @with_db_retry(retries=3, delay=0.01)
        def sync_func() -> str:
            nonlocal call_count
            call_count += 1
            raise _make_operational_error("unique constraint violated")

        with pytest.raises(OperationalError):
            sync_func()
        assert call_count == 1

    def test_preserves_function_metadata(self) -> None:
        """Decorator should preserve original function metadata."""

        @with_db_retry()
        async def my_async_func() -> str:
            """My docstring."""
            return "async"

        @with_db_retry()
        def my_sync_func() -> str:
            """My sync docstring."""
            return "sync"

        assert my_async_func.__name__ == "my_async_func"
        assert my_async_func.__doc__ == "My docstring."
        assert my_sync_func.__name__ == "my_sync_func"
        assert my_sync_func.__doc__ == "My sync docstring."

    @pytest.mark.asyncio
    async def test_exponential_backoff_applied(self) -> None:
        """Should apply exponential backoff between retries."""
        delays: list[float] = []
        call_count = 0

        async def mock_sleep(delay: float) -> None:
            delays.append(delay)

        @with_db_retry(retries=3, delay=1.0, backoff=2.0, max_delay=10.0)
        async def async_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise _make_operational_error("database is locked")
            return "success"

        with patch("shared.database.db_retry.asyncio.sleep", mock_sleep):
            result = await async_func()

        assert result == "success"
        # Delays: 1.0, 2.0, 4.0 (before succeeding on attempt 4)
        assert delays == [1.0, 2.0, 4.0]
