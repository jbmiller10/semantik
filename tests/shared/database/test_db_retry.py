import pytest
from sqlalchemy.exc import OperationalError

from shared.database.db_retry import _is_retryable_error


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
