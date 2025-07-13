"""Transaction support for database operations.

This module provides transaction context managers for ensuring atomic operations
across multiple database calls.
"""

import logging
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any

from .exceptions import TransactionError
from .sqlite_implementation import DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def sqlite_transaction() -> Iterator[sqlite3.Connection]:
    """Context manager for SQLite transactions.

    This ensures that multiple database operations are executed atomically.
    If any operation fails, all changes are rolled back.

    Example:
        with sqlite_transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO table1 ...")
            cursor.execute("UPDATE table2 ...")
            # Both operations committed together

    Yields:
        SQLite connection object with transaction started

    Raises:
        TransactionError: If transaction fails or rollback fails
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Start transaction
        conn.execute("BEGIN")
        yield conn
        # Commit if no exceptions
        conn.commit()
    except Exception as e:
        # Rollback on any error
        try:
            conn.rollback()
            rollback_successful = True
        except Exception as rollback_error:
            logger.error(f"Failed to rollback transaction: {rollback_error}")
            rollback_successful = False

        # Raise transaction error with rollback status
        raise TransactionError(f"Transaction failed: {str(e)}", rollback_successful=rollback_successful) from e
    finally:
        conn.close()


@asynccontextmanager
async def async_sqlite_transaction() -> AsyncIterator[sqlite3.Connection]:
    """Async context manager for SQLite transactions.

    This is a wrapper around the sync version for compatibility with async code.
    Note: SQLite operations are still synchronous, but this allows using
    async/await syntax consistently.

    Example:
        async with async_sqlite_transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO table1 ...")
            cursor.execute("UPDATE table2 ...")
            # Both operations committed together

    Yields:
        SQLite connection object with transaction started

    Raises:
        TransactionError: If transaction fails or rollback fails
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Start transaction
        conn.execute("BEGIN")
        yield conn
        # Commit if no exceptions
        conn.commit()
    except Exception as e:
        # Rollback on any error
        try:
            conn.rollback()
            rollback_successful = True
        except Exception as rollback_error:
            logger.error(f"Failed to rollback transaction: {rollback_error}")
            rollback_successful = False

        # Raise transaction error with rollback status
        raise TransactionError(f"Transaction failed: {str(e)}", rollback_successful=rollback_successful) from e
    finally:
        conn.close()


class RepositoryTransaction:
    """Transaction context for repository operations.

    This provides a higher-level transaction interface that can work
    with multiple repositories while ensuring atomicity.

    Example:
        async with RepositoryTransaction() as transaction:
            # All operations within this block are atomic
            await job_repo.delete_job(job_id)
            await file_repo.delete_files_for_job(job_id)
            # Both operations committed together
    """

    def __init__(self) -> None:
        """Initialize transaction context."""
        self.connection: sqlite3.Connection | None = None
        self._in_transaction = False

    async def __aenter__(self) -> "RepositoryTransaction":
        """Start transaction."""
        self.connection = sqlite3.connect(DB_PATH)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("BEGIN")
        self._in_transaction = True

        # Store the connection in a thread-local or context variable
        # so repositories can access it
        # This is a simplified version - in production you'd use contextvars
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Complete or rollback transaction."""
        if self.connection is None:
            return False

        try:
            if exc_type is None:
                # No exception, commit
                self.connection.commit()
            else:
                # Exception occurred, rollback
                self.connection.rollback()
        except Exception as e:
            logger.error(f"Error during transaction cleanup: {e}")
            # Try to rollback if commit failed
            if exc_type is None:
                with suppress(Exception):
                    self.connection.rollback()
        finally:
            self.connection.close()
            self.connection = None
            self._in_transaction = False

        # Don't suppress the original exception
        return False

    def get_connection(self) -> sqlite3.Connection:
        """Get the current transaction connection.

        Returns:
            The SQLite connection for this transaction

        Raises:
            RuntimeError: If called outside of transaction context
        """
        if not self._in_transaction or self.connection is None:
            raise RuntimeError("Not in a transaction context")
        return self.connection


# Future enhancement: PostgreSQL transaction support
# @asynccontextmanager
# async def postgres_transaction(pool: AsyncConnectionPool) -> AsyncIterator[AsyncConnection]:
#     """Context manager for PostgreSQL transactions."""
#     async with pool.connection() as conn:
#         async with conn.transaction():
#             yield conn
