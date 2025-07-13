"""Database connection pooling for worker processes.

This module provides connection pooling to prevent database connection exhaustion
when scaling Celery workers.
"""

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from queue import Empty, Queue

from shared.config import settings

logger = logging.getLogger(__name__)


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool for worker processes."""

    def __init__(self, database_path: str, max_connections: int = 5):
        """Initialize the connection pool.

        Args:
            database_path: Path to the SQLite database
            max_connections: Maximum number of connections in the pool
        """
        self.database_path = database_path
        self.max_connections = max_connections
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._created_connections = 0

        # Pre-create minimum connections
        min_connections = min(2, max_connections)
        for _ in range(min_connections):
            conn = self._create_connection()
            self._pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._created_connections += 1
        logger.debug(f"Created connection {self._created_connections}/{self.max_connections}")
        return conn

    @contextmanager
    def get_connection(self, timeout: float = 5.0) -> Iterator[sqlite3.Connection]:
        """Get a connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection

        Yields:
            A database connection

        Raises:
            TimeoutError: If no connection is available within timeout
        """
        conn = None
        try:
            # Try to get from pool
            try:
                conn = self._pool.get(block=False)
            except Empty:
                # Create new connection if under limit
                with self._lock:
                    if self._created_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait for a connection to be returned
                        try:
                            conn = self._pool.get(timeout=timeout)
                        except Empty as err:
                            raise TimeoutError(
                                f"No database connection available after {timeout}s. "
                                f"Consider increasing max_connections or scaling workers differently."
                            ) from err

            yield conn

        finally:
            # Return connection to pool
            if conn is not None:
                try:
                    # Test connection is still valid
                    conn.execute("SELECT 1")
                    self._pool.put(conn)
                except sqlite3.Error:
                    # Connection is broken, create a new one for the pool
                    logger.warning("Broken connection detected, creating replacement")
                    with self._lock:
                        self._created_connections -= 1
                    try:
                        new_conn = self._create_connection()
                        self._pool.put(new_conn)
                    except Exception as e:
                        logger.error(f"Failed to create replacement connection: {e}")

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
        with self._lock:
            self._created_connections = 0
        logger.info("Closed all database connections")


# Global connection pool for workers
# Limit connections per worker to prevent exhaustion
_connection_pool = None
_pool_lock = threading.Lock()


def get_connection_pool() -> SQLiteConnectionPool:
    """Get or create the global connection pool.

    Returns:
        The global SQLiteConnectionPool instance
    """
    global _connection_pool

    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:
                # Calculate max connections based on worker configuration
                # Each worker should have a limited pool to prevent exhaustion
                max_connections_per_worker = 3  # Conservative limit
                _connection_pool = SQLiteConnectionPool(
                    str(settings.webui_db), max_connections=max_connections_per_worker
                )
                logger.info(f"Created connection pool with {max_connections_per_worker} max connections")

    return _connection_pool


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """Get a database connection from the pool.

    This is the main interface for worker processes to get database connections.

    Yields:
        A database connection

    Example:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            result = cursor.fetchone()
    """
    pool = get_connection_pool()
    with pool.get_connection() as conn:
        yield conn
