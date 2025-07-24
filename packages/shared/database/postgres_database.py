"""
PostgreSQL database connection and session management.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from packages.shared.config.postgres import PostgresConfig, postgres_config

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Manages PostgreSQL database connections with retry logic."""

    def __init__(self, config: PostgresConfig | None = None):
        """Initialize connection manager with configuration."""
        self.config = config or postgres_config
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    async def initialize(self) -> None:
        """Initialize the database engine and session factory."""
        if self._engine:
            return

        # Create engine with retry logic
        for attempt in range(self.config.DB_RETRY_LIMIT):
            try:
                logger.info(f"Attempting to connect to PostgreSQL (attempt {attempt + 1}/{self.config.DB_RETRY_LIMIT})")

                # Get connect_args and ensure 'echo' is not in it to avoid conflicts
                connect_args = self.config.get_connect_args()
                # Remove 'echo' and 'echo_pool' if they somehow got into connect_args
                connect_args.pop("echo", None)
                connect_args.pop("echo_pool", None)

                # For async engines, use NullPool if pool_size is 0, otherwise use default async pool
                if self.config.DB_POOL_SIZE == 0:
                    self._engine = create_async_engine(
                        self.config.async_database_url,
                        echo=self.config.DB_ECHO,
                        echo_pool=self.config.DB_ECHO_POOL,
                        poolclass=NullPool,
                        connect_args=connect_args,
                    )
                else:
                    # Use default async pool with proper configuration
                    pool_kwargs = self.config.get_pool_kwargs()
                    # Ensure echo parameters are not in pool_kwargs
                    pool_kwargs.pop("echo", None)
                    pool_kwargs.pop("echo_pool", None)

                    self._engine = create_async_engine(
                        self.config.async_database_url,
                        echo=self.config.DB_ECHO,
                        echo_pool=self.config.DB_ECHO_POOL,
                        connect_args=connect_args,
                        pool_size=pool_kwargs["pool_size"],
                        max_overflow=pool_kwargs["max_overflow"],
                        pool_timeout=pool_kwargs["pool_timeout"],
                        pool_recycle=pool_kwargs["pool_recycle"],
                        pool_pre_ping=pool_kwargs["pool_pre_ping"],
                    )

                # Test the connection
                async with self._engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))

                logger.info("Successfully connected to PostgreSQL")
                break

            except (OperationalError, DBAPIError) as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.DB_RETRY_LIMIT - 1:
                    await asyncio.sleep(self.config.DB_RETRY_INTERVAL * (2**attempt))  # Exponential backoff
                else:
                    raise

        # Create session factory
        self._sessionmaker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    async def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with automatic cleanup."""
        if not self._sessionmaker:
            await self.initialize()

        if self._sessionmaker is None:
            raise RuntimeError("Database sessionmaker not initialized")
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def execute_with_retry(self, session: AsyncSession, query: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a query with retry logic."""
        for attempt in range(self.config.DB_RETRY_LIMIT):
            try:
                return await session.execute(query, *args, **kwargs)
            except OperationalError as e:
                if attempt < self.config.DB_RETRY_LIMIT - 1:
                    logger.warning(f"Query failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(self.config.DB_RETRY_INTERVAL)
                else:
                    raise
        return None


# Global connection manager instance
pg_connection_manager = PostgresConnectionManager()


async def get_postgres_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get PostgreSQL database session.

    Yields:
        AsyncSession: Database session for the request
    """
    async with pg_connection_manager.get_session() as session:
        yield session


# PostgreSQL-specific optimizations - removed synchronous event listener
# These settings are now applied via connect_args in the configuration


async def check_postgres_connection() -> bool:
    """Check if PostgreSQL connection is available."""
    try:
        async with pg_connection_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"PostgreSQL connection check failed: {e}")
        return False
