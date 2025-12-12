"""
Async database session management using PostgreSQL.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .postgres_database import get_postgres_db, pg_connection_manager

# Re-export PostgreSQL components for backward compatibility
# This allows existing code to continue using:
# from shared.database.database import AsyncSessionLocal, get_db

AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


async def ensure_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Ensure the global sessionmaker is initialised and return it."""

    global AsyncSessionLocal
    if AsyncSessionLocal is None or pg_connection_manager.sessionmaker is None:
        await pg_connection_manager.initialize()
        AsyncSessionLocal = pg_connection_manager.sessionmaker

    if AsyncSessionLocal is None:
        raise RuntimeError("Database sessionmaker not initialized")
    return AsyncSessionLocal


# Re-export get_db using the PostgreSQL implementation, which already
# initializes the connection manager lazily.
get_db = get_postgres_db
