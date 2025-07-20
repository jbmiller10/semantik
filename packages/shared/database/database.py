"""
Async database session management for SQLAlchemy.
"""

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from packages.shared.config import settings

logger = logging.getLogger(__name__)

# Create async engine
# Using aiosqlite for SQLite async support
engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.webui_db}",
    echo=False,  # Set to True for SQL query logging
    future=True,
    connect_args={
        "timeout": 30,  # Increase timeout to 30 seconds
        "check_same_thread": False,
    },
    pool_pre_ping=True,  # Verify connections before using
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency function to get async database session.

    Yields:
        AsyncSession: Database session for the request
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Set up SQLite optimizations for better concurrency
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Set SQLite pragmas for better performance and concurrency"""
    if "sqlite" in str(engine.url):
        cursor = dbapi_conn.cursor()
        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        # Increase cache size (negative value means KB)
        cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
        # Set busy timeout to 30 seconds
        cursor.execute("PRAGMA busy_timeout=30000")
        # Additional settings for better concurrency
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=30000000000")
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
