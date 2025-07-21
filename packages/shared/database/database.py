"""
Async database session management using PostgreSQL.
"""

import logging

from .postgres_database import get_postgres_db, pg_connection_manager

logger = logging.getLogger(__name__)

# Re-export PostgreSQL components for backward compatibility
# This allows existing code to continue using:
# from shared.database.database import AsyncSessionLocal, get_db

# Create a module-level reference to the sessionmaker
# This will be initialized when the connection manager initializes
AsyncSessionLocal = None


async def _ensure_initialized():
    """Ensure the connection manager is initialized."""
    global AsyncSessionLocal
    if not pg_connection_manager._sessionmaker:
        await pg_connection_manager.initialize()
    AsyncSessionLocal = pg_connection_manager._sessionmaker


# Re-export get_db using the PostgreSQL implementation
get_db = get_postgres_db


# For code that directly uses AsyncSessionLocal, provide a wrapper
class AsyncSessionLocalWrapper:
    """Wrapper to handle direct AsyncSessionLocal usage."""
    
    def __new__(cls):
        """Create a new session using the PostgreSQL sessionmaker."""
        import asyncio
        
        # Ensure connection manager is initialized
        if not pg_connection_manager._sessionmaker:
            # Run initialization in the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't await here
                # This should be rare as most code paths go through get_db()
                raise RuntimeError(
                    "Database not initialized. Use get_db() or initialize the connection manager first."
                )
            else:
                loop.run_until_complete(pg_connection_manager.initialize())
        
        return pg_connection_manager._sessionmaker()
    
    def __call__(self):
        """Support callable syntax."""
        return self.__new__(self.__class__)


# Replace the module-level AsyncSessionLocal with our wrapper
AsyncSessionLocal = AsyncSessionLocalWrapper
