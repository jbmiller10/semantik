"""Repository factory for PostgreSQL implementations.

This module provides factory functions to create PostgreSQL repository instances.
All repositories now use PostgreSQL exclusively - SQLite support has been removed.
"""

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from .base import ApiKeyRepository, AuthRepository, UserRepository
from .database import AsyncSessionLocal, ensure_async_sessionmaker

if TYPE_CHECKING:
    from .repositories.chunk_repository import ChunkRepository
    from .repositories.collection_repository import CollectionRepository
    from .repositories.document_repository import DocumentRepository
    from .repositories.operation_repository import OperationRepository
    from .repositories.projection_run_repository import ProjectionRunRepository

logger = logging.getLogger(__name__)


def create_user_repository(session: AsyncSession) -> UserRepository:
    """Create a user repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL UserRepository instance
    """
    from webui.repositories.postgres import PostgreSQLUserRepository

    result: UserRepository = PostgreSQLUserRepository(session)
    return result


def create_auth_repository(session: AsyncSession) -> AuthRepository:
    """Create an auth repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL AuthRepository instance
    """
    from webui.repositories.postgres import PostgreSQLAuthRepository

    result: AuthRepository = PostgreSQLAuthRepository(session)
    return result


def create_api_key_repository(session: AsyncSession) -> ApiKeyRepository:
    """Create an API key repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL ApiKeyRepository instance
    """
    from webui.repositories.postgres import PostgreSQLApiKeyRepository

    result: ApiKeyRepository = PostgreSQLApiKeyRepository(session)
    return result


def create_operation_repository(session: AsyncSession) -> "OperationRepository":
    """Create an operation repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        OperationRepository instance
    """
    from .repositories.operation_repository import OperationRepository

    return OperationRepository(session)


def create_document_repository(session: AsyncSession) -> "DocumentRepository":
    """Create a document repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        DocumentRepository instance
    """
    from .repositories.document_repository import DocumentRepository

    return DocumentRepository(session)


def create_projection_run_repository(session: AsyncSession) -> "ProjectionRunRepository":
    """Create a projection run repository instance."""

    from .repositories.projection_run_repository import ProjectionRunRepository

    return ProjectionRunRepository(session)


def create_collection_repository(session: AsyncSession) -> "CollectionRepository":
    """Create a collection repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        CollectionRepository instance
    """
    from .repositories.collection_repository import CollectionRepository

    return CollectionRepository(session)


def create_chunk_repository(session: AsyncSession) -> "ChunkRepository":
    """Create a chunk repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        ChunkRepository instance with partition-aware operations
    """
    from .repositories.chunk_repository import ChunkRepository

    return ChunkRepository(session)


def create_all_repositories(session: AsyncSession) -> dict[str, object]:
    """Create all repository instances with the provided session.

    Args:
        session: AsyncSession for database operations

    Returns:
        Dictionary mapping repository names to instances
    """
    return {
        "user": create_user_repository(session),
        "auth": create_auth_repository(session),
        "api_key": create_api_key_repository(session),
        "operation": create_operation_repository(session),
        "document": create_document_repository(session),
        "collection": create_collection_repository(session),
        "chunk": create_chunk_repository(session),
        "projection_run": create_projection_run_repository(session),
    }


# Helper function for dependency injection in FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for dependency injection.

    Yields:
        AsyncSession instance
    """
    sessionmaker = AsyncSessionLocal
    if sessionmaker is None:
        sessionmaker = await ensure_async_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
