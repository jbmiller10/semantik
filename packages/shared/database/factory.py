"""Repository factory for PostgreSQL implementations.

This module provides factory functions to create PostgreSQL repository instances.
All repositories now use PostgreSQL exclusively - SQLite support has been removed.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .base import ApiKeyRepository, AuthRepository, UserRepository
from .database import AsyncSessionLocal

logger = logging.getLogger(__name__)


def create_user_repository(session: AsyncSession) -> UserRepository:
    """Create a user repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL UserRepository instance
    """
    from packages.webui.repositories.postgres import PostgreSQLUserRepository

    return PostgreSQLUserRepository(session)


def create_auth_repository(session: AsyncSession) -> AuthRepository:
    """Create an auth repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL AuthRepository instance
    """
    from packages.webui.repositories.postgres import PostgreSQLAuthRepository

    return PostgreSQLAuthRepository(session)


def create_api_key_repository(session: AsyncSession) -> ApiKeyRepository:
    """Create an API key repository instance.

    Args:
        session: AsyncSession for database operations

    Returns:
        PostgreSQL ApiKeyRepository instance
    """
    from packages.webui.repositories.postgres import PostgreSQLApiKeyRepository

    return PostgreSQLApiKeyRepository(session)


def create_operation_repository() -> Any:
    """Create an operation repository instance with session management.

    This maintains compatibility with existing code that expects
    a repository that manages its own session.

    Returns:
        Async wrapper around OperationRepository
    """
    from .repositories.operation_repository import OperationRepository

    class AsyncOperationRepositoryWrapper:
        """Async wrapper that manages its own database session."""

        def __init__(self) -> None:
            self._session: Any | None = None  # AsyncSession
            self._repo: Any | None = None  # OperationRepository

        async def _ensure_initialized(self) -> None:
            """Ensure repository is initialized with a session."""
            if self._repo is None:
                self._session = AsyncSessionLocal()
                self._repo = OperationRepository(self._session)

        async def __aenter__(self) -> "AsyncOperationRepositoryWrapper":
            await self._ensure_initialized()
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._session:
                await self._session.close()

        def __getattr__(self, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
            """Proxy all attribute access to the repository."""

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self._ensure_initialized()
                result = await getattr(self._repo, name)(*args, **kwargs)
                if self._session:
                    await self._session.commit()  # Auto-commit for compatibility
                return result

            return async_wrapper

    return AsyncOperationRepositoryWrapper()


def create_document_repository() -> Any:
    """Create a document repository instance with session management.

    This maintains compatibility with existing code that expects
    a repository that manages its own session.

    Returns:
        Async wrapper around DocumentRepository
    """
    from .repositories.document_repository import DocumentRepository

    class AsyncDocumentRepositoryWrapper:
        """Async wrapper that manages its own database session."""

        def __init__(self) -> None:
            self._session: Any | None = None  # AsyncSession
            self._repo: Any | None = None  # DocumentRepository

        async def _ensure_initialized(self) -> None:
            """Ensure repository is initialized with a session."""
            if self._repo is None:
                self._session = AsyncSessionLocal()
                self._repo = DocumentRepository(self._session)

        async def __aenter__(self) -> "AsyncDocumentRepositoryWrapper":
            await self._ensure_initialized()
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._session:
                await self._session.close()

        def __getattr__(self, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
            """Proxy all attribute access to the repository."""

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self._ensure_initialized()
                result = await getattr(self._repo, name)(*args, **kwargs)
                if self._session:
                    await self._session.commit()  # Auto-commit for compatibility
                return result

            return async_wrapper

    return AsyncDocumentRepositoryWrapper()


def create_collection_repository() -> Any:
    """Create a collection repository instance.

    Note: Use packages.shared.database.repositories.collection_repository.CollectionRepository
    directly with an async session for new code.

    Returns:
        Async wrapper around CollectionRepository
    """
    from .repositories.collection_repository import CollectionRepository

    class AsyncCollectionRepositoryWrapper:
        """Async wrapper that manages its own database session."""

        def __init__(self) -> None:
            self._session: Any | None = None  # AsyncSession
            self._repo: Any | None = None  # CollectionRepository

        async def _ensure_initialized(self) -> None:
            """Ensure repository is initialized with a session."""
            if self._repo is None:
                self._session = AsyncSessionLocal()
                self._repo = CollectionRepository(self._session)

        async def __aenter__(self) -> "AsyncCollectionRepositoryWrapper":
            await self._ensure_initialized()
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            if self._session:
                await self._session.close()

        def __getattr__(self, name: str) -> Callable[..., Coroutine[Any, Any, Any]]:
            """Proxy all attribute access to the repository."""

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self._ensure_initialized()
                result = await getattr(self._repo, name)(*args, **kwargs)
                if self._session:
                    await self._session.commit()  # Auto-commit for compatibility
                return result

            return async_wrapper

    return AsyncCollectionRepositoryWrapper()


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
        # Legacy repositories that manage their own sessions
        "operation": create_operation_repository(),
        "document": create_document_repository(),
        "collection": create_collection_repository(),
    }


# Helper function for dependency injection in FastAPI
async def get_db_session() -> AsyncSession:
    """Get a database session for dependency injection.

    Yields:
        AsyncSession instance
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
