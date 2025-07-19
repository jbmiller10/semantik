"""Repository factory for dependency injection.

This module provides factory functions to create repository instances,
allowing for easy switching between different implementations.
"""

from typing import Any

from .base import AuthRepository, CollectionRepository, FileRepository, JobRepository, UserRepository
from .sqlite_repository import (
    SQLiteAuthRepository,
    SQLiteCollectionRepository,
    SQLiteFileRepository,
    SQLiteJobRepository,
    SQLiteUserRepository,
)


def create_job_repository() -> JobRepository:
    """Create a job repository instance.

    Returns:
        JobRepository instance

    Note:
        This returns the SQLite implementation for backward compatibility.
        Jobs have been replaced by operations in the new schema.
        This function will be removed in a future phase.
    """
    return SQLiteJobRepository()


def create_user_repository() -> UserRepository:
    """Create a user repository instance.

    Returns:
        UserRepository instance
    """
    return SQLiteUserRepository()


def create_file_repository() -> FileRepository:
    """Create a file repository instance.

    Returns:
        FileRepository instance

    Note:
        This returns the SQLite implementation for backward compatibility.
        Files have been replaced by documents in the new schema.
        This function will be removed in a future phase.
    """
    return SQLiteFileRepository()


def create_collection_repository() -> CollectionRepository:
    """Create a collection repository instance.

    Returns:
        CollectionRepository instance

    Note:
        This returns the SQLite implementation for backward compatibility.
        The new CollectionRepository uses SQLAlchemy and should be used for new code.
        This function will be removed in a future phase.
    """
    return SQLiteCollectionRepository()


def create_auth_repository() -> AuthRepository:
    """Create an auth repository instance.

    Returns:
        AuthRepository instance
    """
    return SQLiteAuthRepository()


def create_all_repositories() -> dict[str, object]:
    """Create all repository instances.

    Returns:
        Dictionary mapping repository names to instances

    Note:
        Job, File, and Collection repositories are using the old SQLite implementation
        for backward compatibility. They will be replaced in a future phase.
    """
    return {
        "job": create_job_repository(),
        "user": create_user_repository(),
        "file": create_file_repository(),
        "collection": create_collection_repository(),
        "auth": create_auth_repository(),
    }


def create_operation_repository() -> Any:
    """Create an operation repository instance.

    Note: This is a compatibility shim for the new async repositories.
    The actual implementation will create a session and repository on first use.
    """
    import asyncio

    from .database import AsyncSessionLocal
    from .repositories.operation_repository import OperationRepository

    class AsyncOperationRepositoryWrapper:
        """Async wrapper that manages its own database session."""

        def __init__(self):
            self._session = None
            self._repo = None

        async def _ensure_initialized(self):
            """Ensure repository is initialized with a session."""
            if self._repo is None:
                self._session = AsyncSessionLocal()
                self._repo = OperationRepository(self._session)

        async def __aenter__(self):
            await self._ensure_initialized()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._session:
                await self._session.close()

        def __getattr__(self, name):
            """Proxy all attribute access to the repository."""

            async def async_wrapper(*args, **kwargs):
                await self._ensure_initialized()
                result = await getattr(self._repo, name)(*args, **kwargs)
                await self._session.commit()  # Auto-commit for compatibility
                return result

            return async_wrapper

    return AsyncOperationRepositoryWrapper()


def create_document_repository() -> Any:
    """Create a document repository instance.

    Note: This is a compatibility shim for the new async repositories.
    The actual implementation will create a session and repository on first use.
    """
    import asyncio

    from .database import AsyncSessionLocal
    from .repositories.document_repository import DocumentRepository

    class AsyncDocumentRepositoryWrapper:
        """Async wrapper that manages its own database session."""

        def __init__(self):
            self._session = None
            self._repo = None

        async def _ensure_initialized(self):
            """Ensure repository is initialized with a session."""
            if self._repo is None:
                self._session = AsyncSessionLocal()
                self._repo = DocumentRepository(self._session)

        async def __aenter__(self):
            await self._ensure_initialized()
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._session:
                await self._session.close()

        def __getattr__(self, name):
            """Proxy all attribute access to the repository."""

            async def async_wrapper(*args, **kwargs):
                await self._ensure_initialized()
                result = await getattr(self._repo, name)(*args, **kwargs)
                await self._session.commit()  # Auto-commit for compatibility
                return result

            return async_wrapper

    return AsyncDocumentRepositoryWrapper()
