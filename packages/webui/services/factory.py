"""Factory functions for creating service instances with dependencies."""

from fastapi import Depends
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db

from .collection_service import CollectionService
from .document_scanning_service import DocumentScanningService
from .operation_service import OperationService
from .resource_manager import ResourceManager
from .search_service import SearchService


def create_collection_service(db: AsyncSession) -> CollectionService:
    """Create a CollectionService instance with all required dependencies.

    This factory function simplifies dependency injection for FastAPI endpoints.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured CollectionService instance

    Example:
        ```python
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from shared.database import get_db
        from webui.services.factory import create_collection_service

        async def get_collection_service(
            db: AsyncSession = Depends(get_db)
        ) -> CollectionService:
            return create_collection_service(db)

        @router.post("/collections")
        async def create_collection(
            request: CreateCollectionRequest,
            service: CollectionService = Depends(get_collection_service),
        ):
            # Use service here
            pass
        ```
    """
    # Create repository instances
    collection_repo = CollectionRepository(db)
    operation_repo = OperationRepository(db)
    document_repo = DocumentRepository(db)

    # Create and return service
    return CollectionService(
        db_session=db,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        document_repo=document_repo,
    )


def create_document_scanning_service(db: AsyncSession) -> DocumentScanningService:
    """Create a DocumentScanningService instance with required dependencies.

    This factory function simplifies dependency injection for file scanning operations.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured DocumentScanningService instance

    Example:
        ```python
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from shared.database import get_db
        from webui.services.factory import create_document_scanning_service

        async def get_document_scanning_service(
            db: AsyncSession = Depends(get_db)
        ) -> DocumentScanningService:
            return create_document_scanning_service(db)

        # In your task or endpoint
        async def scan_and_register_files(
            collection_id: str,
            source_path: str,
            service: DocumentScanningService = Depends(get_document_scanning_service),
        ):
            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id,
                source_path=source_path
            )
            return stats
        ```
    """
    # Create repository instances
    document_repo = DocumentRepository(db)

    # Create and return service
    return DocumentScanningService(
        db_session=db,
        document_repo=document_repo,
    )


def create_resource_manager(db: AsyncSession) -> ResourceManager:
    """Create a ResourceManager instance with required dependencies.

    This factory function creates a resource manager for monitoring and managing
    resource allocation for collection operations.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured ResourceManager instance

    Example:
        ```python
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from shared.database import get_db
        from webui.services.factory import create_resource_manager

        async def get_resource_manager(
            db: AsyncSession = Depends(get_db)
        ) -> ResourceManager:
            return create_resource_manager(db)

        # In your endpoint or service
        async def check_resources(
            user_id: int,
            resource_manager: ResourceManager = Depends(get_resource_manager),
        ):
            can_create = await resource_manager.can_create_collection(user_id)
            return {"can_create": can_create}
        ```
    """
    # Create repository instances
    collection_repo = CollectionRepository(db)
    operation_repo = OperationRepository(db)

    # Create and return resource manager
    return ResourceManager(
        collection_repo=collection_repo,
        operation_repo=operation_repo,
    )


# FastAPI dependency functions


async def get_collection_service(db: AsyncSession = Depends(get_db)) -> CollectionService:
    """FastAPI dependency for CollectionService injection."""
    return create_collection_service(db)


def create_operation_service(db: AsyncSession) -> OperationService:
    """Create an OperationService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured OperationService instance
    """
    # Create repository instances
    operation_repo = OperationRepository(db)

    # Create and return service
    return OperationService(
        db_session=db,
        operation_repo=operation_repo,
    )


async def get_operation_service(db: AsyncSession = Depends(get_db)) -> OperationService:
    """FastAPI dependency for OperationService injection."""
    return create_operation_service(db)


def create_search_service(db: AsyncSession) -> SearchService:
    """Create a SearchService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured SearchService instance
    """
    # Create repository instances
    collection_repo = CollectionRepository(db)

    # Create and return service
    return SearchService(
        db_session=db,
        collection_repo=collection_repo,
    )


async def get_search_service(db: AsyncSession = Depends(get_db)) -> SearchService:
    """FastAPI dependency for SearchService injection."""
    return create_search_service(db)
