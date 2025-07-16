"""Factory functions for creating service instances with dependencies."""

from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession

from .collection_service import CollectionService
from .file_scanning_service import FileScanningService


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


def create_file_scanning_service(db: AsyncSession) -> FileScanningService:
    """Create a FileScanningService instance with required dependencies.

    This factory function simplifies dependency injection for file scanning operations.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured FileScanningService instance

    Example:
        ```python
        from fastapi import Depends
        from sqlalchemy.ext.asyncio import AsyncSession
        from shared.database import get_db
        from webui.services.factory import create_file_scanning_service

        async def get_file_scanning_service(
            db: AsyncSession = Depends(get_db)
        ) -> FileScanningService:
            return create_file_scanning_service(db)

        # In your task or endpoint
        async def scan_and_register_files(
            collection_id: str,
            source_path: str,
            service: FileScanningService = Depends(get_file_scanning_service),
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
    return FileScanningService(
        db_session=db,
        document_repo=document_repo,
    )
