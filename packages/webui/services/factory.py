"""Factory functions for creating service instances with dependencies."""

from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from sqlalchemy.ext.asyncio import AsyncSession

from .collection_service import CollectionService


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
