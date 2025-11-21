"""Factory functions for creating service instances with dependencies."""

import logging

import httpx
from fastapi import Depends
from shared.database import get_db
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository
from shared.managers import QdrantManager
from sqlalchemy.ext.asyncio import AsyncSession
from webui.utils.qdrant_manager import qdrant_manager as qdrant_connection_manager

from .chunking.container import get_chunking_orchestrator as container_get_chunking_orchestrator
from .chunking.container import get_redis_manager as container_get_redis_manager
from .chunking.orchestrator import ChunkingOrchestrator
from .collection_service import CollectionService
from .directory_scan_service import DirectoryScanService
from .document_scanning_service import DocumentScanningService
from .operation_service import OperationService
from .projection_service import ProjectionService
from .redis_manager import RedisManager
from .resource_manager import ResourceManager
from .search_service import SearchService

logger = logging.getLogger(__name__)


def get_redis_manager() -> RedisManager:
    """Backward-compatible wrapper over chunking container Redis manager."""

    manager = container_get_redis_manager()
    logger.debug("Reusing Redis manager from chunking container")
    return manager


def create_collection_service(
    db: AsyncSession,
    *,
    qdrant_manager_override: QdrantManager | None = None,
) -> CollectionService:
    """Create a CollectionService instance with all required dependencies.

    This factory function simplifies dependency injection for FastAPI endpoints.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection
        qdrant_manager_override: Optional pre-built manager, useful for tests

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

    qdrant_manager_instance = qdrant_manager_override
    if qdrant_manager_instance is None:
        try:
            qdrant_client = qdrant_connection_manager.get_client()
            qdrant_manager_instance = QdrantManager(qdrant_client)
        except Exception as exc:  # pragma: no cover - fallback when Qdrant is offline
            logger.warning("Qdrant client unavailable for collection service: %s", exc)

    # Create and return service
    return CollectionService(
        db_session=db,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        document_repo=document_repo,
        qdrant_manager=qdrant_manager_instance,
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

    qdrant_manager_instance = None
    try:
        qdrant_client = qdrant_connection_manager.get_client()
        qdrant_manager_instance = QdrantManager(qdrant_client)
    except Exception as exc:  # pragma: no cover - fallback when Qdrant is offline
        logger.warning("Qdrant client unavailable for resource metrics: %s", exc)

    # Create and return resource manager
    return ResourceManager(
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        qdrant_manager=qdrant_manager_instance,
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


def create_projection_service(db: AsyncSession) -> ProjectionService:
    """Create a ProjectionService instance with required dependencies."""

    projection_repo = ProjectionRunRepository(db)
    operation_repo = OperationRepository(db)
    collection_repo = CollectionRepository(db)

    return ProjectionService(
        db_session=db,
        projection_repo=projection_repo,
        operation_repo=operation_repo,
        collection_repo=collection_repo,
    )


async def get_projection_service(db: AsyncSession = Depends(get_db)) -> ProjectionService:
    """FastAPI dependency for ProjectionService injection."""

    return create_projection_service(db)


def create_search_service(
    db: AsyncSession,
    default_timeout: httpx.Timeout | None = None,
    retry_timeout_multiplier: float = 4.0,
) -> SearchService:
    """Create a SearchService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection
        default_timeout: Optional default timeout configuration for HTTP requests
        retry_timeout_multiplier: Multiplier for timeout values on retry attempts

    Returns:
        Configured SearchService instance
    """
    # Create repository instances
    collection_repo = CollectionRepository(db)

    # Create and return service
    return SearchService(
        db_session=db,
        collection_repo=collection_repo,
        default_timeout=default_timeout,
        retry_timeout_multiplier=retry_timeout_multiplier,
    )


async def get_search_service(db: AsyncSession = Depends(get_db)) -> SearchService:
    """FastAPI dependency for SearchService injection."""
    # Use default timeout configuration, can be customized if needed
    return create_search_service(db)


async def get_directory_scan_service() -> DirectoryScanService:
    """FastAPI dependency for DirectoryScanService injection.

    Note: DirectoryScanService doesn't require database access as it only
    provides preview functionality without persisting data.
    """
    return DirectoryScanService()


async def create_chunking_orchestrator(db: AsyncSession) -> ChunkingOrchestrator:
    """Create orchestrator using composition root."""

    return await container_get_chunking_orchestrator(db)


async def get_chunking_orchestrator(db: AsyncSession = Depends(get_db)) -> ChunkingOrchestrator:
    """FastAPI dependency for orchestrator injection (new architecture)."""

    return await container_get_chunking_orchestrator(db)


async def get_chunking_service(
    db: AsyncSession = Depends(get_db),
) -> ChunkingOrchestrator:
    """Backward-compatible dependency returning the orchestrator."""

    return await get_chunking_orchestrator(db)


# Expose commonly used dependency providers to builtins for tests that
# reference them without importing (legacy tests convenience)
try:  # pragma: no cover
    import builtins as _builtins

    _builtins.get_chunking_service = get_chunking_service  # type: ignore[attr-defined]
    _builtins.get_collection_service = get_collection_service  # type: ignore[attr-defined]
except Exception:
    pass
