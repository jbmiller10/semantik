"""Factory functions for creating service instances with dependencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import httpx
    from sqlalchemy.ext.asyncio import AsyncSession

    from .api_key_service import ApiKeyService
    from .benchmark_dataset_service import BenchmarkDatasetService
    from .benchmark_service import BenchmarkService
    from .mcp_profile_service import MCPProfileService

from fastapi import Depends

from shared.database import get_db
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from shared.database.repositories.projection_run_repository import ProjectionRunRepository
from shared.managers import QdrantManager
from webui.qdrant import qdrant_manager as qdrant_connection_manager

from .chunking.container import (
    get_chunking_orchestrator as container_get_chunking_orchestrator,
    get_redis_manager as container_get_redis_manager,
)
from .chunking.orchestrator import ChunkingOrchestrator
from .collection_service import CollectionService
from .directory_scan_service import DirectoryScanService
from .operation_service import OperationService
from .projection_service import ProjectionService
from .redis_manager import RedisManager
from .search_service import SearchService
from .source_service import SourceService

logger = logging.getLogger(__name__)


def get_redis_manager() -> RedisManager:
    """Backward-compatible wrapper over chunking container Redis manager."""

    manager = cast(RedisManager, container_get_redis_manager())
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
    collection_source_repo = CollectionSourceRepository(db)

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
        collection_source_repo=collection_source_repo,
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

    orchestrator = await container_get_chunking_orchestrator(db)
    return cast(ChunkingOrchestrator, orchestrator)


async def get_chunking_orchestrator(db: AsyncSession = Depends(get_db)) -> ChunkingOrchestrator:
    """FastAPI dependency for orchestrator injection (new architecture)."""

    orchestrator = await container_get_chunking_orchestrator(db)
    return cast(ChunkingOrchestrator, orchestrator)


async def get_chunking_service(
    db: AsyncSession = Depends(get_db),
) -> ChunkingOrchestrator:
    """Backward-compatible dependency returning the orchestrator."""

    return await get_chunking_orchestrator(db)


def create_source_service(db: AsyncSession) -> SourceService:
    """Create a SourceService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured SourceService instance
    """
    from shared.database.repositories.connector_secret_repository import ConnectorSecretRepository
    from shared.utils.encryption import SecretEncryption

    # Create repository instances
    collection_repo = CollectionRepository(db)
    source_repo = CollectionSourceRepository(db)
    operation_repo = OperationRepository(db)

    # Create secret repository only if encryption is configured
    secret_repo: ConnectorSecretRepository | None = None
    if SecretEncryption.is_configured():
        secret_repo = ConnectorSecretRepository(db)
    else:
        logger.debug("Connector secrets encryption not configured - secrets storage disabled")

    # Create and return service
    return SourceService(
        db_session=db,
        collection_repo=collection_repo,
        source_repo=source_repo,
        operation_repo=operation_repo,
        secret_repo=secret_repo,
    )


async def get_source_service(db: AsyncSession = Depends(get_db)) -> SourceService:
    """FastAPI dependency for SourceService injection."""
    return create_source_service(db)


def create_mcp_profile_service(db: AsyncSession) -> MCPProfileService:
    """Create an MCPProfileService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured MCPProfileService instance
    """
    # Lazy import to avoid circular dependency
    from .mcp_profile_service import MCPProfileService

    return MCPProfileService(db_session=db)


async def get_mcp_profile_service(db: AsyncSession = Depends(get_db)) -> MCPProfileService:
    """FastAPI dependency for MCPProfileService injection."""
    return create_mcp_profile_service(db)


def create_api_key_service(db: AsyncSession) -> ApiKeyService:
    """Create an ApiKeyService instance with required dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured ApiKeyService instance
    """
    # Lazy import to avoid circular dependency
    from .api_key_service import ApiKeyService

    return ApiKeyService(db_session=db)


async def get_api_key_service(db: AsyncSession = Depends(get_db)) -> ApiKeyService:
    """FastAPI dependency for ApiKeyService injection."""
    return create_api_key_service(db)


def create_benchmark_dataset_service(db: AsyncSession) -> BenchmarkDatasetService:
    """Create a BenchmarkDatasetService instance with dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured BenchmarkDatasetService instance
    """
    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository

    from .benchmark_dataset_service import BenchmarkDatasetService

    benchmark_dataset_repo = BenchmarkDatasetRepository(db)
    collection_repo = CollectionRepository(db)
    document_repo = DocumentRepository(db)
    operation_repo = OperationRepository(db)

    return BenchmarkDatasetService(
        db_session=db,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        document_repo=document_repo,
        operation_repo=operation_repo,
    )


async def get_benchmark_dataset_service(
    db: AsyncSession = Depends(get_db),
) -> BenchmarkDatasetService:
    """FastAPI dependency for BenchmarkDatasetService injection."""
    return create_benchmark_dataset_service(db)


def create_benchmark_service(db: AsyncSession) -> BenchmarkService:
    """Create a BenchmarkService instance with dependencies.

    Args:
        db: AsyncSession instance from FastAPI's dependency injection

    Returns:
        Configured BenchmarkService instance
    """
    from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
    from shared.database.repositories.benchmark_repository import BenchmarkRepository

    from .benchmark_service import BenchmarkService

    benchmark_repo = BenchmarkRepository(db)
    benchmark_dataset_repo = BenchmarkDatasetRepository(db)
    collection_repo = CollectionRepository(db)
    operation_repo = OperationRepository(db)
    search_service = create_search_service(db)

    return BenchmarkService(
        db_session=db,
        benchmark_repo=benchmark_repo,
        benchmark_dataset_repo=benchmark_dataset_repo,
        collection_repo=collection_repo,
        operation_repo=operation_repo,
        search_service=search_service,
    )


async def get_benchmark_service(
    db: AsyncSession = Depends(get_db),
) -> BenchmarkService:
    """FastAPI dependency for BenchmarkService injection."""
    return create_benchmark_service(db)


# Expose commonly used dependency providers to builtins for tests that
# reference them without importing (legacy tests convenience)
try:  # pragma: no cover
    import builtins as _builtins

    _builtins.get_chunking_service = get_chunking_service  # type: ignore[attr-defined]
    _builtins.get_collection_service = get_collection_service  # type: ignore[attr-defined]
except Exception:
    pass
