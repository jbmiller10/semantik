"""
Collection API v2 endpoints.

This module provides RESTful API endpoints for collection management using
the new collection-centric architecture.

Exception handling is centralized via global exception handlers registered
in webui.middleware.exception_handlers. Service-layer exceptions are automatically
converted to appropriate HTTP responses:
- EntityNotFoundError -> 404
- EntityAlreadyExistsError -> 409
- ValidationError -> 400
- InvalidStateError -> 409
- AccessDeniedError -> 403
- Unhandled exceptions -> 500 (with sanitized error messages)
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from shared.database.models import Collection
from webui.api.schemas import (
    AddSourceRequest,
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    CollectionSyncRunResponse,
    CollectionUpdate,
    DocumentListResponse,
    DocumentResponse,
    EnableSparseIndexRequest,
    ErrorResponse,
    OperationResponse,
    SparseIndexStatusResponse,
    SparseReindexProgressResponse,
    SparseReindexResponse,
    SyncRunListResponse,
)
from webui.auth import get_current_user
from webui.dependencies import get_collection_for_user
from webui.rate_limiter import limiter
from webui.services.collection_service import CollectionService
from webui.services.factory import get_collection_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/collections", tags=["collections-v2"])


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=201,
    responses={
        409: {"model": ErrorResponse, "description": "Collection name already exists"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def create_collection(
    request: Request,  # noqa: ARG001
    create_request: CollectionCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    """Create a new collection.

    Creates a new collection with the specified configuration. The collection
    will be created in a PENDING state and an initial indexing operation will
    be automatically triggered.
    """
    # Build config, omitting fields that are None so the service can apply
    # sensible defaults instead of passing explicit nulls downstream.
    cfg: dict[str, Any] = {
        "embedding_model": create_request.embedding_model,
        "quantization": create_request.quantization,
        "is_public": create_request.is_public,
        "sync_mode": create_request.sync_mode.value,
    }
    if create_request.sync_mode == "continuous":
        cfg["sync_interval_minutes"] = create_request.sync_interval_minutes
    if create_request.chunk_size is not None:
        cfg["chunk_size"] = create_request.chunk_size
    if create_request.chunk_overlap is not None:
        cfg["chunk_overlap"] = create_request.chunk_overlap

    # Always include chunking_strategy and chunking_config for consistency with tests
    cfg["chunking_strategy"] = create_request.chunking_strategy
    cfg["chunking_config"] = create_request.chunking_config

    if create_request.metadata is not None:
        cfg["metadata"] = create_request.metadata

    # Reranker and extraction config (Phase 2 plugin extensibility)
    if create_request.default_reranker_id is not None:
        cfg["default_reranker_id"] = create_request.default_reranker_id
    if create_request.extraction_config is not None:
        cfg["extraction_config"] = create_request.extraction_config

    # Sparse indexing configuration
    if create_request.sparse_index_config is not None:
        cfg["sparse_index_config"] = create_request.sparse_index_config.model_dump()

    collection, operation = await service.create_collection(
        user_id=int(current_user["id"]),
        name=create_request.name,
        description=create_request.description,
        config=cfg,
    )

    # Convert to response model and add operation uuid
    return CollectionResponse(
        id=collection["id"],
        name=collection["name"],
        description=collection["description"],
        owner_id=collection["owner_id"],
        vector_store_name=collection["vector_store_name"],
        embedding_model=collection["embedding_model"],
        quantization=collection["quantization"],
        chunk_size=collection.get("chunk_size"),
        chunk_overlap=collection.get("chunk_overlap"),
        chunking_strategy=collection.get("chunking_strategy"),
        chunking_config=collection.get("chunking_config"),
        is_public=collection["is_public"],
        metadata=collection["metadata"],
        # Reranker and extraction config (Phase 2 plugin extensibility)
        default_reranker_id=collection.get("default_reranker_id"),
        extraction_config=collection.get("extraction_config"),
        created_at=collection["created_at"],
        updated_at=collection["updated_at"],
        document_count=collection["document_count"],
        vector_count=collection.get("vector_count", 0),
        total_size_bytes=collection.get("total_size_bytes", 0),
        status=collection["status"],
        status_message=collection.get("status_message"),
        # Sync policy fields
        sync_mode=collection.get("sync_mode", "one_time") or "one_time",
        sync_interval_minutes=collection.get("sync_interval_minutes"),
        sync_paused_at=collection.get("sync_paused_at"),
        sync_next_run_at=collection.get("sync_next_run_at"),
        # Sync run tracking
        sync_last_run_started_at=collection.get("sync_last_run_started_at"),
        sync_last_run_completed_at=collection.get("sync_last_run_completed_at"),
        sync_last_run_status=collection.get("sync_last_run_status"),
        sync_last_error=collection.get("sync_last_error"),
        initial_operation_id=operation["uuid"],  # Include the initial operation ID
    )


@router.get(
    "",
    response_model=CollectionListResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_collections(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    include_public: bool = Query(True, description="Include public collections"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionListResponse:
    """List collections accessible to the current user.

    Returns a paginated list of collections that the user owns or has access to.
    Public collections are included by default.
    """
    offset = (page - 1) * per_page

    collections, total = await service.list_for_user(
        user_id=int(current_user["id"]),
        offset=offset,
        limit=per_page,
        include_public=include_public,
    )

    # Convert ORM objects to response models
    collection_responses = [CollectionResponse.from_collection(col) for col in collections]

    return CollectionListResponse(
        collections=collection_responses,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/{collection_uuid}",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def get_collection(
    collection: Collection = Depends(get_collection_for_user),
) -> CollectionResponse:
    """Get detailed information about a specific collection.

    Returns full details about a collection including its configuration and statistics.
    """
    return CollectionResponse.from_collection(collection)


@router.put(
    "/{collection_uuid}",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Name already exists"},
    },
)
async def update_collection(
    collection_uuid: str,
    request: CollectionUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    """Update collection metadata.

    Updates the editable fields of a collection. Only the collection owner can
    perform updates. Note that embedding model and chunk settings cannot be
    changed after creation - use reindexing for those changes.
    """
    # Build updates dict from non-None values
    updates: dict[str, Any] = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.description is not None:
        updates["description"] = request.description
    if request.is_public is not None:
        updates["is_public"] = request.is_public
    if request.metadata is not None:
        updates["meta"] = request.metadata
    if request.sync_mode is not None:
        updates["sync_mode"] = request.sync_mode.value
    if request.sync_interval_minutes is not None:
        updates["sync_interval_minutes"] = request.sync_interval_minutes
    # Reranker and extraction config (Phase 2 plugin extensibility)
    if request.default_reranker_id is not None:
        updates["default_reranker_id"] = request.default_reranker_id
    if request.extraction_config is not None:
        updates["extraction_config"] = request.extraction_config

    # Perform update through service
    updated_collection = await service.update(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        updates=updates,
    )

    return CollectionResponse.from_collection(updated_collection)


@router.delete(
    "/{collection_uuid}",
    status_code=204,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Cannot delete - operation in progress"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/hour")
async def delete_collection(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> None:
    """Delete a collection and all associated data.

    Permanently deletes a collection including all documents, vectors, and
    operations. This action cannot be undone. Only the collection owner can
    delete a collection.
    """
    logger.info(f"User {current_user['id']} attempting to delete collection {collection_uuid}")

    await service.delete_collection(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )
    logger.info(f"Successfully deleted collection {collection_uuid}")


# Collection operation endpoints


@router.post(
    "/{collection_uuid}/sources",
    response_model=OperationResponse,
    status_code=202,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Invalid collection state"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/hour")
async def add_source(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    add_source_request: AddSourceRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> OperationResponse:
    """Add a source to the collection.

    Adds a new source (file or directory) to the collection and triggers
    indexing of its contents. Returns an operation that can be tracked.
    """
    operation = await service.add_source(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        source_type=add_source_request.source_type,
        source_config=add_source_request.source_config,
        legacy_source_path=add_source_request.source_path,
        additional_config=add_source_request.config or {},
    )

    # Convert to response model
    return OperationResponse(
        id=operation["uuid"],
        collection_id=operation["collection_id"],
        type=operation["type"],
        status=operation["status"],
        config=operation["config"],
        created_at=operation["created_at"],
        started_at=operation.get("started_at"),
        completed_at=operation.get("completed_at"),
        error_message=operation.get("error_message"),
    )


@router.delete(
    "/{collection_uuid}/sources",
    response_model=OperationResponse,
    status_code=202,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Invalid collection state"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/hour")
async def remove_source(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    source_path: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> OperationResponse:
    """Remove a source from the collection.

    Removes all documents and vectors associated with the specified source path.
    Returns an operation that can be tracked.
    """
    operation = await service.remove_source(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        source_path=source_path,
    )

    # Convert to response model
    return OperationResponse(
        id=operation["uuid"],
        collection_id=operation["collection_id"],
        type=operation["type"],
        status=operation["status"],
        config=operation["config"],
        created_at=operation["created_at"],
        started_at=operation.get("started_at"),
        completed_at=operation.get("completed_at"),
        error_message=operation.get("error_message"),
    )


@router.post(
    "/{collection_uuid}/reindex",
    response_model=OperationResponse,
    status_code=202,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Invalid collection state"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("1/5minutes")
async def reindex_collection(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    config_updates: dict[str, Any] | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> OperationResponse:
    """Reindex the entire collection.

    Triggers a complete reindexing of the collection. This uses blue-green
    deployment for zero downtime. Optionally update configuration like
    embedding model or chunk size.
    """
    operation = await service.reindex_collection(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        config_updates=config_updates,
    )

    # Convert to response model
    return OperationResponse(
        id=operation["uuid"],
        collection_id=operation["collection_id"],
        type=operation["type"],
        status=operation["status"],
        config=operation["config"],
        created_at=operation["created_at"],
        started_at=operation.get("started_at"),
        completed_at=operation.get("completed_at"),
        error_message=operation.get("error_message"),
    )


@router.get(
    "/{collection_uuid}/operations",
    response_model=list[OperationResponse],
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def list_collection_operations(
    collection_uuid: str,
    status: str | None = Query(None, description="Filter by operation status"),
    operation_type: str | None = Query(None, description="Filter by operation type"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> list[OperationResponse]:
    """List operations for a collection.

    Returns a paginated list of operations performed on the collection,
    ordered by creation date (newest first).
    """
    offset = (page - 1) * per_page

    # Delegate all filtering logic to service
    operations, total = await service.list_operations_filtered(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        status=status,
        operation_type=operation_type,
        offset=offset,
        limit=per_page,
    )

    # Convert ORM objects to response models
    return [
        OperationResponse(
            id=op.uuid,
            collection_id=op.collection_id,
            type=op.type.value,
            status=op.status.value,
            config=op.config,
            created_at=op.created_at,
            started_at=op.started_at,
            completed_at=op.completed_at,
            error_message=op.error_message,
        )
        for op in operations
    ]


@router.get(
    "/{collection_uuid}/documents",
    response_model=DocumentListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def list_collection_documents(
    collection_uuid: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    status: str | None = Query(None, description="Filter by document status"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> DocumentListResponse:
    """List documents in a collection.

    Returns a paginated list of documents in the collection.
    """
    offset = (page - 1) * per_page

    # Delegate all filtering logic to service
    documents, total = await service.list_documents_filtered(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        status=status,
        offset=offset,
        limit=per_page,
    )

    # Convert ORM objects to response models
    document_responses = [
        DocumentResponse(
            id=doc.id,
            collection_id=doc.collection_id,
            file_name=doc.file_name,
            file_path=doc.file_path,
            file_size=doc.file_size,
            mime_type=doc.mime_type,
            content_hash=doc.content_hash,
            status=doc.status.value,
            error_message=doc.error_message,
            chunk_count=doc.chunk_count,
            metadata=doc.meta,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
        for doc in documents
    ]

    return DocumentListResponse(
        documents=document_responses,
        total=total,
        page=page,
        per_page=per_page,
    )


# =============================================================================
# Collection Sync Endpoints
# =============================================================================


@router.post(
    "/{collection_uuid}/sync/run",
    response_model=CollectionSyncRunResponse,
    status_code=202,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        409: {"model": ErrorResponse, "description": "Invalid collection state or operation in progress"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/hour")
async def run_collection_sync(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionSyncRunResponse:
    """Trigger a sync run for all sources in the collection.

    Fans out APPEND operations for each source and creates a sync run record
    to track completion aggregation. Returns 409 if collection has active
    operations or is not in a valid state (READY/DEGRADED).
    """
    sync_run = await service.run_collection_sync(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        triggered_by="manual",
    )

    return CollectionSyncRunResponse(
        id=sync_run.id,
        collection_id=sync_run.collection_id,
        triggered_by=sync_run.triggered_by,
        started_at=sync_run.started_at,
        completed_at=sync_run.completed_at,
        status=sync_run.status,
        expected_sources=sync_run.expected_sources,
        completed_sources=sync_run.completed_sources,
        failed_sources=sync_run.failed_sources,
        partial_sources=sync_run.partial_sources,
        error_summary=sync_run.error_summary,
    )


@router.post(
    "/{collection_uuid}/sync/pause",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        400: {"model": ErrorResponse, "description": "Collection is not in continuous sync mode"},
    },
)
async def pause_collection_sync(
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    """Pause continuous sync for a collection.

    Sets sync_paused_at timestamp to pause scheduled sync runs.
    Collection must be in continuous sync mode.
    """
    collection = await service.pause_collection_sync(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )

    return CollectionResponse.from_collection(collection)


@router.post(
    "/{collection_uuid}/sync/resume",
    response_model=CollectionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        400: {"model": ErrorResponse, "description": "Collection is not paused"},
    },
)
async def resume_collection_sync(
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> CollectionResponse:
    """Resume continuous sync for a collection.

    Clears sync_paused_at and recalculates sync_next_run_at.
    Collection must be paused and in continuous sync mode.
    """
    collection = await service.resume_collection_sync(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )

    return CollectionResponse.from_collection(collection)


@router.get(
    "/{collection_uuid}/sync/runs",
    response_model=SyncRunListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def list_collection_sync_runs(
    collection_uuid: str,
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> SyncRunListResponse:
    """List sync runs for a collection.

    Returns a paginated list of sync runs ordered by start time (newest first).
    """
    sync_runs, total = await service.list_collection_sync_runs(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        offset=offset,
        limit=limit,
    )

    return SyncRunListResponse(
        items=[
            CollectionSyncRunResponse(
                id=run.id,
                collection_id=run.collection_id,
                triggered_by=run.triggered_by,
                started_at=run.started_at,
                completed_at=run.completed_at,
                status=run.status,
                expected_sources=run.expected_sources,
                completed_sources=run.completed_sources,
                failed_sources=run.failed_sources,
                partial_sources=run.partial_sources,
                error_summary=run.error_summary,
            )
            for run in sync_runs
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


# =============================================================================
# Sparse Index Management Endpoints (Phase 3)
# =============================================================================


@router.get(
    "/{collection_uuid}/sparse-index",
    response_model=SparseIndexStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def get_sparse_index_status(
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> SparseIndexStatusResponse:
    """Get sparse index status for a collection.

    Returns the current sparse indexing configuration and statistics.
    """
    sparse_config = await service.get_sparse_index_config(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )

    if sparse_config is None:
        return SparseIndexStatusResponse(enabled=False)

    return SparseIndexStatusResponse(
        enabled=sparse_config.get("enabled", False),
        plugin_id=sparse_config.get("plugin_id"),
        sparse_collection_name=sparse_config.get("sparse_collection_name"),
        model_config_data=sparse_config.get("model_config"),
        document_count=sparse_config.get("document_count"),
        created_at=sparse_config.get("created_at"),
        last_indexed_at=sparse_config.get("last_indexed_at"),
    )


@router.post(
    "/{collection_uuid}/sparse-index",
    response_model=SparseIndexStatusResponse,
    status_code=201,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        400: {"model": ErrorResponse, "description": "Invalid request or plugin not found"},
        409: {"model": ErrorResponse, "description": "Sparse indexing already enabled"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/hour")
async def enable_sparse_index(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    enable_request: EnableSparseIndexRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> SparseIndexStatusResponse:
    """Enable sparse indexing for a collection.

    Creates a sparse Qdrant collection and optionally triggers reindexing
    of existing documents.
    """
    sparse_config = await service.enable_sparse_index(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        plugin_id=enable_request.plugin_id,
        model_config=enable_request.model_config_data,
        reindex_existing=enable_request.reindex_existing,
    )

    return SparseIndexStatusResponse(
        enabled=True,
        plugin_id=sparse_config.get("plugin_id"),
        sparse_collection_name=sparse_config.get("sparse_collection_name"),
        model_config_data=sparse_config.get("model_config"),
        document_count=sparse_config.get("document_count", 0),
        created_at=sparse_config.get("created_at"),
        last_indexed_at=sparse_config.get("last_indexed_at"),
    )


@router.delete(
    "/{collection_uuid}/sparse-index",
    status_code=204,
    responses={
        404: {"model": ErrorResponse, "description": "Collection not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/hour")
async def disable_sparse_index(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> None:
    """Disable sparse indexing for a collection.

    Deletes the sparse Qdrant collection and removes configuration.
    """
    await service.disable_sparse_index(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )


@router.post(
    "/{collection_uuid}/sparse-index/reindex",
    response_model=SparseReindexResponse,
    status_code=202,
    responses={
        404: {"model": ErrorResponse, "description": "Collection or sparse indexing not enabled"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("2/hour")
async def trigger_sparse_reindex(
    request: Request,  # noqa: ARG001
    collection_uuid: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> SparseReindexResponse:
    """Trigger a full sparse reindex of the collection.

    Returns a job ID that can be used to track progress.
    """
    job_info = await service.trigger_sparse_reindex(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
    )

    return SparseReindexResponse(
        job_id=job_info["job_id"],
        status=job_info["status"],
        collection_uuid=collection_uuid,
        plugin_id=job_info["plugin_id"],
    )


@router.get(
    "/{collection_uuid}/sparse-index/reindex/{job_id}",
    response_model=SparseReindexProgressResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Collection or job not found"},
        403: {"model": ErrorResponse, "description": "Access denied"},
    },
)
async def get_sparse_reindex_progress(
    collection_uuid: str,
    job_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: CollectionService = Depends(get_collection_service),
) -> SparseReindexProgressResponse:
    """Get progress of a sparse reindex job.

    Polls the Celery task state to report progress.
    """
    progress = await service.get_sparse_reindex_progress(
        collection_id=collection_uuid,
        user_id=int(current_user["id"]),
        job_id=job_id,
    )

    return SparseReindexProgressResponse(
        job_id=job_id,
        status=progress["status"],
        progress=progress.get("progress"),
        documents_processed=progress.get("documents_processed"),
        total_documents=progress.get("total_documents"),
        error=progress.get("error"),
    )
