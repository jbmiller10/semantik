"""
Collection API v2 endpoints.

This module provides RESTful API endpoints for collection management using
the new collection-centric architecture.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from shared.database.models import Collection
from webui.api.schemas import (
    AddSourceRequest,
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdate,
    DocumentListResponse,
    ErrorResponse,
    OperationResponse,
)
from webui.auth import get_current_user
from webui.dependencies import get_collection_for_user
from webui.rate_limiter import limiter
from webui.services.collection_service import CollectionService
from webui.services.factory import get_collection_service

SharedAccessDeniedError: type[BaseException] | None = None
try:  # pragma: no cover - shared module may not be installed in all environments
    from shared.database.exceptions import AccessDeniedError as _SharedAccessDeniedError
except Exception:  # pragma: no cover
    _SharedAccessDeniedError = None
else:
    SharedAccessDeniedError = _SharedAccessDeniedError

if SharedAccessDeniedError is not None and SharedAccessDeniedError is not AccessDeniedError:
    _ACCESS_DENIED_ERRORS: tuple[type[BaseException], ...] = (AccessDeniedError, SharedAccessDeniedError)
else:
    _ACCESS_DENIED_ERRORS = (AccessDeniedError,)

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
    try:
        # Build config, omitting fields that are None so the service can apply
        # sensible defaults instead of passing explicit nulls downstream.
        cfg: dict[str, Any] = {
            "embedding_model": create_request.embedding_model,
            "quantization": create_request.quantization,
            "is_public": create_request.is_public,
        }
        if create_request.chunk_size is not None:
            cfg["chunk_size"] = create_request.chunk_size
        if create_request.chunk_overlap is not None:
            cfg["chunk_overlap"] = create_request.chunk_overlap

        # Always include chunking_strategy and chunking_config for consistency with tests
        cfg["chunking_strategy"] = create_request.chunking_strategy
        cfg["chunking_config"] = create_request.chunking_config

        if create_request.metadata is not None:
            cfg["metadata"] = create_request.metadata

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
            created_at=collection["created_at"],
            updated_at=collection["updated_at"],
            document_count=collection["document_count"],
            vector_count=collection.get("vector_count", 0),
            status=collection["status"],
            status_message=collection.get("status_message"),
            initial_operation_id=operation["uuid"],  # Include the initial operation ID
        )

    except EntityAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=f"Collection with name '{create_request.name}' already exists",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create collection",
        ) from e


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
    try:
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

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list collections",
        ) from e


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
    try:
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

        # Perform update through service
        updated_collection = await service.update(
            collection_id=collection_uuid,
            user_id=int(current_user["id"]),
            updates=updates,
        )

        return CollectionResponse.from_collection(updated_collection)

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="Only the collection owner can update it",
        ) from e
    except EntityAlreadyExistsError as e:
        raise HTTPException(
            status_code=409,
            detail=f"Collection name '{request.name}' already exists",
        ) from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to update collection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update collection",
        ) from e


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

    try:
        await service.delete_collection(
            collection_id=collection_uuid,
            user_id=int(current_user["id"]),
        )
        logger.info(f"Successfully deleted collection {collection_uuid}")

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="Only the collection owner can delete it",
        ) from e
    except InvalidStateError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete collection",
        ) from e


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
    try:
        operation = await service.add_source(
            collection_id=collection_uuid,
            user_id=int(current_user["id"]),
            source_type=add_source_request.source_type,
            source_config=add_source_request.source_config or {},
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

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to modify this collection",
        ) from e
    except InvalidStateError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to add source: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to add source",
        ) from e


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
    try:
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

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to modify this collection",
        ) from e
    except InvalidStateError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to remove source: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to remove source",
        ) from e


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
    try:
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

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to reindex this collection",
        ) from e
    except InvalidStateError as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Failed to reindex collection: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reindex collection",
        ) from e


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
    try:
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

    except ValueError as e:
        # Service method raises ValueError for invalid filters
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this collection",
        ) from e
    except Exception as e:
        logger.error(f"Failed to list operations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list operations",
        ) from e


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
    try:
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
        from webui.api.schemas import DocumentResponse

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

    except ValueError as e:
        # Service method raises ValueError for invalid filters
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except _ACCESS_DENIED_ERRORS as e:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this collection",
        ) from e
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents",
        ) from e
