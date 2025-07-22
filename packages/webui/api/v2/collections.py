"""
Collection API v2 endpoints.

This module provides RESTful API endpoints for collection management using
the new collection-centric architecture.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db
from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from packages.shared.database.models import Collection
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.api.schemas import (
    AddSourceRequest,
    CollectionCreate,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdate,
    DocumentListResponse,
    ErrorResponse,
    OperationResponse,
)
from packages.webui.auth import get_current_user
from packages.webui.dependencies import (
    get_collection_for_user,
    get_collection_repository,
    get_operation_repository,
    get_document_repository,
)
from packages.webui.rate_limiter import limiter
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import get_collection_service

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
        collection, operation = await service.create_collection(
            user_id=int(current_user["id"]),
            name=create_request.name,
            description=create_request.description,
            config={
                "embedding_model": create_request.embedding_model,
                "quantization": create_request.quantization,
                "chunk_size": create_request.chunk_size,
                "chunk_overlap": create_request.chunk_overlap,
                "is_public": create_request.is_public,
                "metadata": create_request.metadata,
            },
        )

        # Convert to response model
        return CollectionResponse(
            id=collection["id"],
            name=collection["name"],
            description=collection["description"],
            owner_id=collection["owner_id"],
            vector_store_name=collection["vector_store_name"],
            embedding_model=collection["embedding_model"],
            quantization=collection["quantization"],
            chunk_size=collection["chunk_size"],
            chunk_overlap=collection["chunk_overlap"],
            is_public=collection["is_public"],
            metadata=collection["metadata"],
            created_at=collection["created_at"],
            updated_at=collection["updated_at"],
            document_count=collection["document_count"],
            vector_count=collection.get("vector_count", 0),
            status=collection["status"],
            status_message=collection.get("status_message"),
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
    repo: CollectionRepository = Depends(get_collection_repository),
) -> CollectionListResponse:
    """List collections accessible to the current user.

    Returns a paginated list of collections that the user owns or has access to.
    Public collections are included by default.
    """
    try:
        offset = (page - 1) * per_page

        collections, total = await repo.list_for_user(
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
    collection_uuid: str,  # noqa: ARG001
    request: CollectionUpdate,
    collection: Collection = Depends(get_collection_for_user),
    repo: CollectionRepository = Depends(get_collection_repository),
    db: AsyncSession = Depends(get_db),
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

        # Perform update
        updated_collection = await repo.update(str(collection.id), updates)

        await db.commit()

        return CollectionResponse.from_collection(updated_collection)

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
    try:
        await service.delete_collection(
            collection_id=collection_uuid,
            user_id=int(current_user["id"]),
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except AccessDeniedError as e:
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
            source_path=add_source_request.source_path,
            source_config=add_source_request.config or {},
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
    except AccessDeniedError as e:
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
    except AccessDeniedError as e:
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
    except AccessDeniedError as e:
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
    repo: OperationRepository = Depends(get_operation_repository),
) -> list[OperationResponse]:
    """List operations for a collection.

    Returns a paginated list of operations performed on the collection,
    ordered by creation date (newest first).
    """
    try:
        offset = (page - 1) * per_page

        # Convert string parameters to enums if provided
        from packages.shared.database.models import OperationStatus, OperationType

        status_enum = None
        if status:
            try:
                status_enum = OperationStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}",
                ) from None

        type_enum = None
        if operation_type:
            try:
                type_enum = OperationType(operation_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type: {operation_type}",
                ) from None

        operations, total = await repo.list_for_collection(
            collection_id=collection_uuid,
            user_id=int(current_user["id"]),
            status=status_enum,
            operation_type=type_enum,
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

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_uuid}' not found",
        ) from e
    except AccessDeniedError as e:
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
    collection: Collection = Depends(get_collection_for_user),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    status: str | None = Query(None, description="Filter by document status"),
    doc_repo: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    """List documents in a collection.

    Returns a paginated list of documents in the collection.
    """
    try:
        offset = (page - 1) * per_page

        # Convert string status to enum if provided
        from packages.shared.database.models import DocumentStatus

        status_enum = None
        if status:
            try:
                status_enum = DocumentStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}",
                ) from None

        documents, total = await doc_repo.list_by_collection(
            collection_id=str(collection.id),
            status=status_enum,
            offset=offset,
            limit=per_page,
        )

        # Convert ORM objects to response models
        from packages.webui.api.schemas import DocumentResponse

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

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents",
        ) from e
