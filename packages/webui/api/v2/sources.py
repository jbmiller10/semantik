"""
Collection Sources API v2 endpoints.

This module provides RESTful API endpoints for managing collection sources,
including sync configuration, pause/resume, and manual sync triggers.
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
from webui.api.schemas import (
    ErrorResponse,
    SourceCreate,
    SourceListResponse,
    SourceResponse,
    SourceUpdate,
)
from webui.auth import get_current_user
from webui.services.factory import get_source_service
from webui.services.source_service import SourceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/collections", tags=["sources-v2"])


def _source_to_response(source: Any) -> SourceResponse:
    """Convert a CollectionSource model to SourceResponse."""
    return SourceResponse(
        id=source.id,
        collection_id=source.collection_id,
        source_type=source.source_type,
        source_path=source.source_path,
        source_config=source.source_config or {},
        document_count=source.document_count,
        size_bytes=source.size_bytes,
        sync_mode=source.sync_mode,
        interval_minutes=source.interval_minutes,
        paused_at=source.paused_at,
        next_run_at=source.next_run_at,
        last_run_started_at=source.last_run_started_at,
        last_run_completed_at=source.last_run_completed_at,
        last_run_status=source.last_run_status,
        last_error=source.last_error,
        last_indexed_at=source.last_indexed_at,
        created_at=source.created_at,
        updated_at=source.updated_at,
    )


@router.get(
    "/{collection_id}/sources",
    response_model=SourceListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
    },
)
async def list_sources(
    collection_id: str,
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceListResponse:
    """List sources for a collection.

    Returns all sources configured for the collection with their sync status.
    """
    try:
        sources, total = await service.list_sources(
            user_id=int(current_user["id"]),
            collection_id=collection_id,
            offset=offset,
            limit=limit,
        )

        return SourceListResponse(
            items=[_source_to_response(s) for s in sources],
            total=total,
            offset=offset,
            limit=limit,
        )

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sources") from e


@router.post(
    "/{collection_id}/sources",
    response_model=SourceResponse,
    status_code=201,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        409: {"model": ErrorResponse, "description": "Source already exists"},
    },
)
async def create_source(
    collection_id: str,
    request: Request,  # noqa: ARG001
    create_request: SourceCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Create a new source for a collection.

    Creates a source with the specified configuration and sync settings.
    The source can be configured for one-time import or continuous sync.
    """
    try:
        source = await service.create_source(
            user_id=int(current_user["id"]),
            collection_id=collection_id,
            source_type=create_request.source_type,
            source_path=create_request.source_path,
            source_config=create_request.source_config,
            sync_mode=create_request.sync_mode.value,
            interval_minutes=create_request.interval_minutes,
        )

        return _source_to_response(source)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except EntityAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to create source: {e}")
        raise HTTPException(status_code=500, detail="Failed to create source") from e


@router.get(
    "/{collection_id}/sources/{source_id}",
    response_model=SourceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
    },
)
async def get_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Get a source by ID."""
    try:
        source = await service.get_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )

        return _source_to_response(source)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except Exception as e:
        logger.error(f"Failed to get source: {e}")
        raise HTTPException(status_code=500, detail="Failed to get source") from e


@router.patch(
    "/{collection_id}/sources/{source_id}",
    response_model=SourceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
    },
)
async def update_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    update_request: SourceUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Update a source's configuration.

    Updates the source configuration and/or sync settings.
    Note: Changing sync_mode to continuous requires interval_minutes.
    """
    try:
        source = await service.update_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
            source_config=update_request.source_config,
            sync_mode=update_request.sync_mode.value if update_request.sync_mode else None,
            interval_minutes=update_request.interval_minutes,
        )

        return _source_to_response(source)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to update source: {e}")
        raise HTTPException(status_code=500, detail="Failed to update source") from e


@router.delete(
    "/{collection_id}/sources/{source_id}",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
        409: {"model": ErrorResponse, "description": "Active operation in progress"},
    },
)
async def delete_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> dict[str, Any]:
    """Delete a source and its documents.

    This triggers a REMOVE_SOURCE operation to delete all documents
    and vectors associated with this source.

    Returns the operation details for tracking.
    """
    try:
        return await service.delete_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except InvalidStateError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to delete source: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete source") from e


@router.post(
    "/{collection_id}/sources/{source_id}/run",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
        409: {"model": ErrorResponse, "description": "Active operation in progress"},
    },
)
async def run_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> dict[str, Any]:
    """Trigger an immediate sync run for a source.

    Creates an APPEND operation for the source and dispatches it.
    For continuous sync sources, also updates the next scheduled run time.

    Returns the operation details for tracking.
    """
    try:
        return await service.run_now(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except InvalidStateError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to run source sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to run source sync") from e


@router.post(
    "/{collection_id}/sources/{source_id}/pause",
    response_model=SourceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Cannot pause non-continuous source"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
    },
)
async def pause_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Pause a continuous sync source.

    Stops the automatic sync schedule for the source.
    Manual sync via /run is still available while paused.
    """
    try:
        source = await service.pause(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )

        return _source_to_response(source)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to pause source: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause source") from e


@router.post(
    "/{collection_id}/sources/{source_id}/resume",
    response_model=SourceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Cannot resume non-continuous source"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Source not found"},
    },
)
async def resume_source(
    collection_id: str,  # noqa: ARG001 - used for routing
    source_id: int,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Resume a paused continuous sync source.

    Restarts the automatic sync schedule for the source.
    The next sync will be scheduled immediately.
    """
    try:
        source = await service.resume(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )

        return _source_to_response(source)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to resume source: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume source") from e
