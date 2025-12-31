"""
Collection Sources API v2 endpoints.

This module provides RESTful API endpoints for managing collection sources.
Note: Sync policy (mode, interval, pause/resume) is managed at collection level.
Sources only track per-source telemetry (last_run_* fields).
"""

import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query

from shared.database.exceptions import (
    AccessDeniedError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from shared.utils.encryption import EncryptionNotConfiguredError
from webui.api.schemas import (
    ErrorResponse,
    SourceListResponse,
    SourceResponse,
    SourceUpdate,
)
from webui.auth import get_current_user
from webui.services.factory import get_source_service
from webui.services.source_service import SourceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/collections", tags=["sources-v2"])


def _source_to_response(source: Any, secret_types: list[str] | None = None) -> SourceResponse:
    """Convert a CollectionSource model to SourceResponse.

    Note: Sync policy (mode, interval, pause) is managed at collection level.
    Sources only track per-source telemetry (last_run_* fields).

    Args:
        source: CollectionSource model instance
        secret_types: List of secret types configured for this source (if known)
    """
    secret_types_set = set(secret_types or [])

    return SourceResponse(
        id=source.id,
        collection_id=source.collection_id,
        source_type=source.source_type,
        source_path=source.source_path,
        source_config=source.source_config or {},
        document_count=source.document_count,
        size_bytes=source.size_bytes,
        # Per-source sync telemetry (sync policy is at collection level)
        last_run_started_at=source.last_run_started_at,
        last_run_completed_at=source.last_run_completed_at,
        last_run_status=source.last_run_status,
        last_error=source.last_error,
        last_indexed_at=source.last_indexed_at,
        created_at=source.created_at,
        updated_at=source.updated_at,
        has_password="password" in secret_types_set,
        has_token="token" in secret_types_set,
        has_ssh_key="ssh_key" in secret_types_set,
        has_ssh_passphrase="ssh_passphrase" in secret_types_set,
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
    Includes secret indicators (has_password, has_token, etc.) for each source.
    """
    try:
        sources_with_secrets, total = await service.list_sources(
            user_id=int(current_user["id"]),
            collection_id=collection_id,
            offset=offset,
            limit=limit,
            include_secret_types=True,
        )

        return SourceListResponse(
            items=[_source_to_response(s, st) for s, st in sources_with_secrets],
            total=total,
            offset=offset,
            limit=limit,
        )

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except Exception as e:
        logger.error("Failed to list sources: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list sources") from e


# Note: create_source endpoint removed - sources are created via
# POST /api/v2/collections/{collection_id}/sources (in collections.py)


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
    """Get a source by ID.

    Returns source details including secret indicators (has_password, etc.).
    """
    try:
        source, secret_types = await service.get_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
            include_secret_types=True,
        )

        return _source_to_response(source, secret_types)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except Exception as e:
        logger.error("Failed to get source: %s", e, exc_info=True)
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

    Note: Sync policy (mode, interval, pause) is managed at collection level.

    Secrets can be updated by providing new values. Set a secret key to an
    empty string to delete that secret.
    """
    try:
        source, secret_types = await service.update_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
            source_config=update_request.source_config,
            secrets=update_request.secrets,
        )

        return _source_to_response(source, secret_types)

    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EncryptionNotConfiguredError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to update source: %s", e, exc_info=True)
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
        result = await service.delete_source(
            user_id=int(current_user["id"]),
            source_id=source_id,
        )
        return cast(dict[str, Any], result)
    except EntityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AccessDeniedError as e:
        raise HTTPException(status_code=403, detail="Access denied") from e
    except InvalidStateError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error("Failed to delete source: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete source") from e


# Note: run_source, pause_source, resume_source endpoints removed.
# Sync operations are now managed at collection level via:
# - POST /api/v2/collections/{collection_id}/sync/run
# - POST /api/v2/collections/{collection_id}/sync/pause
# - POST /api/v2/collections/{collection_id}/sync/resume
