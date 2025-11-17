"""Projection API v2 endpoints (scaffolding)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse

from packages.shared.database.exceptions import AccessDeniedError, EntityNotFoundError, ValidationError
from packages.webui.api.schemas import ErrorResponse
from packages.webui.api.v2.schemas import (
    ProjectionBuildRequest,
    ProjectionListResponse,
    ProjectionMetadataResponse,
    ProjectionSelectionItem,
    ProjectionSelectionRequest,
    ProjectionSelectionResponse,
)
from packages.webui.auth import get_current_user
from packages.webui.services.factory import get_projection_service

if TYPE_CHECKING:
    from packages.webui.services.projection_service import ProjectionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/collections/{collection_id}/projections", tags=["projections-v2"])


def _to_metadata_response(
    collection_id: str, payload: dict[str, Any], *, fallback_id: str
) -> ProjectionMetadataResponse:
    """Normalise arbitrary projection metadata dictionaries."""

    return ProjectionMetadataResponse(
        id=payload.get("projection_id") or payload.get("id") or fallback_id,
        collection_id=payload.get("collection_id", collection_id),
        status=payload.get("status", "pending"),
        reducer=payload.get("reducer", "umap"),
        dimensionality=int(payload.get("dimensionality", 2) or 2),
        created_at=payload.get("created_at"),
        operation_id=payload.get("operation_id") or payload.get("operation_uuid"),
        operation_status=payload.get("operation_status"),
        message=payload.get("message"),
        config=payload.get("config"),
        meta=payload.get("meta"),
    )


@router.post(
    "",
    response_model=ProjectionMetadataResponse,
    status_code=202,
    responses={
        202: {"description": "Projection run accepted"},
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def start_projection(
    collection_id: str,
    request: ProjectionBuildRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
) -> ProjectionMetadataResponse:
    """Schedule a projection build for the given collection (scaffold)."""

    try:
        result = await service.start_projection_build(collection_id, int(current_user["id"]), request.model_dump())
    except (EntityNotFoundError, AccessDeniedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.debug("Projection run scheduled: %s", result)
    return _to_metadata_response(collection_id, result, fallback_id="pending-projection")


@router.get("", response_model=ProjectionListResponse)
async def list_projections(
    collection_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
) -> ProjectionListResponse:
    """List projections for a collection (placeholder)."""

    runs = await service.list_projections(collection_id, int(current_user["id"]))
    projections = [
        _to_metadata_response(collection_id, payload, fallback_id=f"projection-{idx}")
        for idx, payload in enumerate(runs, start=1)
    ]
    return ProjectionListResponse(projections=projections)


@router.get(
    "/{projection_id}",
    response_model=ProjectionMetadataResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_projection(
    collection_id: str,
    projection_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
) -> ProjectionMetadataResponse:
    """Return metadata for a projection run (placeholder)."""

    try:
        payload = await service.get_projection_metadata(collection_id, projection_id, int(current_user["id"]))
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _to_metadata_response(collection_id, payload, fallback_id=projection_id)


@router.get(
    "/{projection_id}/arrays/{artifact_name}",
    responses={
        200: {"description": "Projection artifact", "content": {"application/octet-stream": {}}},
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def stream_projection_artifact(
    collection_id: str,
    projection_id: str,
    artifact_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
):
    """Stream one of the stored projection artifact files."""

    try:
        artifact_path = await service.resolve_artifact_path(
            collection_id,
            projection_id,
            artifact_name,
            int(current_user["id"]),
        )
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_stat = artifact_path.stat()

    def _file_iterator(chunk_size: int = 1024 * 1024):
        with artifact_path.open("rb") as buffer:
            while True:
                data = buffer.read(chunk_size)
                if not data:
                    break
                yield data

    headers = {
        "Cache-Control": "private, max-age=3600",
        "Content-Length": str(file_stat.st_size),
        "X-Content-Type-Options": "nosniff",
        "Content-Disposition": f'attachment; filename="{artifact_path.name}"',
    }

    return StreamingResponse(
        _file_iterator(),
        media_type="application/octet-stream",
        headers=headers,
    )


@router.post(
    "/{projection_id}/select",
    response_model=ProjectionSelectionResponse,
)
async def select_projection_region(
    collection_id: str,
    projection_id: str,
    selection: ProjectionSelectionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
) -> ProjectionSelectionResponse:
    """Open a selection over a projection (placeholder)."""

    payload = await service.select_projection_region(
        collection_id,
        projection_id,
        selection.model_dump(),
        int(current_user["id"]),
    )
    items = [ProjectionSelectionItem(**item) for item in payload.get("items", [])]
    return ProjectionSelectionResponse(
        projection_id=projection_id,
        items=items,
        missing_ids=[int(mid) for mid in payload.get("missing_ids", [])],
        degraded=bool(payload.get("degraded", False)),
    )


@router.delete(
    "/{projection_id}",
    status_code=204,
    responses={
        204: {"description": "Projection deleted"},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
async def delete_projection(
    collection_id: str,
    projection_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ProjectionService = Depends(get_projection_service),
) -> Response:
    try:
        await service.delete_projection(collection_id, projection_id, int(current_user["id"]))
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return Response(status_code=204)
