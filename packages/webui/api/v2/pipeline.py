"""Pipeline API v2 endpoints.

This module provides API endpoints for pipeline operations including
route preview for testing how files would be routed through a pipeline DAG.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from webui.api.v2.pipeline_schemas import RoutePreviewResponse
from webui.auth import get_current_user
from webui.services.pipeline_preview_service import PipelinePreviewService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/pipeline", tags=["pipeline-v2"])


def get_pipeline_preview_service() -> PipelinePreviewService:
    """Dependency for getting the pipeline preview service."""
    return PipelinePreviewService()


@router.post(
    "/preview-route",
    response_model=RoutePreviewResponse,
    summary="Preview file routing through pipeline",
    responses={
        400: {"description": "Invalid DAG or file"},
        413: {"description": "File too large"},
    },
)
async def preview_route(
    file: UploadFile = File(..., description="Sample file to test routing"),
    dag: str = Form(..., description="Pipeline DAG as JSON string"),
    include_parser_metadata: bool = Form(True, description="Whether to run parser and include metadata"),
    _current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PipelinePreviewService = Depends(get_pipeline_preview_service),
) -> RoutePreviewResponse:
    """Preview how a file would be routed through a pipeline DAG.

    This endpoint allows testing pipeline routing decisions without
    actually processing the file. It evaluates all predicates and
    returns detailed information about which edges matched and why.

    The response includes:
    - File information (filename, mime_type, size)
    - Sniff results (detected content characteristics)
    - Routing stages with edge evaluation details
    - The final path through the pipeline
    - Parser metadata (if include_parser_metadata is True)

    Args:
        file: Sample file to test routing
        dag: Pipeline DAG definition as JSON string
        include_parser_metadata: Whether to run parser for metadata
        _current_user: Authenticated user (for auth)
        service: Pipeline preview service

    Returns:
        RoutePreviewResponse with detailed routing information

    Raises:
        HTTPException: If file is too large or DAG is invalid
    """
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {e}",
        ) from e

    # Validate file size (10MB limit for preview)
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large for preview. Maximum size is {max_size / 1024 / 1024:.1f}MB",
        )

    # Parse DAG JSON
    try:
        dag_dict = json.loads(dag)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid DAG JSON: {e}",
        ) from e

    # Get filename
    filename = file.filename or "unknown"

    # Run preview
    try:
        return await service.preview_route(
            file_content=content,
            filename=filename,
            dag=dag_dict,
            include_parser_metadata=include_parser_metadata,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Pipeline preview failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preview failed: {e}",
        ) from e


__all__ = [
    "router",
    "get_pipeline_preview_service",
]
