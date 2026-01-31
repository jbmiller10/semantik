"""Pipeline API v2 endpoints.

This module provides API endpoints for pipeline operations including
route preview for testing how files would be routed through a pipeline DAG.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from shared.plugins import load_plugins
from shared.plugins.registry import plugin_registry
from webui.api.v2.pipeline_schemas import (
    AvailablePredicateFieldsRequest,
    AvailablePredicateFieldsResponse,
    PredicateField,
    RoutePreviewResponse,
)
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


# =============================================================================
# Static field definitions for source and detected categories
# =============================================================================

# Source metadata fields (from connector, always available)
# NOTE: These are top-level FileReference attributes, not nested under metadata.source
SOURCE_FIELDS = [
    PredicateField(value="mime_type", label="MIME Type", category="source"),
    PredicateField(value="extension", label="Extension", category="source"),
    PredicateField(value="source_type", label="Source Type", category="source"),
    PredicateField(value="content_type", label="Content Type", category="source"),
]

# Detected metadata fields (from pre-routing sniff, always available)
DETECTED_FIELDS = [
    PredicateField(value="metadata.detected.is_scanned_pdf", label="Is Scanned PDF", category="detected"),
    PredicateField(value="metadata.detected.is_code", label="Is Code", category="detected"),
    PredicateField(value="metadata.detected.is_structured_data", label="Is Structured Data", category="detected"),
]

# Human-readable labels for parsed fields
PARSED_FIELD_LABELS = {
    "detected_language": "Detected Language",
    "approx_token_count": "Token Count",
    "line_count": "Line Count",
    "has_code_blocks": "Has Code Blocks",
    "page_count": "Page Count",
    "has_tables": "Has Tables",
    "has_images": "Has Images",
    "element_types": "Element Types",
}


def _get_parser_plugin_id_for_node(dag: dict[str, Any], node_id: str) -> str | None:
    """Get the parser plugin_id for a node in the DAG.

    Returns the plugin_id if the node is a parser, None otherwise.
    """
    if node_id == "_source":
        return None

    nodes = dag.get("nodes", [])
    for node in nodes:
        if node.get("id") == node_id and node.get("type") == "parser":
            plugin_id = node.get("plugin_id")
            return str(plugin_id) if plugin_id else None
    return None


@router.post(
    "/available-predicate-fields",
    response_model=AvailablePredicateFieldsResponse,
    summary="Get available predicate fields for an edge",
    responses={
        400: {"description": "Invalid DAG or node"},
    },
)
async def get_available_predicate_fields(
    request: AvailablePredicateFieldsRequest,
    _current_user: dict[str, Any] = Depends(get_current_user),
) -> AvailablePredicateFieldsResponse:
    """Get available predicate fields for an edge based on its source node.

    Returns fields that can be used in edge predicates for routing decisions.
    The available parsed.* fields depend on which parser is the source node:
    - From _source: No parsed.* fields (parser hasn't run yet)
    - From parser node: Only fields that parser emits

    Source and detected fields are always available regardless of source node.

    Args:
        request: Contains DAG and from_node identifier
        _current_user: Authenticated user (for auth)

    Returns:
        AvailablePredicateFieldsResponse with available fields
    """
    fields: list[PredicateField] = []

    # Always include source and detected fields
    fields.extend(SOURCE_FIELDS)
    fields.extend(DETECTED_FIELDS)

    # Check if from_node is a parser - if so, include its emitted fields
    parser_plugin_id = _get_parser_plugin_id_for_node(request.dag, request.from_node)
    if parser_plugin_id:
        # Ensure parser plugins are loaded
        if not plugin_registry.is_loaded(["parser"]):
            load_plugins(plugin_types=["parser"])

        # Get emitted fields for this parser
        emitted_fields = plugin_registry.get_parser_emitted_fields(parser_plugin_id)
        for field_name in emitted_fields:
            label = PARSED_FIELD_LABELS.get(field_name, field_name.replace("_", " ").title())
            fields.append(
                PredicateField(
                    value=f"metadata.parsed.{field_name}",
                    label=label,
                    category="parsed",
                )
            )

    return AvailablePredicateFieldsResponse(fields=fields)


__all__ = [
    "router",
    "get_pipeline_preview_service",
    "get_available_predicate_fields",
]
