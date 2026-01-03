"""Extractor plugin discovery and testing API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry
from webui.api.schemas import ErrorResponse
from webui.auth import get_current_user
from webui.services.extractor_service import get_extractor_service

router = APIRouter(prefix="/api/v2/extractors", tags=["extractors-v2"])


class ExtractorInfo(BaseModel):
    """Extractor plugin information."""

    id: str = Field(..., description="Plugin ID")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Plugin description")
    version: str = Field(..., description="Plugin version")
    builtin: bool = Field(..., description="Whether this is a built-in plugin")
    supported_extractions: list[str] = Field(
        default_factory=list,
        description="Supported extraction types (keywords, entities, etc.)",
    )


class ExtractorListResponse(BaseModel):
    """Response for listing extractors."""

    extractors: list[ExtractorInfo]
    total: int


class ExtractorManifestSchema(BaseModel):
    """Full extractor manifest."""

    id: str
    type: str
    version: str
    display_name: str
    description: str
    author: str | None = None
    license: str | None = None
    homepage: str | None = None
    requires: list[str] = Field(default_factory=list)
    semantik_version: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)


class ExtractTestRequest(BaseModel):
    """Request to test extraction on sample text."""

    text: str = Field(..., description="Text to extract from", min_length=1)
    extractor_ids: list[str] = Field(..., description="Extractor plugin IDs to run")
    extraction_types: list[str] | None = Field(
        None,
        description="Specific extraction types to perform (keywords, entities, etc.)",
    )
    options: dict[str, Any] | None = Field(
        None,
        description="Extractor-specific options",
    )


class ExtractTestResponse(BaseModel):
    """Response from extraction test."""

    keywords: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    entity_types: list[str] = Field(default_factory=list)
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    sentiment: float | None = None
    custom: dict[str, Any] = Field(default_factory=dict)


def _load_extractor_plugins() -> None:
    """Ensure extractor plugins are loaded."""
    load_plugins(plugin_types={"extractor"})


@router.get(
    "",
    response_model=ExtractorListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_extractors(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ExtractorListResponse:
    """List all available extractor plugins."""
    _load_extractor_plugins()

    records = plugin_registry.list_records(plugin_type="extractor")
    extractors = []

    for record in records:
        manifest = record.manifest
        supported = manifest.capabilities.get("supported_extractions", [])

        extractors.append(
            ExtractorInfo(
                id=record.plugin_id,
                display_name=manifest.display_name,
                description=manifest.description,
                version=record.plugin_version,
                builtin=record.source.value == "builtin",
                supported_extractions=supported,
            )
        )

    return ExtractorListResponse(extractors=extractors, total=len(extractors))


@router.get(
    "/{extractor_id}",
    response_model=ExtractorInfo,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Extractor not found"},
    },
)
async def get_extractor(
    extractor_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ExtractorInfo:
    """Get detailed info for an extractor plugin."""
    _load_extractor_plugins()

    record = plugin_registry.get("extractor", extractor_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Extractor not found: {extractor_id}")

    manifest = record.manifest
    supported = manifest.capabilities.get("supported_extractions", [])

    return ExtractorInfo(
        id=record.plugin_id,
        display_name=manifest.display_name,
        description=manifest.description,
        version=record.plugin_version,
        builtin=record.source.value == "builtin",
        supported_extractions=supported,
    )


@router.get(
    "/{extractor_id}/manifest",
    response_model=ExtractorManifestSchema,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Extractor not found"},
    },
)
async def get_extractor_manifest(
    extractor_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ExtractorManifestSchema:
    """Get the full manifest for an extractor plugin."""
    _load_extractor_plugins()

    record = plugin_registry.get("extractor", extractor_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Extractor not found: {extractor_id}")

    manifest = record.manifest
    return ExtractorManifestSchema(
        id=manifest.id,
        type=manifest.type,
        version=manifest.version,
        display_name=manifest.display_name,
        description=manifest.description,
        author=manifest.author,
        license=manifest.license,
        homepage=manifest.homepage,
        requires=list(manifest.requires),
        semantik_version=manifest.semantik_version,
        capabilities=dict(manifest.capabilities),
    )


@router.post(
    "/test",
    response_model=ExtractTestResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def test_extraction(
    request: ExtractTestRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> ExtractTestResponse:
    """Test extraction on sample text.

    This endpoint allows testing extractors without affecting any collections.
    Useful for previewing what metadata will be extracted.
    """
    _load_extractor_plugins()

    # Validate extractor IDs exist
    for extractor_id in request.extractor_ids:
        if plugin_registry.get("extractor", extractor_id) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Extractor not found: {extractor_id}",
            )

    extractor_service = get_extractor_service()
    result = await extractor_service.run_extractors(
        text=request.text,
        extractor_ids=request.extractor_ids,
        extraction_types=request.extraction_types,
        options=request.options,
    )

    # Convert ExtractionResult to response schema
    searchable = result.to_searchable_dict()

    return ExtractTestResponse(
        keywords=searchable.get("keywords", []),
        entities=searchable.get("entities", {}),
        entity_types=searchable.get("entity_types", []),
        language=searchable.get("language"),
        topics=searchable.get("topics", []),
        sentiment=searchable.get("sentiment"),
        custom=searchable.get("custom", {}),
    )
