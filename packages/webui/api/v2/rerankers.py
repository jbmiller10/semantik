"""Reranker plugin discovery API endpoints."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry
from webui.api.schemas import ErrorResponse
from webui.auth import get_current_user

router = APIRouter(prefix="/api/v2/rerankers", tags=["rerankers-v2"])


class RerankerCapabilitiesSchema(BaseModel):
    """Reranker capabilities schema."""

    max_documents: int = Field(..., description="Maximum documents per request")
    max_query_length: int = Field(..., description="Maximum query length")
    max_doc_length: int = Field(..., description="Maximum document length")
    supports_batching: bool = Field(..., description="Supports batch processing")
    models: list[str] = Field(default_factory=list, description="Available model variants")


class RerankerInfo(BaseModel):
    """Reranker plugin information."""

    id: str = Field(..., description="Plugin ID")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Plugin description")
    version: str = Field(..., description="Plugin version")
    builtin: bool = Field(..., description="Whether this is a built-in plugin")
    capabilities: RerankerCapabilitiesSchema | None = Field(None, description="Reranker capabilities")


class RerankerListResponse(BaseModel):
    """Response for listing rerankers."""

    rerankers: list[RerankerInfo]
    total: int


class RerankerManifestSchema(BaseModel):
    """Full reranker manifest."""

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


def _load_reranker_plugins() -> None:
    """Ensure reranker plugins are loaded."""
    load_plugins(plugin_types={"reranker"})


@router.get(
    "",
    response_model=RerankerListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_rerankers(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> RerankerListResponse:
    """List all available reranker plugins."""
    _load_reranker_plugins()

    records = plugin_registry.list_records(plugin_type="reranker")
    rerankers = []

    for record in records:
        manifest = record.manifest
        capabilities = None

        # Get capabilities from manifest
        if manifest.capabilities:
            with contextlib.suppress(Exception):
                capabilities = RerankerCapabilitiesSchema(
                    max_documents=manifest.capabilities.get("max_documents", 100),
                    max_query_length=manifest.capabilities.get("max_query_length", 512),
                    max_doc_length=manifest.capabilities.get("max_doc_length", 512),
                    supports_batching=manifest.capabilities.get("supports_batching", False),
                    models=manifest.capabilities.get("models", []),
                )

        rerankers.append(
            RerankerInfo(
                id=record.plugin_id,
                display_name=manifest.display_name,
                description=manifest.description,
                version=record.plugin_version,
                builtin=record.source.value == "builtin",
                capabilities=capabilities,
            )
        )

    return RerankerListResponse(rerankers=rerankers, total=len(rerankers))


@router.get(
    "/{reranker_id}",
    response_model=RerankerInfo,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Reranker not found"},
    },
)
async def get_reranker(
    reranker_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> RerankerInfo:
    """Get detailed info for a reranker plugin."""
    _load_reranker_plugins()

    record = plugin_registry.get("reranker", reranker_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Reranker not found: {reranker_id}")

    manifest = record.manifest
    capabilities = None

    if manifest.capabilities:
        with contextlib.suppress(Exception):
            capabilities = RerankerCapabilitiesSchema(
                max_documents=manifest.capabilities.get("max_documents", 100),
                max_query_length=manifest.capabilities.get("max_query_length", 512),
                max_doc_length=manifest.capabilities.get("max_doc_length", 512),
                supports_batching=manifest.capabilities.get("supports_batching", False),
                models=manifest.capabilities.get("models", []),
            )

    return RerankerInfo(
        id=record.plugin_id,
        display_name=manifest.display_name,
        description=manifest.description,
        version=record.plugin_version,
        builtin=record.source.value == "builtin",
        capabilities=capabilities,
    )


@router.get(
    "/{reranker_id}/manifest",
    response_model=RerankerManifestSchema,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Reranker not found"},
    },
)
async def get_reranker_manifest(
    reranker_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> RerankerManifestSchema:
    """Get the full manifest for a reranker plugin."""
    _load_reranker_plugins()

    record = plugin_registry.get("reranker", reranker_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Reranker not found: {reranker_id}")

    manifest = record.manifest
    return RerankerManifestSchema(
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
