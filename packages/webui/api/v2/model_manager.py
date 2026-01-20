"""Model manager API endpoints (superuser-only)."""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Collection
from shared.model_manager import (
    ModelType as SharedModelType,
    get_cache_size_info,
    get_curated_model_ids,
    get_curated_models,
    get_installed_models,
)
from webui.api.schemas import ErrorResponse
from webui.api.v2.model_manager_schemas import (
    CacheSizeInfo,
    CuratedModelResponse,
    EmbeddingModelDetails,
    LLMModelDetails,
    ModelListResponse,
    ModelType,
    TaskProgressResponse,
    TaskStatus,
)
from webui.auth import get_current_user
from webui.dependencies import get_db
from webui.model_manager import task_state
from webui.services.factory import get_redis_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/models", tags=["models-v2"])


def _require_superuser(current_user: dict[str, Any]) -> None:
    """Raise 403 if user is not a superuser."""
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required for model management",
        )


async def _get_collections_using_model(db: AsyncSession, model_id: str) -> list[str]:
    """Get collection names using a specific embedding model.

    Args:
        db: Async database session.
        model_id: The embedding model ID to search for.

    Returns:
        List of collection names using this model.
    """
    result = await db.execute(select(Collection.name).where(Collection.embedding_model == model_id))
    return [row[0] for row in result.all()]


def _map_model_type(shared_type: SharedModelType) -> ModelType:
    """Map shared ModelType to schema ModelType."""
    return ModelType(shared_type.value)


@router.get(
    "",
    response_model=ModelListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
    },
)
async def list_models(
    model_type: ModelType | None = Query(default=None, description="Filter by model type"),
    installed_only: bool = Query(default=False, description="Only return installed models"),
    include_cache_size: bool = Query(default=False, description="Include cache size breakdown"),
    force_refresh_cache: bool = Query(default=False, description="Force refresh HF cache scan"),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ModelListResponse:
    """List all curated models with installation status.

    Returns a list of curated models aggregated from multiple sources:
    - Embedding models from MODEL_CONFIGS
    - Local LLM models from model_registry
    - Reranker models (Qwen3-Reranker)
    - SPLADE sparse indexer models

    Requires superuser access.
    """
    _require_superuser(current_user)

    # Get curated models
    curated_models = get_curated_models()

    # Filter by type if specified
    if model_type is not None:
        curated_models = tuple(m for m in curated_models if m.model_type.value == model_type.value)

    # Scan HF cache in a thread pool to avoid blocking
    installed_models = await asyncio.to_thread(get_installed_models, force_refresh=force_refresh_cache)

    # Get Redis client for active operation lookups
    redis_manager = get_redis_manager()
    redis_client = await redis_manager.async_client()

    # Build response models
    response_models: list[CuratedModelResponse] = []

    for model in curated_models:
        is_installed = model.id in installed_models
        size_on_disk_mb = None

        if is_installed:
            size_on_disk_mb = installed_models[model.id].size_on_disk_mb

        # Skip if filtering for installed only and model is not installed
        if installed_only and not is_installed:
            continue

        # Get collections using this embedding model
        used_by_collections: list[str] = []
        if model.model_type == SharedModelType.EMBEDDING:
            used_by_collections = await _get_collections_using_model(db, model.id)

        # Build type-specific details
        embedding_details = None
        llm_details = None

        if model.model_type == SharedModelType.EMBEDDING:
            embedding_details = EmbeddingModelDetails(
                dimension=model.dimension,
                max_sequence_length=model.max_sequence_length,
                pooling_method=model.pooling_method,
                is_asymmetric=model.is_asymmetric,
                query_prefix=model.query_prefix,
                document_prefix=model.document_prefix,
                default_query_instruction=model.default_query_instruction,
            )
        elif model.model_type == SharedModelType.LLM:
            llm_details = LLMModelDetails(
                context_window=model.context_window,
            )

        # Check for active operations in Redis
        active_download_task_id = None
        active_delete_task_id = None
        try:
            active_op = await task_state.get_active_operation(redis_client, model.id)
            if active_op:
                op_type, task_id = active_op
                if op_type == "download":
                    active_download_task_id = task_id
                elif op_type == "delete":
                    active_delete_task_id = task_id
        except Exception as e:
            logger.warning("Failed to get active operation for %s: %s", model.id, e)

        response_models.append(
            CuratedModelResponse(
                id=model.id,
                name=model.name,
                description=model.description,
                model_type=_map_model_type(model.model_type),
                memory_mb=dict(model.memory_mb),
                is_installed=is_installed,
                size_on_disk_mb=size_on_disk_mb,
                used_by_collections=used_by_collections,
                active_download_task_id=active_download_task_id,
                active_delete_task_id=active_delete_task_id,
                embedding_details=embedding_details,
                llm_details=llm_details,
            )
        )

    # Compute cache size info if requested
    cache_size = None
    if include_cache_size:
        curated_ids = get_curated_model_ids()
        cache_breakdown = await asyncio.to_thread(get_cache_size_info, curated_ids)
        cache_size = CacheSizeInfo(
            total_cache_size_mb=cache_breakdown["total_cache_size_mb"],
            managed_cache_size_mb=cache_breakdown["managed_cache_size_mb"],
            unmanaged_cache_size_mb=cache_breakdown["unmanaged_cache_size_mb"],
            unmanaged_repo_count=cache_breakdown["unmanaged_repo_count"],
        )

    return ModelListResponse(models=response_models, cache_size=cache_size)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskProgressResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
        404: {"model": ErrorResponse, "description": "Task not found"},
    },
)
async def get_task_progress(
    task_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TaskProgressResponse:
    """Get progress for a download/delete task.

    Returns the current status, progress bytes (for downloads), and any errors.
    Task progress is stored in Redis and available for polling until TTL expires.

    Requires superuser access.
    """
    _require_superuser(current_user)

    redis_manager = get_redis_manager()
    redis_client = await redis_manager.async_client()

    progress = await task_state.get_task_progress(redis_client, task_id)

    if progress is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    # Map status string to TaskStatus enum
    status_str = progress.get("status", "pending")
    try:
        task_status = TaskStatus(status_str)
    except ValueError:
        task_status = TaskStatus.PENDING

    return TaskProgressResponse(
        task_id=progress["task_id"],
        model_id=progress["model_id"],
        operation=progress["operation"],
        status=task_status,
        bytes_downloaded=progress.get("bytes_downloaded", 0),
        bytes_total=progress.get("bytes_total", 0),
        error=progress.get("error"),
        updated_at=progress.get("updated_at", 0.0),
    )
