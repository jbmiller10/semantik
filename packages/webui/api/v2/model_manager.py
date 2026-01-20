"""Model manager API endpoints (superuser-only)."""

import asyncio
import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.database.models import Collection, LLMProviderConfig, UserPreferences
from shared.model_manager import (
    ModelType as SharedModelType,
    get_cache_size_info,
    get_curated_model_ids,
    get_curated_models,
    get_installed_models,
    get_model_size_on_disk,
    is_model_installed,
)
from webui.api.schemas import ErrorResponse
from webui.api.v2.model_manager_schemas import (
    CacheSizeInfo,
    ConflictType,
    CuratedModelResponse,
    EmbeddingModelDetails,
    LLMModelDetails,
    ModelDownloadRequest,
    ModelListResponse,
    ModelManagerConflictResponse,
    ModelType,
    ModelUsageResponse,
    TaskProgressResponse,
    TaskResponse,
    TaskStatus,
)
from webui.auth import get_current_user
from webui.celery_app import celery_app
from webui.dependencies import get_db
from webui.model_manager import task_state
from webui.model_manager.task_state import CrossOpConflictError
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


async def _count_user_preferences_using_model(db: AsyncSession, model_id: str) -> int:
    """Count users with this model as their default_embedding_model.

    Args:
        db: Async database session.
        model_id: The embedding model ID.

    Returns:
        Number of users with this as their default embedding model.
    """
    result = await db.execute(
        select(func.count()).select_from(UserPreferences).where(UserPreferences.default_embedding_model == model_id)
    )
    return result.scalar() or 0


async def _count_llm_configs_using_model(db: AsyncSession, model_id: str) -> int:
    """Count LLM configs referencing this model (for local LLM models).

    Checks both high_quality_model and low_quality_model fields.

    Args:
        db: Async database session.
        model_id: The LLM model ID.

    Returns:
        Number of LLM configs referencing this model.
    """
    result = await db.execute(
        select(func.count())
        .select_from(LLMProviderConfig)
        .where((LLMProviderConfig.high_quality_model == model_id) | (LLMProviderConfig.low_quality_model == model_id))
    )
    return result.scalar() or 0


async def _get_vecpipe_loaded_models(model_id: str) -> tuple[bool, list[str]]:
    """Query VecPipe /memory/models to check if model is loaded.

    This is best-effort - failures are logged but don't block operations.

    Args:
        model_id: The model ID to check.

    Returns:
        Tuple of (is_loaded: bool, model_types: list[str])
        model_types contains the types of loaded instances (e.g., ["embedding", "reranker"])
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.SEARCH_API_URL}/memory/models")
            response.raise_for_status()
            models = response.json()

            loaded_types: list[str] = []
            for model in models:
                if model.get("model_name") == model_id:
                    model_type = model.get("model_type", "unknown")
                    if model_type not in loaded_types:
                        loaded_types.append(model_type)

            return (len(loaded_types) > 0, loaded_types)
    except Exception as e:
        logger.warning("Failed to query VecPipe loaded models: %s", e)
        return (False, [])


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


@router.post(
    "/download",
    response_model=TaskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Model not in curated list"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
        409: {"model": ModelManagerConflictResponse, "description": "Operation conflict"},
    },
)
async def download_model(
    request: ModelDownloadRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> TaskResponse:
    """Initiate download of a curated model.

    Validates the model is in the curated list, checks if already installed,
    and dispatches a Celery task for async download.

    Returns:
        - TaskResponse with task_id for tracking progress
        - status=already_installed if model exists (idempotent)
        - status=pending if new download initiated

    Requires superuser access.
    """
    _require_superuser(current_user)

    model_id = request.model_id

    # Validate model is in curated list
    curated_ids = get_curated_model_ids()
    if model_id not in curated_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_id}' is not in the curated model list",
        )

    # Check if already installed (idempotent)
    installed = await asyncio.to_thread(is_model_installed, model_id)
    if installed:
        return TaskResponse(
            task_id=None,
            model_id=model_id,
            operation="download",
            status=TaskStatus.ALREADY_INSTALLED,
        )

    # Get Redis client
    redis_manager = get_redis_manager()
    redis_client = await redis_manager.async_client()

    # Try to claim the operation
    task_id = str(uuid.uuid4())
    try:
        claimed, existing_task_id = await task_state.claim_model_operation(redis_client, model_id, "download", task_id)
    except CrossOpConflictError as e:
        # Different operation (delete) is active
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ModelManagerConflictResponse(
                conflict_type=ConflictType.CROSS_OP_EXCLUSION,
                detail=f"Cannot download: {e.active_operation} is active for this model",
                model_id=model_id,
                active_operation=e.active_operation,
                active_task_id=e.active_task_id,
            ).model_dump(),
        ) from e

    if not claimed and existing_task_id:
        # Same operation already active - return existing task for de-duplication
        return TaskResponse(
            task_id=existing_task_id,
            model_id=model_id,
            operation="download",
            status=TaskStatus.RUNNING,
        )

    # Initialize progress and dispatch task
    await task_state.init_task_progress(redis_client, task_id, model_id, "download")
    try:
        celery_app.send_task(
            "webui.tasks.model_manager.download_model",
            args=[model_id, task_id],
        )
    except Exception as e:
        logger.exception("Failed to enqueue download task: model=%s, task_id=%s", model_id, task_id)
        await task_state.update_task_progress(
            redis_client, task_id, status="failed", error=f"Failed to enqueue task: {e}"
        )
        await task_state.release_model_operation_if_owner(redis_client, model_id, "download", task_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue background download task",
        ) from e

    logger.info("Download task dispatched: model=%s, task_id=%s", model_id, task_id)

    return TaskResponse(
        task_id=task_id,
        model_id=model_id,
        operation="download",
        status=TaskStatus.PENDING,
    )


@router.get(
    "/usage",
    response_model=ModelUsageResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
    },
)
async def get_model_usage(
    model_id: str = Query(..., description="HuggingFace model ID"),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ModelUsageResponse:
    """Get usage information for a model before deletion.

    Returns information about what is using this model across the system,
    helping users understand the impact of deleting it.

    Requires superuser access.
    """
    _require_superuser(current_user)

    # Check if installed
    installed = await asyncio.to_thread(is_model_installed, model_id)
    size_on_disk_mb = None
    if installed:
        size_on_disk_mb = await asyncio.to_thread(get_model_size_on_disk, model_id)

    # Get blocking conditions (collections using this model)
    blocked_by_collections = await _get_collections_using_model(db, model_id)

    # Get warning conditions
    user_preferences_count = await _count_user_preferences_using_model(db, model_id)
    llm_config_count = await _count_llm_configs_using_model(db, model_id)
    is_default = model_id == settings.DEFAULT_EMBEDDING_MODEL

    # Query VecPipe for loaded models (best-effort)
    loaded_in_vecpipe, loaded_types = await _get_vecpipe_loaded_models(model_id)

    # Build warnings
    warnings: list[str] = []
    if blocked_by_collections:
        warnings.append(
            f"Model is used by {len(blocked_by_collections)} collection(s): {', '.join(blocked_by_collections[:3])}"
        )
    if user_preferences_count > 0:
        warnings.append(f"{user_preferences_count} user(s) have this as their default embedding model")
    if llm_config_count > 0:
        warnings.append(f"{llm_config_count} LLM configuration(s) reference this model")
    if is_default:
        warnings.append("This is the system default embedding model")
    if loaded_in_vecpipe:
        warnings.append(f"Model is currently loaded in VecPipe GPU memory ({', '.join(loaded_types)})")

    # Determine if deletion is allowed
    can_delete = len(blocked_by_collections) == 0
    requires_confirmation = len(warnings) > 0 and can_delete

    return ModelUsageResponse(
        model_id=model_id,
        is_installed=installed,
        size_on_disk_mb=size_on_disk_mb,
        estimated_freed_size_mb=size_on_disk_mb,
        blocked_by_collections=blocked_by_collections,
        user_preferences_count=user_preferences_count,
        llm_config_count=llm_config_count,
        is_default_embedding_model=is_default,
        loaded_in_vecpipe=loaded_in_vecpipe,
        loaded_vecpipe_model_types=loaded_types,
        warnings=warnings,
        can_delete=can_delete,
        requires_confirmation=requires_confirmation,
    )


@router.delete(
    "/cache",
    response_model=TaskResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
        409: {"model": ModelManagerConflictResponse, "description": "Operation conflict or requires confirmation"},
    },
)
async def delete_model_cache(
    model_id: str = Query(..., description="HuggingFace model ID"),
    confirm: bool = Query(default=False, description="Confirm deletion despite warnings"),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TaskResponse:
    """Initiate deletion of a model from the HuggingFace cache.

    Checks for blocking conditions (collections using model) and warning
    conditions (user preferences, LLM configs, etc.). If warnings exist
    and confirm=false, returns 409 with REQUIRES_CONFIRMATION.

    Returns:
        - TaskResponse with task_id for tracking progress
        - status=not_installed if model doesn't exist (idempotent)
        - status=pending if deletion initiated

    Requires superuser access.
    """
    _require_superuser(current_user)

    # Check if installed (idempotent)
    installed = await asyncio.to_thread(is_model_installed, model_id)
    if not installed:
        return TaskResponse(
            task_id=None,
            model_id=model_id,
            operation="delete",
            status=TaskStatus.NOT_INSTALLED,
        )

    # Check blocking conditions
    blocked_by_collections = await _get_collections_using_model(db, model_id)
    if blocked_by_collections:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ModelManagerConflictResponse(
                conflict_type=ConflictType.IN_USE_BLOCK,
                detail=f"Cannot delete: model is used by {len(blocked_by_collections)} collection(s)",
                model_id=model_id,
                blocked_by_collections=blocked_by_collections,
            ).model_dump(),
        )

    # Build warnings
    warnings: list[str] = []
    user_preferences_count = await _count_user_preferences_using_model(db, model_id)
    llm_config_count = await _count_llm_configs_using_model(db, model_id)
    is_default = model_id == settings.DEFAULT_EMBEDDING_MODEL
    loaded_in_vecpipe, loaded_types = await _get_vecpipe_loaded_models(model_id)

    if user_preferences_count > 0:
        warnings.append(f"{user_preferences_count} user(s) have this as their default embedding model")
    if llm_config_count > 0:
        warnings.append(f"{llm_config_count} LLM configuration(s) reference this model")
    if is_default:
        warnings.append("This is the system default embedding model")
    if loaded_in_vecpipe:
        warnings.append(f"Model is currently loaded in VecPipe GPU memory ({', '.join(loaded_types)})")

    # Require confirmation if warnings exist
    if warnings and not confirm:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ModelManagerConflictResponse(
                conflict_type=ConflictType.REQUIRES_CONFIRMATION,
                detail="Deletion requires confirmation due to active usage",
                model_id=model_id,
                requires_confirmation=True,
                warnings=warnings,
            ).model_dump(),
        )

    # Get Redis client
    redis_manager = get_redis_manager()
    redis_client = await redis_manager.async_client()

    # Try to claim the operation
    task_id = str(uuid.uuid4())
    try:
        claimed, existing_task_id = await task_state.claim_model_operation(redis_client, model_id, "delete", task_id)
    except CrossOpConflictError as e:
        # Different operation (download) is active
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=ModelManagerConflictResponse(
                conflict_type=ConflictType.CROSS_OP_EXCLUSION,
                detail=f"Cannot delete: {e.active_operation} is active for this model",
                model_id=model_id,
                active_operation=e.active_operation,
                active_task_id=e.active_task_id,
            ).model_dump(),
        ) from e

    if not claimed and existing_task_id:
        # Same operation already active - return existing task for de-duplication
        return TaskResponse(
            task_id=existing_task_id,
            model_id=model_id,
            operation="delete",
            status=TaskStatus.RUNNING,
        )

    # Initialize progress and dispatch task
    await task_state.init_task_progress(redis_client, task_id, model_id, "delete")
    try:
        celery_app.send_task(
            "webui.tasks.model_manager.delete_model",
            args=[model_id, task_id],
        )
    except Exception as e:
        logger.exception("Failed to enqueue delete task: model=%s, task_id=%s", model_id, task_id)
        await task_state.update_task_progress(
            redis_client, task_id, status="failed", error=f"Failed to enqueue task: {e}"
        )
        await task_state.release_model_operation_if_owner(redis_client, model_id, "delete", task_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue background delete task",
        ) from e

    logger.info("Delete task dispatched: model=%s, task_id=%s", model_id, task_id)

    return TaskResponse(
        task_id=task_id,
        model_id=model_id,
        operation="delete",
        status=TaskStatus.PENDING,
        warnings=warnings if confirm else [],
    )
