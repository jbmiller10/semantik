"""User preferences API endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends

from shared.database import get_db
from shared.database.repositories.user_preferences_repository import UserPreferencesRepository
from webui.api.schemas import ErrorResponse
from webui.api.v2.user_preferences_schemas import (
    CollectionDefaults,
    InterfacePreferences,
    SearchPreferences,
    UserPreferencesResponse,
    UserPreferencesUpdate,
)
from webui.auth import get_current_user

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.database.models import UserPreferences

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/preferences", tags=["preferences-v2"])


async def _get_preferences_repo(
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesRepository:
    """Get user preferences repository instance."""
    return UserPreferencesRepository(db)


def _to_response(prefs: UserPreferences) -> UserPreferencesResponse:
    """Convert database model to response schema."""
    return UserPreferencesResponse(
        search=SearchPreferences(
            top_k=prefs.search_top_k,
            mode=prefs.search_mode,
            use_reranker=prefs.search_use_reranker,
            rrf_k=prefs.search_rrf_k,
            similarity_threshold=prefs.search_similarity_threshold,
        ),
        collection_defaults=CollectionDefaults(
            embedding_model=prefs.default_embedding_model,
            quantization=prefs.default_quantization,
            chunking_strategy=prefs.default_chunking_strategy,
            chunk_size=prefs.default_chunk_size,
            chunk_overlap=prefs.default_chunk_overlap,
            enable_sparse=prefs.default_enable_sparse,
            sparse_type=prefs.default_sparse_type,
            enable_hybrid=prefs.default_enable_hybrid,
        ),
        interface=InterfacePreferences(
            data_refresh_interval_ms=prefs.data_refresh_interval_ms,
            visualization_sample_limit=prefs.visualization_sample_limit,
            animation_enabled=prefs.animation_enabled,
        ),
        created_at=prefs.created_at,
        updated_at=prefs.updated_at,
    )


@router.get(
    "",
    response_model=UserPreferencesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def get_preferences(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: UserPreferencesRepository = Depends(_get_preferences_repo),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Get the current user's preferences.

    Returns defaults if no preferences have been configured yet.
    """
    user_id = int(current_user["id"])
    prefs = await repo.get_or_create(user_id)
    await db.commit()

    return _to_response(prefs)


@router.put(
    "",
    response_model=UserPreferencesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or validation error"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def update_preferences(
    update: UserPreferencesUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: UserPreferencesRepository = Depends(_get_preferences_repo),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Update the current user's preferences.

    Supports partial updates - only provided fields will be updated.
    """
    user_id = int(current_user["id"])

    # Build update kwargs from provided fields
    update_kwargs: dict[str, Any] = {}

    if update.search is not None:
        update_kwargs["search_top_k"] = update.search.top_k
        update_kwargs["search_mode"] = update.search.mode
        update_kwargs["search_use_reranker"] = update.search.use_reranker
        update_kwargs["search_rrf_k"] = update.search.rrf_k
        update_kwargs["search_similarity_threshold"] = update.search.similarity_threshold

    if update.collection_defaults is not None:
        update_kwargs["default_embedding_model"] = update.collection_defaults.embedding_model
        update_kwargs["default_quantization"] = update.collection_defaults.quantization
        update_kwargs["default_chunking_strategy"] = update.collection_defaults.chunking_strategy
        update_kwargs["default_chunk_size"] = update.collection_defaults.chunk_size
        update_kwargs["default_chunk_overlap"] = update.collection_defaults.chunk_overlap
        update_kwargs["default_enable_sparse"] = update.collection_defaults.enable_sparse
        update_kwargs["default_sparse_type"] = update.collection_defaults.sparse_type
        update_kwargs["default_enable_hybrid"] = update.collection_defaults.enable_hybrid

    if update.interface is not None:
        update_kwargs["data_refresh_interval_ms"] = update.interface.data_refresh_interval_ms
        update_kwargs["visualization_sample_limit"] = update.interface.visualization_sample_limit
        update_kwargs["animation_enabled"] = update.interface.animation_enabled

    prefs = await repo.update(user_id, **update_kwargs)
    await db.commit()

    logger.info("Updated preferences for user %s", user_id)
    return _to_response(prefs)


@router.post(
    "/reset/search",
    response_model=UserPreferencesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def reset_search_preferences(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: UserPreferencesRepository = Depends(_get_preferences_repo),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Reset search preferences to default values."""
    user_id = int(current_user["id"])

    prefs = await repo.reset_search(user_id)
    await db.commit()

    logger.info("Reset search preferences for user %s", user_id)
    return _to_response(prefs)


@router.post(
    "/reset/collection-defaults",
    response_model=UserPreferencesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def reset_collection_defaults(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: UserPreferencesRepository = Depends(_get_preferences_repo),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Reset collection defaults to system default values."""
    user_id = int(current_user["id"])

    prefs = await repo.reset_collection_defaults(user_id)
    await db.commit()

    logger.info("Reset collection defaults for user %s", user_id)
    return _to_response(prefs)


@router.post(
    "/reset/interface",
    response_model=UserPreferencesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def reset_interface_preferences(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: UserPreferencesRepository = Depends(_get_preferences_repo),
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Reset interface preferences to default values."""
    user_id = int(current_user["id"])

    prefs = await repo.reset_interface(user_id)
    await db.commit()

    logger.info("Reset interface preferences for user %s", user_id)
    return _to_response(prefs)
