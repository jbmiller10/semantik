"""
API Key Management v2 endpoints.

This module provides RESTful API endpoints for managing API keys.
API keys enable programmatic access to the Semantik API.

Error Handling:
    All service-layer exceptions (EntityNotFoundError, AccessDeniedError, etc.)
    are handled by global exception handlers registered in middleware/exception_handlers.py.
    Routers should NOT catch and re-raise these as HTTPExceptions.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request, status

from webui.api.schemas import ErrorResponse
from webui.api.v2.api_key_schemas import (
    ApiKeyCreate,
    ApiKeyCreateResponse,
    ApiKeyListResponse,
    ApiKeyResponse,
    ApiKeyUpdate,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.rate_limiter import limiter
from webui.services.api_key_service import ApiKeyService
from webui.services.factory import get_api_key_service

router = APIRouter(prefix="/api/v2/api-keys", tags=["api-keys-v2"])


def _api_key_to_response(api_key: Any) -> ApiKeyResponse:
    """Convert ApiKey model to response schema."""
    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        is_active=api_key.is_active,
        permissions=api_key.permissions,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


@router.post(
    "",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Maximum keys limit reached"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        409: {"model": ErrorResponse, "description": "Key name already exists"},
    },
)
@limiter.limit(RateLimitConfig.API_KEY_CREATE_RATE)
async def create_api_key(
    request: Request,
    data: ApiKeyCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyCreateResponse:
    """Create a new API key.

    Creates an API key for programmatic access. The full key is returned
    only once at creation time and cannot be retrieved later.

    **Key Format**: `smtk_<prefix>_<secret>`

    Rate limit: 5 per hour
    """
    _ = request  # Required for rate limiter
    api_key, raw_key = await service.create(
        data=data,
        user_id=int(current_user["id"]),
    )

    return ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        is_active=api_key.is_active,
        permissions=api_key.permissions,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        api_key=raw_key,
    )


@router.get(
    "",
    response_model=ApiKeyListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
@limiter.limit(RateLimitConfig.API_KEY_LIST_RATE)
async def list_api_keys(
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyListResponse:
    """List all API keys for the current user.

    Returns all API keys owned by the authenticated user.
    The raw key value is never included in list responses.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    api_keys = await service.list_for_user(user_id=int(current_user["id"]))

    return ApiKeyListResponse(
        api_keys=[_api_key_to_response(k) for k in api_keys],
        total=len(api_keys),
    )


@router.get(
    "/{key_id}",
    response_model=ApiKeyResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "API key not found"},
    },
)
@limiter.limit(RateLimitConfig.API_KEY_LIST_RATE)
async def get_api_key(
    request: Request,
    key_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyResponse:
    """Get a specific API key.

    Returns the API key details. The raw key value is never included.

    Rate limit: 60 per minute
    """
    _ = request  # Required for rate limiter
    api_key = await service.get(
        key_id=key_id,
        user_id=int(current_user["id"]),
    )
    return _api_key_to_response(api_key)


@router.patch(
    "/{key_id}",
    response_model=ApiKeyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Maximum keys limit reached"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "API key not found"},
    },
)
@limiter.limit(RateLimitConfig.API_KEY_UPDATE_RATE)
async def update_api_key(
    request: Request,
    key_id: str,
    data: ApiKeyUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyResponse:
    """Update an API key's active status.

    Use this endpoint to revoke (is_active=false) or reactivate
    (is_active=true) an API key. This is a soft revoke - the key
    record is preserved but authentication will be denied.

    Rate limit: 30 per minute
    """
    _ = request  # Required for rate limiter
    api_key = await service.update_active_status(
        key_id=key_id,
        user_id=int(current_user["id"]),
        is_active=data.is_active,
    )
    return _api_key_to_response(api_key)
