"""
MCP Profile API v2 endpoints.

This module provides RESTful API endpoints for managing MCP (Model Context Protocol)
search profiles. MCP profiles define scoped collection access and search defaults
for LLM clients like Claude Desktop.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from shared.config import settings as shared_settings
from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, EntityNotFoundError
from webui.api.schemas import ErrorResponse
from webui.api.v2.mcp_schemas import (
    CollectionSummary,
    MCPClientConfig,
    MCPProfileCreate,
    MCPProfileListResponse,
    MCPProfileResponse,
    MCPProfileUpdate,
)
from webui.auth import get_current_user
from webui.services.factory import get_mcp_profile_service
from webui.services.mcp_profile_service import MCPProfileService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/mcp/profiles", tags=["mcp-profiles-v2"])


def _profile_to_response(profile: Any) -> MCPProfileResponse:
    """Convert MCPProfile model to response schema."""
    return MCPProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        enabled=profile.enabled,
        search_type=profile.search_type,
        result_count=profile.result_count,
        use_reranker=profile.use_reranker,
        score_threshold=profile.score_threshold,
        hybrid_alpha=profile.hybrid_alpha,
        search_mode=profile.search_mode,
        rrf_k=profile.rrf_k,
        collections=[CollectionSummary(id=c.id, name=c.name) for c in profile.collections],
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.post(
    "",
    response_model=MCPProfileResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied to collection"},
        404: {"model": ErrorResponse, "description": "Collection not found"},
        409: {"model": ErrorResponse, "description": "Profile name already exists"},
    },
)
async def create_profile(
    profile: MCPProfileCreate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> MCPProfileResponse:
    """Create a new MCP search profile.

    Creates a profile that exposes specified collections to MCP clients.
    The profile name must be unique per user and follow lowercase naming
    conventions suitable for MCP tool naming.
    """
    try:
        created = await service.create(
            data=profile,
            owner_id=int(current_user["id"]),
        )
        return _profile_to_response(created)

    except EntityAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Failed to create MCP profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create profile",
        ) from e


@router.get(
    "",
    response_model=MCPProfileListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
    },
)
async def list_profiles(
    enabled: bool | None = Query(None, description="Filter by enabled state"),
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> MCPProfileListResponse:
    """List all MCP profiles for the current user.

    Returns all profiles owned by the authenticated user, optionally
    filtered by enabled state.
    """
    try:
        profiles = await service.list_for_user(
            user_id=int(current_user["id"]),
            enabled_only=enabled is True,
        )

        # If enabled filter is False, filter to disabled profiles
        if enabled is False:
            profiles = [p for p in profiles if not p.enabled]

        return MCPProfileListResponse(
            profiles=[_profile_to_response(p) for p in profiles],
            total=len(profiles),
        )

    except Exception as e:
        logger.error("Failed to list MCP profiles: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list profiles",
        ) from e


@router.get(
    "/{profile_id}",
    response_model=MCPProfileResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Profile not found"},
    },
)
async def get_profile(
    profile_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> MCPProfileResponse:
    """Get a specific MCP profile.

    Returns the profile with its associated collections.
    """
    try:
        profile = await service.get(
            profile_id=profile_id,
            owner_id=int(current_user["id"]),
        )
        return _profile_to_response(profile)

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile not found: {profile_id}",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this profile",
        ) from e
    except Exception as e:
        logger.error("Failed to get MCP profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile",
        ) from e


@router.put(
    "/{profile_id}",
    response_model=MCPProfileResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Profile or collection not found"},
        409: {"model": ErrorResponse, "description": "Profile name already exists"},
    },
)
async def update_profile(
    profile_id: str,
    profile: MCPProfileUpdate,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> MCPProfileResponse:
    """Update an MCP profile.

    Updates the specified fields. Collection ordering is preserved
    based on the order of collection_ids in the request.
    """
    try:
        updated = await service.update(
            profile_id=profile_id,
            data=profile,
            owner_id=int(current_user["id"]),
        )
        return _profile_to_response(updated)

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e),
        ) from e
    except EntityAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Failed to update MCP profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile",
        ) from e


@router.delete(
    "/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Profile not found"},
    },
)
async def delete_profile(
    profile_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> None:
    """Delete an MCP profile.

    Permanently removes the profile. This does not affect the
    underlying collections.
    """
    try:
        await service.delete(
            profile_id=profile_id,
            owner_id=int(current_user["id"]),
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile not found: {profile_id}",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this profile",
        ) from e
    except Exception as e:
        logger.error("Failed to delete MCP profile: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete profile",
        ) from e


@router.get(
    "/{profile_id}/config",
    response_model=MCPClientConfig,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        404: {"model": ErrorResponse, "description": "Profile not found"},
    },
)
async def get_profile_config(
    profile_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    service: MCPProfileService = Depends(get_mcp_profile_service),
) -> MCPClientConfig:
    """Get MCP client configuration for a profile.

    Returns a JSON configuration snippet suitable for Claude Desktop
    or other MCP clients. The configuration includes the command,
    arguments, and environment variables needed to connect.

    Note: The auth token in the response is a placeholder. Replace it
    with a valid API key or access token.
    """
    try:
        webui_url = shared_settings.WEBUI_URL or "http://localhost:8080"

        return await service.get_config(
            profile_id=profile_id,
            owner_id=int(current_user["id"]),
            webui_url=webui_url,
        )

    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile not found: {profile_id}",
        ) from e
    except AccessDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this profile",
        ) from e
    except Exception as e:
        logger.error("Failed to get MCP profile config: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get profile config",
        ) from e
