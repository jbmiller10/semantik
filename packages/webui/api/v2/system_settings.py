"""System settings API endpoints (admin-only).

These endpoints allow administrators to view and modify system-wide
configuration settings that would otherwise require environment variables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from shared.database import get_db
from shared.database.repositories.system_settings_repository import SystemSettingsRepository
from webui.api.schemas import ErrorResponse
from webui.auth import get_current_user
from webui.services.system_settings_service import SYSTEM_SETTING_DEFAULTS

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/system-settings", tags=["system-settings"])


# =============================================================================
# Request/Response Schemas
# =============================================================================


class SystemSettingValue(BaseModel):
    """A single system setting with its value and metadata."""

    value: Any = Field(description="The setting value (null means use env var fallback)")
    updated_at: str | None = Field(description="When the setting was last updated")
    updated_by: int | None = Field(description="User ID who last updated the setting")


class SystemSettingsResponse(BaseModel):
    """Response containing all system settings."""

    settings: dict[str, SystemSettingValue] = Field(description="Map of setting keys to their values and metadata")


class SystemSettingsUpdateRequest(BaseModel):
    """Request to update system settings."""

    settings: dict[str, Any] = Field(
        description="Map of setting keys to new values (null to reset to env var fallback)",
        examples=[{"max_collections_per_user": 20, "cache_ttl_seconds": 600}],
    )


class SystemSettingsUpdateResponse(BaseModel):
    """Response after updating system settings."""

    updated: list[str] = Field(description="List of keys that were updated")
    settings: dict[str, SystemSettingValue] = Field(description="Updated settings with their new values")


class EffectiveSettingsResponse(BaseModel):
    """Response containing effective values (resolved through DB -> env -> default chain)."""

    settings: dict[str, Any] = Field(description="Map of setting keys to their effective values")


class DefaultSettingsResponse(BaseModel):
    """Response containing default values for all system settings."""

    defaults: dict[str, Any] = Field(description="Map of setting keys to their default values")


# =============================================================================
# Helper Functions
# =============================================================================


def _require_admin(current_user: dict[str, Any]) -> None:
    """Raise 403 if user is not an admin."""
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to manage system settings",
        )


async def _get_settings_repo(
    db: AsyncSession = Depends(get_db),
) -> SystemSettingsRepository:
    """Get system settings repository instance."""
    return SystemSettingsRepository(db)


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "",
    response_model=SystemSettingsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
    },
)
async def get_system_settings(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: SystemSettingsRepository = Depends(_get_settings_repo),
) -> SystemSettingsResponse:
    """Get all system settings with their metadata (admin-only).

    Returns the raw database values. A null value means the setting will
    fall back to the environment variable or default.
    """
    _require_admin(current_user)

    settings_with_meta = await repo.get_settings_with_metadata()

    # Build response with all known settings
    response_settings: dict[str, SystemSettingValue] = {}

    for key in SYSTEM_SETTING_DEFAULTS:
        if key in settings_with_meta:
            meta = settings_with_meta[key]
            response_settings[key] = SystemSettingValue(
                value=meta["value"],
                updated_at=meta["updated_at"],
                updated_by=meta["updated_by"],
            )
        else:
            # Setting not in DB yet - show as null
            response_settings[key] = SystemSettingValue(
                value=None,
                updated_at=None,
                updated_by=None,
            )

    return SystemSettingsResponse(settings=response_settings)


@router.get(
    "/effective",
    response_model=EffectiveSettingsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
    },
)
async def get_effective_settings(
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: SystemSettingsRepository = Depends(_get_settings_repo),
) -> EffectiveSettingsResponse:
    """Get effective setting values (admin-only).

    Returns settings resolved through the precedence chain:
    1. Database value (if not null)
    2. Environment variable (if set)
    3. Default value
    """
    import os

    _require_admin(current_user)

    # Get raw DB values
    db_settings = await repo.get_all_settings()

    # Resolve effective values
    effective: dict[str, Any] = {}
    for key, default in SYSTEM_SETTING_DEFAULTS.items():
        # Check DB value first
        db_value = db_settings.get(key)
        if db_value is not None:
            effective[key] = db_value
            continue

        # Fall back to env var
        env_value = os.getenv(key.upper())
        if env_value is not None:
            # Parse env value based on default type
            if isinstance(default, bool):
                effective[key] = env_value.lower() in ("true", "1", "yes", "on")
            elif isinstance(default, int):
                try:
                    effective[key] = int(env_value)
                except ValueError:
                    effective[key] = default
            elif isinstance(default, float):
                try:
                    effective[key] = float(env_value)
                except ValueError:
                    effective[key] = default
            else:
                effective[key] = env_value
            continue

        # Use default
        effective[key] = default

    return EffectiveSettingsResponse(settings=effective)


@router.patch(
    "",
    response_model=SystemSettingsUpdateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
    },
)
async def update_system_settings(
    request: SystemSettingsUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    repo: SystemSettingsRepository = Depends(_get_settings_repo),
    db: AsyncSession = Depends(get_db),
) -> SystemSettingsUpdateResponse:
    """Update system settings (admin-only).

    Only settings provided in the request will be updated.
    Set a value to null to reset it to use the environment variable fallback.
    """
    _require_admin(current_user)

    if not request.settings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No settings provided to update",
        )

    # Validate that all keys are known settings
    unknown_keys = set(request.settings.keys()) - set(SYSTEM_SETTING_DEFAULTS.keys())
    if unknown_keys:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown setting keys: {', '.join(sorted(unknown_keys))}",
        )

    user_id = int(current_user["id"])

    # Update settings
    updated_keys = await repo.set_settings(request.settings, user_id=user_id)
    await db.commit()

    # Invalidate cache so future reads get fresh values
    from webui.services.system_settings_service import _service_instance

    if _service_instance is not None:
        _service_instance.invalidate_cache()

    logger.info(
        "Admin user %s updated system settings: %s",
        user_id,
        ", ".join(updated_keys),
    )

    # Fetch updated settings to return
    settings_with_meta = await repo.get_settings_with_metadata()

    response_settings: dict[str, SystemSettingValue] = {}
    for key in updated_keys:
        meta = settings_with_meta.get(key, {})
        response_settings[key] = SystemSettingValue(
            value=meta.get("value"),
            updated_at=meta.get("updated_at"),
            updated_by=meta.get("updated_by"),
        )

    return SystemSettingsUpdateResponse(
        updated=updated_keys,
        settings=response_settings,
    )


@router.get(
    "/defaults",
    response_model=DefaultSettingsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
    },
)
async def get_default_settings(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> DefaultSettingsResponse:
    """Get the default values for all system settings (admin-only).

    These are the fallback values used when a setting is not configured
    in the database and no environment variable is set.
    """
    _require_admin(current_user)

    return DefaultSettingsResponse(defaults=dict(SYSTEM_SETTING_DEFAULTS))
