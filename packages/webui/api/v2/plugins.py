"""Plugin management API endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from shared.database import get_db
from webui.api.schemas import ErrorResponse
from webui.api.v2.plugins_schemas import (
    AvailablePluginInfo,
    AvailablePluginsListResponse,
    PluginConfigUpdateRequest,
    PluginHealthResponse,
    PluginInfo,
    PluginListResponse,
    PluginManifestSchema,
    PluginStatusResponse,
)
from webui.auth import get_current_user
from webui.services.plugin_service import PluginService

router = APIRouter(prefix="/api/v2/plugins", tags=["plugins-v2"])


async def _get_plugin_service(db=Depends(get_db)) -> PluginService:
    return PluginService(db)


@router.get(
    "",
    response_model=PluginListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_plugins(
    plugin_type: str | None = None,
    enabled: bool | None = None,
    include_health: bool = False,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginListResponse:
    """List all installed external plugins."""
    plugins = await service.list_plugins(plugin_type=plugin_type, enabled=enabled, include_health=include_health)
    return PluginListResponse(plugins=[PluginInfo(**plugin) for plugin in plugins])


# --- Available Plugins (from registry) ---


@router.get(
    "/available",
    response_model=AvailablePluginsListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def list_available_plugins(
    plugin_type: str | None = None,
    verified_only: bool = False,
    force_refresh: bool = False,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> AvailablePluginsListResponse:
    """List available plugins from the remote registry.

    Returns plugins that can be installed, with compatibility information
    and install commands.
    """
    from shared.plugins.compatibility import check_compatibility, get_semantik_version
    from shared.plugins.registry_client import fetch_registry, get_registry_source

    # Get current Semantik version
    semantik_version = get_semantik_version()

    # Get installed plugin IDs
    installed_plugins = await service.list_plugins()
    installed_ids = {p["id"] for p in installed_plugins}

    # Fetch registry
    registry = await fetch_registry(force_refresh=force_refresh)

    # Filter plugins
    plugins = registry.plugins
    if plugin_type:
        plugins = [p for p in plugins if p.type == plugin_type]
    if verified_only:
        plugins = [p for p in plugins if p.verified]

    # Build response with compatibility info
    result_plugins = []
    for p in plugins:
        is_compatible, compat_msg = check_compatibility(
            p.min_semantik_version,
            semantik_version,
        )

        result_plugins.append(
            AvailablePluginInfo(
                id=p.id,
                type=p.type,
                name=p.name,
                description=p.description,
                author=p.author,
                repository=p.repository,
                pypi=p.pypi,
                verified=p.verified,
                min_semantik_version=p.min_semantik_version,
                tags=p.tags,
                is_compatible=is_compatible,
                compatibility_message=compat_msg,
                is_installed=p.id in installed_ids,
                install_command=f"pip install {p.pypi}",
            )
        )

    return AvailablePluginsListResponse(
        plugins=result_plugins,
        registry_version=registry.registry_version,
        last_updated=registry.last_updated,
        registry_source=get_registry_source(),
        semantik_version=semantik_version,
    )


@router.post(
    "/available/refresh",
    response_model=AvailablePluginsListResponse,
    responses={401: {"model": ErrorResponse, "description": "Unauthorized"}},
)
async def refresh_available_plugins(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> AvailablePluginsListResponse:
    """Force refresh the available plugins registry cache."""
    return await list_available_plugins(
        plugin_type=None,
        verified_only=False,
        force_refresh=True,
        current_user=current_user,
        service=service,
    )


# --- Installed Plugin Operations ---


@router.get(
    "/{plugin_id}",
    response_model=PluginInfo,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def get_plugin(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginInfo:
    """Get detailed info for a plugin."""
    plugin = await service.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginInfo(**plugin)


@router.get(
    "/{plugin_id}/manifest",
    response_model=PluginManifestSchema,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def get_plugin_manifest(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginManifestSchema:
    """Get the manifest for a plugin."""
    manifest = await service.get_manifest(plugin_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginManifestSchema(**manifest)


@router.get(
    "/{plugin_id}/config-schema",
    response_model=dict[str, Any] | None,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def get_plugin_config_schema(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> dict[str, Any] | None:
    """Get JSON Schema for plugin configuration."""
    plugin = await service.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return await service.get_config_schema(plugin_id)


@router.post(
    "/{plugin_id}/enable",
    response_model=PluginStatusResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def enable_plugin(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginStatusResponse:
    """Enable a plugin (requires restart to take effect)."""
    payload = await service.set_enabled(plugin_id, True)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginStatusResponse(plugin_id=plugin_id, enabled=True, requires_restart=True)


@router.post(
    "/{plugin_id}/disable",
    response_model=PluginStatusResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def disable_plugin(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginStatusResponse:
    """Disable a plugin (requires restart to take effect)."""
    payload = await service.set_enabled(plugin_id, False)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginStatusResponse(plugin_id=plugin_id, enabled=False, requires_restart=True)


@router.put(
    "/{plugin_id}/config",
    response_model=PluginInfo,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid config"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def update_plugin_config(
    plugin_id: str,
    request: PluginConfigUpdateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginInfo:
    """Update plugin configuration (validated against schema if present)."""
    try:
        payload = await service.update_config(plugin_id, request.config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    payload["requires_restart"] = True
    return PluginInfo(**payload)


@router.get(
    "/{plugin_id}/health",
    response_model=PluginHealthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def check_plugin_health(
    plugin_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginHealthResponse:
    """Run a health check for a plugin."""
    payload = await service.check_health(plugin_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginHealthResponse(**payload)
