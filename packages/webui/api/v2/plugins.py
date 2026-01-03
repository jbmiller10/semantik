"""Plugin management API endpoints."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Path

from shared.database import get_db
from shared.plugins.validation import (
    PLUGIN_ID_MAX_LENGTH,
    PLUGIN_ID_REGEX,
    validate_package_name,
    validate_pip_install_target,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from webui.api.schemas import ErrorResponse
from webui.api.v2.plugins_schemas import (
    AvailablePluginInfo,
    AvailablePluginsListResponse,
    PluginConfigUpdateRequest,
    PluginHealthResponse,
    PluginInfo,
    PluginInstallRequest,
    PluginInstallResponse,
    PluginListResponse,
    PluginManifestSchema,
    PluginStatusResponse,
)
from webui.auth import get_current_user
from webui.services.plugin_service import PluginService

router = APIRouter(prefix="/api/v2/plugins", tags=["plugins-v2"])


async def _get_plugin_service(db: AsyncSession = Depends(get_db)) -> PluginService:
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
    from webui.services.plugin_installer import list_installed_packages

    # Get current Semantik version
    semantik_version = get_semantik_version()

    # Get installed plugin IDs (plugins that are loaded and active)
    installed_plugins = await service.list_plugins()
    installed_ids = {p["id"] for p in installed_plugins}

    # Get packages in plugins directory (may include pending-restart plugins)
    pending_packages = set(list_installed_packages())

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

        # Determine if plugin is installed and loaded
        is_installed = p.id in installed_ids

        # Check if package is in plugins dir but not loaded (pending restart)
        # Derive package name from pypi field or plugin ID
        package_name = p.pypi or f"semantik-plugin-{p.id}"
        pkg_dir_name = package_name.replace("-", "_")
        is_pending = pkg_dir_name in pending_packages and not is_installed

        # Build install command from install_command field or pypi field
        if p.install_command:
            install_cmd = p.install_command
        elif p.pypi:
            install_cmd = f"pip install {p.pypi}"
        else:
            install_cmd = ""

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
                is_installed=is_installed,
                pending_restart=is_pending,
                install_command=install_cmd,
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
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
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
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
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
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> dict[str, Any] | None:
    """Get JSON Schema for plugin configuration."""
    plugin = await service.get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    schema: dict[str, Any] | None = await service.get_config_schema(plugin_id)
    return schema


@router.post(
    "/{plugin_id}/enable",
    response_model=PluginStatusResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
    },
)
async def enable_plugin(
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
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
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
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
    request: PluginConfigUpdateRequest,
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
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
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginHealthResponse:
    """Run a health check for a plugin."""
    payload = await service.check_health(plugin_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")
    return PluginHealthResponse(**payload)


# --- Plugin Installation ---


@router.post(
    "/install",
    response_model=PluginInstallResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
        404: {"model": ErrorResponse, "description": "Plugin not found in registry"},
    },
)
async def install_plugin_endpoint(
    request: PluginInstallRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> PluginInstallResponse:
    """Install a plugin from the registry (admin only).

    Installs the plugin to a persistent directory. A container restart
    is required to activate the plugin.
    """
    from shared.plugins.registry_client import fetch_registry
    from shared.plugins.security import audit_log
    from webui.services.plugin_installer import install_plugin

    # Check admin permission
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=403,
            detail="Admin access required to install plugins",
        )

    # Look up plugin in registry
    registry = await fetch_registry()
    registry_entry = next(
        (p for p in registry.plugins if p.id == request.plugin_id),
        None,
    )
    if not registry_entry:
        raise HTTPException(
            status_code=404,
            detail=f"Plugin {request.plugin_id} not found in registry",
        )

    # Build install command
    install_cmd: str | None = None
    if registry_entry.install_command:
        install_cmd = registry_entry.install_command
        # Strip "pip install " prefix if present (registry may contain full command)
        if install_cmd.startswith("pip install "):
            install_cmd = install_cmd[len("pip install ") :]
        if request.version:
            # Append version to git URL (e.g., git+https://...git -> git+https://...git@v1.0.0)
            if ".git" in install_cmd:
                install_cmd = install_cmd.replace(".git", f".git@{request.version}")
            else:
                install_cmd = f"{install_cmd}@{request.version}"
    elif registry_entry.pypi:
        install_cmd = registry_entry.pypi
        if request.version:
            install_cmd = f"{install_cmd}=={request.version}"

    if not install_cmd:
        raise HTTPException(
            status_code=400,
            detail=f"Plugin {request.plugin_id} has no install command",
        )

    try:
        validate_pip_install_target(install_cmd)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Audit log
    audit_log(
        request.plugin_id,
        "plugin.install.started",
        {"user_id": current_user["id"], "version": request.version},
    )

    # Install (run in thread to avoid blocking event loop)
    success, message = await asyncio.to_thread(install_plugin, install_cmd)

    audit_log(
        request.plugin_id,
        "plugin.install.completed" if success else "plugin.install.failed",
        {"success": success, "message": message[:200]},  # Truncate long error messages
    )

    return PluginInstallResponse(
        success=success,
        message=message,
        restart_required=success,
    )


@router.delete(
    "/{plugin_id}/uninstall",
    response_model=PluginInstallResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Admin access required"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
        409: {"model": ErrorResponse, "description": "Multiple matching packages installed"},
    },
)
async def uninstall_plugin_endpoint(
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
    current_user: dict[str, Any] = Depends(get_current_user),
) -> PluginInstallResponse:
    """Uninstall an installed plugin (admin only).

    Removes the plugin from the persistent directory. A container restart
    is required to fully unload the plugin.
    """
    from shared.plugins.registry_client import fetch_registry
    from shared.plugins.security import audit_log
    from webui.services.plugin_installer import is_plugin_installed, list_installed_packages, uninstall_plugin

    # Check admin permission
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=403,
            detail="Admin access required to uninstall plugins",
        )

    def _resolve_installed_package(
        installed_packages: list[str],
        plugin_id: str,
        default_package: str,
    ) -> str | None:
        """Resolve package name for uninstall using installed package hints."""
        normalized = {pkg: pkg.replace("_", "-") for pkg in installed_packages}
        exact_default = [pkg for pkg, norm in normalized.items() if norm == default_package]
        if exact_default:
            return exact_default[0]
        exact_id = [pkg for pkg, norm in normalized.items() if norm == plugin_id]
        if exact_id:
            return exact_id[0]
        suffix_matches = [pkg for pkg, norm in normalized.items() if norm.endswith(f"-{plugin_id}")]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Multiple installed packages match plugin {plugin_id}: "
                    f"{', '.join(sorted(suffix_matches))}"
                ),
            )
        if is_plugin_installed(default_package):
            return default_package
        if is_plugin_installed(plugin_id):
            return plugin_id
        return None

    # Look up plugin in registry to get package name
    registry_entry = None
    try:
        registry = await fetch_registry()
        registry_entry = next(
            (p for p in registry.plugins if p.id == plugin_id),
            None,
        )
    except Exception:
        registry_entry = None

    default_package_name = f"semantik-plugin-{plugin_id}"
    installed_packages = list_installed_packages()

    if registry_entry:
        # Derive package name from pypi field or plugin ID
        package_name = registry_entry.pypi or default_package_name
        # If registry points to a package that's not installed, try local hints
        if installed_packages and package_name not in installed_packages:
            package_name = _resolve_installed_package(
                installed_packages,
                plugin_id,
                default_package_name,
            ) or package_name
    else:
        package_name = _resolve_installed_package(
            installed_packages,
            plugin_id,
            default_package_name,
        )
        if not package_name:
            raise HTTPException(
                status_code=404,
                detail=f"Plugin {plugin_id} not found in registry or installed packages",
            )

    try:
        validate_package_name(package_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    audit_log(
        plugin_id,
        "plugin.uninstall.started",
        {"user_id": current_user["id"]},
    )

    success, message = await asyncio.to_thread(uninstall_plugin, package_name)

    audit_log(
        plugin_id,
        "plugin.uninstall.completed" if success else "plugin.uninstall.failed",
        {"success": success, "message": message},
    )

    return PluginInstallResponse(
        success=success,
        message=message,
        restart_required=success,
    )
