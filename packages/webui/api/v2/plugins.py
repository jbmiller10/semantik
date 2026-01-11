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

from shared.plugins.error_codes import PluginErrorCode
from shared.plugins.exceptions import PluginConfigValidationError
from webui.api.schemas import ErrorResponse
from webui.api.v2.plugins_schemas import (
    AvailablePluginInfo,
    AvailablePluginsListResponse,
    PluginConfigUpdateRequest,
    PluginErrorDetail,
    PluginErrorResponse,
    PluginHealthResponse,
    PluginInfo,
    PluginInstallRequest,
    PluginInstallResponse,
    PluginListResponse,
    PluginManifestSchema,
    PluginStatusResponse,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig
from webui.rate_limiter import rate_limit_dependency
from webui.services.plugin_service import PluginService

router = APIRouter(prefix="/api/v2/plugins", tags=["plugins-v2"])


def _plugin_error(
    status_code: int,
    code: PluginErrorCode,
    detail: str,
    plugin_id: str | None = None,
) -> HTTPException:
    """Create an HTTPException with structured plugin error response."""
    return HTTPException(
        status_code=status_code,
        detail=PluginErrorResponse(
            detail=detail,
            code=code,
            plugin_id=plugin_id,
        ).model_dump(),
    )


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
        raise _plugin_error(404, PluginErrorCode.PLUGIN_NOT_FOUND, f"Plugin not found: {plugin_id}", plugin_id)
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
        raise _plugin_error(404, PluginErrorCode.PLUGIN_NOT_FOUND, f"Plugin not found: {plugin_id}", plugin_id)
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
    return PluginStatusResponse(
        plugin_id=plugin_id,
        enabled=True,
        requires_restart=True,
        sync_warning=payload.get("sync_warning"),
    )


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
    return PluginStatusResponse(
        plugin_id=plugin_id,
        enabled=False,
        requires_restart=True,
        sync_warning=payload.get("sync_warning"),
    )


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
    except PluginConfigValidationError as exc:
        # Return structured validation errors with suggestions
        errors = [
            PluginErrorDetail(
                field=e.get("field"),
                message=e.get("message", "Validation error"),
                suggestion=e.get("suggestion"),
            )
            for e in exc.errors
        ]
        raise HTTPException(
            status_code=400,
            detail=PluginErrorResponse(
                detail=str(exc),
                code=PluginErrorCode.PLUGIN_CONFIG_INVALID,
                plugin_id=plugin_id,
                errors=errors,
            ).model_dump(),
        ) from exc

    if payload is None:
        raise _plugin_error(404, PluginErrorCode.PLUGIN_NOT_FOUND, f"Plugin not found: {plugin_id}", plugin_id)
    payload["requires_restart"] = True
    return PluginInfo(**payload)


@router.get(
    "/{plugin_id}/health",
    response_model=PluginHealthResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        404: {"model": ErrorResponse, "description": "Plugin not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.PLUGIN_HEALTH_RATE))],
)
async def check_plugin_health(
    plugin_id: str = Path(..., pattern=PLUGIN_ID_REGEX, max_length=PLUGIN_ID_MAX_LENGTH),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    service: PluginService = Depends(_get_plugin_service),
) -> PluginHealthResponse:
    """Run a health check for a plugin."""
    payload = await service.check_health(plugin_id)
    if payload is None:
        raise _plugin_error(404, PluginErrorCode.PLUGIN_NOT_FOUND, f"Plugin not found: {plugin_id}", plugin_id)
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
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.PLUGIN_INSTALL_RATE))],
)
async def install_plugin_endpoint(
    body: PluginInstallRequest,
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
        raise _plugin_error(
            403,
            PluginErrorCode.PLUGIN_INSTALL_FAILED,
            "Admin access required to install plugins",
            body.plugin_id,
        )

    # Look up plugin in registry
    registry = await fetch_registry()
    registry_entry = next(
        (p for p in registry.plugins if p.id == body.plugin_id),
        None,
    )
    if not registry_entry:
        raise _plugin_error(
            404,
            PluginErrorCode.PLUGIN_NOT_IN_REGISTRY,
            f"Plugin {body.plugin_id} not found in registry",
            body.plugin_id,
        )

    # Build install command
    install_cmd: str | None = None
    if registry_entry.install_command:
        from shared.plugins.git_url import append_version_to_git_url, is_git_url

        install_cmd = registry_entry.install_command
        # Strip "pip install " prefix if present (registry may contain full command)
        if install_cmd.startswith("pip install "):
            install_cmd = install_cmd[len("pip install ") :]
        if body.version:
            # Append version using proper URL parsing
            if is_git_url(install_cmd):
                install_cmd = append_version_to_git_url(install_cmd, body.version)
            else:
                install_cmd = f"{install_cmd}@{body.version}"
    elif registry_entry.pypi:
        install_cmd = registry_entry.pypi
        if body.version:
            install_cmd = f"{install_cmd}=={body.version}"

    if not install_cmd:
        raise _plugin_error(
            400,
            PluginErrorCode.PLUGIN_NO_INSTALL_COMMAND,
            f"Plugin {body.plugin_id} has no install command",
            body.plugin_id,
        )

    validate_pip_install_target(install_cmd)

    # Audit log
    audit_log(
        body.plugin_id,
        "plugin.install.started",
        {"user_id": current_user["id"], "version": body.version},
    )

    # Install (run in thread to avoid blocking event loop)
    success, message = await asyncio.to_thread(install_plugin, install_cmd)

    audit_log(
        body.plugin_id,
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
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    dependencies=[Depends(rate_limit_dependency(RateLimitConfig.PLUGIN_UNINSTALL_RATE))],
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
        raise _plugin_error(
            403,
            PluginErrorCode.PLUGIN_UNINSTALL_FAILED,
            "Admin access required to uninstall plugins",
            plugin_id,
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
                detail=(f"Multiple installed packages match plugin {plugin_id}: {', '.join(sorted(suffix_matches))}"),
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
            package_name = (
                _resolve_installed_package(
                    installed_packages,
                    plugin_id,
                    default_package_name,
                )
                or package_name
            )
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

    validate_package_name(package_name)

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
