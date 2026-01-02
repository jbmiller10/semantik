"""Plugin installation service for in-app plugin management.

This module provides functions to install and uninstall plugins from
the persistent plugins directory. Installations are performed synchronously
using pip with --target to install into the plugins volume.

Note: PYTHONPATH is configured as /app/packages:/app/plugins, so app
packages always take precedence. Plugins can bring new dependencies
but cannot override existing app packages.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Default plugins directory (can be overridden via environment)
PLUGINS_DIR = Path(os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins"))


def get_plugins_dir() -> Path:
    """Get the plugins directory path.

    Returns:
        Path to the plugins directory.
    """
    return PLUGINS_DIR


def install_plugin(
    install_command: str,
    timeout: int = 300,
) -> tuple[bool, str]:
    """Install a plugin using pip.

    Runs pip install with --target to install the package into the
    persistent plugins directory. This is a synchronous operation that
    blocks until pip completes.

    Note: App packages take precedence over plugin packages via PYTHONPATH
    order, so plugins cannot override core dependencies like pydantic, anyio, etc.

    Args:
        install_command: The pip install target. Can be:
            - A git URL: "git+https://github.com/user/repo.git"
            - A git URL with version: "git+https://github.com/user/repo.git@v1.0.0"
            - A PyPI package: "semantik-plugin-openai"
            - A PyPI package with version: "semantik-plugin-openai==1.0.0"
        timeout: Maximum time in seconds to wait for installation.

    Returns:
        Tuple of (success, message) where success is True if the
        installation completed successfully.
    """
    plugins_dir = get_plugins_dir()
    plugins_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Installing plugin: %s -> %s", install_command, plugins_dir)

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(plugins_dir),
                "--upgrade",
                "--no-cache-dir",
                install_command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            logger.info("Plugin installed successfully: %s", install_command)
            return True, "Successfully installed. Restart required to activate."

        error_msg = result.stderr.strip() or result.stdout.strip()
        logger.error("Plugin installation failed: %s\n%s", install_command, error_msg)
        return False, f"Installation failed:\n{error_msg}"

    except subprocess.TimeoutExpired:
        logger.error("Plugin installation timed out: %s", install_command)
        return False, f"Installation timed out after {timeout} seconds"
    except FileNotFoundError:
        logger.error("pip not found in PATH")
        return False, "pip not found. Please check the container configuration."
    except Exception as exc:
        logger.exception("Unexpected error during plugin installation")
        return False, f"Unexpected error: {exc}"


def uninstall_plugin(package_name: str) -> tuple[bool, str]:
    """Uninstall a plugin by removing its package directory.

    This directly removes the package directory and dist-info from
    the plugins directory. Does not use pip uninstall because the
    plugins directory is not a standard pip target.

    Args:
        package_name: The package name to uninstall (e.g., "semantik-plugin-openai").

    Returns:
        Tuple of (success, message).
    """
    plugins_dir = get_plugins_dir()

    # Convert package name to directory name (replace hyphens with underscores)
    dir_name = package_name.replace("-", "_")
    plugin_path = plugins_dir / dir_name

    logger.info("Uninstalling plugin: %s from %s", package_name, plugins_dir)

    removed_any = False

    # Remove the package directory
    if plugin_path.exists():
        try:
            shutil.rmtree(plugin_path)
            removed_any = True
            logger.info("Removed package directory: %s", plugin_path)
        except Exception as exc:
            logger.error("Failed to remove package directory: %s", exc)
            return False, f"Failed to remove package: {exc}"

    # Remove .dist-info directories
    for dist_info in plugins_dir.glob(f"{dir_name}-*.dist-info"):
        try:
            shutil.rmtree(dist_info)
            removed_any = True
            logger.info("Removed dist-info: %s", dist_info)
        except Exception as exc:
            logger.warning("Failed to remove dist-info %s: %s", dist_info, exc)

    if removed_any:
        return True, f"Uninstalled {package_name}. Restart required."

    return False, f"Plugin {package_name} not found in {plugins_dir}"


def list_installed_packages() -> list[str]:
    """List packages installed in the plugins directory.

    Scans the plugins directory for .dist-info directories and extracts
    the package names.

    Returns:
        List of installed package names.
    """
    plugins_dir = get_plugins_dir()

    if not plugins_dir.exists():
        return []

    installed = []
    for dist_info in plugins_dir.glob("*.dist-info"):
        # Extract package name from dist-info directory name
        # Format: package_name-version.dist-info
        name = dist_info.name.rsplit("-", 2)[0]
        installed.append(name)

    logger.debug("Found %d installed packages in %s", len(installed), plugins_dir)
    return installed


def is_plugin_installed(package_name: str) -> bool:
    """Check if a plugin package is installed in the plugins directory.

    Args:
        package_name: The package name to check.

    Returns:
        True if the package is installed.
    """
    plugins_dir = get_plugins_dir()
    dir_name = package_name.replace("-", "_")
    return (plugins_dir / dir_name).exists()


def get_installed_version(package_name: str) -> str | None:
    """Get the installed version of a plugin package.

    Args:
        package_name: The package name to check.

    Returns:
        Version string if installed, None otherwise.
    """
    plugins_dir = get_plugins_dir()
    dir_name = package_name.replace("-", "_")

    # Look for dist-info directory
    for dist_info in plugins_dir.glob(f"{dir_name}-*.dist-info"):
        # Extract version from directory name
        # Format: package_name-version.dist-info
        parts = dist_info.name[:-10].rsplit("-", 1)  # Remove .dist-info
        if len(parts) == 2:
            return parts[1]

    return None
