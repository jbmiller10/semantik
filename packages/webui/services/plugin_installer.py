"""Plugin installation service for in-app plugin management.

This module provides functions to install and uninstall plugins from
the persistent plugins directory. Installations are performed synchronously
using pip with --target to install into the plugins volume.

Note: PYTHONPATH is configured as /app/packages:/app/plugins, so app
packages always take precedence. Plugins can bring new dependencies
but cannot override existing app packages.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from shared.plugins.validation import validate_package_name, validate_pip_install_target

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# Default plugins directory (can be overridden via environment)
PLUGINS_DIR = Path(os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins"))

# Timeout for acquiring the installation lock (seconds)
INSTALL_LOCK_TIMEOUT = 60


@contextmanager
def _plugin_install_lock(timeout: int = INSTALL_LOCK_TIMEOUT) -> Generator[None, None, None]:
    """Acquire exclusive lock for plugin installation/uninstallation.

    Uses fcntl file locking to prevent concurrent pip operations from
    corrupting the plugins directory. The lock is automatically released
    when the context manager exits or if the process dies.

    Args:
        timeout: Maximum seconds to wait for lock acquisition.

    Raises:
        TimeoutError: If lock cannot be acquired within timeout.
    """
    lock_path = get_plugins_dir() / ".plugin_install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = lock_path.open("w")  # noqa: SIM115 - need file handle for fcntl
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.debug("Acquired plugin installation lock")
                break
            except OSError:
                if time.time() - start > timeout:
                    lock_file.close()
                    raise TimeoutError(
                        "Could not acquire plugin lock. Another installation may be in progress."
                    ) from None
                time.sleep(0.5)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        logger.debug("Released plugin installation lock")


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
        validate_pip_install_target(install_command)
    except ValueError as exc:
        logger.warning("Rejected unsafe install target: %s (%s)", install_command, exc)
        return False, str(exc)

    try:
        # Acquire exclusive lock to prevent concurrent installations
        with _plugin_install_lock():
            # Use system pip directly (venv doesn't have pip, uses uv)
            result = subprocess.run(
                [
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

    except TimeoutError as exc:
        logger.error("Plugin installation lock timeout: %s", exc)
        return False, str(exc)
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

    try:
        validate_package_name(package_name)
    except ValueError as exc:
        logger.warning("Rejected unsafe package name for uninstall: %s (%s)", package_name, exc)
        return False, str(exc)

    # Convert package name to directory name (replace hyphens with underscores)
    dir_name = package_name.replace("-", "_")
    plugins_dir_resolved = plugins_dir.resolve()
    plugin_path = (plugins_dir / dir_name).resolve()
    if not plugin_path.is_relative_to(plugins_dir_resolved):
        return False, "Invalid package name"

    logger.info("Uninstalling plugin: %s from %s", package_name, plugins_dir)

    try:
        # Acquire exclusive lock to prevent concurrent uninstall/install operations
        with _plugin_install_lock():
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
                dist_info_path = dist_info.resolve()
                if not dist_info_path.is_relative_to(plugins_dir_resolved):
                    logger.warning("Skipping unexpected dist-info path outside plugins dir: %s", dist_info_path)
                    continue
                try:
                    shutil.rmtree(dist_info_path)
                    removed_any = True
                    logger.info("Removed dist-info: %s", dist_info_path)
                except Exception as exc:
                    logger.warning("Failed to remove dist-info %s: %s", dist_info, exc)

            if removed_any:
                return True, f"Uninstalled {package_name}. Restart required."

            return False, f"Plugin {package_name} not found in {plugins_dir}"

    except TimeoutError as exc:
        logger.error("Plugin uninstall lock timeout: %s", exc)
        return False, str(exc)


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
