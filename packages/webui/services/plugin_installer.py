"""Plugin installation service for in-app plugin management.

This module provides functions to install and uninstall plugins from
the persistent plugins directory. Installations are performed synchronously
using pip with --target to install into the plugins volume.

Key design: To avoid overriding app packages (like anyio, pydantic), we:
1. Install to a staging directory first
2. Compare against packages already in the app's venv
3. Only copy packages that are NOT already installed in the app
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Default plugins directory (can be overridden via environment)
PLUGINS_DIR = Path(os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins"))

# App's virtual environment site-packages
VENV_SITE_PACKAGES = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def get_plugins_dir() -> Path:
    """Get the plugins directory path.

    Returns:
        Path to the plugins directory.
    """
    return PLUGINS_DIR


def _get_venv_packages() -> set[str]:
    """Get normalized package names from the app's virtual environment.

    Returns:
        Set of normalized package names (lowercase, underscores).
    """
    packages = set()
    if VENV_SITE_PACKAGES.exists():
        for dist_info in VENV_SITE_PACKAGES.glob("*.dist-info"):
            # Extract package name from dist-info directory name
            # Format: package_name-version.dist-info
            name = dist_info.name.rsplit("-", 1)[0]
            # Normalize: lowercase and replace hyphens with underscores
            packages.add(name.lower().replace("-", "_"))
    return packages


def _copy_non_conflicting_packages(staging_dir: Path, target_dir: Path) -> list[str]:
    """Copy packages from staging to target, skipping those in app venv.

    Args:
        staging_dir: Directory where pip installed packages.
        target_dir: Final plugins directory.

    Returns:
        List of package names that were copied.
    """
    venv_packages = _get_venv_packages()
    copied = []
    skipped = []

    # Find all .dist-info directories in staging to identify installed packages
    for dist_info in staging_dir.glob("*.dist-info"):
        # Extract package name
        pkg_name = dist_info.name.rsplit("-", 1)[0]
        pkg_name_normalized = pkg_name.lower().replace("-", "_")

        # Skip if package exists in app's venv
        if pkg_name_normalized in venv_packages:
            skipped.append(pkg_name)
            continue

        # Copy this package's dist-info
        dest_dist_info = target_dir / dist_info.name
        if dest_dist_info.exists():
            shutil.rmtree(dest_dist_info)
        shutil.copytree(dist_info, dest_dist_info)

        # Copy the package directory(s)
        # The package name might be different from dist-info name
        pkg_dir = staging_dir / pkg_name_normalized
        if pkg_dir.exists() and pkg_dir.is_dir():
            dest_pkg_dir = target_dir / pkg_name_normalized
            if dest_pkg_dir.exists():
                shutil.rmtree(dest_pkg_dir)
            shutil.copytree(pkg_dir, dest_pkg_dir)
            copied.append(pkg_name)
        else:
            # Try original name (some packages use different conventions)
            pkg_dir = staging_dir / pkg_name
            if pkg_dir.exists() and pkg_dir.is_dir():
                dest_pkg_dir = target_dir / pkg_name
                if dest_pkg_dir.exists():
                    shutil.rmtree(dest_pkg_dir)
                shutil.copytree(pkg_dir, dest_pkg_dir)
                copied.append(pkg_name)

    # Also copy any standalone .py files (single-file packages)
    for py_file in staging_dir.glob("*.py"):
        if py_file.name != "setup.py":
            dest_file = target_dir / py_file.name
            shutil.copy2(py_file, dest_file)

    # Copy .libs directories if present (for compiled packages)
    for libs_dir in staging_dir.glob("*.libs"):
        pkg_name = libs_dir.name[:-5]  # Remove .libs suffix
        pkg_name_normalized = pkg_name.lower().replace("-", "_")
        if pkg_name_normalized not in venv_packages:
            dest_libs = target_dir / libs_dir.name
            if dest_libs.exists():
                shutil.rmtree(dest_libs)
            shutil.copytree(libs_dir, dest_libs)

    if skipped:
        logger.info("Skipped packages already in app venv: %s", ", ".join(skipped))

    return copied


def install_plugin(
    install_command: str,
    timeout: int = 300,
) -> tuple[bool, str]:
    """Install a plugin using pip.

    Runs pip install with --target to install the package into a staging
    directory, then copies only packages that don't conflict with the
    app's existing dependencies.

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

    # Create a temporary staging directory
    with tempfile.TemporaryDirectory() as staging_dir:
        staging_path = Path(staging_dir)

        try:
            # Install to staging directory first
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    str(staging_path),
                    "--no-cache-dir",
                    install_command,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                logger.error("Plugin installation failed: %s\n%s", install_command, error_msg)
                return False, f"Installation failed:\n{error_msg}"

            # Copy only non-conflicting packages to the plugins directory
            copied = _copy_non_conflicting_packages(staging_path, plugins_dir)

            if copied:
                logger.info("Plugin installed successfully: %s (packages: %s)", install_command, ", ".join(copied))
                return True, f"Successfully installed {', '.join(copied)}. Restart required to activate."

            logger.warning("No new packages installed for: %s", install_command)
            return True, "Plugin dependencies already satisfied by app. Restart may be required."

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
