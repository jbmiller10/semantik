"""Unit tests for plugin_installer module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from webui.services import plugin_installer


@pytest.fixture()
def temp_plugins_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary plugins directory."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setattr(plugin_installer, "PLUGINS_DIR", plugins_dir)
    return plugins_dir


class TestGetPluginsDir:
    """Tests for get_plugins_dir function."""

    def test_returns_plugins_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_plugins_dir returns the PLUGINS_DIR value."""
        test_path = Path("/test/plugins")
        monkeypatch.setattr(plugin_installer, "PLUGINS_DIR", test_path)
        assert plugin_installer.get_plugins_dir() == test_path


class TestInstallPlugin:
    """Tests for install_plugin function."""

    def test_install_success(self, temp_plugins_dir: Path) -> None:
        """Test successful plugin installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            success, message = plugin_installer.install_plugin("test-package")

            assert success is True
            assert "restart required" in message.lower()
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "pip" in call_args
            assert "install" in call_args
            assert "--target" in call_args
            assert "test-package" in call_args

    def test_install_failure(self, temp_plugins_dir: Path) -> None:
        """Test plugin installation failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: package not found",
            )

            success, message = plugin_installer.install_plugin("nonexistent-package")

            assert success is False
            assert "installation failed" in message.lower()
            assert "package not found" in message

    def test_install_failure_uses_stdout_if_no_stderr(self, temp_plugins_dir: Path) -> None:
        """Test that stdout is used for error message if stderr is empty."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="Error from stdout",
                stderr="",
            )

            success, message = plugin_installer.install_plugin("bad-package")

            assert success is False
            assert "Error from stdout" in message

    def test_install_timeout(self, temp_plugins_dir: Path) -> None:
        """Test plugin installation timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=60)

            success, message = plugin_installer.install_plugin("slow-package", timeout=60)

            assert success is False
            assert "timed out" in message.lower()
            assert "60" in message

    def test_install_pip_not_found(self, temp_plugins_dir: Path) -> None:
        """Test installation when pip is not found."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("pip not found")

            success, message = plugin_installer.install_plugin("test-package")

            assert success is False
            assert "pip not found" in message.lower()

    def test_install_unexpected_exception(self, temp_plugins_dir: Path) -> None:
        """Test installation with unexpected exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")

            success, message = plugin_installer.install_plugin("test-package")

            assert success is False
            assert "unexpected error" in message.lower()

    def test_install_creates_plugins_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that install_plugin creates the plugins directory if it doesn't exist."""
        plugins_dir = tmp_path / "new_plugins_dir"
        monkeypatch.setattr(plugin_installer, "PLUGINS_DIR", plugins_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            plugin_installer.install_plugin("test-package")

        assert plugins_dir.exists()

    def test_install_rejects_whitespace_target(self, temp_plugins_dir: Path) -> None:
        """install_plugin should reject targets with whitespace (option-injection defense)."""
        with patch("subprocess.run") as mock_run:
            success, message = plugin_installer.install_plugin("git+https://example.com/repo.git @v1")

        assert success is False
        assert "whitespace" in message.lower()
        mock_run.assert_not_called()


class TestUninstallPlugin:
    """Tests for uninstall_plugin function."""

    def test_uninstall_success(self, temp_plugins_dir: Path) -> None:
        """Test successful plugin uninstallation."""
        # Create a package directory
        pkg_dir = temp_plugins_dir / "test_plugin"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").touch()

        success, message = plugin_installer.uninstall_plugin("test-plugin")

        assert success is True
        assert "uninstalled" in message.lower()
        assert not pkg_dir.exists()

    def test_uninstall_removes_dist_info(self, temp_plugins_dir: Path) -> None:
        """Test that uninstall removes dist-info directories."""
        # Create package and dist-info directories
        pkg_dir = temp_plugins_dir / "test_plugin"
        pkg_dir.mkdir()
        dist_info = temp_plugins_dir / "test_plugin-1.0.0.dist-info"
        dist_info.mkdir()

        success, message = plugin_installer.uninstall_plugin("test-plugin")

        assert success is True
        assert not pkg_dir.exists()
        assert not dist_info.exists()

    def test_uninstall_only_dist_info(self, temp_plugins_dir: Path) -> None:
        """Test uninstall when only dist-info exists (no package dir)."""
        dist_info = temp_plugins_dir / "test_plugin-1.0.0.dist-info"
        dist_info.mkdir()

        success, message = plugin_installer.uninstall_plugin("test-plugin")

        assert success is True
        assert not dist_info.exists()

    def test_uninstall_not_found(self, temp_plugins_dir: Path) -> None:
        """Test uninstall when plugin is not found."""
        success, message = plugin_installer.uninstall_plugin("nonexistent-plugin")

        assert success is False
        assert "not found" in message.lower()

    def test_uninstall_failure_on_rmtree(self, temp_plugins_dir: Path) -> None:
        """Test uninstall failure when rmtree fails."""
        pkg_dir = temp_plugins_dir / "test_plugin"
        pkg_dir.mkdir()

        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Permission denied")

            success, message = plugin_installer.uninstall_plugin("test-plugin")

            assert success is False
            assert "failed to remove" in message.lower()

    def test_uninstall_dist_info_failure_continues(self, temp_plugins_dir: Path) -> None:
        """Test that dist-info removal failure only logs warning."""
        # Create package and dist-info directories
        pkg_dir = temp_plugins_dir / "test_plugin"
        pkg_dir.mkdir()
        dist_info = temp_plugins_dir / "test_plugin-1.0.0.dist-info"
        dist_info.mkdir()

        call_count = 0

        def rmtree_side_effect(_path: Path) -> None:
            nonlocal call_count
            call_count += 1
            # Succeed on first call (package dir), fail on second (dist-info)
            if call_count == 2:
                raise PermissionError("Cannot remove dist-info")

        with patch("shutil.rmtree", side_effect=rmtree_side_effect):
            success, message = plugin_installer.uninstall_plugin("test-plugin")

            # Should still succeed because package dir was removed
            assert success is True

    def test_uninstall_rejects_path_traversal(self, temp_plugins_dir: Path) -> None:
        """uninstall_plugin should reject path traversal attempts."""
        success, message = plugin_installer.uninstall_plugin("../etc")

        assert success is False
        assert "path" in message.lower()


class TestListInstalledPackages:
    """Tests for list_installed_packages function."""

    def test_list_empty(self, temp_plugins_dir: Path) -> None:
        """Test listing when no packages are installed."""
        result = plugin_installer.list_installed_packages()
        assert result == []

    def test_list_with_packages(self, temp_plugins_dir: Path) -> None:
        """Test listing installed packages."""
        # Create some dist-info directories
        (temp_plugins_dir / "package_one-1.0.0.dist-info").mkdir()
        (temp_plugins_dir / "package_two-2.0.0.dist-info").mkdir()

        result = plugin_installer.list_installed_packages()

        assert len(result) == 2
        assert "package_one" in result
        assert "package_two" in result

    def test_list_nonexistent_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test listing when plugins directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        monkeypatch.setattr(plugin_installer, "PLUGINS_DIR", nonexistent)

        result = plugin_installer.list_installed_packages()

        assert result == []


class TestIsPluginInstalled:
    """Tests for is_plugin_installed function."""

    def test_plugin_installed(self, temp_plugins_dir: Path) -> None:
        """Test checking an installed plugin."""
        pkg_dir = temp_plugins_dir / "test_plugin"
        pkg_dir.mkdir()

        assert plugin_installer.is_plugin_installed("test-plugin") is True

    def test_plugin_not_installed(self, temp_plugins_dir: Path) -> None:
        """Test checking a non-installed plugin."""
        assert plugin_installer.is_plugin_installed("nonexistent") is False


class TestGetInstalledVersion:
    """Tests for get_installed_version function."""

    def test_get_version_found(self, temp_plugins_dir: Path) -> None:
        """Test getting version of installed plugin."""
        dist_info = temp_plugins_dir / "test_plugin-1.2.3.dist-info"
        dist_info.mkdir()

        version = plugin_installer.get_installed_version("test-plugin")

        assert version == "1.2.3"

    def test_get_version_not_found(self, temp_plugins_dir: Path) -> None:
        """Test getting version of non-installed plugin."""
        version = plugin_installer.get_installed_version("nonexistent")

        assert version is None

    def test_get_version_multiple_dist_info(self, temp_plugins_dir: Path) -> None:
        """Test getting version when multiple dist-info exist (returns first)."""
        (temp_plugins_dir / "test_plugin-1.0.0.dist-info").mkdir()
        (temp_plugins_dir / "test_plugin-2.0.0.dist-info").mkdir()

        version = plugin_installer.get_installed_version("test-plugin")

        # Should return one of the versions
        assert version in ["1.0.0", "2.0.0"]
