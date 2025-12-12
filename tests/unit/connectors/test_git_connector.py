"""Unit tests for GitConnector."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.connectors.git import (
    DEFAULT_MAX_FILE_SIZE,
    DEFAULT_SHALLOW_DEPTH,
    GitConnector,
)


class TestGitConnectorInit:
    """Test GitConnector initialization and config validation."""

    def test_valid_https_config(self):
        """Test initialization with valid HTTPS config."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "ref": "main",
        })
        assert connector.repo_url == "https://github.com/user/repo.git"
        assert connector.ref == "main"

    def test_valid_ssh_config(self):
        """Test initialization with valid SSH config."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
        })
        assert connector.repo_url == "git@github.com:user/repo.git"

    def test_valid_ssh_protocol_config(self):
        """Test initialization with SSH protocol URL."""
        connector = GitConnector({
            "repo_url": "ssh://git@github.com/user/repo.git",
        })
        assert connector.repo_url == "ssh://git@github.com/user/repo.git"

    def test_missing_repo_url(self):
        """Test initialization fails without repo_url."""
        with pytest.raises(ValueError) as exc_info:
            GitConnector({})
        assert "repo_url" in str(exc_info.value)

    def test_invalid_repo_url(self):
        """Test initialization fails with invalid repo_url."""
        with pytest.raises(ValueError) as exc_info:
            GitConnector({"repo_url": "ftp://example.com/repo"})
        assert "Invalid repo_url" in str(exc_info.value)

    def test_invalid_auth_method(self):
        """Test initialization fails with invalid auth_method."""
        with pytest.raises(ValueError) as exc_info:
            GitConnector({
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "invalid",
            })
        assert "Invalid auth_method" in str(exc_info.value)


class TestGitConnectorProperties:
    """Test GitConnector property methods."""

    @pytest.fixture()
    def connector(self):
        return GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "ref": "develop",
            "auth_method": "https_token",
            "include_globs": ["*.md", "docs/**"],
            "exclude_globs": ["*.min.js"],
            "max_file_size_mb": 5,
            "shallow_depth": 10,
        })

    def test_repo_url(self, connector):
        assert connector.repo_url == "https://github.com/user/repo.git"

    def test_ref(self, connector):
        assert connector.ref == "develop"

    def test_ref_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.ref == "main"

    def test_auth_method(self, connector):
        assert connector.auth_method == "https_token"

    def test_auth_method_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.auth_method == "none"

    def test_include_globs(self, connector):
        assert connector.include_globs == ["*.md", "docs/**"]

    def test_include_globs_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.include_globs == []

    def test_exclude_globs(self, connector):
        assert connector.exclude_globs == ["*.min.js"]

    def test_exclude_globs_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.exclude_globs == []

    def test_max_file_size(self, connector):
        assert connector.max_file_size == 5 * 1024 * 1024

    def test_max_file_size_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.max_file_size == DEFAULT_MAX_FILE_SIZE

    def test_shallow_depth(self, connector):
        assert connector.shallow_depth == 10

    def test_shallow_depth_default(self):
        connector = GitConnector({"repo_url": "https://github.com/user/repo.git"})
        assert connector.shallow_depth == DEFAULT_SHALLOW_DEPTH


class TestGitConnectorCredentials:
    """Test credential handling."""

    def test_set_credentials_token(self):
        """Test setting token credentials."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "auth_method": "https_token",
        })
        connector.set_credentials(token="ghp_xxx")

        assert connector._token == "ghp_xxx"
        assert connector._ssh_key is None

    def test_set_credentials_ssh(self):
        """Test setting SSH credentials."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
            "auth_method": "ssh_key",
        })
        connector.set_credentials(
            ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----",
            ssh_passphrase="secret",
        )

        assert connector._ssh_key == "-----BEGIN OPENSSH PRIVATE KEY-----"
        assert connector._ssh_passphrase == "secret"


class TestGitConnectorAuthUrl:
    """Test authenticated URL building."""

    def test_build_auth_url_with_token(self):
        """Test building authenticated HTTPS URL."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "auth_method": "https_token",
        })
        connector.set_credentials(token="ghp_xxx")

        url = connector._build_auth_url()
        assert url == "https://ghp_xxx@github.com/user/repo.git"

    def test_build_auth_url_no_token(self):
        """Test URL unchanged when no token."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "auth_method": "none",
        })

        url = connector._build_auth_url()
        assert url == "https://github.com/user/repo.git"

    def test_build_auth_url_ssh(self):
        """Test SSH URL unchanged."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
            "auth_method": "ssh_key",
        })
        connector.set_credentials(token="ignored")

        url = connector._build_auth_url()
        assert url == "git@github.com:user/repo.git"


class TestGitConnectorSshEnv:
    """Test SSH environment setup."""

    def test_setup_ssh_env_no_key(self):
        """Test SSH env empty when no key."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "auth_method": "none",
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            env = connector._setup_ssh_env(Path(temp_dir))
            assert env == {}

    def test_setup_ssh_env_with_key(self):
        """Test SSH env setup with key."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
            "auth_method": "ssh_key",
        })
        connector.set_credentials(ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----")

        with tempfile.TemporaryDirectory() as temp_dir:
            env = connector._setup_ssh_env(Path(temp_dir))

            assert "GIT_SSH_COMMAND" in env

            # Check key file was created
            key_file = Path(temp_dir) / "id_rsa"
            assert key_file.exists()
            assert key_file.read_text() == "-----BEGIN OPENSSH PRIVATE KEY-----"

    def test_setup_ssh_env_with_passphrase(self):
        """Test SSH env setup with passphrase."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
            "auth_method": "ssh_key",
        })
        connector.set_credentials(
            ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----",
            ssh_passphrase="secret",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            env = connector._setup_ssh_env(Path(temp_dir))

            # Check SSH wrapper script contains sshpass
            ssh_script = Path(temp_dir) / "ssh_wrapper.sh"
            assert ssh_script.exists()
            content = ssh_script.read_text()
            assert "sshpass" in content


class TestGitConnectorCacheDir:
    """Test cache directory generation."""

    def test_get_cache_dir_https(self):
        """Test cache dir for HTTPS URL."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        cache_dir = connector._get_cache_dir()
        assert "user_repo" in str(cache_dir)

    def test_get_cache_dir_ssh(self):
        """Test cache dir for SSH URL."""
        connector = GitConnector({
            "repo_url": "git@github.com:user/repo.git",
        })

        cache_dir = connector._get_cache_dir()
        assert "user_repo" in str(cache_dir)

    def test_get_cache_dir_with_source_id(self):
        """Test cache dir includes source_id."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        cache_dir = connector._get_cache_dir(source_id=123)
        assert "123_" in str(cache_dir)


class TestGitConnectorAuthenticate:
    """Test authenticate method."""

    @pytest.mark.asyncio()
    async def test_authenticate_success(self):
        """Test successful authentication."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        with patch.object(connector, "_run_git_command") as mock_run:
            mock_run.return_value = (
                0,
                "abc123\trefs/heads/main\ndef456\trefs/heads/develop\nghi789\trefs/tags/v1.0.0",
                "",
            )

            result = await connector.authenticate()

            assert result is True
            assert "main" in connector._refs
            assert "develop" in connector._refs
            assert "v1.0.0" in connector._refs

    @pytest.mark.asyncio()
    async def test_authenticate_failure(self):
        """Test authentication failure raises ValueError."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        with patch.object(connector, "_run_git_command") as mock_run:
            mock_run.return_value = (128, "", "Authentication failed")

            with pytest.raises(ValueError) as exc_info:
                await connector.authenticate()

            assert "Cannot access repository" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_authenticate_exception(self):
        """Test authentication with exception propagates."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        with patch.object(connector, "_run_git_command") as mock_run:
            mock_run.side_effect = ValueError("Timeout")

            with pytest.raises(ValueError) as exc_info:
                await connector.authenticate()

            assert "Timeout" in str(exc_info.value)


class TestGitConnectorGetRefs:
    """Test get_refs method."""

    def test_get_refs_returns_copy(self):
        """Test get_refs returns a copy of refs list."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })
        connector._refs = ["main", "develop"]

        refs = connector.get_refs()

        assert refs == ["main", "develop"]
        # Ensure it's a copy
        refs.append("new-branch")
        assert "new-branch" not in connector._refs


class TestGitConnectorRunGitCommand:
    """Test _run_git_command method."""

    @pytest.mark.asyncio()
    async def test_run_git_command_success(self):
        """Test successful git command execution."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        with patch("shared.connectors.git.asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            code, stdout, stderr = await connector._run_git_command(["status"])

            assert code == 0
            assert stdout == "output"
            assert stderr == ""

    @pytest.mark.asyncio()
    async def test_run_git_command_timeout(self):
        """Test git command timeout."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        with patch("shared.connectors.git.asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = TimeoutError()
            mock_exec.return_value = mock_process

            with pytest.raises(ValueError) as exc_info:
                await connector._run_git_command(["clone"], timeout=1)

            assert "timed out" in str(exc_info.value)


class TestGitConnectorFileMatching:
    """Test file matching logic."""

    def test_matches_patterns_no_globs(self):
        """Test pattern matching with no globs (allow all)."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        # Without include_globs, all paths match
        assert connector._matches_patterns("README.md")
        assert connector._matches_patterns("src/main.py")
        assert connector._matches_patterns("image.png")

    def test_is_supported_extension_default(self):
        """Test file extension support with defaults."""
        from pathlib import Path

        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
        })

        assert connector._is_supported_extension(Path("README.md"))
        assert connector._is_supported_extension(Path("src/main.py"))
        assert not connector._is_supported_extension(Path("image.png"))

    def test_matches_patterns_with_include_globs(self):
        """Test pattern matching with custom include globs."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "include_globs": ["*.txt", "docs/**"],
        })

        assert connector._matches_patterns("notes.txt")
        assert connector._matches_patterns("docs/guide.md")
        assert not connector._matches_patterns("src/main.py")

    def test_matches_patterns_with_exclude_globs(self):
        """Test pattern matching with exclude globs."""
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "exclude_globs": ["*.min.js", "vendor/**"],
        })

        assert not connector._matches_patterns("app.min.js")
        assert not connector._matches_patterns("vendor/lib.js")
        assert connector._matches_patterns("src/app.js")

    def test_is_supported_extension_with_include_globs(self):
        """Test extension support trusts include_globs."""
        from pathlib import Path

        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "include_globs": ["*.png"],  # Custom globs
        })

        # When include_globs are set, extension check returns True
        # (trusting the include patterns)
        assert connector._is_supported_extension(Path("image.png"))
