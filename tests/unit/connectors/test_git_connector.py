"""Unit tests for GitConnector."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from shared.connectors.git import DEFAULT_MAX_FILE_SIZE, DEFAULT_SHALLOW_DEPTH, GitConnector


class TestGitConnectorInit:
    """Test GitConnector initialization and config validation."""

    def test_valid_https_config(self):
        """Test initialization with valid HTTPS config."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "ref": "main",
            }
        )
        assert connector.repo_url == "https://github.com/user/repo.git"
        assert connector.ref == "main"

    def test_valid_ssh_config(self):
        """Test initialization with valid SSH config."""
        connector = GitConnector(
            {
                "repo_url": "git@github.com:user/repo.git",
            }
        )
        assert connector.repo_url == "git@github.com:user/repo.git"

    def test_valid_ssh_protocol_config(self):
        """Test initialization with SSH protocol URL."""
        connector = GitConnector(
            {
                "repo_url": "ssh://git@github.com/user/repo.git",
            }
        )
        assert connector.repo_url == "ssh://git@github.com/user/repo.git"

    def test_missing_repo_url(self):
        """Test initialization fails without repo_url."""
        with pytest.raises(ValueError, match=r"repo_url") as exc_info:
            GitConnector({})
        assert "repo_url" in str(exc_info.value)

    def test_invalid_repo_url(self):
        """Test initialization fails with invalid repo_url."""
        with pytest.raises(ValueError, match=r"Invalid repo_url") as exc_info:
            GitConnector({"repo_url": "ftp://example.com/repo"})
        assert "Invalid repo_url" in str(exc_info.value)

    def test_invalid_auth_method(self):
        """Test initialization fails with invalid auth_method."""
        with pytest.raises(ValueError, match=r"Invalid auth_method") as exc_info:
            GitConnector(
                {
                    "repo_url": "https://github.com/user/repo.git",
                    "auth_method": "invalid",
                }
            )
        assert "Invalid auth_method" in str(exc_info.value)


class TestGitConnectorProperties:
    """Test GitConnector property methods."""

    @pytest.fixture()
    def connector(self):
        return GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "ref": "develop",
                "auth_method": "https_token",
                "include_globs": ["*.md", "docs/**"],
                "exclude_globs": ["*.min.js"],
                "max_file_size_mb": 5,
                "shallow_depth": 10,
            }
        )

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
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "https_token",
            }
        )
        connector.set_credentials(token="ghp_xxx")

        assert connector._token == "ghp_xxx"
        assert connector._ssh_key is None

    def test_set_credentials_ssh(self):
        """Test setting SSH credentials."""
        connector = GitConnector(
            {
                "repo_url": "git@github.com:user/repo.git",
                "auth_method": "ssh_key",
            }
        )
        connector.set_credentials(
            ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----",
            ssh_passphrase="secret",
        )

        assert connector._ssh_key == "-----BEGIN OPENSSH PRIVATE KEY-----"
        assert connector._ssh_passphrase == "secret"


class TestGitConnectorHttpsEnv:
    """Test HTTPS token auth environment setup."""

    def test_setup_https_env_no_token(self):
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "https_token",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(ValueError, match=r"Token not set"):
            connector._setup_https_env(Path(temp_dir))

    def test_setup_https_env_with_token_does_not_embed_token_in_script(self):
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "https_token",
            }
        )
        connector.set_credentials(token="ghp_xxx")

        with tempfile.TemporaryDirectory() as temp_dir:
            env = connector._setup_https_env(Path(temp_dir))

            assert "GIT_ASKPASS" in env
            assert "GIT_ASKPASS_TOKEN_FILE" in env

            askpass_script = Path(env["GIT_ASKPASS"])
            assert askpass_script.exists()
            assert "ghp_xxx" not in askpass_script.read_text()

            token_file = Path(env["GIT_ASKPASS_TOKEN_FILE"])
            assert token_file.exists()
            assert token_file.read_text() == "ghp_xxx"


class TestGitConnectorSshEnv:
    """Test SSH environment setup."""

    def test_setup_ssh_env_no_key(self):
        """Test SSH env empty when no key."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "none",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            env = connector._setup_ssh_env(Path(temp_dir))
            assert env == {}

    def test_setup_ssh_env_with_key(self):
        """Test SSH env setup with key."""
        connector = GitConnector(
            {
                "repo_url": "git@github.com:user/repo.git",
                "auth_method": "ssh_key",
            }
        )
        connector.set_credentials(ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----")

        def which_side_effect(cmd: str):
            if cmd == "ssh":
                return "/usr/bin/ssh"
            return None

        with (
            patch("shared.connectors.git.shutil.which", side_effect=which_side_effect),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            env = connector._setup_ssh_env(Path(temp_dir))

            assert "GIT_SSH_COMMAND" in env

            # Check key file was created
            key_file = Path(temp_dir) / "id_rsa"
            assert key_file.exists()
            assert key_file.read_text() == "-----BEGIN OPENSSH PRIVATE KEY-----"

    def test_setup_ssh_env_with_passphrase(self):
        """Test SSH env setup with passphrase."""
        connector = GitConnector(
            {
                "repo_url": "git@github.com:user/repo.git",
                "auth_method": "ssh_key",
            }
        )
        connector.set_credentials(
            ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----",
            ssh_passphrase="secret",
        )

        def which_side_effect(cmd: str):
            if cmd == "ssh":
                return "/usr/bin/ssh"
            if cmd == "sshpass":
                return "/usr/bin/sshpass"
            return None

        with (
            patch("shared.connectors.git.shutil.which", side_effect=which_side_effect),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            env = connector._setup_ssh_env(Path(temp_dir))
            assert "GIT_SSH_COMMAND" in env

            # Check SSH wrapper script contains sshpass
            ssh_script = Path(temp_dir) / "ssh_wrapper.sh"
            assert ssh_script.exists()
            content = ssh_script.read_text()
            assert "sshpass" in content


class TestGitConnectorCacheDir:
    """Test cache directory generation."""

    def test_get_cache_dir_https(self):
        """Test cache dir for HTTPS URL."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        cache_dir = connector._get_cache_dir()
        assert "user_repo" in str(cache_dir)

    def test_get_cache_dir_ssh(self):
        """Test cache dir for SSH URL."""
        connector = GitConnector(
            {
                "repo_url": "git@github.com:user/repo.git",
            }
        )

        cache_dir = connector._get_cache_dir()
        assert "user_repo" in str(cache_dir)

    def test_get_cache_dir_with_source_id(self):
        """Test cache dir includes source_id."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        cache_dir = connector._get_cache_dir(source_id=123)
        assert "123_" in str(cache_dir)


class TestGitConnectorAuthenticate:
    """Test authenticate method."""

    @pytest.mark.asyncio()
    async def test_authenticate_success(self):
        """Test successful authentication."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

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
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with patch.object(connector, "_run_git_command") as mock_run:
            mock_run.return_value = (128, "", "Authentication failed")

            with pytest.raises(ValueError, match=r"Cannot access repository") as exc_info:
                await connector.authenticate()

            assert "Cannot access repository" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_authenticate_exception(self):
        """Test authentication with exception propagates."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with patch.object(connector, "_run_git_command") as mock_run:
            mock_run.side_effect = ValueError("Timeout")

            with pytest.raises(ValueError, match=r"Timeout") as exc_info:
                await connector.authenticate()

            assert "Timeout" in str(exc_info.value)


class TestGitConnectorGetRefs:
    """Test get_refs method."""

    def test_get_refs_returns_copy(self):
        """Test get_refs returns a copy of refs list."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )
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
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

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
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with patch("shared.connectors.git.asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.side_effect = TimeoutError()
            mock_exec.return_value = mock_process

            with pytest.raises(ValueError, match=r"timed out") as exc_info:
                await connector._run_git_command(["clone"], timeout=1)

            assert "timed out" in str(exc_info.value)


class TestGitConnectorFileMatching:
    """Test file matching logic."""

    def test_matches_patterns_no_globs(self):
        """Test pattern matching with no globs (allow all)."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        # Without include_globs, all paths match
        assert connector._matches_patterns("README.md")
        assert connector._matches_patterns("src/main.py")
        assert connector._matches_patterns("image.png")

    def test_is_supported_extension_default(self):
        """Test file extension support with defaults."""
        from pathlib import Path

        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        assert connector._is_supported_extension(Path("README.md"))
        assert connector._is_supported_extension(Path("src/main.py"))
        assert not connector._is_supported_extension(Path("image.png"))

    def test_matches_patterns_with_include_globs(self):
        """Test pattern matching with custom include globs."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "include_globs": ["*.txt", "docs/**"],
            }
        )

        assert connector._matches_patterns("notes.txt")
        assert connector._matches_patterns("docs/guide.md")
        assert not connector._matches_patterns("src/main.py")

    def test_matches_patterns_with_exclude_globs(self):
        """Test pattern matching with exclude globs."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "exclude_globs": ["*.min.js", "vendor/**"],
            }
        )

        assert not connector._matches_patterns("app.min.js")
        assert not connector._matches_patterns("vendor/lib.js")
        assert connector._matches_patterns("src/app.js")

    def test_is_supported_extension_with_include_globs(self):
        """Test extension support trusts include_globs."""
        from pathlib import Path

        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "include_globs": ["*.png"],  # Custom globs
            }
        )

        # When include_globs are set, extension check returns True
        # (trusting the include patterns)
        assert connector._is_supported_extension(Path("image.png"))


class TestGitConnectorRedactSensitive:
    """Test _redact_sensitive method."""

    def test_redact_sensitive_replaces_token(self):
        """Test that token is redacted from error messages."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "auth_method": "https_token",
            }
        )
        connector.set_credentials(token="ghp_secret123")

        result = connector._redact_sensitive("Error: ghp_secret123 authentication failed")

        assert "ghp_secret123" not in result
        assert "***" in result
        assert "Error:" in result

    def test_redact_sensitive_no_token_set(self):
        """Test redaction returns unchanged text when no token set."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        result = connector._redact_sensitive("Error: some message")
        assert result == "Error: some message"

    def test_redact_sensitive_empty_text(self):
        """Test redaction handles empty text."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )
        connector.set_credentials(token="ghp_secret123")

        assert connector._redact_sensitive("") == ""
        assert connector._redact_sensitive(None) is None


class TestGitConnectorCleanup:
    """Test cleanup method."""

    def test_cleanup_removes_repo_directory(self):
        """Test cleanup() removes the cached repository directory."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock repo directory
            repo_dir = Path(temp_dir) / "test_repo"
            repo_dir.mkdir()
            (repo_dir / "file.txt").write_text("content")

            connector._repo_dir = repo_dir
            assert repo_dir.exists()

            connector.cleanup()

            assert not repo_dir.exists()
            assert connector._repo_dir is None

    def test_cleanup_handles_no_repo_dir(self):
        """Test cleanup() handles case when no repo directory is set."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        # Should not raise
        connector.cleanup()
        assert connector._repo_dir is None


class TestGitConnectorGetBlobSha:
    """Test _get_blob_sha method."""

    @pytest.mark.asyncio()
    async def test_get_blob_sha_returns_sha(self):
        """Test _get_blob_sha returns blob SHA from git ls-tree."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir
            test_file = repo_dir / "README.md"
            test_file.write_text("# Test")

            with patch.object(connector, "_run_git_command") as mock_run:
                mock_run.return_value = (0, "100644 blob abc123def456\tREADME.md", "")

                sha = await connector._get_blob_sha(test_file)

                assert sha == "abc123def456"

    @pytest.mark.asyncio()
    async def test_get_blob_sha_returns_none_for_not_found(self):
        """Test _get_blob_sha returns None when file not in tree."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir
            test_file = repo_dir / "missing.txt"
            test_file.write_text("content")

            with patch.object(connector, "_run_git_command") as mock_run:
                mock_run.return_value = (0, "", "")

                sha = await connector._get_blob_sha(test_file)

                assert sha is None

    @pytest.mark.asyncio()
    async def test_get_blob_sha_returns_none_when_no_repo_dir(self):
        """Test _get_blob_sha returns None when no repo directory set."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )
        connector._repo_dir = None

        sha = await connector._get_blob_sha(Path("/some/file.txt"))

        assert sha is None


class TestGitConnectorCloneOrFetch:
    """Test _clone_or_fetch method."""

    @pytest.mark.asyncio()
    async def test_clone_or_fetch_creates_new_clone(self):
        """Test cloning when cache dir doesn't exist."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "shallow_depth": 1,
            }
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(connector, "_get_cache_dir") as mock_cache_dir,
            patch.object(connector, "_run_git_command") as mock_run,
            patch.object(connector, "_setup_ssh_env", return_value={}),
            patch.object(connector, "_setup_https_env", return_value={}),
        ):
            cache_dir = Path(temp_dir) / "cache"
            mock_cache_dir.return_value = cache_dir

            # Mock successful clone and checkout
            mock_run.side_effect = [
                (0, "", ""),  # clone
                (0, "", ""),  # checkout
                (0, "abc123", ""),  # rev-parse
            ]

            result = await connector._clone_or_fetch()

            assert result == cache_dir
            # Verify clone was called
            clone_call = mock_run.call_args_list[0]
            assert "clone" in clone_call[0][0]
            assert "--depth" in clone_call[0][0]
            assert "1" in clone_call[0][0]

    @pytest.mark.asyncio()
    async def test_clone_or_fetch_fetches_existing_repo(self):
        """Test fetching when .git directory exists."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir(parents=True)
            (cache_dir / ".git").mkdir()

            with (
                patch.object(connector, "_get_cache_dir", return_value=cache_dir),
                patch.object(connector, "_run_git_command") as mock_run,
                patch.object(connector, "_setup_ssh_env", return_value={}),
                patch.object(connector, "_setup_https_env", return_value={}),
            ):
                mock_run.side_effect = [
                    (0, "", ""),  # fetch
                    (0, "", ""),  # checkout
                    (0, "def456", ""),  # rev-parse
                ]

                result = await connector._clone_or_fetch()

                assert result == cache_dir
                # Verify fetch was called (not clone)
                fetch_call = mock_run.call_args_list[0]
                assert "fetch" in fetch_call[0][0]

    @pytest.mark.asyncio()
    async def test_clone_or_fetch_reclones_on_fetch_failure(self):
        """Test re-clone when fetch fails."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir(parents=True)
            (cache_dir / ".git").mkdir()
            (cache_dir / "old_file.txt").write_text("old")

            with (
                patch.object(connector, "_get_cache_dir", return_value=cache_dir),
                patch.object(connector, "_run_git_command") as mock_run,
                patch.object(connector, "_setup_ssh_env", return_value={}),
                patch.object(connector, "_setup_https_env", return_value={}),
            ):
                mock_run.side_effect = [
                    (128, "", "fetch failed"),  # fetch fails
                    (0, "", ""),  # clone succeeds
                    (0, "", ""),  # checkout
                    (0, "abc123", ""),  # rev-parse
                ]

                result = await connector._clone_or_fetch()

                assert result == cache_dir
                # Verify clone was called after fetch failure
                call_args = [call[0][0] for call in mock_run.call_args_list]
                assert any("clone" in args for args in call_args)

    @pytest.mark.asyncio()
    async def test_clone_or_fetch_handles_checkout_retry(self):
        """Test ref fetching when initial checkout fails."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "ref": "feature-branch",
            }
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(connector, "_get_cache_dir") as mock_cache_dir,
            patch.object(connector, "_run_git_command") as mock_run,
            patch.object(connector, "_setup_ssh_env", return_value={}),
            patch.object(connector, "_setup_https_env", return_value={}),
        ):
            cache_dir = Path(temp_dir) / "cache"
            mock_cache_dir.return_value = cache_dir

            mock_run.side_effect = [
                (0, "", ""),  # clone
                (1, "", "ref not found"),  # first checkout fails
                (0, "", ""),  # fetch origin ref:ref
                (0, "", ""),  # second checkout succeeds
                (0, "abc123", ""),  # rev-parse
            ]

            result = await connector._clone_or_fetch()

            assert result == cache_dir
            assert connector._commit_sha == "abc123"


class TestGitConnectorProcessFile:
    """Test _process_file method."""

    @pytest.mark.asyncio()
    async def test_process_file_skips_symlinks(self):
        """Test that symlinks are skipped."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir

            # Create a file and a symlink
            real_file = repo_dir / "real.txt"
            real_file.write_text("content")
            symlink = repo_dir / "link.txt"
            symlink.symlink_to(real_file)

            result = await connector._process_file(symlink, "link.txt")

            assert result is None

    @pytest.mark.asyncio()
    async def test_process_file_skips_large_files(self):
        """Test files exceeding max_file_size are skipped."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "max_file_size_mb": 1,  # 1 MB limit
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir

            # Create a large file (>1 MB)
            large_file = repo_dir / "large.txt"
            large_file.write_text("x" * (2 * 1024 * 1024))  # 2 MB

            result = await connector._process_file(large_file, "large.txt")

            assert result is None

    @pytest.mark.asyncio()
    async def test_process_file_skips_empty_files(self):
        """Test empty files (0 bytes) are skipped."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir

            empty_file = repo_dir / "empty.txt"
            empty_file.write_text("")

            result = await connector._process_file(empty_file, "empty.txt")

            assert result is None

    @pytest.mark.asyncio()
    async def test_process_file_reads_text_files_directly(self):
        """Test text files are read without extraction service."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir
            connector._commit_sha = "abc123"

            md_file = repo_dir / "README.md"
            md_file.write_text("# Hello World\n\nThis is a test.")

            with patch.object(connector, "_get_blob_sha", return_value="blob123"):
                result = await connector._process_file(md_file, "README.md")

            assert result is not None
            assert "Hello World" in result.content
            assert result.unique_id == "git://https://github.com/user/repo.git/README.md"
            assert result.source_type == "git"
            assert result.metadata["blob_sha"] == "blob123"
            assert result.metadata["commit_sha"] == "abc123"

    @pytest.mark.asyncio()
    async def test_process_file_builds_correct_metadata(self):
        """Test that metadata is correctly populated."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "ref": "main",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            connector._repo_dir = repo_dir
            connector._commit_sha = "commit123"

            py_file = repo_dir / "src" / "main.py"
            py_file.parent.mkdir(parents=True)
            py_file.write_text("print('hello')")

            with patch.object(connector, "_get_blob_sha", return_value="blob456"):
                result = await connector._process_file(py_file, "src/main.py")

            assert result is not None
            # Connector identity
            assert result.metadata["source_type"] == "git"
            assert result.metadata["source_path"] == "src/main.py"

            # Parser metadata contract
            assert result.metadata["parser"] == "text"
            assert result.metadata["filename"] == "main.py"
            assert result.metadata["file_extension"] == ".py"
            assert result.metadata["file_type"] == "py"
            assert isinstance(result.metadata["mime_type"], str)
            assert result.metadata["mime_type"]

            assert result.metadata["file_path"] == "src/main.py"
            assert result.metadata["ref"] == "main"
            assert result.metadata["repo_url"] == "https://github.com/user/repo.git"
            assert result.content_hash is not None


class TestGitConnectorLoadDocuments:
    """Test load_documents method."""

    @pytest.mark.asyncio()
    async def test_load_documents_yields_documents(self):
        """Test that load_documents yields IngestedDocument objects."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            (repo_dir / ".git").mkdir()
            (repo_dir / "README.md").write_text("# Test")
            (repo_dir / "src").mkdir()
            (repo_dir / "src" / "main.py").write_text("print('hello')")

            with (
                patch.object(connector, "_clone_or_fetch", return_value=repo_dir),
                patch.object(connector, "_get_blob_sha", return_value="blob123"),
            ):
                connector._repo_dir = repo_dir
                connector._commit_sha = "abc123"

                docs = [doc async for doc in connector.load_documents()]

            assert len(docs) == 2
            paths = [doc.metadata["file_path"] for doc in docs]
            assert "README.md" in paths
            assert "src/main.py" in paths

    @pytest.mark.asyncio()
    async def test_load_documents_skips_git_directory(self):
        """Test that .git directory is skipped during traversal."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            (repo_dir / ".git").mkdir()
            (repo_dir / ".git" / "config").write_text("[core]")
            (repo_dir / "README.md").write_text("# Test")

            with (
                patch.object(connector, "_clone_or_fetch", return_value=repo_dir),
                patch.object(connector, "_get_blob_sha", return_value="blob123"),
            ):
                connector._repo_dir = repo_dir
                connector._commit_sha = "abc123"

                docs = [doc async for doc in connector.load_documents()]

            # Only README.md should be yielded, not .git/config
            assert len(docs) == 1
            assert docs[0].metadata["file_path"] == "README.md"

    @pytest.mark.asyncio()
    async def test_load_documents_respects_include_patterns(self):
        """Test include_globs filtering in load_documents."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "include_globs": ["*.md"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            (repo_dir / ".git").mkdir()
            (repo_dir / "README.md").write_text("# Test")
            (repo_dir / "main.py").write_text("print('hello')")
            (repo_dir / "CHANGELOG.md").write_text("# Changes")

            with (
                patch.object(connector, "_clone_or_fetch", return_value=repo_dir),
                patch.object(connector, "_get_blob_sha", return_value="blob123"),
            ):
                connector._repo_dir = repo_dir
                connector._commit_sha = "abc123"

                docs = [doc async for doc in connector.load_documents()]

            # Only .md files should be yielded
            assert len(docs) == 2
            paths = [doc.metadata["file_path"] for doc in docs]
            assert "README.md" in paths
            assert "CHANGELOG.md" in paths
            assert "main.py" not in paths

    @pytest.mark.asyncio()
    async def test_load_documents_respects_exclude_patterns(self):
        """Test exclude_globs filtering in load_documents."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
                "exclude_globs": ["*.min.js"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir)
            (repo_dir / ".git").mkdir()
            (repo_dir / "app.js").write_text("console.log('hi')")
            (repo_dir / "app.min.js").write_text("console.log('hi')")

            with (
                patch.object(connector, "_clone_or_fetch", return_value=repo_dir),
                patch.object(connector, "_get_blob_sha", return_value="blob123"),
            ):
                connector._repo_dir = repo_dir
                connector._commit_sha = "abc123"

                docs = [doc async for doc in connector.load_documents()]

            # app.min.js should be excluded
            paths = [doc.metadata["file_path"] for doc in docs]
            assert "app.js" in paths
            assert "app.min.js" not in paths


class TestGitConnectorRunGitCommandFileNotFound:
    """Test _run_git_command FileNotFoundError handling."""

    @pytest.mark.asyncio()
    async def test_run_git_command_file_not_found_error(self):
        """Test FileNotFoundError when git binary not installed."""
        connector = GitConnector(
            {
                "repo_url": "https://github.com/user/repo.git",
            }
        )

        with patch("shared.connectors.git.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("git not found")

            with pytest.raises(ValueError, match=r"Git binary not found") as exc_info:
                await connector._run_git_command(["status"])

            assert "Git binary not found" in str(exc_info.value)
