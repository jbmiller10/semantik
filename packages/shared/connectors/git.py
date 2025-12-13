"""Git repository connector for remote document sources."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import logging
import mimetypes
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from shared.config import settings
from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.text_processing.extraction import extract_and_serialize
from shared.utils.hashing import compute_content_hash

logger = logging.getLogger(__name__)

# Default supported file extensions for Git sources
DEFAULT_INCLUDE_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".adoc",  # Documentation
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",  # Code
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",  # More code
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",  # C-family
    ".html",
    ".css",
    ".scss",
    ".yaml",
    ".yml",
    ".json",  # Web/config
    ".sh",
    ".bash",
    ".zsh",  # Scripts
    ".sql",
    ".graphql",  # Query languages
    ".toml",
    ".ini",
    ".cfg",  # Config files
}

# Maximum file size (10 MB default for Git files - smaller than local)
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024

# Default shallow clone depth
DEFAULT_SHALLOW_DEPTH = 1


class GitConnector(BaseConnector):
    """Connector for remote Git repository sources.

    Clones/fetches Git repositories and indexes file contents with
    support for HTTPS token or SSH key authentication.

    Config keys:
        repo_url (required): Git repository URL (HTTPS or SSH)
        ref (optional): Branch, tag, or commit to checkout (default: "main")
        auth_method (optional): "none", "https_token", or "ssh_key" (default: "none")
        include_globs (optional): List of glob patterns to include (e.g., ["*.md", "docs/**"])
        exclude_globs (optional): List of glob patterns to exclude (e.g., ["*.min.js"])
        max_file_size_mb (optional): Maximum file size in MB (default: 10)
        shallow_depth (optional): Shallow clone depth (default: 1, 0 for full)

    Secrets (set via set_credentials):
        token: HTTPS personal access token (for auth_method="https_token")
        ssh_key: SSH private key content (for auth_method="ssh_key")
        ssh_passphrase: SSH key passphrase (optional)

    Example:
        ```python
        connector = GitConnector({
            "repo_url": "https://github.com/user/repo.git",
            "ref": "main",
            "auth_method": "https_token",
            "include_globs": ["*.md", "docs/**"],
        })
        connector.set_credentials(token="ghp_xxx...")

        if await connector.authenticate():
            async for doc in connector.load_documents():
                print(doc.unique_id)
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the Git connector."""
        self._token: str | None = None
        self._ssh_key: str | None = None
        self._ssh_passphrase: str | None = None
        self._repo_dir: Path | None = None
        self._commit_sha: str | None = None
        self._refs: list[str] = []
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate required config keys."""
        if "repo_url" not in self._config:
            raise ValueError("GitConnector requires 'repo_url' in config")

        url = self._config["repo_url"]
        if not url.startswith(("https://", "git@", "ssh://")):
            raise ValueError(f"Invalid repo_url: must be HTTPS or SSH URL, got: {url}")

        auth_method = self._config.get("auth_method", "none")
        if auth_method not in ("none", "https_token", "ssh_key"):
            raise ValueError(f"Invalid auth_method: {auth_method}")

        if auth_method == "https_token" and not url.startswith(("http://", "https://")):
            raise ValueError("auth_method=https_token requires an HTTP(S) repo_url")

    def set_credentials(
        self,
        token: str | None = None,
        ssh_key: str | None = None,
        ssh_passphrase: str | None = None,
    ) -> None:
        """Set authentication credentials.

        Args:
            token: HTTPS personal access token
            ssh_key: SSH private key content
            ssh_passphrase: SSH key passphrase
        """
        self._token = token
        self._ssh_key = ssh_key
        self._ssh_passphrase = ssh_passphrase

    @property
    def repo_url(self) -> str:
        """Get the repository URL."""
        return str(self._config["repo_url"])

    @property
    def ref(self) -> str:
        """Get the ref to checkout."""
        return str(self._config.get("ref", "main"))

    @property
    def auth_method(self) -> str:
        """Get the authentication method."""
        return str(self._config.get("auth_method", "none"))

    @property
    def include_globs(self) -> list[str]:
        """Get include glob patterns."""
        globs = self._config.get("include_globs", [])
        return list(globs) if globs else []

    @property
    def exclude_globs(self) -> list[str]:
        """Get exclude glob patterns."""
        globs = self._config.get("exclude_globs", [])
        return list(globs) if globs else []

    @property
    def max_file_size(self) -> int:
        """Get maximum file size in bytes."""
        mb = self._config.get("max_file_size_mb", DEFAULT_MAX_FILE_SIZE // (1024 * 1024))
        return int(mb) * 1024 * 1024

    @property
    def shallow_depth(self) -> int:
        """Get shallow clone depth (0 for full clone)."""
        return int(self._config.get("shallow_depth", DEFAULT_SHALLOW_DEPTH))

    def _get_cache_dir(self, source_id: int | None = None) -> Path:
        """Get the cache directory for this repository.

        Args:
            source_id: Optional source ID to use in path (for uniqueness)
        """
        # Create a unique directory name from the repo URL
        url_hash = hashlib.sha256(self.repo_url.encode()).hexdigest()[:16]

        # Parse URL for a readable name
        parsed = urlparse(self.repo_url)
        if parsed.scheme:
            # HTTPS URL
            path_parts = parsed.path.strip("/").replace(".git", "").split("/")
            repo_name = "_".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
        else:
            # SSH URL like git@github.com:user/repo.git
            repo_name = self.repo_url.split(":")[-1].replace(".git", "").replace("/", "_")

        # Use source_id if available for uniqueness
        dir_name = f"{source_id}_{repo_name}_{url_hash}" if source_id else f"{repo_name}_{url_hash}"

        return Path(settings.git_cache_dir / dir_name)

    def _redact_sensitive(self, text: str) -> str:
        """Redact secrets (tokens, etc.) from text that may be logged or raised."""
        if not text:
            return text
        if self._token:
            return text.replace(self._token, "***")
        return text

    async def _run_git_command(
        self,
        args: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300,
    ) -> tuple[int, str, str]:
        """Run a git command asynchronously.

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Disable interactive prompts
        full_env["GIT_TERMINAL_PROMPT"] = "0"

        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=full_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return (
                process.returncode or 0,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except FileNotFoundError as exc:
            raise ValueError("Git binary not found - ensure 'git' is installed and on PATH") from exc
        except TimeoutError:
            logger.error(f"Git command timed out: git {' '.join(args)}")
            raise ValueError(f"Git command timed out after {timeout}s") from None

    def _setup_ssh_env(self, temp_dir: Path) -> dict[str, str]:
        """Set up SSH environment for git commands.

        Returns:
            Environment variables dict for subprocess
        """
        env: dict[str, str] = {}

        if self.auth_method != "ssh_key":
            return env

        if not self._ssh_key:
            raise ValueError("SSH key not set - call set_credentials() first")

        if shutil.which("ssh") is None:
            raise ValueError("SSH binary not found - install an OpenSSH client to use ssh_key auth")

        # Write SSH key to temp file
        key_file = temp_dir / "id_rsa"
        key_file.write_text(self._ssh_key)
        key_file.chmod(0o600)

        env["GIT_SSH_KEY_FILE"] = str(key_file)

        # Create SSH wrapper script
        ssh_script = temp_dir / "ssh_wrapper.sh"
        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

        if self._ssh_passphrase:
            if shutil.which("sshpass") is None:
                raise ValueError(
                    "sshpass is required for encrypted SSH keys - install sshpass or use an unencrypted key"
                )

            passphrase_file = temp_dir / "ssh_passphrase"
            passphrase_file.write_text(self._ssh_passphrase)
            passphrase_file.chmod(0o600)
            env["GIT_SSH_PASSPHRASE_FILE"] = str(passphrase_file)

        ssh_script.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            f'SSH_OPTS="{ssh_opts}"\n'
            'if [ -n "${GIT_SSH_PASSPHRASE_FILE:-}" ]; then\n'
            '  exec sshpass -f "$GIT_SSH_PASSPHRASE_FILE" ssh $SSH_OPTS -i "$GIT_SSH_KEY_FILE" "$@"\n'
            "fi\n"
            'exec ssh $SSH_OPTS -i "$GIT_SSH_KEY_FILE" "$@"\n'
        )

        ssh_script.chmod(0o700)

        env["GIT_SSH_COMMAND"] = str(ssh_script)
        return env

    def _setup_https_env(self, temp_dir: Path) -> dict[str, str]:
        """Set up HTTPS token auth via GIT_ASKPASS.

        This avoids embedding tokens in clone URLs (which can be persisted in
        `.git/config` or echoed in error output).
        """
        env: dict[str, str] = {}

        if self.auth_method != "https_token":
            return env

        if not self._token:
            raise ValueError("Token not set - call set_credentials() first")

        parsed = urlparse(self.repo_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("auth_method=https_token requires an HTTP(S) repo_url")

        token_file = temp_dir / "git_token"
        token_file.write_text(self._token)
        token_file.chmod(0o600)

        askpass_script = temp_dir / "git_askpass.sh"
        askpass_script.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            'prompt="${1:-}"\n'
            'token="$(cat "$GIT_ASKPASS_TOKEN_FILE")"\n'
            'case "$prompt" in\n'
            "  *Username*|*username*) printf '%s\\n' \"$token\" ;;\n"
            "  *) printf '\\n' ;;\n"
            "esac\n"
        )
        askpass_script.chmod(0o700)

        env["GIT_ASKPASS"] = str(askpass_script)
        env["GIT_ASKPASS_TOKEN_FILE"] = str(token_file)
        return env

    async def authenticate(self) -> bool:
        """Verify repository access by running git ls-remote.

        Returns:
            True if repository is accessible.

        Raises:
            ValueError: If authentication fails.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env = {}
            env.update(self._setup_ssh_env(temp_path))
            env.update(self._setup_https_env(temp_path))

            # Get all refs (branches and tags)
            code, stdout, stderr = await self._run_git_command(
                ["ls-remote", "--heads", "--tags", self.repo_url],
                env=env,
                timeout=60,
            )

            if code != 0:
                raise ValueError(f"Cannot access repository: {self._redact_sensitive(stderr)}")

            # Parse refs from output
            # Format: sha\trefs/heads/branch or sha\trefs/tags/tag
            self._refs = []
            for line in stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    ref = parts[1]
                    # Convert refs/heads/main to main, refs/tags/v1.0 to v1.0
                    if ref.startswith("refs/heads/"):
                        self._refs.append(ref[11:])  # Remove "refs/heads/"
                    elif ref.startswith("refs/tags/"):
                        self._refs.append(ref[10:])  # Remove "refs/tags/"

            logger.info(f"Successfully authenticated to {self.repo_url}, found {len(self._refs)} refs")
            return True

    def get_refs(self) -> list[str]:
        """Get the list of refs (branches/tags) found during authentication.

        Returns:
            List of ref names (e.g., ["main", "develop", "v1.0"])
        """
        return self._refs.copy()

    async def _clone_or_fetch(self, source_id: int | None = None) -> Path:
        """Clone the repository or fetch updates.

        Args:
            source_id: Optional source ID for cache directory

        Returns:
            Path to the repository directory
        """
        cache_dir = self._get_cache_dir(source_id)
        cache_dir.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env = {}
            env.update(self._setup_ssh_env(temp_path))
            env.update(self._setup_https_env(temp_path))

            if cache_dir.exists() and (cache_dir / ".git").exists():
                # Fetch updates
                logger.info(f"Fetching updates for {self.repo_url}")
                code, _, stderr = await self._run_git_command(
                    ["fetch", "--prune", "origin"],
                    cwd=cache_dir,
                    env=env,
                    timeout=300,
                )
                if code != 0:
                    logger.warning("Fetch failed, will re-clone: %s", self._redact_sensitive(stderr))
                    shutil.rmtree(cache_dir)

            if not cache_dir.exists():
                # Clone repository
                logger.info(f"Cloning {self.repo_url} to {cache_dir}")
                clone_args = ["clone"]

                if self.shallow_depth > 0:
                    clone_args.extend(["--depth", str(self.shallow_depth)])

                clone_args.extend([self.repo_url, str(cache_dir)])

                code, _, stderr = await self._run_git_command(
                    clone_args,
                    env=env,
                    timeout=600,
                )
                if code != 0:
                    raise ValueError(f"Clone failed: {self._redact_sensitive(stderr)}")

            # Checkout the specified ref
            code, _, stderr = await self._run_git_command(
                ["checkout", self.ref],
                cwd=cache_dir,
                timeout=60,
            )
            if code != 0:
                # Try fetching the ref first (might be a remote branch)
                await self._run_git_command(
                    ["fetch", "origin", f"{self.ref}:{self.ref}"],
                    cwd=cache_dir,
                    env=env,
                    timeout=120,
                )
                code, _, stderr = await self._run_git_command(
                    ["checkout", self.ref],
                    cwd=cache_dir,
                    timeout=60,
                )
                if code != 0:
                    raise ValueError(f"Cannot checkout ref '{self.ref}': {self._redact_sensitive(stderr)}")

            # Get current commit SHA
            code, stdout, _ = await self._run_git_command(
                ["rev-parse", "HEAD"],
                cwd=cache_dir,
                timeout=10,
            )
            self._commit_sha = stdout.strip() if code == 0 else None

            self._repo_dir = cache_dir
            return cache_dir

    def _matches_patterns(self, rel_path: str) -> bool:
        """Check if a path matches include/exclude patterns.

        Args:
            rel_path: Relative path from repo root

        Returns:
            True if the file should be included
        """
        # Normalize path separators
        rel_path = rel_path.replace("\\", "/")

        # If include patterns specified, file must match at least one
        if self.include_globs:
            matched = False
            for pattern in self.include_globs:
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(Path(rel_path).name, pattern):
                    matched = True
                    break
            if not matched:
                return False

        # Check exclude patterns
        for pattern in self.exclude_globs:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(Path(rel_path).name, pattern):
                return False

        return True

    def _is_supported_extension(self, path: Path) -> bool:
        """Check if the file extension is supported."""
        # If include_globs are specified, trust them for extension filtering
        if self.include_globs:
            return True

        # Otherwise use default extension list
        return path.suffix.lower() in DEFAULT_INCLUDE_EXTENSIONS

    async def _get_blob_sha(self, file_path: Path) -> str | None:
        """Get the git blob SHA for a file."""
        if not self._repo_dir:
            return None

        rel_path = file_path.relative_to(self._repo_dir)
        code, stdout, _ = await self._run_git_command(
            ["ls-tree", "HEAD", str(rel_path)],
            cwd=self._repo_dir,
            timeout=10,
        )

        if code == 0 and stdout.strip():
            # Format: mode type sha\tpath
            parts = stdout.strip().split()
            if len(parts) >= 3:
                return parts[2]

        return None

    async def load_documents(
        self,
        source_id: int | None = None,
    ) -> AsyncIterator[IngestedDocument]:
        """Yield documents from the Git repository.

        Args:
            source_id: Optional source ID for cache directory uniqueness

        Yields:
            IngestedDocument for each matching file.
        """
        repo_dir = await self._clone_or_fetch(source_id)

        # Walk the repository
        for root, dirs, files in os.walk(repo_dir):
            # Skip .git directory
            if ".git" in dirs:
                dirs.remove(".git")

            for filename in files:
                file_path = Path(root) / filename
                rel_path = str(file_path.relative_to(repo_dir))

                # Check patterns
                if not self._matches_patterns(rel_path):
                    continue

                # Check extension
                if not self._is_supported_extension(file_path):
                    continue

                # Process the file
                doc = await self._process_file(file_path, rel_path)
                if doc is not None:
                    yield doc

    async def _process_file(
        self,
        file_path: Path,
        rel_path: str,
    ) -> IngestedDocument | None:
        """Process a single file from the repository.

        Args:
            file_path: Absolute path to the file
            rel_path: Relative path from repo root

        Returns:
            IngestedDocument or None if file should be skipped
        """
        # Avoid reading through symlinks (can escape repo root)
        if file_path.is_symlink():
            logger.debug(f"Skipping symlinked file: {rel_path}")
            return None

        # Check file size
        try:
            stat = file_path.stat()
            if stat.st_size > self.max_file_size:
                logger.debug(f"Skipping file too large: {rel_path} ({stat.st_size} bytes)")
                return None
            if stat.st_size == 0:
                logger.debug(f"Skipping empty file: {rel_path}")
                return None
            file_size = stat.st_size
        except Exception as e:
            logger.error(f"Cannot access file {rel_path}: {e}")
            return None

        # Get blob SHA for change detection
        blob_sha = await self._get_blob_sha(file_path)

        # Try to extract text content
        content: str | None = None

        # For text files, try direct reading first
        mime_type, _ = mimetypes.guess_type(str(file_path))
        is_likely_text = (
            mime_type
            and (
                mime_type.startswith("text/")
                or mime_type in ("application/json", "application/xml", "application/javascript")
            )
        ) or file_path.suffix.lower() in {".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml"}

        if is_likely_text:
            with contextlib.suppress(Exception):
                content = file_path.read_text(encoding="utf-8", errors="replace")

        # Fall back to extraction service for binary formats
        if content is None:
            try:
                elements = extract_and_serialize(str(file_path))
                content = "\n\n".join(text for text, _ in elements)
            except Exception as e:
                logger.debug(f"Cannot extract content from {rel_path}: {e}")
                # For binary files that can't be extracted, read as text anyway
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    return None

        # Skip empty documents
        if not content or not content.strip():
            logger.debug(f"Skipping empty content: {rel_path}")
            return None

        # Compute content hash
        content_hash = compute_content_hash(content)

        # Build unique ID: git://{repo_url}/{path}
        unique_id = f"git://{self.repo_url}/{rel_path}"

        # Build metadata
        metadata: dict[str, Any] = {
            "file_path": rel_path,
            "file_size": file_size,
            "mime_type": mime_type,
            "blob_sha": blob_sha,
            "commit_sha": self._commit_sha,
            "ref": self.ref,
            "repo_url": self.repo_url,
        }

        return IngestedDocument(
            content=content,
            unique_id=unique_id,
            source_type="git",
            metadata=metadata,
            content_hash=content_hash,
            file_path=None,  # No local file path for git sources
        )

    def cleanup(self) -> None:
        """Remove the cached repository directory."""
        if self._repo_dir and self._repo_dir.exists():
            logger.info(f"Cleaning up repository cache: {self._repo_dir}")
            shutil.rmtree(self._repo_dir, ignore_errors=True)
            self._repo_dir = None
