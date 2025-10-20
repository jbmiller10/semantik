# shared/config/webui.py

import secrets
from pathlib import Path
from typing import Any

from .base import BaseConfig


class WebuiConfig(BaseConfig):
    """
    WebUI-specific configuration.
    Contains settings specific to the web application and API.
    """

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str = "default-secret-key"  # MUST be overridden in .env
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    ALGORITHM: str = "HS256"
    DISABLE_AUTH: bool = False  # Set to True for development only

    # Service Ports
    WEBUI_PORT: int = 8080
    WEBUI_METRICS_PORT: int = 9092

    # Service URLs (for internal API calls)
    WEBUI_URL: str = "http://localhost:8080"
    WEBUI_INTERNAL_HOST: str = "localhost"  # Can be overridden for containerized deployments

    # External service URLs
    SEARCH_API_URL: str = "http://localhost:8000"

    # Search Configuration
    SEARCH_CANDIDATE_MULTIPLIER: int = 3  # How many candidates to retrieve for re-ranking (k * multiplier)

    # Redis Configuration (for Celery and WebSocket pub/sub)
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Feature toggles
    USE_CHUNKING_ORCHESTRATOR: bool = False

    # Document storage configuration
    DOCUMENT_ROOT: str | None = None
    DOCUMENT_ALLOWED_ROOTS: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with JWT secret key validation."""
        super().__init__(**kwargs)

        # Determine document root directory and ensure it exists
        self._document_root: Path | None = None
        self._document_allowed_roots: tuple[Path, ...] = ()
        self._default_document_mounts: tuple[Path, ...] = ()
        if self.DOCUMENT_ROOT:
            raw_document_root = Path(self.DOCUMENT_ROOT).expanduser()
            raw_document_root.mkdir(parents=True, exist_ok=True)
            self._document_root = raw_document_root.resolve()
        extra_roots = []
        raw_allowed = kwargs.get("DOCUMENT_ALLOWED_ROOTS") or self.DOCUMENT_ALLOWED_ROOTS
        if raw_allowed:
            for entry in str(raw_allowed).split(":"):
                entry = entry.strip()
                if not entry:
                    continue
                extra_roots.append(Path(entry).expanduser().resolve())
        self._document_allowed_roots = tuple(extra_roots)

        default_mounts: list[Path] = []
        for candidate in (Path("/mnt/docs"),):
            if candidate.exists():
                default_mounts.append(candidate.resolve())
        self._default_document_mounts = tuple(default_mounts)

        # JWT Secret Key file path (in the data directory)
        jwt_secret_file = self.data_dir / ".jwt_secret"

        # Handle JWT secret key based on environment
        if self.ENVIRONMENT == "production":
            # In production, JWT_SECRET_KEY must be explicitly set via environment variable
            if self.JWT_SECRET_KEY == "default-secret-key" or not self.JWT_SECRET_KEY:
                raise ValueError(
                    "JWT_SECRET_KEY must be explicitly set via environment variable in production. "
                    "Generate a secure key using: openssl rand -hex 32"
                )
        else:
            # In development, allow auto-generation or reading from file
            if self.JWT_SECRET_KEY == "default-secret-key":
                # Check if .jwt_secret file exists
                if jwt_secret_file.exists():
                    # Read the secret from file
                    try:
                        self.JWT_SECRET_KEY = jwt_secret_file.read_text().strip()
                        if not self.JWT_SECRET_KEY:
                            raise ValueError("JWT secret file is empty")
                    except Exception as e:
                        raise ValueError(f"Failed to read JWT secret from {jwt_secret_file}: {e}") from e
                else:
                    # Generate a new secret and save it to file
                    self.JWT_SECRET_KEY = secrets.token_hex(32)
                    try:
                        jwt_secret_file.write_text(self.JWT_SECRET_KEY)
                        # Set secure permissions (readable only by owner)
                        jwt_secret_file.chmod(0o600)
                    except Exception as e:
                        # If we can't write the file, continue with the generated secret
                        # but warn the user
                        import logging

                        logging.warning(
                            f"Generated JWT secret key but failed to save to {jwt_secret_file}: {e}. "
                            "The key will be regenerated on next startup unless JWT_SECRET_KEY is set."
                        )

    @property
    def document_root(self) -> Path | None:
        """Root directory where document content is stored when configured."""

        return self._document_root

    @property
    def document_allowed_roots(self) -> tuple[Path, ...]:
        """Additional directories allowed for serving document content."""

        explicit_roots: list[Path] = []
        if self._document_root is not None:
            explicit_roots.append(self._document_root)
        if self._document_allowed_roots:
            explicit_roots.extend(self._document_allowed_roots)

        roots: list[Path] = []
        if explicit_roots:
            roots.extend(explicit_roots)
        else:
            roots.extend(self._default_document_mounts)

        roots.append(self.loaded_dir.resolve())

        # Preserve order while removing duplicates
        return tuple(dict.fromkeys(roots))
