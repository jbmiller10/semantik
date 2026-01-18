# shared/config/webui.py

from pathlib import Path

from pydantic import field_validator

from .base import BaseConfig


class WebuiConfig(BaseConfig):
    """
    WebUI-specific configuration.
    Contains settings specific to the web application and API.
    """

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str  # MUST be provided via environment
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

    # Resource limits (configurable)
    MAX_COLLECTIONS_PER_USER: int = 10
    MAX_STORAGE_GB_PER_USER: float = 50.0

    # Cache defaults
    CACHE_DEFAULT_TTL_SECONDS: int = 300

    # Redis Configuration (for Celery and WebSocket pub/sub)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Background cleanup circuit breaker configuration
    REDIS_CLEANUP_INTERVAL_SECONDS: int = 60
    REDIS_CLEANUP_MAX_CONSECUTIVE_FAILURES: int = 5
    REDIS_CLEANUP_BACKOFF_MULTIPLIER: float = 2.0
    REDIS_CLEANUP_MAX_BACKOFF_SECONDS: int = 300

    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"

    # Feature toggles
    USE_CHUNKING_ORCHESTRATOR: bool = False

    # Parallel ingestion settings
    PARALLEL_INGESTION_ENABLED: bool = True  # Enable parallel extraction/chunking
    PARALLEL_INGESTION_WORKERS: int = 0  # Number of extraction workers (0 = auto-detect based on CPU count)
    PARALLEL_INGESTION_MAX_WORKERS: int = 0  # Maximum workers cap (0 = no limit, use all CPUs)

    # Document storage configuration
    DOCUMENT_ROOT: str | None = None
    DOCUMENT_ALLOWED_ROOTS: str | None = None

    # Artifact storage configuration (for non-file sources like Git, IMAP)
    MAX_ARTIFACT_BYTES: int = 50 * 1024 * 1024  # 50 MB default max artifact size

    # Connector secrets encryption key (Fernet format: 44 chars, base64-encoded)
    # If not set, connector secrets cannot be stored (passwords, tokens, SSH keys)
    # Generate with: python scripts/generate_secrets_key.py
    CONNECTOR_SECRETS_KEY: str | None = None

    # API Key Management Configuration
    API_KEY_MAX_PER_USER: int = 20
    API_KEY_MAX_EXPIRY_DAYS: int = 3650  # ~10 years (0 = no max)
    API_KEY_DEFAULT_EXPIRY_DAYS: int = 365

    # Runtime override hooks (used in tests)
    _document_root: Path | None = None
    _document_allowed_roots: tuple[Path, ...] | None = None
    _default_document_mounts: tuple[Path, ...] | None = None

    # Keep side-effect free; runtime code is responsible for validating/creating paths and
    # ensuring JWT secrets are present (see shared.config.runtime).

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _validate_jwt_secret(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("JWT_SECRET_KEY must be set via environment (use scripts/generate_jwt_secret.py)")
        return value

    @field_validator("CONNECTOR_SECRETS_KEY")
    @classmethod
    def _validate_connector_secrets_key(cls, value: str | None) -> str | None:
        if not value or not value.strip():
            return None  # Encryption disabled

        value = value.strip()

        # Fernet keys are 32 bytes, base64-encoded = 44 characters
        if len(value) != 44:
            raise ValueError(
                f"CONNECTOR_SECRETS_KEY must be 44 characters (Fernet format), got {len(value)}. "
                "Generate with: python scripts/generate_secrets_key.py"
            )

        # Validate it's valid base64 and can be decoded
        try:
            import base64

            decoded = base64.urlsafe_b64decode(value)
            if len(decoded) != 32:
                raise ValueError(
                    "CONNECTOR_SECRETS_KEY must decode to 32 bytes. "
                    "Generate with: python scripts/generate_secrets_key.py"
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(
                f"CONNECTOR_SECRETS_KEY is not valid base64: {e}. "
                "Generate with: python scripts/generate_secrets_key.py"
            ) from e

        return value

    @field_validator(
        "MAX_COLLECTIONS_PER_USER",
        "CACHE_DEFAULT_TTL_SECONDS",
        "REDIS_CLEANUP_INTERVAL_SECONDS",
        "REDIS_CLEANUP_MAX_CONSECUTIVE_FAILURES",
        "REDIS_CLEANUP_MAX_BACKOFF_SECONDS",
    )
    @classmethod
    def _validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be a positive integer")
        return value

    @field_validator("MAX_STORAGE_GB_PER_USER")
    @classmethod
    def _validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Value must be a positive number")
        return value

    @field_validator("REDIS_CLEANUP_BACKOFF_MULTIPLIER")
    @classmethod
    def _validate_backoff_multiplier(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError("REDIS_CLEANUP_BACKOFF_MULTIPLIER must be >= 1.0")
        return value

    @property
    def document_root_path(self) -> Path | None:
        """Return resolved document root if configured or overridden in tests."""

        if getattr(self, "_document_root", None) is not None:
            return self._document_root

        if not self.DOCUMENT_ROOT:
            return None
        return Path(self.DOCUMENT_ROOT).expanduser().resolve()

    @property
    def document_allowed_roots(self) -> tuple[Path, ...]:
        """Additional directories allowed for serving document content."""

        roots: list[Path] = []
        if self.document_root_path is not None:
            roots.append(self.document_root_path)

        override_roots = getattr(self, "_document_allowed_roots", None)
        if override_roots is not None:
            roots.extend(override_roots)
        else:
            raw_allowed = self.DOCUMENT_ALLOWED_ROOTS
            if raw_allowed:
                for entry in str(raw_allowed).split(":"):
                    entry = entry.strip()
                    if not entry:
                        continue
                    roots.append(Path(entry).expanduser().resolve())

        if not roots:
            default_mounts: list[Path] = []
            default_override = getattr(self, "_default_document_mounts", None)
            candidates = default_override if default_override is not None else (Path("/mnt/docs"),)
            for candidate in candidates:
                if isinstance(candidate, Path) and candidate.exists():
                    default_mounts.append(candidate.resolve())
            roots.extend(default_mounts)

        # Always allow the loaded_dir by default for compatibility
        roots.append(self.loaded_dir.resolve())

        return tuple(dict.fromkeys(roots))

    @property
    def document_root(self) -> Path | None:
        """Root directory where document content is stored when configured."""
        return self.document_root_path

    @property
    def should_enforce_document_roots(self) -> bool:
        """Return True when document access must be constrained to known roots."""

        if self.document_root_path is not None:
            return True

        override_roots = getattr(self, "_document_allowed_roots", None)
        if override_roots:
            return True

        if self.DOCUMENT_ALLOWED_ROOTS:
            return True

        default_mounts = getattr(self, "_default_document_mounts", None)
        candidates = default_mounts if default_mounts is not None else (Path("/mnt/docs"),)
        return any(isinstance(mount, Path) and mount.exists() for mount in candidates)
