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

    # Redis Configuration (for Celery and WebSocket pub/sub)
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Feature toggles
    USE_CHUNKING_ORCHESTRATOR: bool = False

    # Document storage configuration
    DOCUMENT_ROOT: str | None = None
    DOCUMENT_ALLOWED_ROOTS: str | None = None

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
