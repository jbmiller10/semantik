"""Runtime helpers for config that keep imports side-effect free.

These utilities centralize any filesystem preparation or validation that must
run during service startup (FastAPI lifespan or Celery worker boot), rather
than inside config class constructors/imports.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TCH003

from .base import BaseConfig  # noqa: TCH001
from .webui import WebuiConfig  # noqa: TCH001


def ensure_base_directories(config: BaseConfig) -> None:
    """Create common data/log directories lazily at runtime."""

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)


def ensure_webui_directories(config: WebuiConfig) -> None:
    """Ensure WebUI-specific directories exist.

    This is intentionally separate from config construction so imports stay
    pure and test-friendly.
    """

    ensure_base_directories(config)

    for path in _distinct_paths(
        [config.document_root_path, config.loaded_dir, config.ingest_dir, config.extract_dir]
        + list(config.document_allowed_roots),
    ):
        if path is None:
            continue
        path.mkdir(parents=True, exist_ok=True)


def require_jwt_secret(config: WebuiConfig) -> str:
    """Validate that a JWT secret is provided via environment.

    Returns the secret for convenience and raises a ValueError with a clear
    message when missing.
    """

    secret = config.JWT_SECRET_KEY.strip() if config.JWT_SECRET_KEY else ""
    if not secret:
        raise ValueError("JWT_SECRET_KEY is required. Run scripts/generate_jwt_secret.py to create one.")
    return secret


def require_auth_enabled(config: WebuiConfig) -> None:
    """Ensure DISABLE_AUTH is not enabled in production."""
    environment = (config.ENVIRONMENT or "").lower()
    if environment == "production" and config.DISABLE_AUTH:
        raise RuntimeError("DISABLE_AUTH cannot be enabled in production environments.")


def _distinct_paths(paths: list[Path | None]) -> tuple[Path, ...]:
    seen: dict[Path, None] = {}
    for maybe_path in paths:
        if maybe_path is None:
            continue
        seen.setdefault(maybe_path, None)
    return tuple(seen.keys())


__all__ = ["ensure_base_directories", "ensure_webui_directories", "require_jwt_secret", "require_auth_enabled"]
