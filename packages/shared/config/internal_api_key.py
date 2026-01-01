"""Utilities for managing the internal API key shared across services."""

from __future__ import annotations

import hashlib
import logging
import secrets
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

KEY_FILENAME = "internal_api_key"
_DEFAULT_FILE_PERMISSIONS = 0o600


class _ConfigProtocol(Protocol):
    """Subset of configuration attributes required for managing the API key."""

    ENVIRONMENT: str
    INTERNAL_API_KEY: str | None

    @property
    def data_dir(self) -> Path:
        ...


def ensure_internal_api_key(
    config: _ConfigProtocol,
    *,
    key_file: Path | str | None = None,
) -> str:
    """Ensure the internal API key is available and synchronised across processes.

    The logic follows these rules:
    1. If ``config.INTERNAL_API_KEY`` is already populated, persist it to the shared file
       (if missing or mismatched) and return it.
    2. Otherwise, attempt to load the key from the shared file.
    3. If still missing:
       - Raise ``RuntimeError`` in production environments.
       - Generate a new random key in non-production environments, persist it, and update settings.

    Args:
        config: Settings object with ``ENVIRONMENT``/``INTERNAL_API_KEY`` attributes.
        key_file: Optional override for the persistence location (primarily for tests).

    Returns:
        The resolved internal API key.

    Raises:
        RuntimeError: If the key is missing in production.
    """

    resolved_path = Path(key_file) if key_file is not None else _default_key_path(config)

    # 1. Prefer explicit configuration.
    explicit_key = _clean_key(config.INTERNAL_API_KEY)
    if explicit_key:
        config.INTERNAL_API_KEY = explicit_key
        _persist_key_if_needed(resolved_path, explicit_key)
        return explicit_key

    # 2. Attempt to load from disk.
    file_key = _read_key_from_file(resolved_path)
    if file_key:
        config.INTERNAL_API_KEY = file_key
        logger.debug(
            "Loaded internal API key from %s (fingerprint=%s)",
            resolved_path,
            _fingerprint(file_key),
        )
        return file_key

    # 3. Generate or fail depending on environment.
    environment = (config.ENVIRONMENT or "").lower()
    if environment == "production":
        raise RuntimeError(
            "INTERNAL_API_KEY must be explicitly configured in production. "
            f"Set the environment variable or place the key at {resolved_path}."
        )

    generated_key = secrets.token_urlsafe(48)
    config.INTERNAL_API_KEY = generated_key
    _persist_key_if_needed(resolved_path, generated_key, force_write=True)
    logger.info(
        "Generated internal API key for %s environment (fingerprint=%s, path=%s).",
        environment or "unknown",
        _fingerprint(generated_key),
        resolved_path,
    )
    return generated_key


def _default_key_path(config: _ConfigProtocol) -> Path:
    """Return the default filesystem path for storing the API key."""
    try:
        data_dir = config.data_dir
    except Exception as exc:  # pragma: no cover - defensive logging only
        raise RuntimeError("Unable to determine data directory for internal API key persistence") from exc

    return Path(data_dir) / KEY_FILENAME


def _clean_key(raw_key: str | None) -> str | None:
    """Normalise a key by stripping whitespace and validating non-empty values."""
    if raw_key is None:
        return None
    cleaned = raw_key.strip()
    return cleaned or None


def _read_key_from_file(path: Path | str) -> str | None:
    """Read and validate the key from disk."""
    path = Path(path)
    try:
        if not path.exists():
            return None
        contents = path.read_text(encoding="utf-8").strip()
        return contents or None
    except OSError as exc:
        logger.warning("Failed to read internal API key from %s: %s", path, exc)
        return None


def _persist_key_if_needed(path: Path | str, key: str, *, force_write: bool = False) -> None:
    """Persist the key to disk if missing or different."""
    path = Path(path)
    try:
        current = _read_key_from_file(path)
        if not force_write and current == key:
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(key, encoding="utf-8")
        tmp_path.replace(path)

        try:
            path.chmod(_DEFAULT_FILE_PERMISSIONS)
        except OSError as exc:
            logger.debug("Unable to set permissions on %s: %s", path, exc)

        logger.debug(
            "Persisted internal API key to %s (fingerprint=%s)",
            path,
            _fingerprint(key),
        )
    except OSError as exc:
        logger.warning("Failed to persist internal API key to %s: %s", path, exc)


def _fingerprint(key: str) -> str:
    """Return a short fingerprint of the key for safe logging."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


__all__ = ["ensure_internal_api_key", "KEY_FILENAME"]
