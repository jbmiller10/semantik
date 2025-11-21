"""Tests for internal API key management utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.config.internal_api_key import ensure_internal_api_key


class DummyConfig:
    """Minimal configuration object compatible with ensure_internal_api_key."""

    def __init__(self, data_dir: Path, *, environment: str = "development", key: str | None = None) -> None:
        self._data_dir = Path(data_dir)
        self.ENVIRONMENT = environment
        self.INTERNAL_API_KEY = key

    @property
    def data_dir(self) -> Path:
        return self._data_dir


def test_generates_and_persists_key(tmp_path: Path) -> None:
    """When no key is present, generate and persist a new one in non-production."""
    config = DummyConfig(tmp_path, environment="development", key=None)

    key = ensure_internal_api_key(config)

    assert key
    assert key == config.INTERNAL_API_KEY
    stored = (tmp_path / "internal_api_key").read_text(encoding="utf-8").strip()
    assert stored == key


def test_reads_existing_key_file(tmp_path: Path) -> None:
    """Load the key from disk when configuration value is missing."""
    key_path = tmp_path / "internal_api_key"
    key_path.write_text("stored-key\n", encoding="utf-8")

    config = DummyConfig(tmp_path, environment="development", key=None)

    key = ensure_internal_api_key(config)

    assert key == "stored-key"
    assert config.INTERNAL_API_KEY == "stored-key"


def test_explicit_key_updates_file(tmp_path: Path) -> None:
    """Persist explicit configuration values to disk."""
    key_path = tmp_path / "internal_api_key"
    key_path.write_text("old-key", encoding="utf-8")

    config = DummyConfig(tmp_path, environment="development", key="provided-key ")

    key = ensure_internal_api_key(config)

    assert key == "provided-key"
    assert key_path.read_text(encoding="utf-8").strip() == "provided-key"


def test_missing_key_in_production_raises(tmp_path: Path) -> None:
    """Fail fast when running in production without an explicit key."""
    config = DummyConfig(tmp_path, environment="production", key=None)

    with pytest.raises(RuntimeError):
        ensure_internal_api_key(config)
