"""Unit tests for version resolution."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

import shared.version as version_module


def _clear_cache() -> None:
    version_module.get_version.cache_clear()


def test_get_version_prefers_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cache()
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "9.9.9")

    assert version_module.get_version() == "9.9.9"


def test_get_version_falls_back_to_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_cache()

    def _raise(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)

    version_path = tmp_path / "VERSION"
    version_path.write_text("1.2.3")
    monkeypatch.setattr(version_module, "_VERSION_FILE_PATHS", [version_path])

    assert version_module.get_version() == "1.2.3"


def test_get_version_fallback_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_cache()

    def _raise(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    monkeypatch.setattr(version_module, "_VERSION_FILE_PATHS", [Path("/nonexistent")])

    assert version_module.get_version() == "0.0.0"
