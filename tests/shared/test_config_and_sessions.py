from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config.base import BaseConfig
from shared.config.webui import WebuiConfig
from shared.database import database as db_module, pg_connection_manager
from shared.database.database import ensure_async_sessionmaker, get_db


def test_base_config_has_no_side_effects(tmp_path: Path) -> None:
    data_dir = tmp_path / "data_dir"
    logs_dir = tmp_path / "logs_dir"

    # Instantiation should not create directories
    cfg = BaseConfig(DATA_DIR=data_dir, LOGS_DIR=logs_dir)

    assert not data_dir.exists()
    assert not logs_dir.exists()
    # Paths should still resolve correctly
    assert cfg.data_dir == data_dir
    assert cfg.logs_dir == logs_dir


def test_webui_config_requires_jwt_secret() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
        WebuiConfig(JWT_SECRET_KEY="")

    cfg = WebuiConfig(JWT_SECRET_KEY="test-secret")
    assert cfg.JWT_SECRET_KEY == "test-secret"


@pytest.mark.asyncio()
async def test_get_db_returns_stub_in_testing(monkeypatch) -> None:
    monkeypatch.setenv("TESTING", "true")
    # Clear sessionmaker to force stub path
    pg_connection_manager._sessionmaker = None  # type: ignore[attr-defined]

    agen = get_db()
    session = await agen.__anext__()
    try:
        assert isinstance(session, AsyncMock)
    finally:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()


@pytest.mark.asyncio()
async def test_ensure_async_sessionmaker_uses_existing() -> None:
    # Provide a fake sessionmaker to avoid database initialization
    fake_sessionmaker = MagicMock(name="fake_sessionmaker")
    pg_connection_manager._sessionmaker = fake_sessionmaker  # type: ignore[attr-defined]
    db_module.AsyncSessionLocal = fake_sessionmaker

    sessionmaker = await ensure_async_sessionmaker()

    assert sessionmaker is fake_sessionmaker

    # Clean up for other tests
    db_module.AsyncSessionLocal = None
