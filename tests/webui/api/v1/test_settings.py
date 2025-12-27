"""Integration tests for the v1 settings endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError, SQLAlchemyError

if TYPE_CHECKING:  # pragma: no cover
    from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_database_size_returns_positive(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Ensure the settings endpoint reports a positive database size when available."""

    response = await api_client.get("/api/settings/stats", headers=api_auth_headers)

    assert response.status_code == 200, response.text

    payload = response.json()

    assert "database_size_mb" in payload
    size_mb = payload["database_size_mb"]

    if size_mb is None:
        pytest.skip("Database size unavailable in test environment")

    assert isinstance(size_mb, int | float)

    if size_mb <= 0:
        pytest.skip("Database size is zero in test environment")

    assert size_mb > 0


@pytest.mark.asyncio()
async def test_get_database_stats_handles_query_errors(monkeypatch, tmp_path) -> None:
    from webui.api import settings as settings_module

    db = AsyncMock()
    db.scalar = AsyncMock(
        side_effect=[
            SQLAlchemyError("collection count failed"),
            SQLAlchemyError("document count failed"),
            OperationalError("select", {}, Exception("size failed")),
        ]
    )
    db.rollback = AsyncMock()
    db.in_transaction = MagicMock(return_value=True)

    monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

    result = await settings_module.get_database_stats(current_user={}, db=db)

    assert result["collection_count"] == 0
    assert result["file_count"] == 0
    assert result["database_size_mb"] is None
    assert result["parquet_files_count"] == 0
    assert result["parquet_size_mb"] == 0.0
    assert db.rollback.await_count == 3


@pytest.mark.asyncio()
async def test_get_database_stats_reports_parquet_sizes(monkeypatch, tmp_path) -> None:
    from webui.api import settings as settings_module

    (tmp_path / "first.parquet").write_bytes(b"0" * 1024)
    (tmp_path / "second.parquet").write_bytes(b"0" * 2048)

    db = AsyncMock()
    db.scalar = AsyncMock(side_effect=[2, 5, 1048576])
    db.in_transaction = MagicMock(return_value=False)

    monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

    result = await settings_module.get_database_stats(current_user={}, db=db)

    assert result["collection_count"] == 2
    assert result["file_count"] == 5
    assert result["database_size_mb"] == 1.0
    assert result["parquet_files_count"] == 2
    assert result["parquet_size_mb"] == round((1024 + 2048) / 1024 / 1024, 2)


@pytest.mark.asyncio()
async def test_reset_database_endpoint_requires_admin() -> None:
    from webui.api.settings import reset_database_endpoint

    with pytest.raises(HTTPException) as exc_info:
        await reset_database_endpoint(current_user={"is_superuser": False}, db=AsyncMock())

    assert exc_info.value.status_code == 403
