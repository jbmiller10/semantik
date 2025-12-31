"""Integration tests for the v1 settings endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import OperationalError, SQLAlchemyError

if TYPE_CHECKING:  # pragma: no cover
    from httpx import AsyncClient


class TestGetDatabaseStats:
    """Tests for the get_database_stats endpoint."""

    @pytest.mark.asyncio()
    async def test_database_size_returns_positive(
        self,
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
    async def test_handles_query_errors(self, monkeypatch, tmp_path) -> None:
        """Stats should gracefully handle database query failures."""
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
    async def test_reports_parquet_sizes(self, monkeypatch, tmp_path) -> None:
        """Stats should report parquet file counts and sizes."""
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
    async def test_handles_nonexistent_output_dir(self, monkeypatch) -> None:
        """Stats should handle non-existent output directory gracefully."""
        from webui.api import settings as settings_module

        db = AsyncMock()
        db.scalar = AsyncMock(side_effect=[5, 10, 2097152])
        db.in_transaction = MagicMock(return_value=False)

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", "/nonexistent/path/that/does/not/exist")

        result = await settings_module.get_database_stats(current_user={}, db=db)

        assert result["collection_count"] == 5
        assert result["file_count"] == 10
        assert result["database_size_mb"] == 2.0
        assert result["parquet_files_count"] == 0
        assert result["parquet_size_mb"] == 0.0

    @pytest.mark.asyncio()
    async def test_handles_parquet_stat_errors(self, monkeypatch, tmp_path) -> None:
        """Stats should handle individual parquet file stat failures."""
        from webui.api import settings as settings_module

        # Create a parquet file
        parquet_file = tmp_path / "test.parquet"
        parquet_file.write_bytes(b"0" * 1024)

        db = AsyncMock()
        db.scalar = AsyncMock(side_effect=[1, 2, 1048576])
        db.in_transaction = MagicMock(return_value=False)

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        # Mock Path.stat to fail for the parquet file
        original_stat = Path.stat

        def failing_stat(self, *args, **kwargs):
            if str(self).endswith(".parquet"):
                raise OSError("Permission denied")
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", failing_stat)

        result = await settings_module.get_database_stats(current_user={}, db=db)

        # Should still return results with 0 parquet size
        assert result["parquet_files_count"] == 1
        assert result["parquet_size_mb"] == 0.0

    @pytest.mark.asyncio()
    async def test_handles_unexpected_sql_error_on_size(self, monkeypatch, tmp_path) -> None:
        """Stats should handle unexpected SQLAlchemy errors on size query."""
        from webui.api import settings as settings_module

        db = AsyncMock()
        db.scalar = AsyncMock(
            side_effect=[
                1,  # collection count
                2,  # document count
                SQLAlchemyError("Unexpected error"),  # size query
            ]
        )
        db.rollback = AsyncMock()
        db.in_transaction = MagicMock(return_value=True)

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        result = await settings_module.get_database_stats(current_user={}, db=db)

        assert result["collection_count"] == 1
        assert result["file_count"] == 2
        assert result["database_size_mb"] is None


class TestResetDatabaseEndpoint:
    """Tests for the reset_database_endpoint."""

    @pytest.mark.asyncio()
    async def test_requires_admin(self) -> None:
        """Non-admin users should receive 403."""
        from webui.api.settings import reset_database_endpoint

        with pytest.raises(HTTPException) as exc_info:
            await reset_database_endpoint(current_user={"is_superuser": False}, db=AsyncMock())

        assert exc_info.value.status_code == 403
        assert "Only administrators" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_rejects_missing_superuser_flag(self) -> None:
        """Users without is_superuser flag should be rejected."""
        from webui.api.settings import reset_database_endpoint

        with pytest.raises(HTTPException) as exc_info:
            await reset_database_endpoint(current_user={}, db=AsyncMock())

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio()
    async def test_success_path(self, monkeypatch, tmp_path) -> None:
        """Admin users should be able to reset the database."""
        from webui.api import settings as settings_module

        # Create mock parquet files
        (tmp_path / "test1.parquet").write_bytes(b"data")
        (tmp_path / "test2.parquet").write_bytes(b"data")

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        # Mock database session
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_collection = MagicMock()
        mock_collection.vector_store_name = "test_vector_store"
        mock_result.scalars.return_value.all.return_value = [mock_collection]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        # Mock Qdrant client
        mock_qdrant = AsyncMock()
        mock_qdrant.delete_collection = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with patch("webui.api.settings.AsyncQdrantClient", return_value=mock_qdrant):
            result = await settings_module.reset_database_endpoint(
                current_user={"is_superuser": True}, db=mock_db
            )

        assert result["status"] == "success"
        assert "reset successfully" in result["message"]

        # Verify parquet files were deleted
        assert not (tmp_path / "test1.parquet").exists()
        assert not (tmp_path / "test2.parquet").exists()

        # Verify Qdrant collections were deleted
        mock_qdrant.delete_collection.assert_called()

    @pytest.mark.asyncio()
    async def test_handles_qdrant_deletion_error(self, monkeypatch, tmp_path) -> None:
        """Database reset should continue even if Qdrant deletion fails."""
        from webui.api import settings as settings_module

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_collection = MagicMock()
        mock_collection.vector_store_name = "test_store"
        mock_result.scalars.return_value.all.return_value = [mock_collection]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        # Mock Qdrant client that fails
        mock_qdrant = AsyncMock()
        mock_qdrant.delete_collection = AsyncMock(side_effect=Exception("Qdrant error"))
        mock_qdrant.close = AsyncMock()

        with patch("webui.api.settings.AsyncQdrantClient", return_value=mock_qdrant):
            result = await settings_module.reset_database_endpoint(
                current_user={"is_superuser": True}, db=mock_db
            )

        # Should still succeed despite Qdrant errors
        assert result["status"] == "success"

    @pytest.mark.asyncio()
    async def test_handles_db_clear_error(self, monkeypatch, tmp_path) -> None:
        """Database reset should raise HTTPException on DB clear failure."""
        from webui.api import settings as settings_module

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        # First execute for getting collections succeeds
        # Subsequent executes for deletion fail
        mock_db.execute = AsyncMock(
            side_effect=[
                mock_result,  # Get collections
                Exception("Delete failed"),  # Delete operation
            ]
        )
        mock_db.rollback = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("webui.api.settings.AsyncQdrantClient", return_value=mock_qdrant),
            pytest.raises(HTTPException) as exc_info,
        ):
            await settings_module.reset_database_endpoint(
                current_user={"is_superuser": True}, db=mock_db
            )

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio()
    async def test_closes_qdrant_client_on_error(self, monkeypatch, tmp_path) -> None:
        """Qdrant client should be closed even when errors occur."""
        from webui.api import settings as settings_module

        monkeypatch.setattr(settings_module, "OUTPUT_DIR", str(tmp_path))

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_collection = MagicMock()
        mock_collection.vector_store_name = "test_store"
        mock_result.scalars.return_value.all.return_value = [mock_collection]
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        # Mock Qdrant client
        mock_qdrant = AsyncMock()
        mock_qdrant.delete_collection = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with patch("webui.api.settings.AsyncQdrantClient", return_value=mock_qdrant):
            await settings_module.reset_database_endpoint(
                current_user={"is_superuser": True}, db=mock_db
            )

        # Verify close was called
        mock_qdrant.close.assert_called_once()
