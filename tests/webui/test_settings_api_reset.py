"""Tests for reset database endpoint resource cleanup."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from packages.webui.api.settings import reset_database_endpoint


@pytest.mark.asyncio()
async def test_reset_database_closes_qdrant_client() -> None:
    """The reset endpoint should close the AsyncQdrantClient after operations."""

    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_result
    mock_session.scalar.return_value = 0
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    with (
        patch("packages.webui.api.settings.AsyncQdrantClient") as mock_client_cls,
        patch("pathlib.Path.glob", return_value=[]),
    ):
        mock_client = mock_client_cls.return_value
        mock_client.delete_collection = AsyncMock()
        mock_client.aclose = AsyncMock()

        response = await reset_database_endpoint(current_user={"is_superuser": True}, db=mock_session)

        assert response["status"] == "success"
        mock_client.aclose.assert_awaited_once()
