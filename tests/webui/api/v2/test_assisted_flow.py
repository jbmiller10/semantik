"""Tests for assisted flow API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient


class TestStartAssistedFlow:
    """Test POST /api/v2/assisted-flow/start endpoint."""

    @pytest.mark.asyncio
    async def test_start_returns_session_id(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Start endpoint returns session ID."""
        source_id = 42

        with patch(
            "webui.api.v2.assisted_flow.get_source_stats",
            new_callable=AsyncMock,
        ) as mock_get_stats:
            mock_get_stats.return_value = {
                "source_name": "Test Source",
                "source_type": "directory",
                "source_path": "/test",
                "source_config": {},
            }

            response = await api_client.post(
                "/api/v2/assisted-flow/start",
                json={"source_id": source_id},
                headers=api_auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["source_name"] == "Test Source"

    @pytest.mark.asyncio
    async def test_start_source_not_found(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Start endpoint returns error when source not found."""
        from shared.database.exceptions import EntityNotFoundError

        source_id = 999

        with patch(
            "webui.api.v2.assisted_flow.get_source_stats",
            new_callable=AsyncMock,
        ) as mock_get_stats:
            mock_get_stats.side_effect = EntityNotFoundError(
                "collection_source", str(source_id)
            )

            response = await api_client.post(
                "/api/v2/assisted-flow/start",
                json={"source_id": source_id},
                headers=api_auth_headers,
            )

        # EntityNotFoundError should be converted to 500 by the endpoint
        # (or 404 if global exception handler catches it)
        assert response.status_code in [404, 500]


class TestSendMessageStream:
    """Test POST /api/v2/assisted-flow/{session_id}/messages/stream endpoint."""

    @pytest.mark.asyncio
    async def test_message_session_not_found(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Returns error when session not found."""
        with patch(
            "webui.api.v2.assisted_flow.get_session_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_get_client.return_value = None

            response = await api_client.post(
                "/api/v2/assisted-flow/unknown_session/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_message_streams_response(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Streams SSE events from SDK client."""
        mock_client = MagicMock()

        # Mock receive_response as an async generator
        async def mock_receive():
            yield MagicMock(type="text", content="Hello")

        mock_client.query = AsyncMock()
        mock_client.receive_response = mock_receive

        with patch(
            "webui.api.v2.assisted_flow.get_session_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_get_client.return_value = mock_client

            response = await api_client.post(
                "/api/v2/assisted-flow/test_session/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")
