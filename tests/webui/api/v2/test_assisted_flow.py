"""Tests for assisted flow API endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from webui.auth import get_current_user
from webui.main import app


class TestStartAssistedFlow:
    """Test POST /api/v2/assisted-flow/start endpoint."""

    @pytest.mark.asyncio()
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

            with patch(
                "webui.api.v2.assisted_flow.create_sdk_session",
                new_callable=AsyncMock,
            ) as mock_create_session:
                mock_create_session.return_value = ("af_deadbeefdeadbeef", MagicMock())

                response = await api_client.post(
                    "/api/v2/assisted-flow/start",
                    json={"source_id": source_id},
                    headers=api_auth_headers,
                )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["source_name"] == "Test Source"

    @pytest.mark.asyncio()
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
            mock_get_stats.side_effect = EntityNotFoundError("collection_source", str(source_id))

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

    @pytest.mark.asyncio()
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

    @pytest.mark.asyncio()
    async def test_message_streams_response(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Streams SSE events from SDK client."""
        mock_client = MagicMock()

        # Mock receive_response as an async generator
        async def mock_receive():
            from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

            yield AssistantMessage(content=[TextBlock(text="Hello")], model="haiku")
            yield ResultMessage(
                subtype="stop",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="default",
            )

        mock_client.receive_response = mock_receive

        with (
            patch(
                "webui.api.v2.assisted_flow.get_session_client",
                new_callable=AsyncMock,
            ) as mock_get_client,
            patch(
                "webui.api.v2.assisted_flow.send_message",
                new_callable=AsyncMock,
            ) as mock_send_message,
        ):
            mock_get_client.return_value = mock_client
            mock_send_message.return_value = mock_client

            response = await api_client.post(
                "/api/v2/assisted-flow/test_session/messages/stream",
                json={"message": "Hello"},
                headers=api_auth_headers,
            )

        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")


class TestSubmitAnswer:
    """Test POST /api/v2/assisted-flow/{session_id}/answer endpoint."""

    @pytest.mark.asyncio()
    async def test_submit_answer_success(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Submitting a valid answer resolves the pending question."""
        mock_manager = MagicMock()
        mock_manager.submit_answer = AsyncMock(return_value=True)

        with (
            patch(
                "webui.api.v2.assisted_flow.get_session_client",
                new_callable=AsyncMock,
            ) as mock_get_client,
            patch(
                "webui.api.v2.assisted_flow.get_question_manager",
                return_value=mock_manager,
            ),
        ):
            mock_get_client.return_value = MagicMock()

            response = await api_client.post(
                "/api/v2/assisted-flow/test_session/answer",
                json={
                    "question_id": "q_123",
                    "answers": {"Which model?": "gpt-4"},
                },
                headers=api_auth_headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_manager.submit_answer.assert_awaited_once_with(
            "q_123", {"Which model?": "gpt-4"}
        )

    @pytest.mark.asyncio()
    async def test_submit_answer_session_not_found(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Returns 404 when session does not exist."""
        with patch(
            "webui.api.v2.assisted_flow.get_session_client",
            new_callable=AsyncMock,
        ) as mock_get_client:
            mock_get_client.return_value = None

            response = await api_client.post(
                "/api/v2/assisted-flow/nonexistent_session/answer",
                json={
                    "question_id": "q_123",
                    "answers": {"Which model?": "gpt-4"},
                },
                headers=api_auth_headers,
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio()
    async def test_submit_answer_question_not_found(
        self,
        api_client: AsyncClient,
        api_auth_headers: dict,
    ) -> None:
        """Returns 404 when question_id is not found or already answered."""
        mock_manager = MagicMock()
        mock_manager.submit_answer = AsyncMock(return_value=False)

        with (
            patch(
                "webui.api.v2.assisted_flow.get_session_client",
                new_callable=AsyncMock,
            ) as mock_get_client,
            patch(
                "webui.api.v2.assisted_flow.get_question_manager",
                return_value=mock_manager,
            ),
        ):
            mock_get_client.return_value = MagicMock()

            response = await api_client.post(
                "/api/v2/assisted-flow/test_session/answer",
                json={
                    "question_id": "q_unknown",
                    "answers": {"Which model?": "gpt-4"},
                },
                headers=api_auth_headers,
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio()
    async def test_submit_answer_invalid_user_id(
        self,
        db_session,
        use_fakeredis,
        reset_redis_manager,
    ) -> None:
        """Returns 401 when user_id is not a valid integer."""
        _ = use_fakeredis
        _ = reset_redis_manager

        async def override_get_current_user_invalid() -> dict[str, Any]:
            return {
                "id": None,
                "username": "invalid",
                "email": "invalid@test.com",
                "full_name": "Invalid User",
            }

        original_overrides = dict(app.dependency_overrides)
        app.dependency_overrides[get_current_user] = override_get_current_user_invalid

        try:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/v2/assisted-flow/test_session/answer",
                    json={
                        "question_id": "q_123",
                        "answers": {"Which model?": "gpt-4"},
                    },
                )

            assert response.status_code == 401
            assert "invalid user session" in response.json()["detail"].lower()
        finally:
            app.dependency_overrides = original_overrides
