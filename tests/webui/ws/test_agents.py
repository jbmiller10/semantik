"""Tests for agent WebSocket streaming endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from shared.agents.exceptions import AgentInterruptedError, SessionNotFoundError
from shared.agents.types import AgentMessage, MessageRole, MessageType
from shared.database.exceptions import AccessDeniedError

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture()
def mock_agent_service() -> AsyncMock:
    """Create a mock agent service."""
    service = AsyncMock()
    service.verify_websocket_access = AsyncMock()
    service.execute = AsyncMock()
    service.interrupt = AsyncMock()
    return service


@pytest.fixture()
def mock_session() -> MagicMock:
    """Create a mock agent session."""
    session = MagicMock()
    session.id = "uuid-123"
    session.external_id = "abc12345"
    session.agent_plugin_id = "claude-agent"
    session.agent_config = {}
    session.collection_id = None
    session.user_id = 1
    return session


@pytest.fixture()
def mock_user() -> dict[str, Any]:
    """Create a mock user dict."""
    return {"id": 1, "email": "test@example.com", "is_active": True}


@pytest.fixture()
def auth_token() -> str:
    """Create a test JWT token."""
    from webui.auth import create_access_token

    return create_access_token({"sub": "1", "email": "test@example.com"})


class TestAgentWebSocketEndpoint:
    """Tests for agent WebSocket endpoint."""

    @pytest.mark.asyncio()
    async def test_connection_requires_auth(self) -> None:
        """Connection without token should be rejected."""
        from webui.main import app

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            client = TestClient(app)
            # WebSocket close raises generic exception from starlette
            with pytest.raises(Exception):  # noqa: B017, PT011
                with client.websocket_connect("/ws/agents/abc12345"):
                    pass

    @pytest.mark.asyncio()
    async def test_connection_with_valid_token(
        self, mock_agent_service: AsyncMock, mock_session: MagicMock, mock_user: dict, auth_token: str
    ) -> None:
        """Connection with valid token should succeed."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.return_value = mock_session

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            # Mock the async generator for get_db
            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            with client.websocket_connect(
                "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
            ) as ws:
                # Send ping to verify connection works
                ws.send_json({"type": "ping"})
                response = ws.receive_json()
                assert response["type"] == "pong"

    @pytest.mark.asyncio()
    async def test_connection_session_not_found(
        self, mock_agent_service: AsyncMock, mock_user: dict, auth_token: str
    ) -> None:
        """Connection to non-existent session should be rejected."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.side_effect = SessionNotFoundError(
            "Session not found: invalid"
        )

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            # WebSocket close raises generic exception from starlette
            with pytest.raises(Exception):  # noqa: B017, PT011
                with client.websocket_connect(
                    "/ws/agents/invalid", subprotocols=[f"access_token.{auth_token}"]
                ):
                    pass

    @pytest.mark.asyncio()
    async def test_connection_access_denied(
        self, mock_agent_service: AsyncMock, mock_user: dict, auth_token: str
    ) -> None:
        """Connection to another user's session should be rejected."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.side_effect = AccessDeniedError("Access denied")

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            # WebSocket close raises generic exception from starlette
            with pytest.raises(Exception):  # noqa: B017, PT011
                with client.websocket_connect(
                    "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
                ):
                    pass

    @pytest.mark.asyncio()
    async def test_execute_streams_messages(
        self, mock_agent_service: AsyncMock, mock_session: MagicMock, mock_user: dict, auth_token: str
    ) -> None:
        """Execute should stream messages to client."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.return_value = mock_session

        # Create async generator for execute
        async def mock_execute(*_args: Any, **_kwargs: Any):
            yield AgentMessage(
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content="Hello",
            )
            yield AgentMessage(
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content="Hi there!",
                is_partial=True,
            )
            yield AgentMessage(
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content="Hi there! How can I help?",
            )

        mock_agent_service.execute = mock_execute

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            with client.websocket_connect(
                "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
            ) as ws:
                # Send execute message
                ws.send_json({"type": "execute", "prompt": "Hello"})

                # Receive streamed messages
                msg1 = ws.receive_json()
                assert msg1["type"] == "message"
                assert msg1["message"]["content"] == "Hello"
                assert msg1["message"]["role"] == "user"

                msg2 = ws.receive_json()
                assert msg2["type"] == "message"
                assert msg2["message"]["content"] == "Hi there!"
                assert msg2["message"]["is_partial"] is True

                msg3 = ws.receive_json()
                assert msg3["type"] == "message"
                assert msg3["message"]["content"] == "Hi there! How can I help?"

                complete = ws.receive_json()
                assert complete["type"] == "complete"

    @pytest.mark.asyncio()
    async def test_execute_handles_interrupt(
        self, mock_agent_service: AsyncMock, mock_session: MagicMock, mock_user: dict, auth_token: str
    ) -> None:
        """Execute should handle interruption properly."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.return_value = mock_session

        # Create async generator that raises interrupt
        async def mock_execute(*_args: Any, **_kwargs: Any):
            yield AgentMessage(
                role=MessageRole.ASSISTANT,
                type=MessageType.TEXT,
                content="Starting...",
            )
            raise AgentInterruptedError("Interrupted by user")

        mock_agent_service.execute = mock_execute

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            with client.websocket_connect(
                "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
            ) as ws:
                # Send execute message
                ws.send_json({"type": "execute", "prompt": "Hello"})

                # Receive first message
                msg1 = ws.receive_json()
                assert msg1["type"] == "message"

                # Receive interrupted message
                interrupted = ws.receive_json()
                assert interrupted["type"] == "interrupted"

    @pytest.mark.asyncio()
    async def test_execute_handles_error(
        self, mock_agent_service: AsyncMock, mock_session: MagicMock, mock_user: dict, auth_token: str
    ) -> None:
        """Execute should handle errors properly."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.return_value = mock_session

        # Create async generator that raises an error
        async def mock_execute(*_args: Any, **_kwargs: Any):
            raise ValueError("Something went wrong")
            yield  # Make it a generator

        mock_agent_service.execute = mock_execute

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            with client.websocket_connect(
                "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
            ) as ws:
                # Send execute message
                ws.send_json({"type": "execute", "prompt": "Hello"})

                # Receive error message
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "Something went wrong" in error["error"]["message"]

    @pytest.mark.asyncio()
    async def test_unknown_message_type(
        self, mock_agent_service: AsyncMock, mock_session: MagicMock, mock_user: dict, auth_token: str
    ) -> None:
        """Unknown message type should return error."""
        from webui.main import app

        mock_agent_service.verify_websocket_access.return_value = mock_session

        with (
            patch("webui.main.pg_connection_manager") as mock_pg,
            patch("webui.main.ws_manager") as mock_ws,
            patch("webui.ws.agents.get_current_user_websocket", return_value=mock_user),
            patch("webui.ws.agents.create_agent_service", return_value=mock_agent_service),
            patch("webui.ws.agents.get_db") as mock_get_db,
        ):
            mock_pg.initialize = AsyncMock()
            mock_ws.startup = AsyncMock()
            mock_ws.shutdown = AsyncMock()

            async def mock_db_gen():
                mock_db = AsyncMock()
                yield mock_db

            mock_get_db.return_value = mock_db_gen()

            client = TestClient(app)
            with client.websocket_connect(
                "/ws/agents/abc12345", subprotocols=[f"access_token.{auth_token}"]
            ) as ws:
                # Send unknown message type
                ws.send_json({"type": "unknown_type"})

                # Receive error message
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "Unknown message type" in error["error"]["message"]


class TestVerifyWebSocketAccess:
    """Tests for AgentService.verify_websocket_access method."""

    @pytest.mark.asyncio()
    async def test_verify_access_success(self) -> None:
        """Successful access verification."""
        from webui.services.agent_service import AgentService

        mock_db = AsyncMock()
        mock_repo = AsyncMock()

        mock_session = MagicMock()
        mock_session.user_id = 1
        mock_repo.get_by_external_id.return_value = mock_session

        service = AgentService(mock_db, mock_repo)
        result = await service.verify_websocket_access("session-123", 1)

        assert result == mock_session
        mock_repo.get_by_external_id.assert_called_once_with("session-123")

    @pytest.mark.asyncio()
    async def test_verify_access_session_not_found(self) -> None:
        """Session not found raises error."""
        from webui.services.agent_service import AgentService

        mock_db = AsyncMock()
        mock_repo = AsyncMock()
        mock_repo.get_by_external_id.return_value = None

        service = AgentService(mock_db, mock_repo)

        with pytest.raises(SessionNotFoundError):
            await service.verify_websocket_access("nonexistent", 1)

    @pytest.mark.asyncio()
    async def test_verify_access_wrong_user(self) -> None:
        """Access by wrong user raises error."""
        from webui.services.agent_service import AgentService

        mock_db = AsyncMock()
        mock_repo = AsyncMock()

        mock_session = MagicMock()
        mock_session.user_id = 1
        mock_repo.get_by_external_id.return_value = mock_session

        service = AgentService(mock_db, mock_repo)

        with pytest.raises(AccessDeniedError):
            await service.verify_websocket_access("session-123", 2)  # Different user

    @pytest.mark.asyncio()
    async def test_verify_access_no_user_id(self) -> None:
        """Session with no user_id allows any user."""
        from webui.services.agent_service import AgentService

        mock_db = AsyncMock()
        mock_repo = AsyncMock()

        mock_session = MagicMock()
        mock_session.user_id = None  # No user restriction
        mock_repo.get_by_external_id.return_value = mock_session

        service = AgentService(mock_db, mock_repo)
        result = await service.verify_websocket_access("session-123", 999)

        assert result == mock_session
