"""Integration tests for WebSocket subprotocol authentication.

This file tests the WebSocket authentication via Sec-WebSocket-Protocol header
as implemented in webui/api/v2/operations.py (lines 270-305).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from starlette.websockets import WebSocket

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class TestWebSocketSubprotocolAuth:
    """Tests for WebSocket subprotocol-based authentication."""

    @pytest.fixture()
    def mock_websocket_with_subprotocol(self) -> AsyncMock:
        """Create mock WebSocket with subprotocol header."""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {}
        mock.headers = {"sec-websocket-protocol": "access_token.valid-jwt-token"}
        mock.received_messages = []

        async def track_send_json(data: dict[str, Any]) -> None:
            mock.received_messages.append(data)

        mock.send_json.side_effect = track_send_json
        return mock

    @pytest.fixture()
    def mock_websocket_with_query_param_auth(self) -> AsyncMock:
        """Create mock WebSocket using deprecated query param auth."""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {"token": "valid-jwt-token"}
        mock.headers = {}  # No subprotocol header
        mock.received_messages = []
        return mock

    @pytest.fixture()
    def mock_user(self) -> dict[str, Any]:
        """Create mock user dict."""
        return {"id": 1, "username": "testuser"}

    def _create_db_mock(self) -> AsyncMock:
        """Create a mock async generator for get_db."""

        async def mock_db_generator() -> AsyncGenerator[Any, None]:
            mock_session = AsyncMock()
            yield mock_session

        return mock_db_generator

    # Test 1: Origin validation rejection (lines 270-273)
    @pytest.mark.asyncio()
    async def test_websocket_rejects_invalid_origin(self) -> None:
        """WebSocket should reject connections from non-allowed origins."""
        from webui.api.v2.operations import operation_websocket

        mock = AsyncMock(spec=WebSocket)
        mock.close = AsyncMock()
        mock.headers = {"origin": "https://malicious-site.com"}
        mock.query_params = {}

        with patch(
            "webui.api.v2.operations._validate_websocket_origin",
            return_value=False,
        ):
            await operation_websocket(mock, "test-op-id")

        mock.close.assert_called_once_with(code=4003, reason="Origin not allowed")

    # Test 2: Subprotocol token extraction success (lines 275-286)
    @pytest.mark.asyncio()
    async def test_websocket_extracts_token_from_subprotocol(
        self, mock_websocket_with_subprotocol: AsyncMock, mock_user: dict[str, Any]
    ) -> None:
        """WebSocket should extract JWT from Sec-WebSocket-Protocol header."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch("webui.api.v2.operations.get_current_user_websocket") as mock_auth,
            patch("webui.api.v2.operations.OperationService") as mock_service_class,
            patch("webui.api.v2.operations.get_db") as mock_get_db,
            patch("webui.api.v2.operations.ws_manager") as mock_ws_manager,
            patch("webui.api.v2.operations.OperationRepository"),
        ):
            mock_auth.return_value = mock_user
            mock_service_instance = AsyncMock()
            mock_service_instance.verify_websocket_access = AsyncMock(return_value=None)
            mock_service_class.return_value = mock_service_instance

            mock_get_db.return_value = self._create_db_mock()()
            mock_ws_manager.connect = AsyncMock(return_value="conn-id")
            mock_ws_manager.disconnect = AsyncMock()

            mock_websocket_with_subprotocol.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

            await operation_websocket(mock_websocket_with_subprotocol, "test-op-id")

            # Verify token was extracted from subprotocol
            mock_auth.assert_called_once_with("valid-jwt-token")

            # Verify subprotocol was echoed back in connect call
            mock_ws_manager.connect.assert_called_once_with(
                mock_websocket_with_subprotocol,
                "1",
                "test-op-id",
                subprotocol="access_token.valid-jwt-token",
            )

    # Test 3: Fallback to query params (lines 288-292)
    @pytest.mark.asyncio()
    async def test_websocket_fallback_to_query_param_auth(
        self, mock_websocket_with_query_param_auth: AsyncMock, mock_user: dict[str, Any]
    ) -> None:
        """WebSocket should fallback to query param auth when no subprotocol."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch("webui.api.v2.operations.get_current_user_websocket") as mock_auth,
            patch("webui.api.v2.operations.OperationService") as mock_service_class,
            patch("webui.api.v2.operations.get_db") as mock_get_db,
            patch("webui.api.v2.operations.ws_manager") as mock_ws_manager,
            patch("webui.api.v2.operations.OperationRepository"),
            patch("webui.api.v2.operations.logger") as mock_logger,
        ):
            mock_auth.return_value = mock_user
            mock_service_instance = AsyncMock()
            mock_service_instance.verify_websocket_access = AsyncMock(return_value=None)
            mock_service_class.return_value = mock_service_instance

            mock_get_db.return_value = self._create_db_mock()()
            mock_ws_manager.connect = AsyncMock(return_value="conn-id")
            mock_ws_manager.disconnect = AsyncMock()

            mock_websocket_with_query_param_auth.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

            await operation_websocket(mock_websocket_with_query_param_auth, "test-op-id")

            # Verify query param token was used
            mock_auth.assert_called_once_with("valid-jwt-token")

            # Verify deprecation warning was logged
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "deprecated" in warning_call.lower()

            # Verify subprotocol is None when using query param
            mock_ws_manager.connect.assert_called_once_with(
                mock_websocket_with_query_param_auth,
                "1",
                "test-op-id",
                subprotocol=None,
            )

    # Test 4: ValueError during authentication (lines 294-301)
    @pytest.mark.asyncio()
    async def test_websocket_closes_on_auth_value_error(self, mock_websocket_with_subprotocol: AsyncMock) -> None:
        """WebSocket should close with 1008 on authentication ValueError."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch(
                "webui.api.v2.operations.get_current_user_websocket",
                side_effect=ValueError("Invalid authentication token"),
            ),
        ):
            await operation_websocket(mock_websocket_with_subprotocol, "test-op-id")

            mock_websocket_with_subprotocol.close.assert_called_once_with(
                code=1008, reason="Invalid authentication token"
            )

    # Test 5: Generic exception during authentication (lines 302-305)
    @pytest.mark.asyncio()
    async def test_websocket_closes_on_auth_generic_exception(self, mock_websocket_with_subprotocol: AsyncMock) -> None:
        """WebSocket should close with 1011 on unexpected auth error."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch(
                "webui.api.v2.operations.get_current_user_websocket",
                side_effect=RuntimeError("Database connection failed"),
            ),
            patch("webui.api.v2.operations.logger") as mock_logger,
        ):
            await operation_websocket(mock_websocket_with_subprotocol, "test-op-id")

            mock_websocket_with_subprotocol.close.assert_called_once_with(code=1011, reason="Internal server error")
            mock_logger.error.assert_called()

    # Test 6: Multiple subprotocols in header
    @pytest.mark.asyncio()
    async def test_websocket_handles_multiple_subprotocols(self, mock_user: dict[str, Any]) -> None:
        """WebSocket should find access_token protocol among multiple protocols."""
        from webui.api.v2.operations import operation_websocket

        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {}
        mock.headers = {"sec-websocket-protocol": "graphql-ws, access_token.jwt-token-here, json"}

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch("webui.api.v2.operations.get_current_user_websocket") as mock_auth,
            patch("webui.api.v2.operations.OperationService") as mock_service_class,
            patch("webui.api.v2.operations.get_db") as mock_get_db,
            patch("webui.api.v2.operations.ws_manager") as mock_ws_manager,
            patch("webui.api.v2.operations.OperationRepository"),
        ):
            mock_auth.return_value = mock_user
            mock_service_instance = AsyncMock()
            mock_service_instance.verify_websocket_access = AsyncMock(return_value=None)
            mock_service_class.return_value = mock_service_instance

            mock_get_db.return_value = self._create_db_mock()()
            mock_ws_manager.connect = AsyncMock(return_value="conn-id")
            mock_ws_manager.disconnect = AsyncMock()

            mock.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

            await operation_websocket(mock, "test-op-id")

            # Should extract the correct token
            mock_auth.assert_called_once_with("jwt-token-here")

    # Test 7: Operation not found
    @pytest.mark.asyncio()
    async def test_websocket_closes_on_operation_not_found(
        self, mock_websocket_with_subprotocol: AsyncMock, mock_user: dict[str, Any]
    ) -> None:
        """WebSocket should close with 1008 when operation not found."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch(
                "webui.api.v2.operations.get_current_user_websocket",
                return_value=mock_user,
            ),
            patch("webui.api.v2.operations.OperationService") as mock_service_class,
            patch("webui.api.v2.operations.get_db") as mock_get_db,
            patch("webui.api.v2.operations.OperationRepository"),
        ):
            mock_service_instance = AsyncMock()
            mock_service_instance.verify_websocket_access = AsyncMock(
                side_effect=EntityNotFoundError("Operation", "test-op-id")
            )
            mock_service_class.return_value = mock_service_instance

            mock_get_db.return_value = self._create_db_mock()()

            await operation_websocket(mock_websocket_with_subprotocol, "test-op-id")

            mock_websocket_with_subprotocol.close.assert_called_once()
            close_call = mock_websocket_with_subprotocol.close.call_args
            assert close_call[1]["code"] == 1008
            assert "not found" in close_call[1]["reason"]

    # Test 8: Access denied
    @pytest.mark.asyncio()
    async def test_websocket_closes_on_access_denied(
        self, mock_websocket_with_subprotocol: AsyncMock, mock_user: dict[str, Any]
    ) -> None:
        """WebSocket should close with 1008 when access denied."""
        from webui.api.v2.operations import operation_websocket

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch(
                "webui.api.v2.operations.get_current_user_websocket",
                return_value=mock_user,
            ),
            patch("webui.api.v2.operations.OperationService") as mock_service_class,
            patch("webui.api.v2.operations.get_db") as mock_get_db,
            patch("webui.api.v2.operations.OperationRepository"),
        ):
            mock_service_instance = AsyncMock()
            mock_service_instance.verify_websocket_access = AsyncMock(
                side_effect=AccessDeniedError("user-1", "Operation", "test-op-id")
            )
            mock_service_class.return_value = mock_service_instance

            mock_get_db.return_value = self._create_db_mock()()

            await operation_websocket(mock_websocket_with_subprotocol, "test-op-id")

            mock_websocket_with_subprotocol.close.assert_called_once()
            close_call = mock_websocket_with_subprotocol.close.call_args
            assert close_call[1]["code"] == 1008
            assert "access" in close_call[1]["reason"].lower()


class TestGlobalWebSocketSubprotocolAuth:
    """Tests for global WebSocket endpoint subprotocol authentication."""

    @pytest.fixture()
    def mock_websocket_global(self) -> AsyncMock:
        """Create mock WebSocket for global endpoint."""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {}
        mock.headers = {"sec-websocket-protocol": "access_token.global-jwt-token"}
        return mock

    @pytest.mark.asyncio()
    async def test_global_websocket_subprotocol_auth(self, mock_websocket_global: AsyncMock) -> None:
        """Global WebSocket should support subprotocol authentication."""
        from webui.api.v2.operations import operation_websocket_global

        mock_user = {"id": 10, "username": "globaluser"}

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch("webui.api.v2.operations.get_current_user_websocket") as mock_auth,
            patch("webui.api.v2.operations.ws_manager") as mock_ws_manager,
        ):
            mock_auth.return_value = mock_user
            mock_ws_manager.connect = AsyncMock(return_value="global-conn-id")
            mock_ws_manager.disconnect = AsyncMock()

            mock_websocket_global.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

            await operation_websocket_global(mock_websocket_global)

            mock_auth.assert_called_once_with("global-jwt-token")
            mock_ws_manager.connect.assert_called_once_with(
                mock_websocket_global,
                "10",
                subprotocol="access_token.global-jwt-token",
            )

    @pytest.mark.asyncio()
    async def test_global_websocket_rejects_invalid_origin(self) -> None:
        """Global WebSocket should reject invalid origins."""
        from webui.api.v2.operations import operation_websocket_global

        mock = AsyncMock(spec=WebSocket)
        mock.close = AsyncMock()
        mock.headers = {"origin": "https://evil.com"}
        mock.query_params = {}

        with patch(
            "webui.api.v2.operations._validate_websocket_origin",
            return_value=False,
        ):
            await operation_websocket_global(mock)

        mock.close.assert_called_once_with(code=4003, reason="Origin not allowed")

    @pytest.mark.asyncio()
    async def test_global_websocket_fallback_to_query_param(self) -> None:
        """Global WebSocket should fallback to query param auth."""
        from webui.api.v2.operations import operation_websocket_global

        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {"token": "query-token"}
        mock.headers = {}  # No subprotocol

        mock_user = {"id": 20, "username": "queryuser"}

        with (
            patch(
                "webui.api.v2.operations._validate_websocket_origin",
                return_value=True,
            ),
            patch("webui.api.v2.operations.get_current_user_websocket") as mock_auth,
            patch("webui.api.v2.operations.ws_manager") as mock_ws_manager,
            patch("webui.api.v2.operations.logger"),
        ):
            mock_auth.return_value = mock_user
            mock_ws_manager.connect = AsyncMock(return_value="conn-id")
            mock_ws_manager.disconnect = AsyncMock()

            mock.receive = AsyncMock(side_effect=[{"type": "websocket.disconnect"}])

            await operation_websocket_global(mock)

            mock_auth.assert_called_once_with("query-token")
