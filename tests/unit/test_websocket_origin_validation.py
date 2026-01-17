"""Unit tests for WebSocket origin validation helpers."""

import pytest
from starlette.datastructures import URL

from webui.api.v2.operations import _validate_websocket_origin


class StubWebSocket:
    """Minimal WebSocket stub for origin validation tests."""

    def __init__(self, *, origin: str | None, url: str) -> None:
        self.headers: dict[str, str] = {}
        if origin is not None:
            self.headers["origin"] = origin
        self.url = URL(url)


@pytest.mark.asyncio()
async def test_validate_websocket_origin_allows_ip_same_origin() -> None:
    websocket = StubWebSocket(
        origin="http://192.168.1.122:8080",
        url="ws://192.168.1.122:8080/ws/operations",
    )

    assert await _validate_websocket_origin(websocket) is True


@pytest.mark.asyncio()
async def test_validate_websocket_origin_rejects_cross_origin_hostname() -> None:
    websocket = StubWebSocket(
        origin="http://evil.example:8080",
        url="ws://192.168.1.122:8080/ws/operations",
    )

    assert await _validate_websocket_origin(websocket) is False


@pytest.mark.asyncio()
async def test_validate_websocket_origin_rejects_cross_origin_ip() -> None:
    websocket = StubWebSocket(
        origin="http://192.168.1.123:8080",
        url="ws://192.168.1.122:8080/ws/operations",
    )

    assert await _validate_websocket_origin(websocket) is False
