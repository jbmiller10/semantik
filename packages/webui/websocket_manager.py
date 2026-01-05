"""Legacy Redis Streams WebSocket manager (deprecated).

This module is kept for backward compatibility. The app uses the scalable
Pub/Sub manager in ``webui.websocket.scalable_manager``. Tests that cover the
legacy manager should import ``webui.websocket.legacy_stream_manager``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "webui.websocket_manager is deprecated; use webui.websocket.legacy_stream_manager",
    DeprecationWarning,
    stacklevel=2,
)

from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager, ws_manager  # noqa: F401,E402

__all__ = ["RedisStreamWebSocketManager", "ws_manager"]
