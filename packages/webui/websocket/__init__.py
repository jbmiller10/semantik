"""WebSocket scaling implementation with Redis Pub/Sub for horizontal scaling."""

from .scalable_manager import ScalableWebSocketManager

__all__ = ["ScalableWebSocketManager"]