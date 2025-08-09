"""Integration module for the scalable WebSocket architecture."""

import logging

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse

from packages.webui.auth import get_current_user
from packages.webui.websocket import (
    ScalableWebSocketManager,
    WebSocketHealthMonitor,
    sse_endpoint,
    sse_manager,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["websocket"])

# Initialize components
ws_manager = ScalableWebSocketManager()
health_monitor = WebSocketHealthMonitor()


async def get_ws_manager() -> ScalableWebSocketManager:
    """Dependency to get WebSocket manager."""
    return ws_manager


async def verify_ws_auth(websocket: WebSocket, token: str | None = Query(None)) -> str | None:
    """
    Verify WebSocket authentication.

    Args:
        websocket: WebSocket connection
        token: JWT token from query parameter

    Returns:
        User ID if authenticated
    """
    if not token:
        await websocket.close(code=1008, reason="Missing authentication")
        return None

    try:
        # Validate JWT token
        # This is a simplified example - use your actual auth logic
        from packages.webui.auth import verify_token

        user = await verify_token(token)
        return user.id
    except Exception as e:
        logger.error(f"WebSocket auth failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return None


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    channel: str = Query(...),
    token: str | None = Query(None),
    manager: ScalableWebSocketManager = Depends(get_ws_manager),
) -> None:
    """
    Main WebSocket endpoint with horizontal scaling support.

    Query Parameters:
        channel: Channel to subscribe to
        token: JWT authentication token
    """
    # Authenticate
    user_id = await verify_ws_auth(websocket, token)
    if not user_id:
        return

    # Connect
    connection_id = await manager.connect(
        websocket=websocket,
        user_id=user_id,
        channel=channel,
        metadata={"user_agent": websocket.headers.get("user-agent")},
    )

    if not connection_id:
        return

    # Register with health monitor
    health_monitor.register_connection(connection_id)

    try:
        # Handle messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_json()

                # Record message for health monitoring
                health_monitor.record_message(connection_id)

                # Handle message
                await manager.handle_message(connection_id, data)

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                health_monitor.record_error(connection_id, str(e))
                break

    finally:
        # Cleanup
        health_monitor.unregister_connection(connection_id)
        await manager.disconnect(connection_id)


@router.websocket("/operation/{operation_id}")
async def operation_websocket(
    websocket: WebSocket,
    operation_id: str,
    token: str | None = Query(None),
    manager: ScalableWebSocketManager = Depends(get_ws_manager),
) -> None:
    """
    WebSocket endpoint for operation-specific updates.

    Path Parameters:
        operation_id: Operation identifier

    Query Parameters:
        token: JWT authentication token
    """
    # Authenticate
    user_id = await verify_ws_auth(websocket, token)
    if not user_id:
        return

    # Connect to operation channel
    channel = f"operation:{operation_id}"
    connection_id = await manager.connect(
        websocket=websocket, user_id=user_id, channel=channel, metadata={"operation_id": operation_id}
    )

    if not connection_id:
        return

    try:
        while True:
            try:
                data = await websocket.receive_json()
                await manager.handle_message(connection_id, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Operation WebSocket error: {e}")
                break

    finally:
        await manager.disconnect(connection_id)


@router.get("/sse/connect")
async def sse_connect(
    request: Request, channel: str = Query(...), current_user=Depends(get_current_user)
) -> StreamingResponse:
    """
    Server-Sent Events endpoint as WebSocket fallback.

    Query Parameters:
        channel: Channel to subscribe to
    """
    return await sse_endpoint.connect(request=request, user_id=current_user.id, channel=channel)


@router.get("/health")
async def websocket_health() -> dict:
    """
    WebSocket service health check endpoint.

    Returns:
        Health status and metrics
    """
    metrics = health_monitor.get_metrics()

    # Determine health status
    if metrics.unhealthy_connections > metrics.healthy_connections:
        status_code = "degraded"
    elif metrics.total_connections == 0:
        status_code = "idle"
    else:
        status_code = "healthy"

    return {
        "status": status_code,
        "metrics": metrics.model_dump(),
        "instance_id": ws_manager.config.instance_id,
        "connections": {
            "total": metrics.total_connections,
            "healthy": metrics.healthy_connections,
            "unhealthy": metrics.unhealthy_connections,
        },
        "performance": {
            "avg_latency_ms": round(metrics.average_latency_ms, 2),
            "p95_latency_ms": round(metrics.p95_latency_ms, 2),
            "p99_latency_ms": round(metrics.p99_latency_ms, 2),
            "messages_per_second": round(metrics.messages_per_second, 2),
        },
    }


@router.get("/stats")
async def websocket_stats(current_user=Depends(get_current_user)) -> dict:  # noqa: ARG001
    """
    Detailed WebSocket statistics (requires authentication).

    Returns:
        Detailed statistics and connection information
    """
    # Get registry stats
    registry_stats = await ws_manager.registry.get_stats() if ws_manager.registry else {}

    # Get router stats
    router_stats = ws_manager.router.get_stats()

    # Get health stats
    health_stats = health_monitor.get_detailed_stats()

    return {
        "instance": {
            "id": ws_manager.config.instance_id,
            "uptime": health_monitor.start_time,
            "config": {
                "max_connections_per_user": ws_manager.config.max_connections_per_user,
                "max_total_connections": ws_manager.config.max_total_connections,
                "heartbeat_interval": ws_manager.config.heartbeat_interval,
                "message_rate_limit": ws_manager.config.message_rate_limit,
            },
        },
        "registry": registry_stats,
        "router": router_stats,
        "health": health_stats,
        "sse": {
            "active_connections": len(sse_manager.connections),
            "message_history_size": sum(len(messages) for messages in sse_manager.message_history.values()),
        },
    }


@router.post("/broadcast")
async def broadcast_message(
    message: dict,
    current_user=Depends(get_current_user),  # noqa: ARG001
    manager: ScalableWebSocketManager = Depends(get_ws_manager),
) -> dict:
    """
    Broadcast a message to all connections (admin only).

    Body:
        message: Message to broadcast

    Returns:
        Number of connections reached
    """
    # Check admin permission (implement your own logic)
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")

    count = await manager.broadcast(message)

    return {"success": True, "connections_reached": count}


@router.post("/send")
async def send_to_channel(
    channel: str,
    message: dict,
    current_user=Depends(get_current_user),  # noqa: ARG001
    manager: ScalableWebSocketManager = Depends(get_ws_manager),
) -> dict:
    """
    Send a message to a specific channel.

    Body:
        channel: Target channel
        message: Message to send

    Returns:
        Number of connections reached
    """
    count = await manager.send_message(channel=channel, message=message)

    # Also send via SSE
    sse_count = await sse_manager.send_to_channel(channel=channel, event="message", data=message)

    return {
        "success": True,
        "websocket_connections": count,
        "sse_connections": sse_count,
        "total_connections": count + sse_count,
    }


async def startup_websocket() -> None:
    """Initialize WebSocket components on application startup."""
    logger.info("Initializing WebSocket components...")

    # Start WebSocket manager
    await ws_manager.startup()

    # Start health monitor
    await health_monitor.start()

    logger.info("WebSocket components initialized successfully")


async def shutdown_websocket() -> None:
    """Cleanup WebSocket components on application shutdown."""
    logger.info("Shutting down WebSocket components...")

    # Stop health monitor
    await health_monitor.stop()

    # Shutdown WebSocket manager
    await ws_manager.shutdown()

    logger.info("WebSocket components shut down successfully")
