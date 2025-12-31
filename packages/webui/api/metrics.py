"""
Metrics and monitoring routes for the Web UI
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends

from shared.config import settings as webui_settings
from webui.auth import User, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["metrics"])

# Start metrics server if metrics port is configured
METRICS_PORT = webui_settings.WEBUI_METRICS_PORT
METRICS_AVAILABLE = False
generate_latest: Any | None = None
registry: Any | None = None
metrics_updater_thread = None


def update_metrics_loop() -> None:
    """Background thread to continuously update metrics"""
    import time

    from shared.metrics.prometheus import metrics_collector

    while True:
        try:
            metrics_collector.update_resource_metrics(force=True)
        except Exception as e:
            logger.warning("Error in metrics update loop: %s", e, exc_info=True)
        time.sleep(1)


if METRICS_PORT:
    try:
        from prometheus_client import generate_latest as _generate_latest

        from shared.metrics.prometheus import registry as _registry, start_metrics_server

        generate_latest = _generate_latest
        registry = _registry
        start_metrics_server(METRICS_PORT)
        logger.info("Metrics server started on port %s", METRICS_PORT)
        METRICS_AVAILABLE = True

        # Start background metrics updater thread
        import threading

        metrics_updater_thread = threading.Thread(target=update_metrics_loop, daemon=True)
        metrics_updater_thread.start()
        logger.info("Started background metrics updater thread")
    except Exception as e:
        logger.warning("Failed to start metrics server: %s", e, exc_info=True)


@router.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)) -> dict[str, Any]:  # noqa: ARG001
    """Get current Prometheus metrics"""
    if not METRICS_PORT:
        return {"available": False, "error": "Metrics port not configured"}

    # If local metrics server isn't available, try to fetch from the search API metrics server
    if not METRICS_AVAILABLE:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{METRICS_PORT}/metrics")
                if response.status_code == 200:
                    return {"available": True, "metrics_port": METRICS_PORT, "data": response.text}
                return {"error": "Metrics server not responding", "metrics_port": METRICS_PORT}
        except Exception as e:
            logger.error("Failed to fetch metrics from port %s: %s", METRICS_PORT, e, exc_info=True)
            return {"error": f"Metrics not available: {str(e)}", "metrics_port": METRICS_PORT}

    try:
        # Generate metrics in Prometheus format
        if generate_latest is not None and registry is not None:
            metrics_data = generate_latest(registry)
            return {"available": True, "metrics_port": METRICS_PORT, "data": metrics_data.decode("utf-8")}
        return {"error": "Metrics not initialized", "metrics_port": METRICS_PORT}
    except Exception as e:
        logger.error("Failed to generate metrics: %s", e, exc_info=True)
        return {"error": str(e), "metrics_port": METRICS_PORT}
