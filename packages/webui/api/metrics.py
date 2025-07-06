"""
Metrics and monitoring routes for the Web UI
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends

from webui.auth import User, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["metrics"])

# Start metrics server if metrics port is configured
METRICS_PORT = int(os.getenv("WEBUI_METRICS_PORT", "9092"))
METRICS_AVAILABLE = False
generate_latest: Any | None = None
registry: Any | None = None
metrics_updater_thread = None


def update_metrics_loop() -> None:
    """Background thread to continuously update metrics"""
    import time

    from vecpipe.metrics import metrics_collector

    while True:
        try:
            metrics_collector.update_resource_metrics(force=True)
        except Exception as e:
            logger.warning(f"Error in metrics update loop: {e}")
        time.sleep(1)


if METRICS_PORT:
    try:
        from prometheus_client import generate_latest as _generate_latest
        from vecpipe.metrics import registry as _registry
        from vecpipe.metrics import start_metrics_server

        generate_latest = _generate_latest
        registry = _registry
        start_metrics_server(METRICS_PORT)
        logger.info(f"Metrics server started on port {METRICS_PORT}")
        METRICS_AVAILABLE = True

        # Start background metrics updater thread
        import threading

        metrics_updater_thread = threading.Thread(target=update_metrics_loop, daemon=True)
        metrics_updater_thread.start()
        logger.info("Started background metrics updater thread")
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")


@router.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)) -> dict[str, Any]:  # noqa: ARG001
    """Get current Prometheus metrics"""
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
            logger.error(f"Failed to fetch metrics from port {METRICS_PORT}: {e}")
            return {"error": f"Metrics not available: {str(e)}", "metrics_port": METRICS_PORT}

    try:
        # Generate metrics in Prometheus format
        if generate_latest is not None and registry is not None:
            metrics_data = generate_latest(registry)
            return {"available": True, "metrics_port": METRICS_PORT, "data": metrics_data.decode("utf-8")}
        else:
            return {"error": "Metrics not initialized", "metrics_port": METRICS_PORT}
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return {"error": str(e), "metrics_port": METRICS_PORT}
