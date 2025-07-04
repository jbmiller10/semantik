"""
Metrics and monitoring routes for the Web UI
"""

import logging
import os

from fastapi import APIRouter, Depends

from ..auth import User, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["metrics"])

# Start metrics server if metrics port is configured
METRICS_PORT = int(os.getenv("WEBUI_METRICS_PORT", "9092"))
METRICS_AVAILABLE = False
generate_latest = None
registry = None
metrics_updater_thread = None


def update_metrics_loop():
    """Background thread to continuously update metrics"""
    import time

    from packages.vecpipe.metrics import metrics_collector

    while True:
        try:
            metrics_collector.update_resource_metrics(force=True)
        except Exception as e:
            logger.warning(f"Error in metrics update loop: {e}")
        time.sleep(1)


if METRICS_PORT:
    try:
        from packages.vecpipe.metrics import generate_latest, registry, start_metrics_server

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
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get current Prometheus metrics"""
    # If local metrics server isn't available, try to fetch from the search API metrics server
    if not METRICS_AVAILABLE:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"http://localhost:{METRICS_PORT}/metrics")
                if response.status_code == 200:
                    return {"available": True, "metrics_port": METRICS_PORT, "data": response.text}
                else:
                    return {"error": "Metrics server not responding", "metrics_port": METRICS_PORT}
        except Exception as e:
            logger.error(f"Failed to fetch metrics from port {METRICS_PORT}: {e}")
            return {"error": f"Metrics not available: {str(e)}", "metrics_port": METRICS_PORT}

    try:
        # Generate metrics in Prometheus format
        metrics_data = generate_latest(registry)
        return {"available": True, "metrics_port": METRICS_PORT, "data": metrics_data.decode("utf-8")}
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return {"error": str(e), "metrics_port": METRICS_PORT}
