"""
Metrics and monitoring routes for the Web UI
"""

import os
import logging

from fastapi import APIRouter, Depends

from webui.auth import get_current_user, User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["metrics"])

# Start metrics server if metrics port is configured
METRICS_PORT = int(os.getenv("WEBUI_METRICS_PORT", "9092"))
METRICS_AVAILABLE = False
generate_latest = None
registry = None

if METRICS_PORT:
    try:
        from vecpipe.metrics import start_metrics_server, generate_latest, registry

        start_metrics_server(METRICS_PORT)
        logger.info(f"Metrics server started on port {METRICS_PORT}")
        METRICS_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Failed to start metrics server: {e}")


@router.get("/metrics")
async def get_metrics(current_user: User = Depends(get_current_user)):
    """Get current Prometheus metrics"""
    if not METRICS_AVAILABLE:
        return {"error": "Metrics not available", "metrics_port": METRICS_PORT}

    try:
        # Generate metrics in Prometheus format
        metrics_data = generate_latest(registry)
        return {"available": True, "metrics_port": METRICS_PORT, "data": metrics_data.decode("utf-8")}
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return {"error": str(e), "metrics_port": METRICS_PORT}
