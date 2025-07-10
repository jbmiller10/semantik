"""Health check endpoints for monitoring service status"""

import logging
from typing import Any

from fastapi import APIRouter
from shared.embedding import get_embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("/")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/embedding")
async def embedding_health() -> dict[str, Any]:
    """Check embedding service health and status."""
    try:
        service = await get_embedding_service()

        if service.is_initialized:
            try:
                model_info = service.get_model_info()
                return {"status": "healthy", "initialized": True, "model": model_info}
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
                return {
                    "status": "degraded",
                    "initialized": True,
                    "error": "Failed to retrieve model information",
                    "details": str(e),
                }
        else:
            return {"status": "unhealthy", "initialized": False, "message": "Embedding service not initialized"}

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": "Failed to access embedding service", "details": str(e)}


@router.get("/ready")
async def readiness_check() -> dict[str, Any]:
    """Readiness probe for Kubernetes/container orchestration."""
    try:
        service = await get_embedding_service()

        if service.is_initialized:
            # Try a simple embedding to ensure service is truly ready
            try:
                _ = await service.embed_single("health check")
                return {"ready": True, "status": "Service is ready to handle requests"}
            except Exception as e:
                logger.error(f"Readiness probe failed: {e}")
                return {"ready": False, "status": "Service initialized but not ready", "error": str(e)}
        else:
            return {"ready": False, "status": "Service not initialized"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"ready": False, "status": "Service unavailable", "error": str(e)}
