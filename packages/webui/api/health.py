"""Health check endpoints for monitoring service status"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from shared.embedding import get_embedding_service
from shared.database.connection_pool import get_db_connection
from webui.utils.qdrant_manager import qdrant_manager
from webui.websocket_manager import ws_manager

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


@router.get("/healthz")
async def liveness_probe() -> dict[str, str]:
    """
    Liveness probe endpoint for Kubernetes/container orchestration.
    
    This endpoint only checks if the FastAPI process is running.
    It does not check external dependencies like Redis, database, or Qdrant.
    
    Returns:
        dict: Simple health status indicating the service is alive
    """
    return {"status": "healthy", "check": "liveness"}


@router.get("/readyz")
async def readiness_probe() -> dict[str, Any]:
    """
    Readiness probe endpoint for Kubernetes/container orchestration.
    
    This endpoint checks connectivity to all external dependencies:
    - Redis (for job queue and WebSocket management)
    - Database (SQLite connection pool)
    - Qdrant (vector database)
    
    Returns:
        dict: Detailed readiness status for each dependency
        
    Raises:
        HTTPException: 503 Service Unavailable if any dependency is unreachable
    """
    services = {}
    all_healthy = True
    
    # Check Redis connection
    try:
        if ws_manager.redis:
            await ws_manager.redis.ping()
            services["redis"] = {"status": "healthy", "message": "Redis connection successful"}
        else:
            services["redis"] = {"status": "unhealthy", "message": "Redis connection not initialized"}
            all_healthy = False
    except Exception as e:
        services["redis"] = {"status": "unhealthy", "message": f"Redis connection failed: {str(e)}"}
        all_healthy = False
    
    # Check database connection
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
            services["database"] = {"status": "healthy", "message": "Database connection successful"}
    except Exception as e:
        services["database"] = {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}
        all_healthy = False
    
    # Check Qdrant connection
    try:
        client = qdrant_manager.get_client()
        # Simple verification that connection works
        collections = client.get_collections()
        services["qdrant"] = {
            "status": "healthy", 
            "message": "Qdrant connection successful",
            "collections_count": len(collections.collections)
        }
    except Exception as e:
        services["qdrant"] = {"status": "unhealthy", "message": f"Qdrant connection failed: {str(e)}"}
        all_healthy = False
    
    response = {
        "ready": all_healthy,
        "check": "readiness",
        "services": services
    }
    
    if not all_healthy:
        raise HTTPException(status_code=503, detail=response)
    
    return response
