"""Health check endpoints for monitoring service status"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from shared.database.connection_pool import get_db_connection
from shared.embedding import get_embedding_service
from webui.utils.qdrant_manager import qdrant_manager
from webui.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])

# Health check timeout configuration
HEALTH_CHECK_TIMEOUT = 5.0  # seconds


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


async def _check_redis_health() -> dict[str, Any]:
    """Check Redis connection health with timeout."""
    try:
        if not ws_manager.redis:
            return {"status": "unhealthy", "message": "Redis connection not initialized"}

        # Test Redis connection with timeout
        await asyncio.wait_for(ws_manager.redis.ping(), timeout=HEALTH_CHECK_TIMEOUT)
        return {"status": "healthy", "message": "Redis connection successful"}
    except TimeoutError:
        return {"status": "unhealthy", "message": "Redis connection timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Redis connection failed: {str(e)}"}


async def _check_database_health() -> dict[str, Any]:
    """Check database connection health with timeout."""
    try:
        # Use asyncio to add timeout to database operation
        def _db_check():
            with get_db_connection() as conn:
                conn.execute("SELECT 1")
                return True

        await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, _db_check), timeout=HEALTH_CHECK_TIMEOUT)
        return {"status": "healthy", "message": "Database connection successful"}
    except TimeoutError:
        return {"status": "unhealthy", "message": "Database connection timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}


async def _check_qdrant_health() -> dict[str, Any]:
    """Check Qdrant connection health with timeout."""
    try:

        def _qdrant_check():
            client = qdrant_manager.get_client()
            collections = client.get_collections()
            return len(collections.collections)

        collections_count = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _qdrant_check), timeout=HEALTH_CHECK_TIMEOUT
        )
        return {"status": "healthy", "message": "Qdrant connection successful", "collections_count": collections_count}
    except TimeoutError:
        return {"status": "unhealthy", "message": "Qdrant connection timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Qdrant connection failed: {str(e)}"}


async def _check_embedding_health() -> dict[str, Any]:
    """Check embedding service health with timeout."""
    try:
        service = await asyncio.wait_for(get_embedding_service(), timeout=HEALTH_CHECK_TIMEOUT)

        if not service.is_initialized:
            return {"status": "unhealthy", "message": "Embedding service not initialized"}

        # Quick embedding test with timeout
        await asyncio.wait_for(service.embed_single("health check"), timeout=HEALTH_CHECK_TIMEOUT)
        return {"status": "healthy", "message": "Embedding service ready"}
    except TimeoutError:
        return {"status": "unhealthy", "message": "Embedding service timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Embedding service failed: {str(e)}"}


@router.get("/readyz")
async def readiness_probe() -> JSONResponse:
    """
    Readiness probe endpoint for Kubernetes/container orchestration.

    This endpoint checks connectivity to all external dependencies:
    - Redis (for job queue and WebSocket management)
    - Database (SQLite connection pool)
    - Qdrant (vector database)
    - Embedding service (for AI functionality)

    Returns:
        JSONResponse: Detailed readiness status for each dependency

    Returns HTTP 503 if any dependency is unreachable
    """
    # Run all health checks concurrently for better performance
    redis_task = asyncio.create_task(_check_redis_health())
    database_task = asyncio.create_task(_check_database_health())
    qdrant_task = asyncio.create_task(_check_qdrant_health())
    embedding_task = asyncio.create_task(_check_embedding_health())

    # Wait for all checks to complete
    redis_result, database_result, qdrant_result, embedding_result = await asyncio.gather(
        redis_task, database_task, qdrant_task, embedding_task, return_exceptions=True
    )

    # Handle any exceptions from gather
    services = {}
    all_healthy = True

    for name, result in [
        ("redis", redis_result),
        ("database", database_result),
        ("qdrant", qdrant_result),
        ("embedding", embedding_result),
    ]:
        if isinstance(result, Exception):
            services[name] = {"status": "unhealthy", "message": f"Health check failed: {str(result)}"}
            all_healthy = False
        else:
            services[name] = result
            if result["status"] != "healthy":
                all_healthy = False

    response = {"ready": all_healthy, "check": "readiness", "services": services}

    status_code = 200 if all_healthy else 503
    return JSONResponse(status_code=status_code, content=response)
