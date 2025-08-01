"""Health check endpoints for monitoring service status"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from shared.database import check_postgres_connection
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


@router.get("/search-api")
async def search_api_health() -> dict[str, Any]:
    """Check Search API (vecpipe) health and status."""
    import os

    import httpx

    search_api_url = os.getenv("SEARCH_API_URL", "http://vecpipe:8000")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{search_api_url}/health", timeout=HEALTH_CHECK_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                return {"status": "healthy", "message": "Search API is ready", "api_response": data}
            return {
                "status": "unhealthy",
                "message": f"Search API returned status {response.status_code}",
                "details": response.text,
            }
    except httpx.TimeoutException:
        return {"status": "unhealthy", "error": "Search API connection timeout"}
    except Exception as e:
        logger.error(f"Search API health check failed: {e}")
        return {"status": "unhealthy", "error": "Failed to access Search API", "details": str(e)}


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
        # Use the PostgreSQL connection check with timeout
        result = await asyncio.wait_for(check_postgres_connection(), timeout=HEALTH_CHECK_TIMEOUT)
        if result:
            return {"status": "healthy", "message": "Database connection successful"}
        return {"status": "unhealthy", "message": "Database connection failed"}
    except TimeoutError:
        return {"status": "unhealthy", "message": "Database connection timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Database connection failed: {str(e)}"}


async def _check_qdrant_health() -> dict[str, Any]:
    """Check Qdrant connection health with timeout."""
    try:

        def _qdrant_check() -> int:
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


async def _check_search_api_health() -> dict[str, Any]:
    """Check Search API (vecpipe) health with timeout."""
    import os

    import httpx

    search_api_url = os.getenv("SEARCH_API_URL", "http://vecpipe:8000")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{search_api_url}/health", timeout=HEALTH_CHECK_TIMEOUT)
            data = response.json()

            # Check if all components are healthy
            if response.status_code == 200 and data.get("status") == "healthy":
                return {"status": "healthy", "message": "Search API connection successful"}
            if response.status_code == 200 and data.get("status") == "degraded":
                # Extract component statuses for better diagnostics
                components = data.get("components", {})
                unhealthy_components = [
                    f"{name}: {info.get('error', info.get('message', 'unknown'))}"
                    for name, info in components.items()
                    if info.get("status") != "healthy"
                ]
                return {
                    "status": "degraded",
                    "message": "Search API is degraded",
                    "unhealthy_components": unhealthy_components,
                }
            return {"status": "unhealthy", "message": f"Search API returned status {response.status_code}"}
    except httpx.TimeoutException:
        return {"status": "unhealthy", "message": "Search API connection timeout"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Search API connection failed: {str(e)}"}


async def _check_embedding_service_health() -> dict[str, Any]:
    """Check WebUI's embedding service health."""
    try:
        from shared.embedding import get_embedding_service

        # Use the async version
        service = await get_embedding_service()

        if service and hasattr(service, "is_initialized") and service.is_initialized:
            model_info = service.get_model_info() if hasattr(service, "get_model_info") else {}
            return {
                "status": "healthy",
                "message": "Embedding service initialized",
                "model": model_info.get("model_name", "unknown"),
                "mock_mode": getattr(service, "mock_mode", False),
            }
        return {"status": "unhealthy", "message": "Embedding service not initialized"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Embedding service check failed: {str(e)}"}


@router.get("/readyz")
async def readiness_probe() -> JSONResponse:
    """
    Readiness probe endpoint for Kubernetes/container orchestration.

    This endpoint checks connectivity to all external dependencies:
    - Redis (for operation queue and WebSocket management)
    - Database (PostgreSQL)
    - Qdrant (vector database)
    - Search API (for search functionality)
    - Embedding Service (for local embedding operations)

    Returns:
        JSONResponse: Detailed readiness status for each dependency

    Returns HTTP 503 if any dependency is unreachable
    """
    # Run all health checks concurrently for better performance
    redis_task = asyncio.create_task(_check_redis_health())
    database_task = asyncio.create_task(_check_database_health())
    qdrant_task = asyncio.create_task(_check_qdrant_health())
    search_api_task = asyncio.create_task(_check_search_api_health())
    embedding_task = asyncio.create_task(_check_embedding_service_health())

    # Wait for all checks to complete
    results = await asyncio.gather(
        redis_task, database_task, qdrant_task, search_api_task, embedding_task, return_exceptions=True
    )
    redis_result: dict[str, Any] | BaseException = results[0]
    database_result: dict[str, Any] | BaseException = results[1]
    qdrant_result: dict[str, Any] | BaseException = results[2]
    search_api_result: dict[str, Any] | BaseException = results[3]
    embedding_result: dict[str, Any] | BaseException = results[4]

    # Handle any exceptions from gather
    services = {}
    all_healthy = True

    for name, result in [
        ("redis", redis_result),
        ("database", database_result),
        ("qdrant", qdrant_result),
        ("search_api", search_api_result),
        ("embedding", embedding_result),
    ]:
        if isinstance(result, BaseException):
            services[name] = {"status": "unhealthy", "message": f"Health check failed: {str(result)}"}
            all_healthy = False
        else:
            services[name] = result
            # Allow degraded status for search_api
            # Allow embedding service to be unhealthy since WebUI doesn't use it directly
            if name == "embedding":
                # Don't mark the whole service as unhealthy if embedding is not initialized
                pass
            elif result["status"] not in ["healthy", "degraded"]:
                all_healthy = False

    response = {"ready": all_healthy, "check": "readiness", "services": services}

    status_code = 200 if all_healthy else 503
    return JSONResponse(status_code=status_code, content=response)
