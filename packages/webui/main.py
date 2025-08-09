"""
Main entry point for Document Embedding Web UI
Creates and configures the FastAPI application
"""

import logging
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import Response

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import settings as shared_settings
from shared.database import pg_connection_manager
from shared.embedding import configure_global_embedding_service

logger = logging.getLogger(__name__)

from .api import auth, health, internal, metrics, models, root, settings  # noqa: E402
from .api.chunking_exception_handlers import register_chunking_exception_handlers  # noqa: E402
from .api.v2 import chunking as v2_chunking  # noqa: E402
from .api.v2 import collections as v2_collections  # noqa: E402
from .api.v2 import directory_scan as v2_directory_scan  # noqa: E402
from .api.v2 import documents as v2_documents  # noqa: E402
from .api.v2 import operations as v2_operations  # noqa: E402
from .api.v2 import partition_monitoring as v2_partition_monitoring  # noqa: E402
from .api.v2 import search as v2_search  # noqa: E402
from .api.v2 import system as v2_system  # noqa: E402
from .api.v2.directory_scan import directory_scan_websocket  # noqa: E402
from .api.v2.operations import operation_websocket  # noqa: E402
from .background_tasks import start_background_tasks, stop_background_tasks  # noqa: E402
from .middleware.correlation import CorrelationMiddleware, configure_logging_with_correlation  # noqa: E402
from .middleware.rate_limit import RateLimitMiddleware  # noqa: E402
from .rate_limiter import limiter, rate_limit_exceeded_handler  # noqa: E402
from .websocket_manager import ws_manager  # noqa: E402


def rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Wrapper to ensure proper type signature for rate limit handler"""
    if isinstance(exc, RateLimitExceeded):
        # Use our custom handler with circuit breaker support
        return rate_limit_exceeded_handler(request, exc)
    # This shouldn't happen, but handle gracefully
    return Response(content="Rate limit error", status_code=429)


def _configure_embedding_service() -> None:
    """Configure the global embedding service at app startup.

    This centralizes the embedding service configuration to avoid redundant calls
    across multiple API modules.
    """
    try:
        logger.info("Configuring global embedding service at app startup")
        configure_global_embedding_service(mock_mode=shared_settings.USE_MOCK_EMBEDDINGS)
        logger.info(f"Embedding service configured with mock_mode={shared_settings.USE_MOCK_EMBEDDINGS}")
    except Exception as e:
        logger.error(f"Failed to configure embedding service: {e}")
        # Re-raise to prevent app startup with misconfigured service
        raise RuntimeError(f"Critical error: Failed to configure embedding service: {e}") from e


def _configure_internal_api_key() -> None:
    """Configure internal API key, generating one if using the default value."""
    if shared_settings.INTERNAL_API_KEY == "change-me-in-production":
        # Generate a secure random key
        generated_key = secrets.token_urlsafe(32)
        shared_settings.INTERNAL_API_KEY = generated_key
        logger.warning(
            f"Generated internal API key for development. "
            f"Set INTERNAL_API_KEY environment variable for production. "
            f"Current key: {generated_key}"
        )
    else:
        logger.info("Using configured internal API key")


def _validate_cors_origins(origins: list[str]) -> list[str]:
    """Validate CORS origins and return only valid URLs.

    Args:
        origins: List of origin URLs to validate

    Returns:
        List of valid origin URLs
    """
    valid_origins = []

    for origin in origins:
        # Check for wildcards or null
        if origin in ["*", "null"]:
            logger.warning(
                f"Wildcard or null origin detected in CORS configuration: '{origin}' - "
                f"this is insecure in production!"
            )
            if shared_settings.ENVIRONMENT == "production":
                logger.error(f"Rejecting insecure origin '{origin}' in production environment")
                continue
            # In development, allow wildcards
            valid_origins.append(origin)
            continue

        # Validate URL format
        try:
            parsed = urlparse(origin)
            if parsed.scheme and parsed.netloc:
                valid_origins.append(origin)
            else:
                logger.warning(f"Invalid CORS origin format: {origin}")
        except Exception as e:
            logger.error(f"Error parsing CORS origin '{origin}': {e}")

    return valid_origins


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting up WebUI application...")

    # Configure logging with correlation support
    configure_logging_with_correlation()
    logger.info("Logging configured with correlation ID support")

    # Initialize PostgreSQL connection
    logger.info("Initializing PostgreSQL connection...")
    await pg_connection_manager.initialize()
    logger.info("PostgreSQL connection initialized")

    # Initialize WebSocket manager
    logger.info("Initializing WebSocket manager...")
    await ws_manager.startup()
    logger.info("WebSocket manager initialization complete")

    # Ensure default data exists
    try:
        from .startup_tasks import ensure_default_data

        await ensure_default_data()
    except Exception as e:
        logger.error(f"Error running startup tasks: {e}")
        # Don't fail startup if default data can't be created

    # Configure global embedding service
    _configure_embedding_service()

    # Configure internal API key
    _configure_internal_api_key()

    # Start background tasks for Redis cleanup
    logger.info("Starting background tasks...")
    try:
        await start_background_tasks()
        logger.info("Background tasks started successfully")
    except Exception as e:
        logger.error(f"Failed to start background tasks: {e}")
        # Don't fail startup if background tasks can't start

    yield

    # Shutdown
    logger.info("Shutting down WebUI application...")

    # Stop background tasks
    logger.info("Stopping background tasks...")
    try:
        await stop_background_tasks()
        logger.info("Background tasks stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping background tasks: {e}")
        # Continue shutdown even if background tasks fail to stop

    # Clean up WebSocket manager
    logger.info("Shutting down WebSocket manager...")
    await ws_manager.shutdown()
    logger.info("WebSocket manager shutdown complete")

    # Close PostgreSQL connection
    logger.info("Closing PostgreSQL connection...")
    await pg_connection_manager.close()
    logger.info("PostgreSQL connection closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Document Embedding Web UI",
        description="Create and search document embeddings",
        version="1.1.0",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    # Parse comma-separated origins from configuration
    raw_origins = [origin.strip() for origin in shared_settings.CORS_ORIGINS.split(",") if origin.strip()]

    # Validate origins
    cors_origins = _validate_cors_origins(raw_origins)

    if not cors_origins:
        logger.warning(
            "No valid CORS origins configured - frontend requests may be blocked. "
            "Set CORS_ORIGINS environment variable with valid origin URLs."
        )

    # Add correlation middleware BEFORE other middleware for proper context propagation
    app.add_middleware(CorrelationMiddleware)

    # Add rate limit middleware to set user in request.state
    app.add_middleware(RateLimitMiddleware)

    # Configure CORS with more restrictive settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],  # Explicit methods
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
            "X-Correlation-ID",
        ],  # Added correlation header
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    # Register chunking exception handlers
    register_chunking_exception_handlers(app)

    # Include routers with their specific prefixes
    app.include_router(auth.router)
    app.include_router(metrics.router)
    app.include_router(settings.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(internal.router)

    # Include v2 API routers
    app.include_router(v2_chunking.router)
    app.include_router(v2_collections.router)
    app.include_router(v2_directory_scan.router)
    app.include_router(v2_documents.router)
    app.include_router(v2_operations.router)
    app.include_router(v2_partition_monitoring.router)
    app.include_router(v2_search.router)
    app.include_router(v2_system.router)

    # Mount static files BEFORE catch-all route
    # Mount static files with proper path resolution
    base_dir = Path(__file__).resolve().parent
    static_dir = base_dir / "static"
    assets_dir = static_dir / "assets"

    # Ensure directories exist (for tests and fresh environments)
    static_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Mount assets directory for React build
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    # Mount static files
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # Include root router AFTER static file mounts to ensure catch-all doesn't intercept static files
    app.include_router(root.router)  # No prefix for static + root

    # Mount WebSocket endpoints at the app level
    @app.websocket("/ws/operations/{operation_id}")
    async def operation_ws(websocket: WebSocket, operation_id: str) -> None:
        await operation_websocket(websocket, operation_id)

    @app.websocket("/ws/directory-scan/{scan_id}")
    async def directory_scan_ws(websocket: WebSocket, scan_id: str) -> None:
        await directory_scan_websocket(websocket, scan_id)

    # Add health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint for Docker health monitoring"""
        return {"status": "healthy", "service": "webui"}

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=shared_settings.WEBUI_PORT)
