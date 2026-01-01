"""Main entry point for Document Embedding Web UI."""

from __future__ import annotations

# ruff: noqa: E402
import logging
import sys
from collections.abc import AsyncIterator  # noqa: TCH003
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure repo-local packages are importable when running this file directly.
WEBUI_DIR = Path(__file__).resolve().parent
PACKAGES_DIR = WEBUI_DIR.parent
REPO_ROOT = PACKAGES_DIR.parent
for path in (PACKAGES_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# When executed directly (not via -m), set __package__ so relative imports work.
# Preserve interpreter-provided values like "packages.webui" when running
# with ``python -m packages.webui.main`` to avoid breaking absolute imports.
if __name__ == "__main__" and not __package__:
    __package__ = "webui"

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request  # noqa: TCH002
from starlette.responses import Response  # noqa: TCH002

from shared.config import settings as shared_settings
from shared.config.internal_api_key import ensure_internal_api_key
from shared.config.runtime import ensure_webui_directories, require_auth_enabled, require_jwt_secret
from shared.database import pg_connection_manager
from shared.embedding import configure_global_embedding_service

from .api import auth, health, internal, metrics, models, root, settings
from .api.chunking_exception_handlers import register_chunking_exception_handlers
from .api.v2 import (
    chunking as v2_chunking,
    collections as v2_collections,
    connectors as v2_connectors,
    directory_scan as v2_directory_scan,
    documents as v2_documents,
    embedding as v2_embedding,
    operations as v2_operations,
    partition_monitoring as v2_partition_monitoring,
    plugins as v2_plugins,
    projections as v2_projections,
    search as v2_search,
    sources as v2_sources,
    system as v2_system,
)
from .api.v2.directory_scan import directory_scan_websocket
from .api.v2.operations import operation_websocket, operation_websocket_global
from .background_tasks import start_background_tasks, stop_background_tasks
from .middleware.correlation import CorrelationMiddleware, configure_logging_with_correlation
from .middleware.csp import CSPMiddleware
from .middleware.exception_handlers import register_global_exception_handlers
from .middleware.rate_limit import RateLimitMiddleware
from .rate_limiter import limiter, rate_limit_exceeded_handler
from .websocket.scalable_manager import scalable_ws_manager as ws_manager

logger = logging.getLogger(__name__)


def rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Wrapper to ensure proper type signature for rate limit handler"""
    if isinstance(exc, RateLimitExceeded):
        # Use our custom handler with circuit breaker support
        return rate_limit_exceeded_handler(request, exc)
    # This shouldn't happen, but handle gracefully
    from fastapi.responses import JSONResponse

    return JSONResponse(content={"detail": "Rate limit error"}, status_code=429)


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
    """Ensure the internal API key is configured and persisted."""
    import hashlib

    try:
        key = ensure_internal_api_key(shared_settings)
        fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        logger.info("Internal API key configured (fingerprint=%s)", fingerprint)
    except RuntimeError as exc:
        logger.error("Internal API key configuration failed: %s", exc)
        raise


def _configure_connector_secrets_encryption() -> None:
    """Initialize the connector secrets encryption if configured.

    This must be called at app startup before any services that use
    encrypted secrets are created.
    """
    from shared.utils.encryption import SecretEncryption

    key = shared_settings.CONNECTOR_SECRETS_KEY
    if key:
        try:
            enabled = SecretEncryption.initialize(key)
            if enabled:
                logger.info(
                    "Connector secrets encryption enabled (key_id=%s)",
                    SecretEncryption.get_key_id(),
                )
            else:
                logger.info("Connector secrets encryption disabled (no key configured)")
        except ValueError as exc:
            logger.error("Invalid CONNECTOR_SECRETS_KEY: %s", exc)
            raise
    else:
        SecretEncryption.initialize(None)
        logger.info("Connector secrets encryption disabled (CONNECTOR_SECRETS_KEY not set)")


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
            logger.error(
                "SECURITY: Rejecting insecure origin '%s' from CORS_ORIGINS. "
                "Use explicit origins like 'http://localhost:5173' instead.",
                origin,
            )
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

    # Prepare filesystem and required secrets
    ensure_webui_directories(shared_settings)
    require_jwt_secret(shared_settings)
    require_auth_enabled(shared_settings)
    logger.info("Runtime directories prepared and JWT secret validated")

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

    # Configure connector secrets encryption
    _configure_connector_secrets_encryption()

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


def create_app(skip_lifespan: bool = False) -> FastAPI:
    """Create and configure the FastAPI application

    Args:
        skip_lifespan: Skip lifespan events (for testing)
    """
    import os

    # Check if we're in testing mode
    is_testing = os.getenv("TESTING", "false").lower() in ("true", "1", "yes")

    app_kwargs: dict[str, Any] = {
        "title": "Document Embedding Web UI",
        "description": "Create and search document embeddings",
        "version": "1.1.0",
    }

    # Only add lifespan if not skipping and not testing
    if not skip_lifespan and not is_testing:
        app_kwargs["lifespan"] = lifespan

    app = FastAPI(**app_kwargs)

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

    # Add CSP middleware for XSS prevention
    app.add_middleware(CSPMiddleware)

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

    # Add rate limit middleware to set user in request.state after SlowAPI middleware so user context is available
    app.add_middleware(RateLimitMiddleware)

    # Register global exception handlers
    register_global_exception_handlers(app)

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
    app.include_router(v2_connectors.router)
    app.include_router(v2_directory_scan.router)
    app.include_router(v2_documents.router)
    app.include_router(v2_embedding.router)
    app.include_router(v2_operations.router)
    app.include_router(v2_plugins.router)
    app.include_router(v2_projections.router)
    app.include_router(v2_partition_monitoring.router)
    app.include_router(v2_search.router)
    app.include_router(v2_sources.router)
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
    @app.websocket("/ws/operations")
    async def operation_ws_global(websocket: WebSocket) -> None:
        await operation_websocket_global(websocket)

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

# Make app available as a built-in for tests that reference `app` without import
try:  # pragma: no cover
    import builtins as _builtins

    _builtins.app = app  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=shared_settings.WEBUI_PORT)
