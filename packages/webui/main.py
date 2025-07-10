"""
Main entry point for Document Embedding Web UI
Creates and configures the FastAPI application
"""

import logging
import secrets
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import Response

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shared.config import settings as shared_settings
from shared.embedding import configure_global_embedding_service

logger = logging.getLogger(__name__)

from .api import (  # noqa: E402
    auth,
    collections,
    documents,
    files,
    health,
    internal,
    jobs,
    metrics,
    models,
    root,
    search,
    settings,
)
from .api.files import scan_websocket  # noqa: E402
from .api.jobs import websocket_endpoint  # noqa: E402
from .rate_limiter import limiter  # noqa: E402


def rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Wrapper to ensure proper type signature for rate limit handler"""
    if isinstance(exc, RateLimitExceeded):
        return _rate_limit_exceeded_handler(request, exc)
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Document Embedding Web UI", description="Create and search document embeddings", version="1.1.0"
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    # Configure global embedding service at app startup
    _configure_embedding_service()

    # Configure internal API key
    _configure_internal_api_key()

    # Include routers with their specific prefixes
    app.include_router(auth.router)
    app.include_router(jobs.router)
    app.include_router(files.router)
    app.include_router(collections.router)
    app.include_router(metrics.router)
    app.include_router(settings.router)
    app.include_router(models.router)
    app.include_router(search.router)
    app.include_router(documents.router)
    app.include_router(health.router)
    app.include_router(internal.router)
    app.include_router(root.router)  # No prefix for static + root

    # Mount WebSocket endpoints at the app level
    @app.websocket("/ws/{job_id}")
    async def job_websocket(websocket: WebSocket, job_id: str) -> None:
        await websocket_endpoint(websocket, job_id)

    @app.websocket("/ws/scan/{scan_id}")
    async def scan_ws(websocket: WebSocket, scan_id: str) -> None:
        await scan_websocket(websocket, scan_id)

    # Add health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint for Docker health monitoring"""
        return {"status": "healthy", "service": "webui"}

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

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=shared_settings.WEBUI_PORT)
