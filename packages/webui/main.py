"""
Main entry point for Document Embedding Web UI
Creates and configures the FastAPI application
"""

import os
import sys

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .api import auth, collections, documents, files, jobs, metrics, models, root, search, settings
from .api.files import scan_websocket
from .api.jobs import websocket_endpoint
from .rate_limiter import limiter


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Document Embedding Web UI", description="Create and search document embeddings", version="1.1.0"
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    app.include_router(root.router)  # No prefix for static + root

    # Mount WebSocket endpoints at the app level
    @app.websocket("/ws/{job_id}")
    async def job_websocket(websocket: WebSocket, job_id: str) -> None:
        await websocket_endpoint(websocket, job_id)

    @app.websocket("/ws/scan/{scan_id}")
    async def scan_ws(websocket: WebSocket, scan_id: str) -> None:
        await scan_websocket(websocket, scan_id)

    # Mount static files with proper path resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")

    # Mount assets directory for React build
    app.mount("/assets", StaticFiles(directory=os.path.join(static_dir, "assets")), name="assets")

    # Mount static files
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from vecpipe.config import settings

    uvicorn.run(app, host="0.0.0.0", port=settings.WEBUI_PORT)
