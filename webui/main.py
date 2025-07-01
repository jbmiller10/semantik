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

from webui.api import auth, documents, files, jobs, metrics, models, root, search, settings
from webui.api.files import scan_websocket
from webui.api.jobs import websocket_endpoint
from webui.rate_limiter import limiter


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
    app.include_router(metrics.router)
    app.include_router(settings.router)
    app.include_router(models.router)
    app.include_router(search.router)
    app.include_router(documents.router)
    app.include_router(root.router)  # No prefix for static + root

    # Mount WebSocket endpoints at the app level
    @app.websocket("/ws/{job_id}")
    async def job_websocket(websocket: WebSocket, job_id: str):
        await websocket_endpoint(websocket, job_id)

    @app.websocket("/ws/scan/{scan_id}")
    async def scan_ws(websocket: WebSocket, scan_id: str):
        await scan_websocket(websocket, scan_id)

    # Mount static files with proper path resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Mount the SvelteKit build output at root for proper asset loading
    app.mount("/", StaticFiles(directory=os.path.join(base_dir, "static"), html=True), name="static")

    return app


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    from vecpipe.config import settings

    uvicorn.run(app, host="0.0.0.0", port=settings.WEBUI_PORT)
