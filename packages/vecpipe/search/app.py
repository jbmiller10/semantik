"""FastAPI application factory for vecpipe search."""

from fastapi import FastAPI

from vecpipe.search.lifespan import lifespan
from vecpipe.search.router import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Document Vector Search API",
        description="Unified search API with vector similarity, hybrid search, and Qwen3 support",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

__all__ = ["app", "create_app"]
