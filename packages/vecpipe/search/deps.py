"""FastAPI dependency injection helpers for VecPipe."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    import httpx
    from qdrant_client import AsyncQdrantClient

    from vecpipe.llm_model_manager import LLMModelManager
    from vecpipe.model_manager import ModelManager
    from vecpipe.search.runtime import VecpipeRuntime
    from vecpipe.sparse_model_manager import SparseModelManager


def get_runtime(request: Request) -> VecpipeRuntime:
    """Get VecPipe runtime from app.state."""
    runtime: VecpipeRuntime | None = getattr(request.app.state, "vecpipe_runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="VecPipe runtime not initialized")
    if runtime.is_closed:
        raise HTTPException(status_code=503, detail="VecPipe runtime is shutting down")
    return runtime


def get_qdrant_http(request: Request) -> httpx.AsyncClient:
    return cast("httpx.AsyncClient", get_runtime(request).qdrant_http)


def get_qdrant_sdk(request: Request) -> AsyncQdrantClient:
    return cast("AsyncQdrantClient", get_runtime(request).qdrant_sdk)


def get_model_manager(request: Request) -> ModelManager:
    return cast("ModelManager", get_runtime(request).model_manager)


def get_sparse_manager(request: Request) -> SparseModelManager:
    return cast("SparseModelManager", get_runtime(request).sparse_manager)


def get_llm_manager(request: Request) -> LLMModelManager | None:
    return cast("LLMModelManager | None", get_runtime(request).llm_manager)


def require_llm_manager(request: Request) -> LLMModelManager:
    """Get LLM manager, raising 503 if disabled."""
    mgr = get_runtime(request).llm_manager
    if mgr is None:
        raise HTTPException(status_code=503, detail="Local LLM support is disabled")
    return cast("LLMModelManager", mgr)
