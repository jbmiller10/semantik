"""
Version 2 API modules for collection-centric architecture.

This package contains the new collection-centric API endpoints that replace
the legacy operation-centric endpoints.
"""

from webui.api.v2.agent import router as agent_router
from webui.api.v2.collections import router as collections_router
from webui.api.v2.documents import router as documents_router
from webui.api.v2.embedding import router as embedding_router
from webui.api.v2.llm_settings import router as llm_settings_router
from webui.api.v2.model_manager import router as model_manager_router
from webui.api.v2.operations import router as operations_router
from webui.api.v2.pipeline import router as pipeline_router
from webui.api.v2.plugins import router as plugins_router
from webui.api.v2.projections import router as projections_router
from webui.api.v2.search import router as search_router
from webui.api.v2.system import router as system_router
from webui.api.v2.system_settings import router as system_settings_router

__all__ = [
    "agent_router",
    "collections_router",
    "documents_router",
    "embedding_router",
    "llm_settings_router",
    "model_manager_router",
    "operations_router",
    "pipeline_router",
    "plugins_router",
    "projections_router",
    "search_router",
    "system_router",
    "system_settings_router",
]
