"""
Version 2 API modules for collection-centric architecture.

This package contains the new collection-centric API endpoints that replace
the legacy operation-centric endpoints.
"""

from webui.api.v2.collections import router as collections_router
from webui.api.v2.documents import router as documents_router
from webui.api.v2.embedding import router as embedding_router
from webui.api.v2.operations import router as operations_router
from webui.api.v2.projections import router as projections_router
from webui.api.v2.search import router as search_router
from webui.api.v2.system import router as system_router

__all__ = [
    "collections_router",
    "documents_router",
    "embedding_router",
    "operations_router",
    "projections_router",
    "search_router",
    "system_router",
]
