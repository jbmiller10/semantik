"""Services module for business logic.

This package exposes common service factories lazily to avoid import cycles
between FastAPI routers and background task modules.
"""

from typing import Any

__all__ = ["CollectionService", "create_collection_service", "ProjectionService", "create_projection_service"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    if name == "CollectionService":
        from .collection_service import CollectionService as _CollectionService

        return _CollectionService
    if name == "create_collection_service":
        from .factory import create_collection_service as _create_collection_service

        return _create_collection_service
    if name == "ProjectionService":
        from .projection_service import ProjectionService as _ProjectionService

        return _ProjectionService
    if name == "create_projection_service":
        from .factory import create_projection_service as _create_projection_service

        return _create_projection_service
    raise AttributeError(f"module 'packages.webui.services' has no attribute {name!r}")
