"""Services module for business logic.

This package exposes common service factories lazily to avoid import cycles
between FastAPI routers and background task modules.
"""

from typing import Any

__all__ = ["CollectionService", "create_collection_service"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    if name == "CollectionService":
        from .collection_service import CollectionService as _CollectionService

        return _CollectionService
    if name == "create_collection_service":
        from .factory import create_collection_service as _create_collection_service

        return _create_collection_service
    raise AttributeError(f"module 'packages.webui.services' has no attribute {name!r}")
