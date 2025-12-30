"""Unified Qdrant manager provider used across WebUI services and tasks."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, cast

from qdrant_client import QdrantClient

from shared.config import settings
from shared.managers.qdrant_manager import QdrantManager
from webui.utils.retry import exponential_backoff_retry

logger = logging.getLogger(__name__)


class UnifiedQdrantManager(QdrantManager):
    """Thin wrapper that exposes the underlying client for legacy call sites."""

    def __init__(self, qdrant_client: QdrantClient):
        super().__init__(qdrant_client)

    def get_client(self) -> QdrantClient:
        """Return the underlying Qdrant client (compat with old manager)."""
        return cast(QdrantClient, self.client)


_manager_lock = Lock()
_manager_instance: UnifiedQdrantManager | None = None


@exponential_backoff_retry(max_retries=3, initial_delay=1.0, max_delay=8.0, exceptions=(Exception,))
def _build_client() -> QdrantClient:
    """Create and verify a Qdrant client with retries."""
    url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    api_key = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
    client = QdrantClient(url=url, api_key=api_key)
    # Basic connectivity check; will raise on failure to be retried by decorator
    client.get_collections()
    logger.info("Connected to Qdrant at %s", url)
    return client


def get_qdrant_manager() -> UnifiedQdrantManager:
    """Return a singleton UnifiedQdrantManager instance."""
    global _manager_instance

    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                client = _build_client()
                _manager_instance = UnifiedQdrantManager(client)
    return _manager_instance


def set_qdrant_manager_for_tests(manager: UnifiedQdrantManager | None) -> None:
    """Allow tests to override or reset the singleton instance."""
    global _manager_instance
    with _manager_lock:
        _manager_instance = manager


class _LazyQdrantManagerProxy:
    """Defer Qdrant client construction until first real use.

    Importing ``webui.qdrant`` should never attempt an external connection so
    unit tests and startup environments without Qdrant remain import-safe. The
    proxy forwards attribute access to the singleton created by
    :func:`get_qdrant_manager`.
    """

    def _resolve(self) -> UnifiedQdrantManager:
        return get_qdrant_manager()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
        instantiated = _manager_instance is not None
        return f"<LazyQdrantManagerProxy instantiated={instantiated}>"

    def __bool__(self) -> bool:
        # Preserve truthiness checks without forcing a connection
        return True


# Singleton placeholder exposed to the rest of the codebase
qdrant_manager = _LazyQdrantManagerProxy()

__all__ = [
    "UnifiedQdrantManager",
    "get_qdrant_manager",
    "set_qdrant_manager_for_tests",
    "qdrant_manager",
]
