from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from qdrant_client import AsyncQdrantClient

    from vecpipe.llm_model_manager import LLMModelManager
    from vecpipe.model_manager import ModelManager
    from vecpipe.sparse_model_manager import SparseModelManager

logger = logging.getLogger(__name__)


@dataclass
class VecpipeRuntime:
    """Runtime container for VecPipe resources."""

    qdrant_http: httpx.AsyncClient
    qdrant_sdk: AsyncQdrantClient
    model_manager: ModelManager  # Can be ModelManager or GovernedModelManager
    sparse_manager: SparseModelManager
    executor: ThreadPoolExecutor
    llm_manager: LLMModelManager | None = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Idempotent shutdown of all resources."""
        if self._closed:
            return
        self._closed = True

        # Shutdown order mirrors lifespan.py cleanup (lines 188-203)
        try:
            await self.qdrant_http.aclose()
        except Exception:
            logger.exception("Error closing qdrant_http")

        try:
            await self.qdrant_sdk.close()
        except Exception:
            logger.exception("Error closing qdrant_sdk")

        if self.llm_manager is not None:
            try:
                await self.llm_manager.shutdown()
            except Exception:
                logger.exception("Error shutting down llm_manager")

        try:
            await self.sparse_manager.shutdown()
        except Exception:
            logger.exception("Error shutting down sparse_manager")

        # Use async shutdown for GovernedModelManager to avoid deadlock
        try:
            if hasattr(self.model_manager, "shutdown_async"):
                await self.model_manager.shutdown_async()
            else:
                self.model_manager.shutdown()
        except Exception:
            logger.exception("Error shutting down model_manager")

        try:
            self.executor.shutdown(wait=True)
        except Exception:
            logger.exception("Error shutting down executor")

        logger.info("VecpipeRuntime closed")
