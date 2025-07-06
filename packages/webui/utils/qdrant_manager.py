"""
Qdrant connection manager with pooling and retry logic
"""

import logging
from threading import Lock
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from packages.vecpipe.config import settings

from .retry import exponential_backoff_retry

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """
    Singleton manager for Qdrant connections with connection pooling.
    Reuses connections and provides retry logic for operations.
    """

    _instance: Optional["QdrantConnectionManager"] = None
    _lock = Lock()

    def __new__(cls) -> "QdrantConnectionManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._client: QdrantClient | None = None
        self._url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        self._initialized = True
        logger.info(f"QdrantConnectionManager initialized for {self._url}")

    def _create_client(self) -> QdrantClient:
        """Create a new Qdrant client instance."""
        return QdrantClient(url=self._url)

    def _verify_connection(self, client: QdrantClient) -> bool:
        """Verify that the client can connect to Qdrant."""
        try:
            client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Failed to verify Qdrant connection: {e}")
            return False

    @exponential_backoff_retry(max_retries=3, initial_delay=1.0, max_delay=8.0, exceptions=(Exception,))
    def get_client(self) -> QdrantClient:
        """
        Get a Qdrant client with connection verification and retry logic.

        Returns:
            QdrantClient: A verified working client

        Raises:
            Exception: If connection cannot be established after retries
        """
        # If we have a cached client, verify it's still working
        if self._client:
            if self._verify_connection(self._client):
                return self._client
            logger.warning("Cached Qdrant client is no longer valid, creating new one")
            self._client = None

        # Create and verify new client
        client = self._create_client()
        if not self._verify_connection(client):
            raise Exception(f"Cannot connect to Qdrant at {self._url}")

        # Cache the working client
        self._client = client
        logger.info(f"Successfully connected to Qdrant at {self._url}")
        return client

    @exponential_backoff_retry(max_retries=3, initial_delay=1.0, max_delay=8.0, exceptions=(Exception,))
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        optimizers_config: dict | None = None,
    ) -> None:
        """
        Create a Qdrant collection with retry logic.

        Args:
            collection_name: Name of the collection to create
            vector_size: Dimension of vectors
            distance: Distance metric to use
            optimizers_config: Optional optimizer configuration
        """
        client = self.get_client()

        if optimizers_config is None:
            optimizers_config = {"indexing_threshold": 20000, "memmap_threshold": 0}

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
            optimizers_config=optimizers_config,
        )
        logger.info(f"Successfully created collection {collection_name}")

    @exponential_backoff_retry(max_retries=3, initial_delay=1.0, max_delay=8.0, exceptions=(Exception,))
    def verify_collection(self, collection_name: str) -> dict:
        """
        Verify a collection exists and get its info with retry logic.

        Args:
            collection_name: Name of the collection to verify

        Returns:
            dict: Collection information

        Raises:
            Exception: If collection doesn't exist or verification fails
        """
        client = self.get_client()
        return client.get_collection(collection_name)

    def close(self):
        """Close the cached client connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Closed Qdrant client connection")


# Global instance
qdrant_manager = QdrantConnectionManager()
