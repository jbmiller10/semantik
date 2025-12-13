"""API endpoint tests for collection retrieval."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from fastapi import status

from shared.database.models import Collection, CollectionStatus
from webui.dependencies import get_collection_for_user
from webui.main import app


@pytest.mark.asyncio()
class TestCollectionGetEndpoint:
    """Test the GET /api/v2/collections/{collection_uuid} endpoint."""

    async def test_get_collection_includes_total_size_bytes(self, test_client) -> None:
        """Ensure collection stats include total_size_bytes (used by UI Total Size)."""
        collection_uuid = "test-collection-uuid"
        total_size_bytes = 1_234_567

        mock_collection = MagicMock(spec=Collection)
        mock_collection.id = collection_uuid
        mock_collection.name = "Test Collection"
        mock_collection.description = "Test description"
        mock_collection.owner_id = 1
        mock_collection.vector_store_name = "test_vector_store"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.chunking_strategy = None
        mock_collection.chunking_config = None
        mock_collection.is_public = False
        mock_collection.meta = {}
        mock_collection.created_at = datetime.now(UTC)
        mock_collection.updated_at = datetime.now(UTC)
        mock_collection.document_count = 3
        mock_collection.vector_count = 42
        mock_collection.total_size_bytes = total_size_bytes
        mock_collection.status = CollectionStatus.READY
        mock_collection.status_message = None

        async def override_get_collection_for_user(collection_uuid: str) -> Collection:  # type: ignore[override]
            mock_collection.id = collection_uuid
            return mock_collection

        app.dependency_overrides[get_collection_for_user] = override_get_collection_for_user

        try:
            response = test_client.get(f"/api/v2/collections/{collection_uuid}")
            assert response.status_code == status.HTTP_200_OK
            payload = response.json()

            assert "total_size_bytes" in payload
            assert payload["total_size_bytes"] == total_size_bytes
        finally:
            del app.dependency_overrides[get_collection_for_user]
