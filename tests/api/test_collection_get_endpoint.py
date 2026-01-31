"""API endpoint tests for collection retrieval."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from shared.database.exceptions import DatabaseOperationError
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
        # Sync policy fields (now at collection level)
        mock_collection.sync_mode = "one_time"
        mock_collection.sync_interval_minutes = None
        mock_collection.sync_paused_at = None
        mock_collection.sync_next_run_at = None
        mock_collection.sync_last_run_started_at = None
        mock_collection.sync_last_run_completed_at = None
        mock_collection.sync_last_run_status = None
        mock_collection.sync_last_error = None
        mock_collection.default_reranker_id = None
        mock_collection.extraction_config = {}

        async def override_get_collection_for_user(collection_uuid: str) -> Collection:  # type: ignore[override]
            mock_collection.id = collection_uuid
            return mock_collection

        app.dependency_overrides[get_collection_for_user] = override_get_collection_for_user

        # Mock DocumentRepository to return error_count
        mock_doc_repo = MagicMock()
        mock_doc_repo.count_failed_by_collection = AsyncMock(return_value=0)

        try:
            with patch("webui.api.v2.collections.DocumentRepository", return_value=mock_doc_repo):
                response = test_client.get(f"/api/v2/collections/{collection_uuid}")
                assert response.status_code == status.HTTP_200_OK
                payload = response.json()

                assert "total_size_bytes" in payload
                assert payload["total_size_bytes"] == total_size_bytes
                assert "error_count" in payload
                assert payload["error_count"] == 0
        finally:
            del app.dependency_overrides[get_collection_for_user]

    async def test_get_collection_graceful_on_count_failure(self, test_client) -> None:
        """Test endpoint returns 200 with error_count=0 when count query fails."""
        collection_uuid = "test-collection-uuid"

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
        mock_collection.total_size_bytes = 1000
        mock_collection.status = CollectionStatus.READY
        mock_collection.status_message = None
        mock_collection.sync_mode = "one_time"
        mock_collection.sync_interval_minutes = None
        mock_collection.sync_paused_at = None
        mock_collection.sync_next_run_at = None
        mock_collection.sync_last_run_started_at = None
        mock_collection.sync_last_run_completed_at = None
        mock_collection.sync_last_run_status = None
        mock_collection.sync_last_error = None
        mock_collection.default_reranker_id = None
        mock_collection.extraction_config = {}

        async def override_get_collection_for_user(collection_uuid: str) -> Collection:  # type: ignore[override]
            mock_collection.id = collection_uuid
            return mock_collection

        app.dependency_overrides[get_collection_for_user] = override_get_collection_for_user

        # Mock DocumentRepository to raise DatabaseOperationError
        mock_doc_repo = MagicMock()
        mock_doc_repo.count_failed_by_collection = AsyncMock(
            side_effect=DatabaseOperationError("count_failed", "documents", "DB error")
        )

        try:
            with patch("webui.api.v2.collections.DocumentRepository", return_value=mock_doc_repo):
                response = test_client.get(f"/api/v2/collections/{collection_uuid}")

                # Should still return 200, not 500
                assert response.status_code == status.HTTP_200_OK
                payload = response.json()
                assert payload["error_count"] == 0  # Graceful fallback
        finally:
            del app.dependency_overrides[get_collection_for_user]
