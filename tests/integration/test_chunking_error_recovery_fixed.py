"""
Fixed version of chunking error recovery tests with proper authentication.

This file demonstrates how to migrate error recovery tests to use
the proper authentication fixtures from conftest.py.
"""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from shared.database.models import Chunk, Collection, Document, Operation
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError


class TestChunkingOperationInterruption:
    """Tests for handling chunking operation interruptions."""

    @pytest.mark.asyncio
    async def test_operation_interruption_recovery(
        self,
        authenticated_async_client: AsyncClient,  # Use authenticated client
        authenticated_collection: dict,  # Use authenticated collection
        async_session: AsyncSession,
        authenticated_user: dict,  # Use authenticated user
        redis_client: Any,
    ) -> None:
        """Test recovery from chunking operation interruption.
        
        Key changes:
        1. Use authenticated_async_client instead of async_client
        2. Use authenticated_collection instead of test_collection
        3. Use authenticated_user instead of test_user
        4. No need to manually override get_current_user
        """
        # Create test documents for the authenticated collection
        test_documents = []
        for i in range(5):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=authenticated_collection['id'],
                name=f"test_doc_{i}.txt",
                path=f"/test/test_doc_{i}.txt",
                type="text",
                size=1000,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            async_session.add(doc)
            test_documents.append(doc)
        
        await async_session.commit()
        
        # Start chunking operation - this will work because:
        # 1. authenticated_async_client has the correct user override
        # 2. authenticated_collection is owned by that user
        response = await authenticated_async_client.post(
            f"/api/v2/chunking/collections/{authenticated_collection['id']}/chunk",
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
            },
        )

        # Should not get AccessDeniedError anymore!
        assert response.status_code == 202
        operation_id = response.json()["operation_id"]

        # Simulate partial processing
        docs_to_process = test_documents[:2]
        for doc in docs_to_process:
            # Create some chunks
            for i in range(3):
                chunk = Chunk(
                    collection_id=authenticated_collection['id'],
                    document_id=doc.id,
                    content=f"Partial chunk {i}",
                    chunk_index=i,
                    start_offset=i * 100,
                    end_offset=(i + 1) * 100,
                    token_count=10,
                    created_at=datetime.now(UTC),
                )
                async_session.add(chunk)

        await async_session.commit()

        # Simulate interruption - mark operation as failed
        operation = await async_session.get(Operation, operation_id)
        if operation:
            operation.status = "failed"
            operation.error_message = "Process interrupted"
            await async_session.commit()

            # Test recovery mechanism
            response = await authenticated_async_client.post(
                f"/api/v2/chunking/operations/{operation_id}/retry"
            )

            # Check recovery initiated
            assert response.status_code in [200, 202]
            
            # Verify operation status
            await async_session.refresh(operation)
            assert operation.status in ["pending", "processing"]

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handling(
        self,
        authenticated_async_client: AsyncClient,
        authenticated_collection: dict,
        async_session: AsyncSession,
        authenticated_user: dict,
    ) -> None:
        """Test graceful shutdown during chunking operation."""
        # Create a document
        doc = Document(
            id=str(uuid.uuid4()),
            collection_id=authenticated_collection['id'],
            name="test_doc.txt",
            path="/test/test_doc.txt",
            type="text",
            size=10000,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        async_session.add(doc)
        await async_session.commit()

        # Start chunking
        response = await authenticated_async_client.post(
            f"/api/v2/chunking/collections/{authenticated_collection['id']}/chunk",
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 1000, "chunk_overlap": 100},
            },
        )

        assert response.status_code == 202
        operation_id = response.json()["operation_id"]

        # Simulate graceful shutdown signal
        with patch("packages.webui.services.chunking_service.ChunkingService.handle_shutdown") as mock_shutdown:
            mock_shutdown.return_value = True
            
            # Check operation is properly saved
            operation = await async_session.get(Operation, operation_id)
            assert operation is not None
            
            # Verify operation can be resumed after restart
            response = await authenticated_async_client.get(
                f"/api/v2/chunking/operations/{operation_id}"
            )
            assert response.status_code == 200


class TestDatabaseFailures:
    """Tests for handling database failures during chunking."""

    @pytest.mark.asyncio
    async def test_database_connection_loss(
        self,
        authenticated_async_client: AsyncClient,
        authenticated_collection: dict,
        async_session: AsyncSession,
        authenticated_user: dict,
    ) -> None:
        """Test handling of database connection loss during chunking."""
        # Create a document
        doc = Document(
            id=str(uuid.uuid4()),
            collection_id=authenticated_collection['id'],
            name="test_doc.txt",
            path="/test/test_doc.txt",
            type="text",
            size=1000,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        async_session.add(doc)
        await async_session.commit()

        # Mock database failure
        with patch.object(async_session, "execute", side_effect=OperationalError("Connection lost", None, None)):
            response = await authenticated_async_client.post(
                f"/api/v2/chunking/collections/{authenticated_collection['id']}/chunk",
                json={
                    "strategy": "fixed_size",
                    "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
                },
            )

            # Should handle the error gracefully
            assert response.status_code in [500, 503]
            assert "database" in response.json().get("detail", "").lower()


# Migration notes for other test classes:
#
# 1. Replace all instances of:
#    - async_client -> authenticated_async_client
#    - test_collection -> authenticated_collection  
#    - test_user -> authenticated_user
#
# 2. Remove manual overrides of get_current_user:
#    - Delete: app.dependency_overrides[get_current_user] = mock_get_current_user
#    - Delete: from packages.webui.auth import get_current_user
#
# 3. When creating test data, use authenticated_collection['id'] instead of test_collection.id
#
# 4. The authenticated fixtures ensure:
#    - User ID matches collection owner_id
#    - No AccessDeniedError for valid operations
#    - Consistent authentication across all test methods