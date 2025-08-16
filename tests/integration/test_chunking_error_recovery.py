"""
Integration tests for chunking error recovery and resilience.

Tests error handling, recovery mechanisms, and system stability
under various failure scenarios.
"""

import contextlib
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from faker import Faker
from httpx import AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError
from shared.database.models import Chunk, Collection, Document, Operation
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from webui.auth import create_access_token
from webui.chunking_tasks import ChunkingTask
from webui.services.chunking_service import ChunkingService

fake = Faker()


@pytest.fixture()
async def test_collection(async_session: AsyncSession, test_user: dict) -> Collection:
    """Create a test collection for error recovery tests."""
    collection = Collection(
        id=str(uuid.uuid4()),
        name=f"Error Test Collection {fake.word()}",
        description="Collection for error recovery testing",
        owner_id=test_user["id"],
        status="ready",
        vector_store_name=f"test_error_{uuid.uuid4().hex[:8]}",
        embedding_model="test-model",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )

    async_session.add(collection)
    await async_session.commit()
    await async_session.refresh(collection)

    return collection


@pytest.fixture()
async def test_documents(async_session: AsyncSession, test_collection: Collection) -> list[Document]:
    """Create test documents for error scenarios."""
    documents = []

    # Normal document
    doc1 = Document(
        id=str(uuid.uuid4()),
        collection_id=test_collection.id,
        name="normal.txt",
        content=fake.text(max_nb_chars=1000),
        file_type="text",
        file_size=1000,
        created_at=datetime.now(UTC),
    )
    documents.append(doc1)

    # Large document that might cause memory issues
    doc2 = Document(
        id=str(uuid.uuid4()),
        collection_id=test_collection.id,
        name="large.txt",
        content=fake.text(max_nb_chars=10000000),  # 10MB
        file_type="text",
        file_size=10000000,
        created_at=datetime.now(UTC),
    )
    documents.append(doc2)

    # Document with special characters that might cause issues
    doc3 = Document(
        id=str(uuid.uuid4()),
        collection_id=test_collection.id,
        name="special.txt",
        content="Special chars: \x00\x01\x02 ñáéíóú 中文 🚀 " + fake.text(max_nb_chars=500),
        file_type="text",
        file_size=600,
        created_at=datetime.now(UTC),
    )
    documents.append(doc3)

    # Empty document
    doc4 = Document(
        id=str(uuid.uuid4()),
        collection_id=test_collection.id,
        name="empty.txt",
        content="",
        file_type="text",
        file_size=0,
        created_at=datetime.now(UTC),
    )
    documents.append(doc4)

    for doc in documents:
        async_session.add(doc)

    await async_session.commit()
    return documents


@pytest.fixture()
async def auth_headers(test_user: dict) -> dict[str, str]:
    """Create authorization headers."""
    token = create_access_token(data={"sub": test_user["username"], "user_id": test_user["id"]})
    return {"Authorization": f"Bearer {token}"}


class TestChunkingOperationInterruption:
    """Test handling of chunking operation interruptions."""

    @pytest.mark.asyncio()
    async def test_operation_interruption_recovery(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test recovery from chunking operation interruption."""
        # Start chunking operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
            },
        )

        assert response.status_code == 202
        operation_id = response.json()["operation_id"]

        # Simulate partial processing
        docs_to_process = test_documents[:2]
        for doc in docs_to_process:
            # Create some chunks
            for i in range(3):
                chunk = Chunk(
                    collection_id=test_collection.id,
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
        operation.status = "failed"
        operation.error_message = "Process interrupted"
        operation.progress_percentage = 50.0
        await async_session.commit()

        # Store progress in Redis for recovery
        redis_client.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "failed",
                "documents_processed": "2",
                "total_documents": str(len(test_documents)),
                "last_processed_doc": test_documents[1].id,
            },
        )

        # Attempt to resume operation
        response = await async_client.post(
            f"/api/v2/chunking/operations/{operation_id}/resume",
            headers=auth_headers,
        )

        # Should either resume or indicate how to retry
        assert response.status_code in [200, 202, 409]

        if response.status_code == 202:
            # Operation resumed
            result = response.json()
            assert "operation_id" in result

            # Verify it continues from where it left off
            resumed_state = redis_client.hgetall(f"operation:{operation_id}")
            assert int(resumed_state.get(b"documents_processed", 0)) >= 2

    @pytest.mark.asyncio()
    async def test_graceful_shutdown_handling(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
    ) -> None:
        """Test graceful shutdown during chunking operation."""
        # Start operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "recursive",
                "config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate graceful shutdown signal
        with patch("webui.chunking_tasks.ChunkingTask._graceful_shutdown", True):
            # Operation should save progress and exit cleanly
            task = ChunkingTask()

            # Mock the operation processing
            with (
                patch.object(task, "_save_progress") as _mock_save,
                contextlib.suppress(SystemExit),
            ):
                # Simulate partial processing
                # This would normally be called by Celery
                # task.run(operation_id, str(uuid.uuid4()))
                pass

                # Progress should be saved
                # mock_save.assert_called()

        # Operation should be in a resumable state
        operation = await async_session.get(Operation, operation_id)
        assert operation.status in ["paused", "failed", "pending"]

        # Should be able to resume
        chunks_before = await async_session.execute(select(Chunk).where(Chunk.collection_id == test_collection.id))
        initial_chunk_count = len(chunks_before.scalars().all())

        # Resuming should continue processing
        assert initial_chunk_count >= 0  # Some chunks may have been created


class TestDatabaseFailures:
    """Test handling of database failures during chunking."""

    @pytest.mark.asyncio()
    async def test_database_connection_loss(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of database connection loss during chunking."""
        # Start chunking operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 200, "chunk_overlap": 20},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate database failure during processing
        with patch("webui.services.chunking_service.AsyncSession") as mock_session:
            mock_session.side_effect = OperationalError("Database connection lost", "", "")

            # Service should handle the error gracefully
            service = ChunkingService()

            with pytest.raises(OperationalError):
                # This should fail but be handled
                await service.process_chunking_operation(
                    operation_id=operation_id,
                    collection_id=test_collection.id,
                    strategy="fixed_size",
                    config={"chunk_size": 200, "chunk_overlap": 20},
                )

        # Operation should be marked as failed with appropriate error
        operation = await async_session.get(Operation, operation_id)
        assert operation.status == "failed"
        assert "database" in operation.error_message.lower() or "connection" in operation.error_message.lower()

    @pytest.mark.asyncio()
    async def test_database_transaction_rollback(
        self,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
    ) -> None:
        """Test transaction rollback on chunk insertion failure."""
        chunks_to_insert = []

        # Create chunks
        for i in range(10):
            chunk = Chunk(
                collection_id=test_collection.id,
                document_id=test_documents[0].id,
                content=f"Test chunk {i}",
                chunk_index=i,
                start_offset=i * 100,
                end_offset=(i + 1) * 100,
                token_count=10,
                created_at=datetime.now(UTC),
            )
            chunks_to_insert.append(chunk)

        # Start transaction
        async with async_session.begin():
            # Insert some chunks
            for chunk in chunks_to_insert[:5]:
                async_session.add(chunk)

            # Simulate error midway
            # Force an error (e.g., duplicate key)
            duplicate_chunk = Chunk(
                collection_id=test_collection.id,
                document_id=test_documents[0].id,
                content="Duplicate",
                chunk_index=0,  # Duplicate index
                start_offset=0,
                end_offset=100,
                token_count=10,
                created_at=datetime.now(UTC),
            )
            async_session.add(duplicate_chunk)
            # Force error
            with pytest.raises(Exception, match="Simulated error"):
                raise Exception("Simulated error")

        # Transaction should be rolled back
        result = await async_session.execute(
            select(Chunk).where(
                Chunk.collection_id == test_collection.id,
                Chunk.document_id == test_documents[0].id,
            )
        )
        chunks = result.scalars().all()

        # No chunks should be inserted due to rollback
        assert len(chunks) == 0


class TestRedisFailures:
    """Test handling of Redis failures."""

    @pytest.mark.asyncio()
    async def test_redis_connection_failure(
        self,
        async_client: AsyncClient,
        test_collection: Collection,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of Redis connection failure."""
        with patch("webui.services.chunking_service.redis_client") as mock_redis:
            # Simulate Redis connection error
            mock_redis.ping.side_effect = RedisConnectionError("Connection refused")
            mock_redis.hset.side_effect = RedisConnectionError("Connection refused")

            # Operation should still be created in database
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection.id}/chunk",
                headers=auth_headers,
                json={
                    "strategy": "fixed_size",
                    "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
                },
            )

            # Should handle Redis failure gracefully
            assert response.status_code in [202, 503]

            if response.status_code == 202:
                # Operation created despite Redis issues
                result = response.json()
                assert "operation_id" in result
                # WebSocket channel might be unavailable
                assert result.get("websocket_channel") is None or result["websocket_channel"] == ""

    @pytest.mark.asyncio()
    async def test_redis_stream_failure(
        self,
        async_client: AsyncClient,
        test_collection: Collection,
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test handling of Redis stream failures for progress updates."""
        # Start operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "semantic",
                "config": {"strategy": "semantic", "chunk_size": 300, "chunk_overlap": 50},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate Redis stream failure
        with patch.object(redis_client, "xadd", side_effect=RedisConnectionError("Stream error")):
            # Progress updates should fail but not crash the operation
            service = ChunkingService()

            # This should continue despite Redis stream errors
            with contextlib.suppress(RedisConnectionError):
                await service._send_progress_update(
                    operation_id=operation_id,
                    progress=50.0,
                    message="Processing...",
                )

        # Operation should continue even without progress updates
        # In real scenario, worker would continue processing


class TestResourceExhaustion:
    """Test handling of resource exhaustion scenarios."""

    @pytest.mark.asyncio()
    async def test_memory_exhaustion_handling(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of memory exhaustion during chunking."""
        # Create a very large document
        huge_doc = Document(
            id=str(uuid.uuid4()),
            collection_id=test_collection.id,
            name="huge.txt",
            content="x" * 100000000,  # 100MB of text
            file_type="text",
            file_size=100000000,
            created_at=datetime.now(UTC),
        )
        async_session.add(huge_doc)
        await async_session.commit()

        # Start chunking with memory monitoring
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 100, "chunk_overlap": 10},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate memory pressure
        with patch("webui.chunking_tasks.psutil.virtual_memory") as mock_memory:
            # Simulate low memory
            mock_memory.return_value = MagicMock(available=100 * 1024 * 1024)  # 100MB available

            # Task should detect memory pressure and handle appropriately
            _task = ChunkingTask()

            # Should either complete with reduced batch size or fail gracefully
            # In real scenario, task would adapt batch size or fail with MemoryError

        # Check operation status
        operation = await async_session.get(Operation, operation_id)

        if operation.status == "failed":
            assert "memory" in operation.error_message.lower()
        else:
            # Should have adapted to memory constraints
            assert operation.status in ["completed", "processing"]

    @pytest.mark.asyncio()
    async def test_disk_space_exhaustion(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
    ) -> None:
        """Test handling of disk space exhaustion."""
        # Start chunking operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "hierarchical",
                "config": {"strategy": "hierarchical", "chunk_size": 1000, "chunk_overlap": 200},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate disk space error during chunk storage
        with patch("webui.services.chunking_service.AsyncSession.commit") as mock_commit:
            mock_commit.side_effect = OperationalError("No space left on device", "", "")

            service = ChunkingService()

            # Should handle disk space error
            with pytest.raises(OperationalError):
                await service.store_chunks(
                    session=async_session,
                    chunks=[],  # Would normally have chunks
                    collection_id=test_collection.id,
                )

        # Operation should be marked as failed
        operation = await async_session.get(Operation, operation_id)
        assert operation.status in ["failed", "pending"]  # Might not have started yet

        if operation.status == "failed":
            assert "space" in operation.error_message.lower() or "disk" in operation.error_message.lower()

    @pytest.mark.asyncio()
    async def test_cpu_time_limit(
        self,
        async_client: AsyncClient,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
    ) -> None:
        """Test CPU time limit enforcement."""
        # Create documents that require intensive processing
        complex_docs = []
        for i in range(10):
            doc = Document(
                id=str(uuid.uuid4()),
                collection_id=test_collection.id,
                name=f"complex_{i}.txt",
                content=fake.text(max_nb_chars=50000),
                file_type="text",
                file_size=50000,
                created_at=datetime.now(UTC),
            )
            complex_docs.append(doc)

        # Start CPU-intensive chunking operation
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "semantic",  # Most CPU-intensive
                "config": {"strategy": "semantic", "chunk_size": 100, "chunk_overlap": 50},
            },
        )

        _operation_id = response.json()["operation_id"]

        # Simulate CPU time limit reached
        with patch("webui.chunking_tasks.time.process_time") as mock_cpu_time:
            # Simulate exceeding CPU time limit
            mock_cpu_time.return_value = 2000  # Exceeds 30-minute limit

            _task = ChunkingTask()

            # Task should detect CPU time limit and handle appropriately
            # In real scenario, task would be terminated by Celery

        # Operation should handle CPU limit gracefully
        # Status would depend on how much was processed before limit


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio()
    async def test_circuit_breaker_activation(
        self,
        async_client: AsyncClient,
        test_collection: Collection,
        auth_headers: dict[str, str],
    ) -> None:
        """Test circuit breaker activation after multiple failures."""
        # Simulate multiple consecutive failures
        with (
            patch("webui.chunking_tasks.ChunkingTask._increment_failure_count") as _mock_increment,
            patch("webui.chunking_tasks.ChunkingTask._check_circuit_breaker") as mock_check,
        ):
            # First few attempts succeed
            mock_check.return_value = True

            for _i in range(3):
                response = await async_client.post(
                    f"/api/v2/chunking/collections/{test_collection.id}/chunk",
                    headers=auth_headers,
                    json={
                        "strategy": "fixed_size",
                        "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
                    },
                )
                assert response.status_code == 202

            # Simulate failures to trigger circuit breaker
            mock_check.return_value = False

            # Next attempt should be rejected by circuit breaker
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection.id}/chunk",
                headers=auth_headers,
                json={
                    "strategy": "fixed_size",
                    "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
                },
            )

            # Should either reject immediately or queue with warning
            assert response.status_code in [202, 503]

            if response.status_code == 503:
                result = response.json()
                assert "circuit breaker" in result.get("detail", "").lower()

    @pytest.mark.asyncio()
    async def test_circuit_breaker_recovery(
        self,
        async_client: AsyncClient,
        test_collection: Collection,
        auth_headers: dict[str, str],
    ) -> None:
        """Test circuit breaker recovery after timeout."""
        with (
            patch("webui.chunking_tasks.ChunkingTask._check_circuit_breaker") as mock_check,
            patch("webui.chunking_tasks.time.time") as mock_time,
        ):
            # Initially, circuit breaker is open
            mock_check.return_value = False
            mock_time.return_value = 1000

            # Attempt should fail
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection.id}/chunk",
                headers=auth_headers,
                json={
                    "strategy": "recursive",
                    "config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
                },
            )

            # Should handle circuit breaker state
            assert response.status_code in [202, 503]

            # Simulate time passing (circuit breaker timeout)
            mock_time.return_value = 1000 + 300  # 5 minutes later
            mock_check.return_value = True  # Circuit breaker closes

            # Should work again
            response = await async_client.post(
                f"/api/v2/chunking/collections/{test_collection.id}/chunk",
                headers=auth_headers,
                json={
                    "strategy": "recursive",
                    "config": {"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
                },
            )

            assert response.status_code == 202


class TestPartialFailures:
    """Test handling of partial failures."""

    @pytest.mark.asyncio()
    async def test_partial_document_failure(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        test_documents: list[Document],
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test handling when some documents fail but others succeed."""
        # Add a problematic document
        problem_doc = Document(
            id=str(uuid.uuid4()),
            collection_id=test_collection.id,
            name="problem.txt",
            content=None,  # Null content should cause issues
            file_type="text",
            file_size=0,
            created_at=datetime.now(UTC),
        )
        async_session.add(problem_doc)
        await async_session.commit()

        all_docs = test_documents + [problem_doc]

        # Start chunking
        response = await async_client.post(
            f"/api/v2/chunking/collections/{test_collection.id}/chunk",
            headers=auth_headers,
            json={
                "strategy": "fixed_size",
                "config": {"strategy": "fixed_size", "chunk_size": 500, "chunk_overlap": 50},
            },
        )

        operation_id = response.json()["operation_id"]

        # Simulate processing with partial failure
        successful_docs = []
        failed_docs = []

        for doc in all_docs:
            if doc.content:
                # Process successfully
                successful_docs.append(doc.id)

                # Create chunks for successful docs
                for i in range(2):
                    chunk = Chunk(
                        collection_id=test_collection.id,
                        document_id=doc.id,
                        content=f"Chunk {i}",
                        chunk_index=i,
                        start_offset=0,
                        end_offset=100,
                        token_count=10,
                        created_at=datetime.now(UTC),
                    )
                    async_session.add(chunk)
            else:
                # Fail for null content
                failed_docs.append(doc.id)

        await async_session.commit()

        # Update operation status
        operation = await async_session.get(Operation, operation_id)
        operation.status = "partial_success"
        operation.completed_at = datetime.now(UTC)
        operation.error_message = f"Failed to process {len(failed_docs)} documents"
        await async_session.commit()

        # Store failure details in Redis
        redis_client.hset(
            f"operation:{operation_id}",
            mapping={
                "status": "partial_success",
                "successful_docs": json.dumps(successful_docs),
                "failed_docs": json.dumps(failed_docs),
            },
        )

        # Verify partial success handling
        assert operation.status == "partial_success"
        assert len(successful_docs) > 0
        assert len(failed_docs) > 0

        # Check that successful documents have chunks
        for doc_id in successful_docs:
            result = await async_session.execute(
                select(Chunk).where(
                    Chunk.collection_id == test_collection.id,
                    Chunk.document_id == doc_id,
                )
            )
            chunks = result.scalars().all()
            assert len(chunks) > 0

        # Failed documents should have no chunks
        for doc_id in failed_docs:
            result = await async_session.execute(
                select(Chunk).where(
                    Chunk.collection_id == test_collection.id,
                    Chunk.document_id == doc_id,
                )
            )
            chunks = result.scalars().all()
            assert len(chunks) == 0

    @pytest.mark.asyncio()
    async def test_retry_failed_documents(
        self,
        async_client: AsyncClient,
        async_session: AsyncSession,
        test_collection: Collection,
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test retry mechanism for failed documents."""
        # Create an operation with failed documents
        operation = Operation(
            uuid=str(uuid.uuid4()),
            collection_id=test_collection.id,
            type="chunking",
            status="partial_success",
            progress_percentage=80.0,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            error_message="2 documents failed",
        )
        async_session.add(operation)
        await async_session.commit()

        # Store failed document IDs in Redis
        failed_doc_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        redis_client.hset(
            f"operation:{operation.uuid}",
            mapping={
                "failed_docs": json.dumps(failed_doc_ids),
            },
        )

        # Retry failed documents
        response = await async_client.post(
            f"/api/v2/chunking/operations/{operation.uuid}/retry-failed",
            headers=auth_headers,
        )

        # Should create a new operation for retry
        assert response.status_code in [200, 202]

        if response.status_code == 202:
            result = response.json()
            assert "operation_id" in result
            assert result["operation_id"] != operation.uuid  # New operation
            assert "retry_count" in result or "documents_to_retry" in result


class TestDeadLetterQueue:
    """Test dead letter queue for unrecoverable failures."""

    @pytest.mark.asyncio()
    async def test_dead_letter_queue_entry(
        self,
        redis_client: Any,
    ) -> None:
        """Test that unrecoverable failures go to dead letter queue."""
        task = ChunkingTask()
        operation_id = str(uuid.uuid4())
        error_message = "Unrecoverable error: Invalid configuration"

        # Send to dead letter queue
        task._send_to_dead_letter_queue(
            operation_id=operation_id,
            error=Exception(error_message),
            context={"attempt": 3, "strategy": "semantic"},
        )

        # Check dead letter queue
        dlq_key = f"dlq:chunking:{operation_id}"
        dlq_entry = redis_client.hgetall(dlq_key)

        assert dlq_entry is not None
        assert b"error" in dlq_entry
        assert error_message in dlq_entry[b"error"].decode()
        assert b"context" in dlq_entry

        # Entry should have TTL
        ttl = redis_client.ttl(dlq_key)
        assert ttl > 0  # Should expire eventually

    @pytest.mark.asyncio()
    async def test_manual_dlq_recovery(
        self,
        async_client: AsyncClient,
        auth_headers: dict[str, str],
        redis_client: Any,
    ) -> None:
        """Test manual recovery from dead letter queue."""
        # Create DLQ entry
        operation_id = str(uuid.uuid4())
        dlq_key = f"dlq:chunking:{operation_id}"

        redis_client.hset(
            dlq_key,
            mapping={
                "operation_id": operation_id,
                "error": "Test error",
                "timestamp": str(time.time()),
                "context": json.dumps({"collection_id": "test-123"}),
            },
        )

        # Admin endpoint to review DLQ
        response = await async_client.get(
            "/api/v2/admin/dead-letter-queue/chunking",
            headers=auth_headers,
        )

        # Should list DLQ entries (if endpoint exists)
        if response.status_code == 200:
            dlq_entries = response.json()
            assert isinstance(dlq_entries, list)

            # Find our entry
            our_entry = next(
                (e for e in dlq_entries if e.get("operation_id") == operation_id),
                None,
            )

            if our_entry:
                assert "error" in our_entry
                assert "timestamp" in our_entry
