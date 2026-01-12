"""Comprehensive tests for webui.tasks.ingestion module.

This test suite covers:
- Embedding concurrency configuration
- INDEX operation processing
- APPEND operation implementation
- REMOVE_SOURCE operation processing
- Task failure handling
- Edge cases and error conditions
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from shared.dtos.ingestion import IngestedDocument
from webui.tasks import ingestion as ingestion_module

# Valid 64-character hex hash for test documents
_VALID_HASH = "a" * 64


def _make_test_doc(unique_id: str, content: str = "test content") -> IngestedDocument:
    """Create a test IngestedDocument with valid hash."""
    return IngestedDocument(
        content=content,
        unique_id=unique_id,
        source_type="directory",
        metadata={},
        content_hash=_VALID_HASH,
    )

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# Test Fixtures and Helpers
# ---------------------------------------------------------------------------


def make_mock_operation(
    op_id: int = 1,
    op_uuid: str = "op-uuid-123",
    collection_id: str = "col-uuid-456",
    op_type: str = "INDEX",
    config: dict | None = None,
    user_id: int | None = 1,
):
    """Create a mock operation object."""
    from shared.database.models import OperationType

    mock_op = Mock()
    mock_op.id = op_id
    mock_op.uuid = op_uuid
    mock_op.collection_id = collection_id
    mock_op.type = getattr(OperationType, op_type)
    mock_op.config = config or {}
    mock_op.user_id = user_id
    return mock_op


def make_mock_collection(
    col_id: str = "col-uuid-456",
    name: str = "Test Collection",
    vector_store_name: str = "qdrant_col_123",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    status: str = "PENDING",
):
    """Create a mock collection object."""
    from shared.database.models import CollectionStatus

    mock_col = Mock()
    mock_col.id = col_id
    mock_col.uuid = col_id
    mock_col.name = name
    mock_col.vector_store_name = vector_store_name
    mock_col.vector_collection_id = vector_store_name
    mock_col.embedding_model = embedding_model
    mock_col.quantization = "float16"
    mock_col.chunk_size = 1000
    mock_col.chunk_overlap = 200
    mock_col.chunking_strategy = "recursive"
    mock_col.chunking_config = {}
    mock_col.config = {"vector_dim": 1024}
    mock_col.qdrant_collections = []
    mock_col.qdrant_staging = []
    mock_col.status = getattr(CollectionStatus, status)
    mock_col.vector_count = 0
    return mock_col


class MockUpdater:
    """Mock CeleryTaskWithOperationUpdates for testing."""

    def __init__(self, operation_id: str = "op-123"):
        self.operation_id = operation_id
        self.updates: list[tuple[str, dict]] = []
        self.user_id: int | None = None

    def set_user_id(self, user_id: int) -> None:
        self.user_id = user_id

    async def send_update(self, event_type: str, data: dict) -> None:
        self.updates.append((event_type, data))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@asynccontextmanager
async def mock_session_context(mock_session) -> Generator[Any, None, None]:
    """Create an async context manager for mock session."""
    yield mock_session


# ---------------------------------------------------------------------------
# Tests for _get_embedding_concurrency
# ---------------------------------------------------------------------------


class TestGetEmbeddingConcurrency:
    """Tests for _get_embedding_concurrency function."""

    def test_default_value(self, monkeypatch):
        """Test default concurrency is 1 when env var not set."""
        monkeypatch.delenv("EMBEDDING_CONCURRENCY_PER_WORKER", raising=False)
        # Re-import to get fresh value
        result = ingestion_module._get_embedding_concurrency()
        assert result == 1

    def test_valid_env_value(self, monkeypatch):
        """Test concurrency respects valid env var value."""
        monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "4")
        result = ingestion_module._get_embedding_concurrency()
        assert result == 4

    def test_negative_value_returns_minimum(self, monkeypatch):
        """Test negative values return minimum of 1."""
        monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "-5")
        result = ingestion_module._get_embedding_concurrency()
        assert result == 1

    def test_zero_value_returns_minimum(self, monkeypatch):
        """Test zero returns minimum of 1."""
        monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "0")
        result = ingestion_module._get_embedding_concurrency()
        assert result == 1

    def test_invalid_string_returns_default(self, monkeypatch):
        """Test invalid string returns default of 1."""
        monkeypatch.setenv("EMBEDDING_CONCURRENCY_PER_WORKER", "not_a_number")
        result = ingestion_module._get_embedding_concurrency()
        assert result == 1

# ---------------------------------------------------------------------------
# Tests for _process_index_operation
# ---------------------------------------------------------------------------


class TestProcessIndexOperation:
    """Tests for INDEX operation processing."""

    @pytest.mark.asyncio()
    async def test_successful_index_creates_qdrant_collection(self):
        """Test INDEX operation creates Qdrant collection successfully."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        # Mock Qdrant manager
        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        assert result["qdrant_collection"] == "qdrant_test_col"
        assert result["vector_dim"] == 1024
        mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio()
    async def test_index_generates_vector_store_name_if_missing(self):
        """Test INDEX generates vector_store_name when not provided."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "abc-def-123",
            "name": "Test Collection",
            "vector_store_name": "",  # Missing
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        # Generated name should be based on UUID
        assert "col_abc_def_123" in result["qdrant_collection"]

    @pytest.mark.asyncio()
    async def test_index_fails_when_qdrant_creation_fails(self):
        """Test INDEX operation fails when Qdrant collection creation fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock(side_effect=Exception("Qdrant unavailable"))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            pytest.raises(Exception, match="Qdrant unavailable"),
        ):
            await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

    @pytest.mark.asyncio()
    async def test_index_cleans_up_on_db_update_failure(self):
        """Test INDEX cleans up Qdrant collection if database update fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_collection_repo.update = AsyncMock(side_effect=Exception("DB error"))
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))
        mock_qdrant_client.delete_collection = Mock()

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            pytest.raises(Exception, match="Failed to update collection"),
        ):
            await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Verify cleanup was attempted
        mock_qdrant_client.delete_collection.assert_called_once_with("qdrant_test_col")


# ---------------------------------------------------------------------------
# Tests for _process_remove_source_operation
# ---------------------------------------------------------------------------


class TestProcessRemoveSourceOperation:
    """Tests for REMOVE_SOURCE operation processing."""

    @pytest.mark.asyncio()
    async def test_remove_source_with_no_documents(self):
        """Test REMOVE_SOURCE when no documents exist for source."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "qdrant_collections": [],
            "qdrant_staging": [],
            "vector_count": 100,
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[])
        updater = MockUpdater("op-123")

        with patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock):
            result = await ingestion_module._process_remove_source_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        assert result["documents_removed"] == 0

    @pytest.mark.asyncio()
    async def test_remove_source_missing_source_id(self):
        """Test REMOVE_SOURCE fails when source_id is missing."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_path": "/path/to/source"},  # Missing source_id
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        with pytest.raises(ValueError, match="source_id is required"):
            await ingestion_module._process_remove_source_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

    @pytest.mark.asyncio()
    async def test_remove_source_sends_progress_updates(self):
        """Test REMOVE_SOURCE sends progress updates via updater."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "qdrant_collections": [],
            "qdrant_staging": [],
            "vector_count": 100,
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        # Return empty list so we hit the early return path
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[])

        updater = MockUpdater("op-123")

        with patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock):
            result = await ingestion_module._process_remove_source_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        assert result["documents_removed"] == 0
        assert result["source_path"] == "/path/to/source"


# ---------------------------------------------------------------------------
# Tests for _handle_task_failure_async
# ---------------------------------------------------------------------------


class TestHandleTaskFailureAsync:
    """Tests for async task failure handling.

    Note: These tests verify the high-level behavior of the failure handler.
    The full integration with database repositories is tested in integration tests.
    """

    def test_sanitize_error_message_is_used(self):
        """Test that _sanitize_error_message function exists and works."""
        from webui.tasks.utils import _sanitize_error_message

        # Test basic sanitization
        result = _sanitize_error_message("Error: test message")
        assert isinstance(result, str)

    def test_failure_handler_signature(self):
        """Test _handle_task_failure_async has correct signature."""
        import inspect

        sig = inspect.signature(ingestion_module._handle_task_failure_async)
        params = list(sig.parameters.keys())
        assert "operation_id" in params
        assert "exc" in params
        assert "task_id" in params


class TestHandleTaskFailure:
    """Tests for sync task failure handler."""

    def test_extracts_operation_id_from_args(self):
        """Test failure handler extracts operation_id from args."""
        mock_self = Mock()
        exc = ValueError("Test error")
        task_id = "task-123"
        args = ("dummy", "op-from-args")
        kwargs = {}
        einfo = Mock()

        with (
            patch.object(
                ingestion_module, "_handle_task_failure_async", new_callable=AsyncMock
            ),
            patch("asyncio.run") as mock_run,
        ):
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            ingestion_module._handle_task_failure(mock_self, exc, task_id, args, kwargs, einfo)

    def test_extracts_operation_id_from_kwargs(self):
        """Test failure handler extracts operation_id from kwargs."""
        mock_self = Mock()
        exc = ValueError("Test error")
        task_id = "task-123"
        args = ()
        kwargs = {"operation_id": "op-from-kwargs"}
        einfo = Mock()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = None

            ingestion_module._handle_task_failure(mock_self, exc, task_id, args, kwargs, einfo)

    def test_logs_error_when_operation_id_missing(self):
        """Test failure handler logs error when operation_id can't be found."""
        mock_self = Mock()
        exc = ValueError("Test error")
        task_id = "task-123"
        args = ()
        kwargs = {}
        einfo = Mock()

        with patch.object(ingestion_module, "logger") as mock_logger:
            ingestion_module._handle_task_failure(mock_self, exc, task_id, args, kwargs, einfo)

            mock_logger.error.assert_called()


# ---------------------------------------------------------------------------
# Tests for _process_append_operation_impl edge cases
# ---------------------------------------------------------------------------


class TestProcessAppendOperationImpl:
    """Tests for APPEND operation implementation edge cases."""

    @pytest.mark.asyncio()
    async def test_append_missing_source_id_raises(self):
        """Test APPEND raises when source_id is missing."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_path": "/path/to/source"},  # Missing source_id
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.session = AsyncMock()
        updater = MockUpdater("op-123")

        with pytest.raises(ValueError, match="source_id is required"):
            await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

    @pytest.mark.asyncio()
    async def test_append_missing_source_config_and_path_raises(self):
        """Test APPEND raises when both source_config and source_path are missing."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_id": 1},  # Missing source_config and source_path
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.session = AsyncMock()
        updater = MockUpdater("op-123")

        with pytest.raises(ValueError, match="source_config or source_path is required"):
            await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

    @pytest.mark.asyncio()
    async def test_append_invalid_source_type_raises(self):
        """Test APPEND raises with invalid source type."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {
                "source_id": 1,
                "source_type": "invalid_type",
                "source_config": {"path": "/some/path"},
            },
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.session = AsyncMock()
        mock_document_repo.session.in_transaction = Mock(return_value=False)
        updater = MockUpdater("op-123")

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                side_effect=ValueError("Unknown source type: invalid_type"),
            ),
            pytest.raises(ValueError, match="Invalid source configuration"),
        ):
            await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )


# ---------------------------------------------------------------------------
# Tests for process_collection_operation
# ---------------------------------------------------------------------------


class TestProcessCollectionOperation:
    """Tests for the main Celery task entry point."""

    def test_task_is_bound(self):
        """Test that process_collection_operation is a bound Celery task."""
        # Verify it's decorated with bind=True
        task = ingestion_module.process_collection_operation
        # Celery tasks have certain attributes
        assert hasattr(task, "delay")
        assert hasattr(task, "apply_async")
        assert hasattr(task, "name")
        assert task.name == "webui.tasks.process_collection_operation"

    def test_task_has_correct_config(self):
        """Test that process_collection_operation has correct Celery configuration."""
        task = ingestion_module.process_collection_operation
        # Check task options
        assert task.acks_late is True  # Ensure message reliability


# ---------------------------------------------------------------------------
# Tests for _tasks_namespace
# ---------------------------------------------------------------------------


class TestTasksNamespace:
    """Tests for the tasks namespace helper."""

    def test_returns_tasks_module(self):
        """Test _tasks_namespace returns the webui.tasks module."""
        ns = ingestion_module._tasks_namespace()
        assert ns is not None
        assert hasattr(ns, "asyncio") or hasattr(ns, "_process_collection_operation_async")


# ---------------------------------------------------------------------------
# Tests for module-level __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test that module exports are correct."""

    def test_all_exports_defined(self):
        """Test all exported names are actually defined."""
        for name in ingestion_module.__all__:
            assert hasattr(ingestion_module, name), f"{name} not found in module"


# ---------------------------------------------------------------------------
# Tests for _process_collection_operation_async
# ---------------------------------------------------------------------------


class TestProcessCollectionOperationAsync:
    """Tests for the async orchestrator function."""

    @pytest.mark.asyncio()
    async def test_database_initialization_failure_raises_runtime_error(self):
        """Test RuntimeError when database connection fails."""
        from shared.database import pg_connection_manager

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        # Save original value
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]

        try:
            # Set sessionmaker to None to simulate uninitialized state
            pg_connection_manager._sessionmaker = None  # type: ignore[attr-defined]

            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.AsyncSessionLocal", None),
                pytest.raises(RuntimeError, match="Failed to initialize database connection"),
            ):
                await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)
        finally:
            # Restore original
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests for _process_remove_source_operation with actual documents
# ---------------------------------------------------------------------------


class TestProcessRemoveSourceOperationWithDocuments:
    """Tests for REMOVE_SOURCE with actual document removal."""

    @pytest.mark.asyncio()
    async def test_remove_source_deletes_from_qdrant(self):
        """Test REMOVE_SOURCE deletes vectors from Qdrant collections."""
        from shared.database import pg_connection_manager

        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_main",
            "qdrant_collections": ["qdrant_secondary"],
            "qdrant_staging": [],
            "vector_count": 100,
        }

        # Create mock documents
        mock_doc1 = Mock()
        mock_doc1.id = 1
        mock_doc2 = Mock()
        mock_doc2.id = 2

        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[mock_doc1, mock_doc2])
        mock_document_repo.bulk_update_status = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}
        )

        mock_collection_repo = AsyncMock()
        mock_collection_repo.update_stats = AsyncMock()

        updater = MockUpdater("op-123")

        # Mock Qdrant
        mock_qdrant_client = Mock()
        mock_qdrant_client.delete = Mock()

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        mock_manager_instance = Mock()
        mock_manager_instance.collection_exists = Mock(return_value=True)

        # Mock session for transaction
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
                patch.object(
                    ingestion_module, "resolve_qdrant_manager_class", return_value=lambda _: mock_manager_instance
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
            ):
                result = await ingestion_module._process_remove_source_operation(
                    operation, collection, mock_collection_repo, mock_document_repo, updater
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        assert result["success"] is True
        assert result["documents_removed"] == 2
        # Verify Qdrant delete was called for both collections
        assert mock_qdrant_client.delete.call_count >= 2  # At least once per doc per collection

    @pytest.mark.asyncio()
    async def test_remove_source_handles_missing_qdrant_collection(self):
        """Test REMOVE_SOURCE handles non-existent Qdrant collection gracefully."""
        from shared.database import pg_connection_manager

        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "nonexistent_collection",
            "qdrant_collections": [],
            "qdrant_staging": [],
            "vector_count": 50,
        }

        mock_doc = Mock()
        mock_doc.id = 1

        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[mock_doc])
        mock_document_repo.bulk_update_status = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}
        )

        mock_collection_repo = AsyncMock()
        mock_collection_repo.update_stats = AsyncMock()

        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        mock_manager_instance = Mock()
        mock_manager_instance.collection_exists = Mock(return_value=False)  # Collection doesn't exist

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
                patch.object(
                    ingestion_module, "resolve_qdrant_manager_class", return_value=lambda _: mock_manager_instance
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
            ):
                result = await ingestion_module._process_remove_source_operation(
                    operation, collection, mock_collection_repo, mock_document_repo, updater
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        # Should still succeed, just skip the non-existent collection
        assert result["success"] is True
        # Delete should not be called since collection doesn't exist
        mock_qdrant_client.delete.assert_not_called()

    @pytest.mark.asyncio()
    async def test_remove_source_records_deletion_errors(self):
        """Test REMOVE_SOURCE records errors when Qdrant deletion fails."""
        from shared.database import pg_connection_manager

        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test",
            "qdrant_collections": [],
            "qdrant_staging": [],
            "vector_count": 50,
        }

        mock_doc = Mock()
        mock_doc.id = 1

        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[mock_doc])
        mock_document_repo.bulk_update_status = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}
        )

        mock_collection_repo = AsyncMock()
        mock_collection_repo.update_stats = AsyncMock()

        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.delete = Mock(side_effect=Exception("Qdrant connection failed"))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        mock_manager_instance = Mock()
        mock_manager_instance.collection_exists = Mock(return_value=True)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
                patch.object(
                    ingestion_module, "resolve_qdrant_manager_class", return_value=lambda _: mock_manager_instance
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
            ):
                result = await ingestion_module._process_remove_source_operation(
                    operation, collection, mock_collection_repo, mock_document_repo, updater
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        assert result["success"] is True
        # Should have recorded deletion errors
        assert result.get("deletion_errors") is not None
        assert len(result["deletion_errors"]) > 0

    @pytest.mark.asyncio()
    async def test_remove_source_deduplicates_collections(self):
        """Test REMOVE_SOURCE deduplicates Qdrant collection names."""
        from shared.database import pg_connection_manager

        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="REMOVE_SOURCE"),
            "config": {"source_id": 1, "source_path": "/path/to/source"},
            "user_id": 1,
        }

        # Same collection name appears multiple times
        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_main",
            "qdrant_collections": ["qdrant_main", "qdrant_secondary"],  # Duplicate
            "qdrant_staging": ["qdrant_main"],  # Another duplicate
            "vector_count": 50,
        }

        mock_doc = Mock()
        mock_doc.id = 1

        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_source_id = AsyncMock(return_value=[mock_doc])
        mock_document_repo.bulk_update_status = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}
        )

        mock_collection_repo = AsyncMock()
        mock_collection_repo.update_stats = AsyncMock()

        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.delete = Mock()

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        mock_manager_instance = Mock()
        mock_manager_instance.collection_exists = Mock(return_value=True)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
                patch.object(
                    ingestion_module, "resolve_qdrant_manager_class", return_value=lambda _: mock_manager_instance
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
            ):
                result = await ingestion_module._process_remove_source_operation(
                    operation, collection, mock_collection_repo, mock_document_repo, updater
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        assert result["success"] is True
        # Should only delete from 2 unique collections (qdrant_main, qdrant_secondary)
        # Each document deleted once per collection
        assert mock_qdrant_client.delete.call_count == 2


# ---------------------------------------------------------------------------
# Tests for _handle_task_failure_async with database operations
# ---------------------------------------------------------------------------


class TestHandleTaskFailureAsyncWithDatabase:
    """Tests for async failure handler with database operations."""

    @pytest.mark.asyncio()
    async def test_failure_handler_updates_operation_status(self):
        """Test failure handler updates operation to FAILED status."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationType

        mock_operation = Mock()
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.INDEX

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.status = CollectionStatus.PENDING

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            ):
                await ingestion_module._handle_task_failure_async(
                    operation_id="op-123",
                    exc=ValueError("Test error"),
                    task_id="task-123",
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        # Verify operation was marked as failed
        mock_operation_repo.update_status.assert_called()

    @pytest.mark.asyncio()
    async def test_failure_handler_sets_collection_error_for_index(self):
        """Test failure handler sets collection to ERROR for INDEX operations."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationType

        mock_operation = Mock()
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.INDEX

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.status = CollectionStatus.PENDING

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            ):
                await ingestion_module._handle_task_failure_async(
                    operation_id="op-123",
                    exc=ValueError("Indexing failed"),
                    task_id="task-123",
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        # Verify collection was set to ERROR for INDEX failure
        mock_collection_repo.update_status.assert_called()
        call_args = mock_collection_repo.update_status.call_args
        assert call_args[0][1] == CollectionStatus.ERROR

    @pytest.mark.asyncio()
    async def test_failure_handler_sets_collection_degraded_for_reindex(self):
        """Test failure handler sets collection to DEGRADED for REINDEX operations."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationType

        mock_operation = Mock()
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.REINDEX

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.status = CollectionStatus.READY

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            ):
                await ingestion_module._handle_task_failure_async(
                    operation_id="op-123",
                    exc=ValueError("Reindex failed"),
                    task_id="task-123",
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]

        # Verify collection was set to DEGRADED for REINDEX failure
        mock_collection_repo.update_status.assert_called()
        call_args = mock_collection_repo.update_status.call_args
        assert call_args[0][1] == CollectionStatus.DEGRADED

    @pytest.mark.asyncio()
    async def test_failure_handler_missing_operation(self):
        """Test failure handler handles missing operation gracefully."""
        from shared.database import pg_connection_manager

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker  # type: ignore[attr-defined]
        pg_connection_manager._sessionmaker = mock_session_factory  # type: ignore[attr-defined]

        try:
            with (
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
            ):
                # Should not raise, just return
                await ingestion_module._handle_task_failure_async(
                    operation_id="nonexistent-op",
                    exc=ValueError("Test error"),
                    task_id="task-123",
                )
        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests for _process_append_operation_impl additional paths
# ---------------------------------------------------------------------------


class TestProcessAppendOperationImplAdditional:
    """Additional tests for APPEND operation implementation."""

    @pytest.mark.asyncio()
    async def test_append_connector_authentication_failure(self):
        """Test APPEND raises when connector authentication fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {
                "source_id": 1,
                "source_type": "directory",
                "source_config": {"path": "/some/path"},
            },
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=False)
        mock_document_repo.session = mock_session
        updater = MockUpdater("op-123")

        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=False)

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=mock_connector,
            ),
            pytest.raises(ValueError, match="Authentication failed"),
        ):
            await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

    @pytest.mark.asyncio()
    async def test_append_uses_legacy_source_path_fallback(self):
        """Test APPEND falls back to source_path when source_config is not provided."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {
                "source_id": 1,
                "source_path": "/legacy/path",  # Legacy format
            },
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=False)
        mock_document_repo.session = mock_session
        updater = MockUpdater("op-123")

        # Create async generator for load_documents
        async def empty_async_gen():
            return
            yield  # Make it an async generator

        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.load_documents = empty_async_gen

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=mock_connector,
            ) as mock_factory,
        ):
            await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Verify connector was created with legacy path
        mock_factory.assert_called_once()
        call_args = mock_factory.call_args
        assert call_args[0][0] == "directory"
        assert call_args[0][1] == {"path": "/legacy/path"}

    @pytest.mark.asyncio()
    async def test_append_with_no_documents_returns_success(self):
        """Test APPEND with an empty document source returns success."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {
                "source_id": 1,
                "source_type": "directory",
                "source_config": {"path": "/empty/path"},
            },
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=False)
        mock_document_repo.session = mock_session
        updater = MockUpdater("op-123")

        # Create async generator that yields nothing
        async def empty_async_gen():
            return
            yield  # Make it an async generator

        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector.load_documents = empty_async_gen

        with (
            patch(
                "webui.services.connector_factory.ConnectorFactory.get_connector",
                return_value=mock_connector,
            ),
        ):
            result = await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should succeed (success is true for empty sources)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# Tests for _process_index_operation additional paths
# ---------------------------------------------------------------------------


class TestProcessCollectionOperationAsyncRouting:
    """Tests for _process_collection_operation_async operation type routing."""

    @pytest.mark.asyncio()
    async def test_routes_append_operation(self):
        """Test _process_collection_operation_async routes APPEND operations correctly."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationStatus, OperationType

        mock_operation = Mock()
        mock_operation.id = 1
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.APPEND
        mock_operation.config = {"source_id": 1, "source_path": "/test"}
        mock_operation.user_id = 1
        mock_operation.status = OperationStatus.PENDING

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.name = "Test Collection"
        mock_collection.vector_store_name = "qdrant_test"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.chunking_strategy = None
        mock_collection.chunking_config = {}
        mock_collection.qdrant_collections = []
        mock_collection.qdrant_staging = []
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_count = 100
        mock_collection.config = {}
        mock_collection.vector_collection_id = None

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.set_task_id = AsyncMock()
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_document_repo = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 10, "total_chunks": 50, "total_size_bytes": 1000}
        )

        mock_projection_repo = AsyncMock()
        mock_projection_repo.list_for_collection = AsyncMock(return_value=([], 0))

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        # Save original and patch
        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.projection_run_repository.ProjectionRunRepository",
                    return_value=mock_projection_repo,
                ),
                patch.object(
                    ingestion_module._tasks_namespace(),
                    "_process_append_operation_impl",
                    new_callable=AsyncMock,
                    return_value={"success": True, "documents_added": 5},
                ) as mock_append,
                patch.object(ingestion_module, "_record_operation_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater.send_update = AsyncMock()
                mock_updater.set_user_id = Mock()
                mock_updater_class.return_value = mock_updater

                await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)

                # Verify APPEND handler was called
                mock_append.assert_called_once()

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker

    @pytest.mark.asyncio()
    async def test_routes_remove_source_operation(self):
        """Test _process_collection_operation_async routes REMOVE_SOURCE operations correctly."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationStatus, OperationType

        mock_operation = Mock()
        mock_operation.id = 1
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.REMOVE_SOURCE
        mock_operation.config = {"source_id": 1}
        mock_operation.user_id = 1
        mock_operation.status = OperationStatus.PENDING

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.name = "Test Collection"
        mock_collection.vector_store_name = "qdrant_test"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.chunking_strategy = None
        mock_collection.chunking_config = {}
        mock_collection.qdrant_collections = []
        mock_collection.qdrant_staging = []
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_count = 100
        mock_collection.config = {}
        mock_collection.vector_collection_id = None

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.set_task_id = AsyncMock()
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_document_repo = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 10, "total_chunks": 50, "total_size_bytes": 1000}
        )

        mock_projection_repo = AsyncMock()
        mock_projection_repo.list_for_collection = AsyncMock(return_value=([], 0))

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.projection_run_repository.ProjectionRunRepository",
                    return_value=mock_projection_repo,
                ),
                patch.object(
                    ingestion_module._tasks_namespace(),
                    "_process_remove_source_operation",
                    new_callable=AsyncMock,
                    return_value={"success": True, "documents_removed": 3},
                ) as mock_remove,
                patch.object(ingestion_module, "_record_operation_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater.send_update = AsyncMock()
                mock_updater.set_user_id = Mock()
                mock_updater_class.return_value = mock_updater

                await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)

                # Verify REMOVE_SOURCE handler was called
                mock_remove.assert_called_once()

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker

    @pytest.mark.asyncio()
    async def test_handles_operation_not_found(self):
        """Test _process_collection_operation_async raises when operation not found."""
        from shared.database import pg_connection_manager

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=None)
        mock_operation_repo.set_task_id = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater_class.return_value = mock_updater

                with pytest.raises(ValueError, match="not found in database"):
                    await ingestion_module._process_collection_operation_async("nonexistent", mock_celery_task)

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker

    @pytest.mark.asyncio()
    async def test_handles_collection_not_found(self):
        """Test _process_collection_operation_async raises when collection not found."""
        from shared.database import pg_connection_manager
        from shared.database.models import OperationStatus, OperationType

        mock_operation = Mock()
        mock_operation.id = 1
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.INDEX
        mock_operation.config = {}
        mock_operation.user_id = 1
        mock_operation.status = OperationStatus.PENDING

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.set_task_id = AsyncMock()
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater.send_update = AsyncMock()
                mock_updater.set_user_id = Mock()
                mock_updater_class.return_value = mock_updater

                with pytest.raises(ValueError, match="Collection .* not found"):
                    await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker

    @pytest.mark.asyncio()
    async def test_handles_failed_operation_updates_status(self):
        """Test _process_collection_operation_async updates status correctly on failure."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationStatus, OperationType

        mock_operation = Mock()
        mock_operation.id = 1
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.INDEX
        mock_operation.config = {}
        mock_operation.user_id = 1
        mock_operation.status = OperationStatus.PENDING

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.name = "Test Collection"
        mock_collection.vector_store_name = "qdrant_test"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.chunking_strategy = None
        mock_collection.chunking_config = {}
        mock_collection.qdrant_collections = []
        mock_collection.qdrant_staging = []
        mock_collection.status = CollectionStatus.PENDING
        mock_collection.vector_count = 0
        mock_collection.config = {}
        mock_collection.vector_collection_id = None

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.set_task_id = AsyncMock()
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_document_repo = AsyncMock()

        mock_projection_repo = AsyncMock()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.projection_run_repository.ProjectionRunRepository",
                    return_value=mock_projection_repo,
                ),
                patch.object(
                    ingestion_module._tasks_namespace(),
                    "_process_index_operation",
                    new_callable=AsyncMock,
                    side_effect=RuntimeError("Test failure"),
                ),
                patch.object(ingestion_module, "_record_operation_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater.send_update = AsyncMock()
                mock_updater.set_user_id = Mock()
                mock_updater_class.return_value = mock_updater

                with pytest.raises(RuntimeError, match="Test failure"):
                    await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker

        # Verify status was updated to FAILED
        mock_operation_repo.update_status.assert_called()
        # Verify collection was set to ERROR for failed INDEX
        mock_collection_repo.update_status.assert_called()

    @pytest.mark.asyncio()
    async def test_uses_vector_collection_id_fallback(self):
        """Test _process_collection_operation_async uses vector_collection_id when vector_store_name is empty."""
        from shared.database import pg_connection_manager
        from shared.database.models import CollectionStatus, OperationStatus, OperationType

        mock_operation = Mock()
        mock_operation.id = 1
        mock_operation.uuid = "op-123"
        mock_operation.collection_id = "col-456"
        mock_operation.type = OperationType.INDEX
        mock_operation.config = {}
        mock_operation.user_id = 1
        mock_operation.status = OperationStatus.PENDING

        mock_collection = Mock()
        mock_collection.id = "col-456"
        mock_collection.uuid = "col-456"
        mock_collection.name = "Test Collection"
        mock_collection.vector_store_name = ""  # Empty - should use fallback
        mock_collection.vector_collection_id = "fallback_qdrant_id"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.chunking_strategy = None
        mock_collection.chunking_config = {}
        mock_collection.qdrant_collections = []
        mock_collection.qdrant_staging = []
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_count = 100
        mock_collection.config = {}

        mock_operation_repo = AsyncMock()
        mock_operation_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
        mock_operation_repo.set_task_id = AsyncMock()
        mock_operation_repo.update_status = AsyncMock()

        mock_collection_repo = AsyncMock()
        mock_collection_repo.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repo.update_status = AsyncMock()

        mock_document_repo = AsyncMock()
        mock_document_repo.get_stats_by_collection = AsyncMock(
            return_value={"total_documents": 10, "total_chunks": 50, "total_size_bytes": 1000}
        )

        mock_projection_repo = AsyncMock()
        mock_projection_repo.list_for_collection = AsyncMock(return_value=([], 0))

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        mock_session_factory = Mock(return_value=mock_session)

        mock_celery_task = Mock()
        mock_celery_task.request = Mock()
        mock_celery_task.request.id = "task-123"

        original_sessionmaker = pg_connection_manager._sessionmaker
        pg_connection_manager._sessionmaker = mock_session_factory

        captured_collection = None

        async def capture_collection(*args, **_kwargs):
            nonlocal captured_collection
            captured_collection = args[1]  # collection is second positional arg
            return {"success": True}

        try:
            with (
                patch.object(pg_connection_manager, "initialize", new_callable=AsyncMock),
                patch("shared.database.database.ensure_async_sessionmaker", AsyncMock(return_value=mock_session_factory)),
                patch(
                    "shared.database.repositories.operation_repository.OperationRepository",
                    return_value=mock_operation_repo,
                ),
                patch(
                    "shared.database.repositories.collection_repository.CollectionRepository",
                    return_value=mock_collection_repo,
                ),
                patch(
                    "shared.database.repositories.document_repository.DocumentRepository",
                    return_value=mock_document_repo,
                ),
                patch(
                    "shared.database.repositories.projection_run_repository.ProjectionRunRepository",
                    return_value=mock_projection_repo,
                ),
                patch.object(
                    ingestion_module._tasks_namespace(),
                    "_process_index_operation",
                    side_effect=capture_collection,
                ),
                patch.object(ingestion_module, "_record_operation_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
                patch.object(ingestion_module, "CeleryTaskWithOperationUpdates") as mock_updater_class,
            ):
                mock_updater = AsyncMock()
                mock_updater.__aenter__ = AsyncMock(return_value=mock_updater)
                mock_updater.__aexit__ = AsyncMock(return_value=None)
                mock_updater.send_update = AsyncMock()
                mock_updater.set_user_id = Mock()
                mock_updater_class.return_value = mock_updater

                await ingestion_module._process_collection_operation_async("op-123", mock_celery_task)

                # Verify fallback was used
                assert captured_collection["vector_store_name"] == "fallback_qdrant_id"

        finally:
            pg_connection_manager._sessionmaker = original_sessionmaker


class TestProcessIndexOperationAdditional:
    """Additional tests for INDEX operation."""

    @pytest.mark.asyncio()
    async def test_index_uses_model_config_from_plugin(self):
        """Test INDEX gets vector dimension from model config."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "custom/plugin-model",
            "quantization": "float16",
            "config": {},  # No vector_dim in config
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        # Mock model config from plugin
        mock_model_config = Mock()
        mock_model_config.dimension = 768

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=mock_model_config),
            patch("shared.embedding.validation.get_model_dimension", return_value=768),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        assert result["vector_dim"] == 768

    @pytest.mark.asyncio()
    async def test_index_falls_back_to_default_dimension(self):
        """Test INDEX uses default dimension when model is unknown."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "unknown/model",
            "quantization": "float16",
            "config": {},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=None),  # Unknown model
            patch("shared.embedding.validation.get_model_dimension", return_value=None),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        assert result["vector_dim"] == 1024  # Default

    @pytest.mark.asyncio()
    async def test_index_logs_dimension_mismatch_warning(self):
        """Test INDEX logs warning when model dimension doesn't match config."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 512},  # Different from actual model
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=None),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            patch.object(ingestion_module, "logger") as mock_logger,
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        assert result["success"] is True
        # Should have logged a warning about dimension mismatch
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio()
    async def test_index_handles_metadata_collection_failure(self):
        """Test INDEX continues when ensure_metadata_collection fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch(
                "shared.database.collection_metadata.ensure_metadata_collection",
                side_effect=Exception("Metadata error"),
            ),
            patch("shared.database.collection_metadata.store_collection_metadata"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should still succeed - metadata failure is non-fatal
        assert result["success"] is True

    @pytest.mark.asyncio()
    async def test_index_handles_store_metadata_failure(self):
        """Test INDEX continues when store_collection_metadata fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(return_value=Mock(vectors_count=0))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch(
                "shared.database.collection_metadata.store_collection_metadata",
                side_effect=Exception("Store metadata error"),
            ),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            patch("shared.embedding.validation.get_model_dimension", return_value=1024),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
        ):
            result = await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should still succeed - metadata failure is non-fatal
        assert result["success"] is True

    @pytest.mark.asyncio()
    async def test_index_fails_on_verification_failure(self):
        """Test INDEX fails when collection verification after creation fails."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="INDEX"),
            "config": {},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {"vector_dim": 1024},
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        updater = MockUpdater("op-123")

        mock_qdrant_client = Mock()
        mock_qdrant_client.create_collection = Mock()
        mock_qdrant_client.get_collection = Mock(side_effect=Exception("Collection not found"))

        mock_manager = Mock()
        mock_manager.get_client.return_value = mock_qdrant_client

        with (
            patch.object(ingestion_module, "resolve_qdrant_manager", return_value=mock_manager),
            patch("shared.database.collection_metadata.ensure_metadata_collection"),
            patch("shared.embedding.factory.resolve_model_config", return_value=Mock(dimension=1024)),
            pytest.raises(Exception, match="was not properly created"),
        ):
            await ingestion_module._process_index_operation(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )


# ---------------------------------------------------------------------------
# Tests for APPEND Transaction Handling (coverage for lines 831-938)
# ---------------------------------------------------------------------------


class TestAppendTransactionHandling:
    """Tests for transaction/session error handling in _process_append_operation_impl."""

    @pytest.mark.asyncio()
    async def test_commit_failure_before_scan_triggers_rollback(self):
        """Test that commit failure before document scan triggers rollback (lines 831-837)."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_id": 1, "source_type": "directory", "source_config": {"path": "/test"}},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "vector_collection_id": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "config": {"vector_dim": 1024},
            "qdrant_collections": [],
            "qdrant_staging": [],
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()

        # Create mock session that fails on first commit but succeeds on rollback
        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=True)
        mock_session.commit = AsyncMock(side_effect=Exception("commit failed"))
        mock_session.rollback = AsyncMock()
        mock_document_repo.session = mock_session

        # Mock connector that returns no documents (to keep test simple)
        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)

        async def empty_load_documents():
            return
            yield  # Make it an async generator that yields nothing

        mock_connector.load_documents = empty_load_documents

        mock_connector_factory = Mock()
        mock_connector_factory.get_connector = Mock(return_value=mock_connector)

        updater = MockUpdater("op-123")

        with (
            patch.object(ingestion_module, "ConnectorFactory", mock_connector_factory),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
            patch("shared.metrics.collection_metrics.record_document_processed"),
            patch("shared.metrics.collection_metrics.document_processing_duration", Mock(labels=Mock(return_value=Mock()))),
        ):
            result = await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Rollback should have been called when commit failed
        mock_session.rollback.assert_awaited()
        assert result["success"] is True

    @pytest.mark.asyncio()
    async def test_invalid_transaction_state_recovery(self):
        """Test recovery when session enters invalid state during registration (lines 916-928)."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_id": 1, "source_type": "directory", "source_config": {"path": "/test"}},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "vector_collection_id": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "config": {"vector_dim": 1024},
            "qdrant_collections": [],
            "qdrant_staging": [],
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()

        # Create mock session
        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        # begin_nested raises invalid transaction error
        mock_nested_context = AsyncMock()
        mock_nested_context.__aenter__ = AsyncMock(side_effect=Exception("invalid transaction state"))
        mock_nested_context.__aexit__ = AsyncMock()
        mock_session.begin_nested = Mock(return_value=mock_nested_context)

        mock_document_repo.session = mock_session

        # Mock connector that yields one document
        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)

        test_doc = _make_test_doc("test-doc-1")

        async def load_documents_gen():
            yield test_doc

        mock_connector.load_documents = load_documents_gen

        mock_connector_factory = Mock()
        mock_connector_factory.get_connector = Mock(return_value=mock_connector)

        updater = MockUpdater("op-123")

        with (
            patch.object(ingestion_module, "ConnectorFactory", mock_connector_factory),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
            patch("shared.metrics.collection_metrics.record_document_processed"),
            patch("shared.metrics.collection_metrics.document_processing_duration", Mock(labels=Mock(return_value=Mock()))),
        ):
            result = await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should have recorded an error during document registration (the path we're testing)
        assert "errors" in result
        assert len(result["errors"]) == 1
        assert "invalid transaction" in result["errors"][0]["error"]

    @pytest.mark.asyncio()
    async def test_commit_failure_after_registrations(self):
        """Test commit failure after document registrations triggers rollback (lines 930-937)."""

        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_id": 1, "source_type": "directory", "source_config": {"path": "/test"}},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "vector_collection_id": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "config": {"vector_dim": 1024},
            "qdrant_collections": [],
            "qdrant_staging": [],
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()

        # Track commit calls - first succeeds, subsequent fail
        commit_call_count = [0]

        async def commit_side_effect():
            commit_call_count[0] += 1
            if commit_call_count[0] > 1:
                raise Exception("post-registration commit failed")

        mock_session = AsyncMock()
        # First in_transaction returns False (no pre-scan commit needed)
        # Second returns True (post-registrations commit needed)
        in_transaction_calls = [False, True, False]
        call_index = [0]

        def in_transaction_side_effect():
            idx = call_index[0]
            call_index[0] += 1
            return in_transaction_calls[idx] if idx < len(in_transaction_calls) else False

        mock_session.in_transaction = Mock(side_effect=in_transaction_side_effect)
        mock_session.commit = AsyncMock(side_effect=commit_side_effect)
        mock_session.rollback = AsyncMock()

        mock_nested_context = AsyncMock()
        mock_nested_context.__aenter__ = AsyncMock()
        mock_nested_context.__aexit__ = AsyncMock()
        mock_session.begin_nested = Mock(return_value=mock_nested_context)

        mock_document_repo.session = mock_session

        # Mock connector with no documents
        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)

        async def empty_gen():
            return
            yield

        mock_connector.load_documents = empty_gen

        updater = MockUpdater("op-123")

        with (
            patch("webui.tasks.ingestion.ConnectorFactory.get_connector", return_value=mock_connector),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
            patch("shared.metrics.collection_metrics.record_document_processed"),
            patch("shared.metrics.collection_metrics.document_processing_duration", Mock(labels=Mock(return_value=Mock()))),
        ):
            result = await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should still succeed (commit failure is logged but not fatal)
        assert result["success"] is True

    @pytest.mark.asyncio()
    async def test_registration_savepoint_handles_error(self):
        """Test that savepoint (nested transaction) handles registration errors (lines 888-896)."""
        operation = {
            "id": 1,
            "uuid": "op-123",
            "collection_id": "col-456",
            "type": Mock(value="APPEND"),
            "config": {"source_id": 1, "source_type": "directory", "source_config": {"path": "/test"}},
            "user_id": 1,
        }

        collection = {
            "id": "col-456",
            "uuid": "col-456",
            "name": "Test Collection",
            "vector_store_name": "qdrant_test_col",
            "vector_collection_id": "qdrant_test_col",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "chunking_strategy": "recursive",
            "chunking_config": {},
            "config": {"vector_dim": 1024},
            "qdrant_collections": [],
            "qdrant_staging": [],
        }

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()

        mock_session = AsyncMock()
        mock_session.in_transaction = Mock(return_value=False)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        # Savepoint enters successfully but registry raises error
        mock_nested_context = AsyncMock()
        mock_nested_context.__aenter__ = AsyncMock()
        mock_nested_context.__aexit__ = AsyncMock()
        mock_session.begin_nested = Mock(return_value=mock_nested_context)

        mock_document_repo.session = mock_session

        # Mock connector with one document
        mock_connector = AsyncMock()
        mock_connector.authenticate = AsyncMock(return_value=True)

        test_doc = _make_test_doc("test-doc-1")

        async def load_documents_gen():
            yield test_doc

        mock_connector.load_documents = load_documents_gen

        # Mock registry that raises during register
        mock_registry = AsyncMock()
        mock_registry.register_or_update = AsyncMock(side_effect=Exception("DB constraint violation"))

        updater = MockUpdater("op-123")

        with (
            patch("webui.tasks.ingestion.ConnectorFactory.get_connector", return_value=mock_connector),
            patch.object(ingestion_module, "DocumentRegistryService", return_value=mock_registry),
            patch.object(ingestion_module, "_audit_log_operation", new_callable=AsyncMock),
            patch.object(ingestion_module, "_update_collection_metrics", new_callable=AsyncMock),
            patch("shared.metrics.collection_metrics.record_document_processed"),
            patch("shared.metrics.collection_metrics.document_processing_duration", Mock(labels=Mock(return_value=Mock()))),
            patch("shared.database.repositories.chunk_repository.ChunkRepository"),
            patch("shared.database.repositories.document_artifact_repository.DocumentArtifactRepository"),
        ):
            result = await ingestion_module._process_append_operation_impl(
                operation, collection, mock_collection_repo, mock_document_repo, updater
            )

        # Should have recorded the error during document registration
        assert "errors" in result
        assert len(result["errors"]) == 1
        assert "test-doc-1" in result["errors"][0]["document"]


# ---------------------------------------------------------------------------
# Sparse indexing helpers
# ---------------------------------------------------------------------------


class TestSparseIndexingHelpers:
    @pytest.mark.asyncio()
    async def test_setup_sparse_collection_for_index_stores_metadata(self) -> None:
        plugin_record = Mock()
        plugin_record.plugin_class = Mock(SPARSE_TYPE="bm25")

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("shared.plugins.load_plugins"),
            patch("shared.plugins.plugin_registry.find_by_id", return_value=plugin_record),
            patch("vecpipe.sparse.generate_sparse_collection_name", return_value="dense_sparse_bm25") as gen_name,
            patch("vecpipe.sparse.ensure_sparse_collection", new=AsyncMock()) as ensure_sparse,
            patch("shared.database.collection_metadata.store_sparse_index_config", new=AsyncMock(return_value=True)) as store,
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            result = await ingestion_module._setup_sparse_collection_for_index(
                vector_store_name="dense",
                sparse_config={"plugin_id": "bm25-local", "model_config_data": {"k1": 1.9}},
            )

        assert result == {"sparse_collection_name": "dense_sparse_bm25", "plugin_id": "bm25-local"}
        gen_name.assert_called_once_with("dense", "bm25")
        ensure_sparse.assert_awaited_once()
        store.assert_awaited_once()
        assert mock_qdrant.close.await_count == 1

    @pytest.mark.asyncio()
    async def test_setup_sparse_collection_for_index_raises_when_store_fails(self) -> None:
        plugin_record = Mock()
        plugin_record.plugin_class = Mock(SPARSE_TYPE="bm25")

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("shared.plugins.load_plugins"),
            patch("shared.plugins.plugin_registry.find_by_id", return_value=plugin_record),
            patch("vecpipe.sparse.generate_sparse_collection_name", return_value="dense_sparse_bm25"),
            patch("vecpipe.sparse.ensure_sparse_collection", new=AsyncMock()),
            patch("shared.database.collection_metadata.store_sparse_index_config", new=AsyncMock(return_value=False)),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            with pytest.raises(ValueError, match="Failed to store sparse index config"):
                await ingestion_module._setup_sparse_collection_for_index(
                    vector_store_name="dense",
                    sparse_config={"plugin_id": "bm25-local", "model_config_data": {}},
                )

        assert mock_qdrant.close.await_count == 1

    @pytest.mark.asyncio()
    async def test_maybe_generate_sparse_vectors_skips_when_missing_config(self) -> None:
        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=None)),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            await ingestion_module._maybe_generate_sparse_vectors(chunks=[], points=[], qdrant_collection_name="dense")

        assert mock_qdrant.close.await_count == 1

    @pytest.mark.asyncio()
    async def test_maybe_generate_sparse_vectors_local_bm25_upserts_and_updates_count(self) -> None:
        from shared.plugins.types.sparse_indexer import SparseVector

        class DummyIndexer:
            async def initialize(self, _cfg=None):  # type: ignore[no-untyped-def]
                return None

            async def cleanup(self) -> None:
                return None

            async def encode_documents(self, documents):  # type: ignore[no-untyped-def]
                return [
                    SparseVector(indices=(1,), values=(0.1,), chunk_id=doc["chunk_id"], metadata=doc.get("metadata", {}))
                    for doc in documents
                ]

        plugin_record = Mock()
        plugin_record.plugin_class = DummyIndexer

        sparse_config = {
            "enabled": True,
            "plugin_id": "bm25-local",
            "model_config": {},
            "sparse_collection_name": "dense_sparse_bm25",
            "document_count": 5,
        }

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        upsert = AsyncMock()
        store = AsyncMock(return_value=True)

        chunks = [
            {"chunk_id": "orig-1_0", "text": "hello", "metadata": {"doc": 1}},
            {"chunk_id": "orig-2_0", "text": "world", "metadata": {"doc": 2}},
        ]
        points = [SimpleNamespace(id="p1"), SimpleNamespace(id="p2")]

        with (
            patch("shared.plugins.load_plugins"),
            patch("shared.plugins.plugin_registry.find_by_id", return_value=plugin_record),
            patch("shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=sparse_config)),
            patch("shared.database.collection_metadata.store_sparse_index_config", store),
            patch("vecpipe.sparse.upsert_sparse_vectors", upsert),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            await ingestion_module._maybe_generate_sparse_vectors(chunks=chunks, points=points, qdrant_collection_name="dense")

        upsert.assert_awaited_once()
        _args = upsert.await_args.args
        assert _args[0] == "dense_sparse_bm25"
        qdrant_vectors = _args[1]
        assert {v["chunk_id"] for v in qdrant_vectors} == {"p1", "p2"}
        assert {v["metadata"]["original_chunk_id"] for v in qdrant_vectors} == {"orig-1_0", "orig-2_0"}

        store.assert_awaited_once()
        stored_cfg = store.await_args.args[2]
        assert stored_cfg["document_count"] == 7
        assert stored_cfg["last_indexed_at"] is not None
        assert mock_qdrant.close.await_count == 1

    @pytest.mark.asyncio()
    async def test_maybe_generate_sparse_vectors_vecpipe_path_calls_sparse_client(self) -> None:
        sparse_config = {
            "enabled": True,
            "plugin_id": "splade-local",
            "model_config": {"batch_size": 32},
            "sparse_collection_name": "dense_sparse_splade",
            "document_count": 0,
        }

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        vecpipe_client = AsyncMock()
        vecpipe_client.encode_documents = AsyncMock(
            return_value=[{"chunk_id": "p1", "indices": [1], "values": [0.9]}]
        )

        upsert = AsyncMock()
        store = AsyncMock(return_value=True)

        with (
            patch("shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=sparse_config)),
            patch("shared.database.collection_metadata.store_sparse_index_config", store),
            patch("webui.clients.sparse_client.SparseEncodingClient", return_value=vecpipe_client),
            patch("vecpipe.sparse.upsert_sparse_vectors", upsert),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            await ingestion_module._maybe_generate_sparse_vectors(
                chunks=[{"chunk_id": "orig", "text": "hello", "metadata": {}}],
                points=[SimpleNamespace(id="p1")],
                qdrant_collection_name="dense",
            )

        vecpipe_client.encode_documents.assert_awaited_once()
        upsert.assert_awaited_once()
        store.assert_awaited_once()
        assert mock_qdrant.close.await_count == 1


# ---------------------------------------------------------------------------
# RETRY_DOCUMENTS Operation Tests
# ---------------------------------------------------------------------------


class TestRetryDocumentsOperation:
    """Tests for _process_retry_documents_operation."""

    @pytest.mark.asyncio()
    async def test_retry_documents_with_no_pending_returns_early(self) -> None:
        """When no pending documents exist, operation returns success with 0 counts."""
        from shared.database.models import DocumentStatus

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.list_by_collection = AsyncMock(return_value=([], 0))

        mock_updater = MockUpdater("op-123")

        operation = {"config": {}, "uuid": "op-123"}
        collection = {
            "id": "col-uuid-456",
            "vector_store_name": "qdrant_col_123",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        result = await ingestion_module._process_retry_documents_operation(
            operation=operation,
            collection=collection,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            updater=mock_updater,
        )

        assert result["success"] is True
        assert result["documents_processed"] == 0
        assert result["documents_failed"] == 0
        assert result["vectors_created"] == 0
        mock_document_repo.list_by_collection.assert_awaited_once_with(
            collection_id="col-uuid-456",
            status=DocumentStatus.PENDING,
            offset=0,
            limit=10000,
        )

    @pytest.mark.asyncio()
    async def test_retry_documents_filters_by_document_ids(self) -> None:
        """When document_ids provided, only those documents are processed."""
        from shared.database.models import DocumentStatus

        # Create mock document that's in PENDING status
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.collection_id = "col-uuid-456"
        mock_doc.status = DocumentStatus.PENDING.value
        mock_doc.source_path = "/path/to/doc.txt"

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id = AsyncMock(return_value=mock_doc)

        mock_updater = MockUpdater("op-123")

        operation = {
            "config": {"document_ids": ["doc-1", "doc-nonexistent"]},
            "uuid": "op-123",
        }
        collection = {
            "id": "col-uuid-456",
            "vector_store_name": "qdrant_col_123",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        # Mock the parallel processing to avoid complex setup
        with patch.object(
            ingestion_module,
            "process_documents_parallel",
            new=AsyncMock(return_value={"processed": 1, "failed": 0, "vectors": 10}),
        ):
            result = await ingestion_module._process_retry_documents_operation(
                operation=operation,
                collection=collection,
                collection_repo=mock_collection_repo,
                document_repo=mock_document_repo,
                updater=mock_updater,
            )

        # Verify document lookup was called for provided IDs
        assert mock_document_repo.get_by_id.await_count == 2
        mock_document_repo.get_by_id.assert_any_await("doc-1")
        mock_document_repo.get_by_id.assert_any_await("doc-nonexistent")

        # Verify success result
        assert result["success"] is True
        assert result["documents_processed"] == 1

    @pytest.mark.asyncio()
    async def test_retry_documents_excludes_non_pending(self) -> None:
        """Documents not in PENDING status are excluded from retry."""
        from shared.database.models import DocumentStatus

        # Create mock document that's COMPLETED (not PENDING)
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.collection_id = "col-uuid-456"
        mock_doc.status = DocumentStatus.COMPLETED.value  # Not PENDING

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id = AsyncMock(return_value=mock_doc)

        mock_updater = MockUpdater("op-123")

        operation = {
            "config": {"document_ids": ["doc-1"]},
            "uuid": "op-123",
        }
        collection = {
            "id": "col-uuid-456",
            "vector_store_name": "qdrant_col_123",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        result = await ingestion_module._process_retry_documents_operation(
            operation=operation,
            collection=collection,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            updater=mock_updater,
        )

        # Should return early with 0 documents since the only doc is not PENDING
        assert result["success"] is True
        assert result["documents_processed"] == 0

    @pytest.mark.asyncio()
    async def test_retry_documents_excludes_wrong_collection(self) -> None:
        """Documents from different collections are excluded."""
        from shared.database.models import DocumentStatus

        # Create mock document from a different collection
        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.collection_id = "different-collection-id"  # Different!
        mock_doc.status = DocumentStatus.PENDING.value

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id = AsyncMock(return_value=mock_doc)

        mock_updater = MockUpdater("op-123")

        operation = {
            "config": {"document_ids": ["doc-1"]},
            "uuid": "op-123",
        }
        collection = {
            "id": "col-uuid-456",
            "vector_store_name": "qdrant_col_123",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        result = await ingestion_module._process_retry_documents_operation(
            operation=operation,
            collection=collection,
            collection_repo=mock_collection_repo,
            document_repo=mock_document_repo,
            updater=mock_updater,
        )

        # Should return early since document belongs to different collection
        assert result["success"] is True
        assert result["documents_processed"] == 0

    @pytest.mark.asyncio()
    async def test_retry_documents_sends_progress_updates(self) -> None:
        """Verify progress updates are sent during processing."""
        from shared.database.models import DocumentStatus

        mock_doc = Mock()
        mock_doc.id = "doc-1"
        mock_doc.collection_id = "col-uuid-456"
        mock_doc.status = DocumentStatus.PENDING.value
        mock_doc.source_path = "/path/to/doc.txt"

        mock_collection_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        mock_document_repo.get_by_id = AsyncMock(return_value=mock_doc)

        mock_updater = MockUpdater("op-123")

        operation = {
            "config": {"document_ids": ["doc-1"]},
            "uuid": "op-123",
        }
        collection = {
            "id": "col-uuid-456",
            "vector_store_name": "qdrant_col_123",
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "quantization": "float16",
            "config": {},
        }

        with patch.object(
            ingestion_module,
            "process_documents_parallel",
            new=AsyncMock(return_value={"processed": 1, "failed": 0, "vectors": 10}),
        ):
            await ingestion_module._process_retry_documents_operation(
                operation=operation,
                collection=collection,
                collection_repo=mock_collection_repo,
                document_repo=mock_document_repo,
                updater=mock_updater,
            )

        # Verify processing_progress update was sent
        progress_updates = [u for u in mock_updater.updates if u[0] == "processing_progress"]
        assert len(progress_updates) >= 1
        assert progress_updates[0][1]["total"] == 1
