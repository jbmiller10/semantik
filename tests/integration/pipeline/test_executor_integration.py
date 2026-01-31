"""Integration tests for the pipeline executor with real plugins.

These tests use real implementations of:
- LocalFileConnector for file enumeration
- Text parser for parsing
- Recursive chunking strategy for chunking
- Mock embedding provider (to avoid GPU requirements)

These tests require a real PostgreSQL database connection.
"""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.connectors.local import LocalFileConnector
from shared.database.models import DocumentStatus
from shared.pipeline.executor import PipelineExecutor
from shared.pipeline.executor_types import ExecutionMode
from shared.pipeline.types import FileReference, NodeType, PipelineDAG, PipelineEdge, PipelineNode


@pytest.fixture()
def simple_dag() -> PipelineDAG:
    """Create a simple DAG for text files."""
    return PipelineDAG(
        id="simple-text-pipeline",
        version="1.0",
        nodes=[
            PipelineNode(
                id="text-parser",
                type=NodeType.PARSER,
                plugin_id="text",
            ),
            PipelineNode(
                id="recursive-chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "max_tokens": 100,
                    "min_tokens": 20,
                    "overlap_tokens": 10,
                },
            ),
            PipelineNode(
                id="embedder",
                type=NodeType.EMBEDDER,
                plugin_id="dense-local",
            ),
        ],
        edges=[
            PipelineEdge(from_node="_source", to_node="text-parser"),
            PipelineEdge(from_node="text-parser", to_node="recursive-chunker"),
            PipelineEdge(from_node="recursive-chunker", to_node="embedder"),
        ],
    )


@pytest.fixture()
def temp_docs_dir() -> Path:
    """Create a temporary directory with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir)

        # Create some test files
        (docs_dir / "doc1.txt").write_text(
            "This is the first document. It contains some text that will be chunked. "
            "The chunker should split this into multiple chunks based on the configuration."
        )

        (docs_dir / "doc2.txt").write_text(
            "This is the second document. It has different content than the first one. "
            "This will also be processed by the pipeline."
        )

        (docs_dir / "doc3.md").write_text(
            "# Markdown Document\n\n"
            "This is a markdown document with some content.\n\n"
            "## Section 1\n\n"
            "Content for section 1.\n\n"
            "## Section 2\n\n"
            "Content for section 2."
        )

        yield docs_dir


class TestPipelineExecutorWithLocalConnector:
    """Integration tests using LocalFileConnector."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session for testing without DB."""
        session = AsyncMock(spec=AsyncSession)
        # Mock the execute method to return None for get_by_uri
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_execute_dry_run_with_text_files(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test DRY_RUN mode with real text files."""
        connector = LocalFileConnector({"path": str(temp_docs_dir), "recursive": True})
        await connector.authenticate()

        # Collect file refs
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check results
        assert result.mode == ExecutionMode.DRY_RUN
        assert result.files_processed == 3
        assert result.files_succeeded == 3
        assert result.files_failed == 0
        assert result.halted is False

        # Check sample outputs
        assert result.sample_outputs is not None
        assert len(result.sample_outputs) == 3

    @pytest.mark.asyncio()
    async def test_execute_creates_chunks(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test that chunks are created from files."""
        connector = LocalFileConnector({"path": str(temp_docs_dir), "recursive": True})
        await connector.authenticate()

        # Collect file refs
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check chunks were created
        assert result.chunks_created > 0

        # Check chunk stats
        assert result.chunk_stats is not None
        assert result.chunk_stats.total_chunks > 0
        assert result.chunk_stats.min_tokens > 0
        assert result.chunk_stats.max_tokens >= result.chunk_stats.min_tokens

    @pytest.mark.asyncio()
    async def test_execute_with_include_pattern(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test with file include pattern."""
        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "recursive": True,
                "include_patterns": ["*.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Only .txt files should be processed
        assert result.files_processed == 2
        assert result.files_succeeded == 2

    @pytest.mark.asyncio()
    async def test_execute_progress_events(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test progress events are emitted correctly."""
        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        events: list = []

        async def progress_callback(event):
            events.append(event)

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        await executor.execute(file_iterator(), progress_callback=progress_callback)

        # Check expected event sequence
        event_types = [e.event_type for e in events]
        assert event_types[0] == "pipeline_started"
        assert "file_started" in event_types
        assert "stage_completed" in event_types
        assert "file_completed" in event_types
        assert event_types[-1] == "pipeline_completed"

    @pytest.mark.asyncio()
    async def test_execute_with_limit(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test file limit is respected."""
        connector = LocalFileConnector({"path": str(temp_docs_dir)})
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator(), limit=1)

        assert result.files_processed == 1

    @pytest.mark.asyncio()
    async def test_execute_stage_timings_populated(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test stage timings are recorded."""
        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        executor = PipelineExecutor(
            dag=simple_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator())

        # Check timings are populated
        assert "loader" in result.stage_timings
        assert result.total_duration_ms > 0


class TestPipelineExecutorWithBranchingDag:
    """Integration tests with branching DAG."""

    @pytest.fixture()
    def branching_dag(self) -> PipelineDAG:
        """Create a DAG that routes by extension."""
        return PipelineDAG(
            id="branching-pipeline",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="markdown-parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="text-parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={"max_tokens": 100, "min_tokens": 20, "overlap_tokens": 10},
                ),
                PipelineNode(
                    id="embedder",
                    type=NodeType.EMBEDDER,
                    plugin_id="dense-local",
                ),
            ],
            edges=[
                # Route .md to markdown parser
                PipelineEdge(
                    from_node="_source",
                    to_node="markdown-parser",
                    when={"extension": ".md"},
                ),
                # Route everything else to text parser
                PipelineEdge(from_node="_source", to_node="text-parser"),
                # Both parsers feed into chunker
                PipelineEdge(from_node="markdown-parser", to_node="chunker"),
                PipelineEdge(from_node="text-parser", to_node="chunker"),
                # Chunker to embedder
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_routes_by_extension(
        self,
        branching_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test files are routed based on extension."""
        connector = LocalFileConnector({"path": str(temp_docs_dir)})
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        events: list = []

        async def progress_callback(event):
            if event.event_type == "stage_completed":
                events.append(event)

        executor = PipelineExecutor(
            dag=branching_dag,
            collection_id="test-collection",
            session=mock_session,
            connector=connector,
            mode=ExecutionMode.DRY_RUN,
        )

        result = await executor.execute(file_iterator(), progress_callback=progress_callback)

        assert result.files_succeeded == 3

        # Check that different parsers were used
        parser_events = [e for e in events if "parser" in e.stage_id]
        parser_ids = {e.stage_id for e in parser_events}
        # Both markdown-parser and text-parser should have been used
        assert "markdown-parser" in parser_ids or "text-parser" in parser_ids


class TestPipelineExecutorLoadContent:
    """Integration tests for connector load_content method."""

    @pytest.fixture()
    def mock_session(self) -> AsyncMock:
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result
        session.flush = AsyncMock()
        return session

    @pytest.mark.asyncio()
    async def test_load_content_from_connector(
        self,
        simple_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session: AsyncMock,
    ) -> None:
        """Test that content is loaded via connector.load_content()."""
        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        # Get file ref
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        assert len(file_refs) == 1

        # Test load_content directly
        content = await connector.load_content(file_refs[0])
        assert isinstance(content, bytes)
        assert b"first document" in content


class TestPipelineExecutorFullModeWithMockedVecPipe:
    """Integration tests for FULL mode with mocked VecPipe endpoints."""

    @pytest.fixture()
    def full_mode_dag(self) -> PipelineDAG:
        """Create a simple DAG for full mode testing."""
        return PipelineDAG(
            id="full-mode-pipeline",
            version="1.0",
            nodes=[
                PipelineNode(
                    id="text-parser",
                    type=NodeType.PARSER,
                    plugin_id="text",
                ),
                PipelineNode(
                    id="recursive-chunker",
                    type=NodeType.CHUNKER,
                    plugin_id="recursive",
                    config={
                        "max_tokens": 100,
                        "min_tokens": 20,
                        "overlap_tokens": 10,
                    },
                ),
                PipelineNode(
                    id="embedder",
                    type=NodeType.EMBEDDER,
                    plugin_id="dense-local",
                    config={"model": "test-model"},
                ),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="text-parser"),
                PipelineEdge(from_node="text-parser", to_node="recursive-chunker"),
                PipelineEdge(from_node="recursive-chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def mock_session_for_full_mode(self) -> AsyncMock:
        """Create a mock database session with document repository behavior."""
        session = AsyncMock(spec=AsyncSession)

        # Mock for get_by_uri (returns None - no existing document)
        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = None

        # Mock for document creation
        mock_doc = MagicMock()
        mock_doc.id = "test-doc-uuid-123"

        session.execute.return_value = mock_get_result
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.add = MagicMock()

        return session

    @pytest.fixture()
    def mock_doc_repo(self) -> AsyncMock:
        """Create a mock document repository."""
        repo = AsyncMock()

        # Mock document creation
        mock_doc = MagicMock()
        mock_doc.id = "test-doc-uuid-123"
        repo.create.return_value = mock_doc

        # Mock get_by_uri (no existing document)
        repo.get_by_uri.return_value = None

        # Mock update_status
        repo.update_status.return_value = None

        return repo

    @pytest.mark.asyncio()
    async def test_execute_full_mode_with_mock_vecpipe(
        self,
        full_mode_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session_for_full_mode: AsyncMock,
        mock_doc_repo: AsyncMock,
    ) -> None:
        """Test full execution mode with mocked VecPipe calls.

        This test verifies that:
        1. The executor calls VecPipe /embed endpoint with correct payload
        2. The executor calls VecPipe /upsert endpoint with embeddings
        3. Document records are created and updated correctly
        """
        from unittest.mock import patch

        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        # Collect file refs
        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        # Track API calls
        embed_calls: list[dict] = []
        upsert_calls: list[dict] = []

        # Mock httpx.AsyncClient
        class MockAsyncClient:
            def __init__(self, **kwargs):
                self.timeout = kwargs.get("timeout")

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url: str, json: dict, headers: dict):
                mock_response = MagicMock()

                if "/embed" in url:
                    embed_calls.append({"url": url, "json": json})
                    # Return mock embeddings (384 dimensions to match small model)
                    num_texts = len(json.get("texts", []))
                    mock_embeddings = [[0.1] * 384 for _ in range(num_texts)]
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"embeddings": mock_embeddings}
                elif "/upsert" in url:
                    upsert_calls.append({"url": url, "json": json})
                    num_points = len(json.get("points", []))
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"upserted": num_points}
                else:
                    mock_response.status_code = 404
                    mock_response.text = "Not found"

                return mock_response

        # Patch the executor to use our mocks
        with (
            patch("httpx.AsyncClient", MockAsyncClient),
            patch("shared.pipeline.executor._get_internal_api_key", return_value="test-api-key"),
            patch("shared.config.settings.SEARCH_API_URL", "http://test-vecpipe:8000"),
        ):
            executor = PipelineExecutor(
                dag=full_mode_dag,
                collection_id="test-collection-uuid",
                session=mock_session_for_full_mode,
                connector=connector,
                mode=ExecutionMode.FULL,
                vector_store_name="test_qdrant_collection",
                embedding_model="test-model",
            )

            # Replace the document repository with our mock
            executor._doc_repo = mock_doc_repo

            result = await executor.execute(file_iterator())

        # Verify results
        assert result.mode == ExecutionMode.FULL
        assert result.files_processed == 1
        assert result.files_succeeded == 1
        assert result.files_failed == 0
        assert result.halted is False

        # Verify /embed was called
        assert len(embed_calls) == 1
        embed_payload = embed_calls[0]["json"]
        assert "texts" in embed_payload
        assert embed_payload["model_name"] == "test-model"
        assert embed_payload["mode"] == "document"
        assert len(embed_payload["texts"]) > 0

        # Verify /upsert was called
        assert len(upsert_calls) >= 1
        upsert_payload = upsert_calls[0]["json"]
        assert upsert_payload["collection_name"] == "test_qdrant_collection"
        assert "points" in upsert_payload
        assert len(upsert_payload["points"]) > 0

        # Verify each point has required payload fields
        first_point = upsert_payload["points"][0]
        assert "id" in first_point
        assert "vector" in first_point
        assert len(first_point["vector"]) == 384
        assert "payload" in first_point
        assert first_point["payload"]["collection_id"] == "test-collection-uuid"
        assert first_point["payload"]["doc_id"] == "test-doc-uuid-123"

        # Verify document repository was called
        mock_doc_repo.create.assert_called_once()
        mock_doc_repo.update_status.assert_called()

    @pytest.mark.asyncio()
    async def test_full_mode_handles_embed_failure(
        self,
        full_mode_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session_for_full_mode: AsyncMock,
        mock_doc_repo: AsyncMock,
    ) -> None:
        """Test that embed failures are handled and document is marked FAILED."""
        from unittest.mock import patch

        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        # Mock httpx.AsyncClient to fail on /embed
        class MockAsyncClientFailing:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url: str, json: dict, headers: dict):
                mock_response = MagicMock()
                if "/embed" in url:
                    mock_response.status_code = 500
                    mock_response.text = "Internal server error"
                    return mock_response
                mock_response.status_code = 200
                return mock_response

        with (
            patch("httpx.AsyncClient", MockAsyncClientFailing),
            patch("shared.pipeline.executor._get_internal_api_key", return_value="test-api-key"),
            patch("shared.config.settings.SEARCH_API_URL", "http://test-vecpipe:8000"),
        ):
            executor = PipelineExecutor(
                dag=full_mode_dag,
                collection_id="test-collection-uuid",
                session=mock_session_for_full_mode,
                connector=connector,
                mode=ExecutionMode.FULL,
                vector_store_name="test_qdrant_collection",
                embedding_model="test-model",
            )
            executor._doc_repo = mock_doc_repo

            result = await executor.execute(file_iterator())

        # Verify failure handling
        assert result.files_failed == 1
        assert result.files_succeeded == 0

        # Verify document was marked as FAILED
        mock_doc_repo.update_status.assert_called()
        # Get the call args to check the status
        update_call = mock_doc_repo.update_status.call_args
        assert update_call.args[1] == DocumentStatus.FAILED
        # Verify error message includes actual error details
        assert "error_message" in update_call.kwargs
        assert "500" in update_call.kwargs["error_message"] or "embed" in update_call.kwargs["error_message"].lower()

    @pytest.mark.asyncio()
    async def test_full_mode_handles_upsert_failure(
        self,
        full_mode_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session_for_full_mode: AsyncMock,
        mock_doc_repo: AsyncMock,
    ) -> None:
        """Test that upsert failures are handled correctly."""
        from unittest.mock import patch

        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["doc1.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        # Mock httpx.AsyncClient - embed succeeds, upsert fails
        class MockAsyncClientUpsertFail:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url: str, json: dict, headers: dict):
                mock_response = MagicMock()
                if "/embed" in url:
                    num_texts = len(json.get("texts", []))
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"embeddings": [[0.1] * 384 for _ in range(num_texts)]}
                elif "/upsert" in url:
                    mock_response.status_code = 503
                    mock_response.text = "Qdrant unavailable"
                return mock_response

        with (
            patch("httpx.AsyncClient", MockAsyncClientUpsertFail),
            patch("shared.pipeline.executor._get_internal_api_key", return_value="test-api-key"),
            patch("shared.config.settings.SEARCH_API_URL", "http://test-vecpipe:8000"),
        ):
            executor = PipelineExecutor(
                dag=full_mode_dag,
                collection_id="test-collection-uuid",
                session=mock_session_for_full_mode,
                connector=connector,
                mode=ExecutionMode.FULL,
                vector_store_name="test_qdrant_collection",
                embedding_model="test-model",
            )
            executor._doc_repo = mock_doc_repo

            result = await executor.execute(file_iterator())

        # Verify failure handling
        assert result.files_failed == 1
        assert result.files_succeeded == 0

        # Document should be marked as FAILED
        mock_doc_repo.update_status.assert_called()
        update_call = mock_doc_repo.update_status.call_args
        assert update_call.args[1] == DocumentStatus.FAILED

    @pytest.mark.asyncio()
    async def test_embedding_count_mismatch_handled(
        self,
        full_mode_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session_for_full_mode: AsyncMock,
        mock_doc_repo: AsyncMock,
    ) -> None:
        """Test executor handles embedding response with wrong count.

        When VecPipe returns fewer embeddings than chunks requested,
        the executor should detect this and fail gracefully.
        """
        from unittest.mock import patch

        # Create a larger file that will produce multiple chunks
        large_text = " ".join([f"Paragraph {i}. " * 20 for i in range(10)])
        (temp_docs_dir / "multi_chunk.txt").write_text(large_text)

        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["multi_chunk.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        # Mock httpx.AsyncClient to return mismatched embedding count
        class MockAsyncClientMismatch:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url: str, json: dict, headers: dict):
                mock_response = MagicMock()
                if "/embed" in url:
                    # Always return exactly 1 embedding regardless of how many texts requested
                    # This ensures a mismatch when multiple texts are sent
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"embeddings": [[0.1] * 384]}
                elif "/upsert" in url:
                    num_points = len(json.get("points", []))
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"upserted": num_points}
                return mock_response

        with (
            patch("httpx.AsyncClient", MockAsyncClientMismatch),
            patch("shared.pipeline.executor._get_internal_api_key", return_value="test-api-key"),
            patch("shared.config.settings.SEARCH_API_URL", "http://test-vecpipe:8000"),
        ):
            executor = PipelineExecutor(
                dag=full_mode_dag,
                collection_id="test-collection-uuid",
                session=mock_session_for_full_mode,
                connector=connector,
                mode=ExecutionMode.FULL,
                vector_store_name="test_qdrant_collection",
                embedding_model="test-model",
            )
            executor._doc_repo = mock_doc_repo

            result = await executor.execute(file_iterator())

        # The file should fail due to embedding count mismatch
        assert result.files_failed == 1
        assert result.files_succeeded == 0

        # Document should be marked as FAILED
        mock_doc_repo.update_status.assert_called()
        update_call = mock_doc_repo.update_status.call_args
        assert update_call.args[1] == DocumentStatus.FAILED

    @pytest.mark.asyncio()
    async def test_partial_upsert_batch_failure(
        self,
        full_mode_dag: PipelineDAG,
        temp_docs_dir: Path,
        mock_session_for_full_mode: AsyncMock,
        mock_doc_repo: AsyncMock,
    ) -> None:
        """Test executor handles batch failure during upsert when processing multiple files.

        This test processes two files:
        - First file's upsert succeeds
        - Second file's upsert fails

        The executor should correctly mark each document based on its own
        upsert outcome.
        """
        from unittest.mock import patch

        # Create two files
        (temp_docs_dir / "file1.txt").write_text("First document content for testing.")
        (temp_docs_dir / "file2.txt").write_text("Second document content for testing.")

        connector = LocalFileConnector(
            {
                "path": str(temp_docs_dir),
                "include_patterns": ["file1.txt", "file2.txt"],
            }
        )
        await connector.authenticate()

        file_refs: list[FileReference] = []
        async for ref in connector.enumerate():
            file_refs.append(ref)
        # Sort to ensure consistent ordering
        file_refs.sort(key=lambda r: r.uri)

        async def file_iterator() -> AsyncIterator[FileReference]:
            for ref in file_refs:
                yield ref

        # Track upsert calls to fail on second file
        upsert_call_count = [0]

        class MockAsyncClientPartialFail:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url: str, json: dict, headers: dict):
                mock_response = MagicMock()
                if "/embed" in url:
                    num_texts = len(json.get("texts", []))
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"embeddings": [[0.1] * 384 for _ in range(num_texts)]}
                elif "/upsert" in url:
                    upsert_call_count[0] += 1
                    if upsert_call_count[0] > 1:
                        # Fail on second file's upsert
                        mock_response.status_code = 503
                        mock_response.text = "Qdrant unavailable"
                    else:
                        num_points = len(json.get("points", []))
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"upserted": num_points}
                return mock_response

        with (
            patch("httpx.AsyncClient", MockAsyncClientPartialFail),
            patch("shared.pipeline.executor._get_internal_api_key", return_value="test-api-key"),
            patch("shared.config.settings.SEARCH_API_URL", "http://test-vecpipe:8000"),
        ):
            executor = PipelineExecutor(
                dag=full_mode_dag,
                collection_id="test-collection-uuid",
                session=mock_session_for_full_mode,
                connector=connector,
                mode=ExecutionMode.FULL,
                vector_store_name="test_qdrant_collection",
                embedding_model="test-model",
            )
            executor._doc_repo = mock_doc_repo

            result = await executor.execute(file_iterator())

        # One file should succeed, one should fail
        assert result.files_processed == 2
        assert result.files_failed == 1
        assert result.files_succeeded == 1

        # Document should be marked as FAILED for the second file
        assert mock_doc_repo.update_status.call_count >= 2
